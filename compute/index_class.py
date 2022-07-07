import os, faiss, time, collections, pickle, gc, json, h5py

import orjson

import numpy as np

from tqdm import tqdm
from inspect import currentframe, getframeinfo
from debug import d_print
from pathlib import Path

from feature_extractor import parallel_feature_extraction


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Index:
    preproc_str = "OPQ8_32"
    ivf_str = "IVF256"
    pqflat_str = "PQ16"

    ngpu = 1
    nprobe = 8
    tempmem = -1
    ncent = 256

    lean = False

    feature_type = 'SURF3'
    marriage_type = None
    marriage_length = 16

    def __init__(self, gpu=False, cache_dir=None, feature_type='SURF3', marriage_type=None, marriage_length=16):
        self.preproc = None
        self.coarse_quantizer = None
        self.trained_index = None

        self.ID_counter = -1
        self.ID_list = []
        self.feature_to_ID = {} # this dict isn't computed
        self.ID_to_path = {}
        #self.path_to_ID = {} #
        #self.ID_to_feature = {} #
        self.path_to_num_features = {} #
        self.feature_dict = {}
        self.feature_list = []
        self.global_feature_list = []

        self.feature_type = feature_type
        self.marriage_type = marriage_type
        if self.feature_type == 'SURF3':
            self.d = 64
        elif self.feature_type == 'PHASH':
            self.d = 16
        elif self.feature_type == 'VGG':
            self.d = 512
        elif self.feature_type == 'MOBILE':
            self.d = 576
        elif self.feature_type == 'DINO':
            self.d = 2048
        elif self.feature_type == 'MARRIAGE':
            if not self.marriage_type:
                print("You've set a feature_type of marriage without specifying the marriage_type")
                exit(420)
            self.feature_type = 'SURF3'
            self.marriage_length = marriage_length
            self.d = 64 + self.marriage_length
        else:
            print('Unrecognized feature type')
            exit(69)

        self.cache_dir = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        if self.cache_dir:
            self.preproc_file = os.path.join(self.cache_dir, 'preproc.cache')
            self.coarse_quant_file = os.path.join(self.cache_dir, 'coarse_quantizer.cache')
            self.trained_index_file = os.path.join(self.cache_dir, 'trained_index.cache')
            self.ID_to_path_file = os.path.join(self.cache_dir, 'ID_to_path.cache')
            self.ID_list_file = os.path.join(self.cache_dir, 'ID_list.cache')
            self.ID_counter_file = os.path.join(self.cache_dir, 'ID_counter.cache')
            self.feature_to_ID_file = os.path.join(self.cache_dir, 'feature_to_ID.cache')
            self.feature_list_file = os.path.join(self.cache_dir, 'feature_list.cache')
            self.feature_dict_file = os.path.join(self.cache_dir, 'feature_dict.cache')
            self.feature_dict_file_pointer = None

        self.gpu = gpu
        if self.gpu:
            self.res = faiss.StandardGpuResources()

    def ID_features(self, feature_dict):
        def adder(feature):
            self.ID_counter += 1
            return self.ID_counter, feature

        ID_list = []
        feature_list = []

        for img_path, img_data_dict in tqdm(feature_dict.items(), desc='IDing img features'):
            feature_dict[img_path]['keypoints'] = []
            feature_dict[img_path]['feature_dict'] = {t_c: feature for t_c, feature in map(adder, img_data_dict['feature_dict'].values())}
            for k in list(feature_dict[img_path]['feature_dict'].keys()):
                self.ID_to_path[k] = img_path
            ID_list.extend(list(feature_dict[img_path]['feature_dict'].keys()))
            feature_list.extend(list(feature_dict[img_path]['feature_dict'].values()))
            gc.collect(0)

        return {id: feat for id, feat in zip(ID_list, feature_list)}

    def features_from_path_list(self, path_list, ID=False):
        d_print('LOG', getframeinfo(currentframe()), 'Extracting image features from a list')

        all_img_data_dict, raw_feature_list = parallel_feature_extraction(path_list, self.feature_type, self.marriage_type, self.marriage_length)
        d_print('LOG', getframeinfo(currentframe()), 'Feature extraction complete')
        # self.feature_dict = copy.deepcopy(all_img_data_dict)

        if ID:
            d_print('LOG', getframeinfo(currentframe()), 'IDing images')
            self.ID_features(all_img_data_dict)

        # self.feature_dict = all_img_data_dict

        d_print('LOG', getframeinfo(currentframe()), 'Concatenating features for training')
        print(np.array(raw_feature_list).shape)

        # self.feature_list = np.array(raw_feature_list, dtype=np.float32)
        # self.feature_list = raw_feature_list
        return raw_feature_list, all_img_data_dict


    def train_index(self, image_list, training_features=None, write=True):
        if image_list:
            training_features = self.features_from_path_list(image_list, ID=True)

        d_print('LOG', getframeinfo(currentframe()), 'Training preprocessor')
        s_t = time.time()
        if self.preproc_str.startswith('OPQ'):
            fi = self.preproc_str[3:].split('_')
            m = int(fi[0]) #number of subspaces decomposed (subspaces_outputdimension)
            dout = int(fi[1]) if len(fi) == 2 else d #output dimension should be a multiple of the number of subspaces?

            self.preproc = faiss.OPQMatrix(self.d, m, dout)

            # if self.gpu:
            #     self.preproc = faiss.index_cpu_to_gpu(self.res, 0, self.preproc)

            self.preproc.train(training_features)

        d_print('LOG', getframeinfo(currentframe()), 'Training coarse quantizer')
        # Train the coarse quantizer centroids
        if self.preproc:
            nt = max(10000, 256 * self.ncent)
            d = self.preproc.d_out
            clus = faiss.Clustering(d, self.ncent)
            clus.verbose = True
            clus.min_points_per_centroid = 1
            clus.max_points_per_centroid = 10000000

            x = self.preproc.apply_py(training_features)
            index = faiss.IndexFlatL2(d)
            clus.train(x, index)
            centroids = faiss.vector_float_to_array(clus.centroids).reshape(self.ncent, d)

            if self.gpu:
                self.coarse_quantizer = faiss.GpuIndexFlatL2(self.res, self.preproc.d_out)
            else:
                self.coarse_quantizer = faiss.IndexFlatL2(self.preproc.d_out)
            self.coarse_quantizer.add(centroids)

        d_print('LOG', getframeinfo(currentframe()), 'Training index')
        # Train the codebooks for the index model
        if self.preproc and self.coarse_quantizer:
            d = self.preproc.d_out
            m = int(self.pqflat_str[2:])

            self.trained_index = faiss.IndexIVFPQ(self.coarse_quantizer, d, self.ncent, m, 8)

            if self.gpu:
                self.trained_index = faiss.index_cpu_to_gpu(self.res, 0, self.trained_index)

            x = self.preproc.apply_py(training_features)
            self.trained_index.train(x)

        d_print('LOG', getframeinfo(currentframe()), f'Training finished in {time.time() - s_t}')

        if self.cache_dir and write:
            self.write_index()

    def add_to_index(self, image_list, write=True, feature_list=None, ids=None):
        d_print('LOG', getframeinfo(currentframe()), 'Adding images to index')
        s_t = time.time()

        if image_list:
            feature_list = self.features_from_path_list(image_list, ID=True)
            self.trained_index.add_with_ids(self.preproc.apply_py(feature_list), np.array(self.ID_list))
        else:
            self.trained_index.add_with_ids(self.preproc.apply_py(feature_list), ids)

        d_print('LOG', getframeinfo(currentframe()), f'Adding finished in {time.time() - s_t}')

        if self.cache_dir and write:
            self.write_index()

    def query_index(self, image_list, query_feature_list=None, recall=1000, gpu=False):
        if not self.trained_index:
            d_print('LOG', getframeinfo(currentframe()), 'Reading index from cache')
            self.preproc = faiss.read_index(self.preproc_file)
            self.trained_index = faiss.read_index(self.trained_index_file)

        query_feature_list_paths = list(zip(*query_feature_list))[0]
        # query_feature_list = np.array(list(zip(*query_feature_list))[1])
        query_feature_list_ = [np.array(q, dtype=np.float32) for q in list(zip(*query_feature_list))[1]]

        # query_feature_list = np.array([np.array(q, dtype=np.float32) for q in query_feature_list])

        if image_list:
            d_print('LOG', getframeinfo(currentframe()), 'Extracting image features for querying')
            query_feature_list = self.features_from_path_list(image_list)

        if gpu:
            if gpu and not self.gpu:
                self.res = faiss.StandardGpuResources()
                self.trained_index = faiss.index_cpu_to_gpu(self.res, 0, self.trained_index)
                self.gpu = gpu

        d_print('LOG', getframeinfo(currentframe()), 'Querying image features')
        s_t = time.time()
        # print(self.preproc.apply_py(np.array(query_feature_list_)).shape)
        # exit(69)
        D, I = self.trained_index.search(self.preproc.apply_py(np.array(query_feature_list_)), recall)
        d_print('LOG', getframeinfo(currentframe()), f'Querying finished in {time.time() - s_t}')

        query_path_result_dict = collections.defaultdict(dict)
        feature_count = 0
        for image, feature, d, i in tqdm(list(zip(query_feature_list_paths, query_feature_list, D, I)), desc='Cleaning query results'):
            query_path_result_dict[image][feature_count] = (d, i)
            feature_count += 1

        return query_path_result_dict

    def query_result_to_image(self, image_result_dict):
        voted_images = collections.Counter()

        for hashed_feature, result_tuple in image_result_dict.items():
            for i in result_tuple[1]:
                voted_images[self.ID_to_path[i]] += 1

        return voted_images

    # computes path_to_num_features for simplicity
    def compute_path_to_num_features(self):
        for path, value in self.feature_dict.items():
            self.path_to_num_features[path] = len(value['feature_dict'])

    def read_index(self, lean=False):
        self.lean = lean

        d_print('LOG', getframeinfo(currentframe()), 'Reading index cache files', l2='Reading coarse quantizer')
        try:
            self.coarse_quantizer = faiss.deserialize_index(np.load(self.coarse_quant_file + '.npy', allow_pickle=True))
        except:
            self.coarse_quantizer = None
            d_print('WARNING', getframeinfo(currentframe()), 'Reading index cache files', l2='No coarse_quant_file, setting to None')
        d_print('LOG', getframeinfo(currentframe()), 'Reading index cache files', l2='Reading preprocessor')
        try:
            self.preproc = faiss.read_VectorTransform(self.preproc_file)
        except:
            self.preproc = None
            d_print('WARNING', getframeinfo(currentframe()), 'Reading index cache files', l2='No preproc_file, setting to None')
        d_print('LOG', getframeinfo(currentframe()), 'Reading index cache files', l2='Reading trained index')
        try:
            self.trained_index = faiss.read_index(self.trained_index_file)
        except:
            self.trained_index = None
            d_print('WARNING', getframeinfo(currentframe()), 'Reading index cache files', l2='No trained_index_file, setting to None')
        d_print('LOG', getframeinfo(currentframe()), 'Reading index cache files', l2='Reading ID_to_path')
        with open(self.ID_to_path_file, 'rb') as f: self.ID_to_path = pickle.load(f)
        d_print('LOG', getframeinfo(currentframe()), 'Reading index cache files', l2='Reading ID_list')
        with open(self.ID_list_file, 'rb') as f: self.ID_list = pickle.load(f)
        d_print('LOG', getframeinfo(currentframe()), 'Reading index cache files', l2='Reading ID_counter')
        with open(self.ID_counter_file, 'rb') as f: self.ID_counter = pickle.load(f)
        d_print('LOG', getframeinfo(currentframe()), 'Reading index cache files', l2='Reading feature_list')
        with open(self.feature_to_ID_file, 'rb') as f: self.feature_to_ID = pickle.load(f)
        try:
            self.h5_read_index(lean)
        except:
            d_print('LOG', getframeinfo(currentframe()), 'Reading index cache files', l2='Reading feature_list')
            with open(self.feature_list_file, 'rb') as f: self.feature_list = pickle.load(f)
            d_print('LOG', getframeinfo(currentframe()), 'Reading index cache files', l2='Reading feature_dict')
            with open(self.feature_dict_file, 'rb') as f: self.feature_dict = pickle.load(f)
        # self.d = len(self.feature_list[0])
        self.d = 80
        # self.compute_path_to_num_features()

    def write_index(self):
        d_print('LOG', getframeinfo(currentframe()), 'Writing index cache files', l2='Writing trained preprocessor')
        if self.preproc: faiss.write_VectorTransform(self.preproc, self.preproc_file)
        d_print('LOG', getframeinfo(currentframe()), 'Writing index cache files', l2='Writing coarse quantizer')
        if self.coarse_quantizer: np.save(self.coarse_quant_file, faiss.serialize_index(self.coarse_quantizer))
        d_print('LOG', getframeinfo(currentframe()), 'Writing index cache files', l2='Writing trained index')
        if self.trained_index: faiss.write_index(self.trained_index, self.trained_index_file)
        d_print('LOG', getframeinfo(currentframe()), 'Writing index cache files', l2='Writing ID_to_path')
        with open(self.ID_to_path_file, 'wb') as f: pickle.dump(self.ID_to_path, f)
        d_print('LOG', getframeinfo(currentframe()), 'Writing index cache files', l2='Writing ID_list')
        with open(self.ID_list_file, 'wb') as f: pickle.dump(self.ID_list, f)
        d_print('LOG', getframeinfo(currentframe()), 'Writing index cache files', l2='Writing ID_counter')
        with open(self.ID_counter_file, 'wb') as f: pickle.dump(self.ID_counter, f)
        d_print('LOG', getframeinfo(currentframe()), 'Writing index cache files', l2='Writing ID_counter')
        with open(self.feature_to_ID_file, 'wb') as f: pickle.dump(self.feature_to_ID, f)
        self.h5_write_index()
        # try:
        #     self.h5_write_index()
        # except:
        #     d_print('LOG', getframeinfo(currentframe()), 'Writing index cache files', l2='Writing feature_list')
        #     with open(self.feature_list_file, 'wb') as f: pickle.dump(self.feature_list, f, protocol=4)
        #     d_print('LOG', getframeinfo(currentframe()), 'Writing index cache files', l2='Writing feature_dict')
        #     with open(self.feature_dict_file, 'wb') as f: dill.dump(self.feature_dict, f)


    def h5_write_index(self):
        d_print('LOG', getframeinfo(currentframe()), 'Writing index cache files', l2='Writing feature_list (h5)')
        hf = h5py.File(self.feature_list_file, 'w')
        print('sneed', len(self.feature_list))
        hf.create_dataset('feature_list', data=np.array(self.feature_list))
        hf.close()
        # with open(self.feature_list_file, 'wb') as f: pickle.dump(self.feature_list, f, protocol=4)
        d_print('LOG', getframeinfo(currentframe()), 'Writing index cache files', l2='Writing feature_dict (h5)')
        # s_t = time.time()
        # self.feature_dict = orjson.loads(json.dumps(self.feature_dict, cls=NumpyEncoder) + '\n')
        # e_t = time.time()
        # print(f'Done un-defaultdicting ({e_t - s_t})')
        # s_t = time.time()
        # dd.io.save(self.feature_dict_file, self.feature_dict, compression='blosc')
        # e_t = time.time()
        # print(f'Done deepdishing ({e_t - s_t})')
        if not self.lean:
            if self.feature_dict_file_pointer:
                self.feature_dict_file_pointer.close()
            else:
                with open(self.feature_dict_file, 'w') as f:
                    json.dump(self.feature_dict, f, cls=NumpyEncoder)
            # f.write(json.dumps(self.feature_dict, cls=NumpyEncoder) + '\n')
        # dicttoh5(self.feature_dict, self.feature_dict_file)

    def h5_read_index(self, lean):
        d_print('LOG', getframeinfo(currentframe()), 'Reading index cache files', l2='Reading feature_list (h5)')
        hf = h5py.File(self.feature_list_file, 'r')
        # print(hf.keys())
        self.feature_list = np.array(hf.get('feature_list'))
        hf.close()

        if not lean:
            d_print('LOG', getframeinfo(currentframe()), 'Reading index cache files', l2='Reading feature_dict (h5)')
        # self.feature_dict = json_stream.load(self.feature_dict_file, persistent=True)
        # f = open(self.feature_dict_file, 'rb')
        # self.feature_dict_file_pointer = f
            with open(self.feature_dict_file, 'rb') as f:
                self.feature_dict = orjson.loads(f.readline())
        # with open(self.feature_dict_file, 'r') as f:
        #     self.feature_dict = ujson.load(f)
        # self.feature_dict = json_stream.load(f, persistent=True)
        # with mmap.mmap(self.feature_dict_file_pointer.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
            # self.feature_dict = json_stream.load(mmap_obj, persistent=True)
            # mmap_obj.seek(0)
            # file_size = os.path.getsize(self.feature_dict_file)
            # json_gen = (mmap_obj.read_byte() for b in tqdm(range(0, file_size)))
            # buffer_for_json = mmap_obj[:file_size]

            # Content is JSON, so load it
            # self.feature_dict = orjson.loads(buffer_for_json.decode("utf-8"))
            # self.feature_dict = orjson.loads(buffer_for_json.decode("utf-8"))
            # self.feature_dict = orjson.loads(memoryview(mmap_obj))

        # try:
        #     s_t = time.time()
        #     self.feature_dict = dd.io.load(self.feature_dict_file)
        #     e_t = time.time()
        #     print(f'Done undeepdishing ({e_t - s_t})')
        # except:
        #     # f = open(self.feature_dict_file, 'rb')
        #     # self.feature_dict_file_pointer = f
        #     # self.feature_dict = bigjson.load(f)
        #     with open(self.feature_dict_file, 'r') as f:
                # self.feature_dict = json.load(f)
        # print(type(self.feature_dict))
        # self.feature_dict = dicttoh5(self.feature_dict_file)
