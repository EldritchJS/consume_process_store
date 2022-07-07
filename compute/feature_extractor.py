import matplotlib
matplotlib.use("Agg")
import cv2, imagehash, time, gc
import concurrent.futures
import copyreg
import numpy as np
import functools

from tqdm import tqdm
from collections import defaultdict

import torch
import torch.multiprocessing as mp
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from multiprocessing.context import SpawnContext

from PIL import Image


# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# python magic to write custom pickle method for cv2.KeyPoints:
#    https://stackoverflow.com/questions/10045363/pickling-cv2-keypoint-causes-picklingerror/48832618
def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)


copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)


class PytorchContext(SpawnContext):
    def SimpleQueue(self):
        return mp.SimpleQueue()

    def Queue(self, maxsize=0):
        return mp.Queue(maxsize, ctx=self.get_context())


class global_features():
    def __init__(self, image_path, model):
        self.image_features = torch.zeros(576)
        self.image_path = image_path
        self.model = model

    def save_feature(self, m, i, o):
        self.image_features = o.data[0]

    def extract_features(self):
        layer = self.model._modules.get('avgpool')

        h = layer.register_forward_hook(self.save_feature)

        yeet = Image.open(self.image_path).convert('RGB')
        self.model.eval()
        scaler = transforms.Resize([224, 224])
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        transform = transforms.Compose([to_tensor, scaler, normalize])

        yote = transform(yeet).unsqueeze(0)

        self.model(yote.to(0))
        h.remove()

        return self.image_features


hessianThreshold = 100
nOctaves = 6
nOctaveLayers = 5
extended = False
upright = False
keepTopNCount = 2500
distanceThreshold = 50
global_context = 16


def extract_image_features(image_path, feature_type, marriage_type=None, model=None):
    keypoints, features, w, h = feature_detection_and_description(image_path, feature_type, model=model)

    if not np.array(features).any():
        return None

    if marriage_type:
        _, g_features, _, _ = feature_detection_and_description(image_path, feature_type=marriage_type, model=model)
        # return {'keypoints': keypoints, 'feature_dict': dict(zip(np.arange(0, len(features) + 1, 1), features)), 'img_size': (w, h), 'global_feature': g_features[0]}
        return {'feature_dict': dict(zip(range(0, len(features) + 1, 1), features)), 'img_size': (w, h), 'global_feature': g_features[0]}

    return {'feature_dict': dict(zip(range(0, len(features) + 1, 1), features)), 'img_size': (w, h)}
    # return {'keypoints': keypoints, 'feature_dict': dict(zip(np.arange(0, len(features) + 1, 1), features)), 'img_size': (w, h)}


def marry_features(feature_dict, marriage_length):
    st_t = time.time()
    # print(feature_dict)
    global_feature_list = []
    global_feature_list = [(p, x['global_feature']) for p, x in feature_dict.items()]
    global_feature_list = [(p, x) for p, x in zip(list(zip(*global_feature_list))[0], StandardScaler().fit_transform(list(zip(*global_feature_list))[1]))]
    pca = PCA(n_components=marriage_length)
    global_feature_list = [(p, x) for p, x in zip(list(zip(*global_feature_list))[0], pca.fit_transform(list(zip(*global_feature_list))[1]))]
    global_feature_dict = dict(global_feature_list)
    ed_t = time.time()
    print(f'PCA took {ed_t - st_t}')

    st_t = time.time()
    raw_feature_list = []
    for img_path, img_dict in tqdm(feature_dict.items(), desc='Images to marry'):
        # for dummy_id, feat in img_dict['feature_dict'].items():
            # if len(feat) != 64:
            #     continue
            # married_feat = np.append(feat, global_feature_dict[img_path])
        feature_dict[img_path]['feature_dict'] = {d_id: np.array(np.append(feat, global_feature_dict[img_path]), dtype=np.float32) for d_id, feat in feature_dict[img_path]['feature_dict'].items()}
            # feature_dict[img_path]['feature_dict'][dummy_id] = np.array(married_feat, dtype=np.float32)
        raw_feature_list.extend(list(feature_dict[img_path]['feature_dict'].values()))
        gc.collect(0)
    ed_t = time.time()
    print(f'Marrying (dict_comp, gc(0)) took {ed_t - st_t}')

    return feature_dict, raw_feature_list


def parallel_feature_extraction(image_path_list, feature_type, marriage_type, marriage_length, PE=1):
    # def create_feature_dict():
    #     return defaultdict(dict)
    # feature_dict = defaultdict(lambda: defaultdict(dict))
    # feature_dict = {{{}}}
    feature_dict = {}
    raw_feature_list = []

    model = None
    if marriage_type == 'MOBILE' or feature_type == 'MOBILE':
        model = models.mobilenet_v3_small(pretrained=True)
        model.share_memory()
        model.to(0)
    elif marriage_type == 'DINO' or feature_type == 'DINO':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        model.share_memory()
        # model.to(0)
    # elif feature_type == 'VGG':

    marriage_extract = functools.partial(extract_image_features, feature_type=feature_type, marriage_type=marriage_type, model=model)

    with concurrent.futures.ProcessPoolExecutor(max_workers=PE, mp_context=mp) as executor:
        for image_path, img_data_dict in tqdm(zip(image_path_list, executor.map(marriage_extract, image_path_list)), total=len(image_path_list)):
            if img_data_dict == None:
                continue
            feature_dict[image_path.replace('/', ':')] = img_data_dict
            raw_feature_list.extend(list(img_data_dict['feature_dict'].values()))

    # global_feature_list = []
    # if marriage_type:
    #     print("[LOG]: Marrying...")
    #     global_feature_list = [(p, x['feature_dict'][len(x['feature_dict']) - 1]) for p, x in feature_dict.items()]
    #     global_feature_list = [(p, x) for p, x in zip(list(zip(*global_feature_list))[0], StandardScaler().fit_transform(list(zip(*global_feature_list))[1]))]
    #     pca = PCA(n_components=marriage_length)
    #     global_feature_list = [(p, x) for p, x in zip(list(zip(*global_feature_list))[0], pca.fit_transform(list(zip(*global_feature_list))[1]))]
    #     global_feature_dict = dict(global_feature_list)
    #     print("[LOG]: PCA finished, appending...")

    #     raw_feature_list = []
    #     for img_path, img_dict in tqdm(feature_dict.items(), desc='Images to marry'):
    #         # extended_feature_dict = {}
    #         # gc.collect()
    #         for dummy_id, feat in tqdm(img_dict['feature_dict'].items(), desc='Features to marry'):
    #             if len(feat) != 64:
    #                 continue
    #             married_feat = np.append(feat, global_feature_dict[img_path])
    #             feature_dict[img_path]['feature_dict'][dummy_id] = np.array(married_feat, dtype=np.float32)
    #             # extended_feature_dict[dummy_id] = np.array(married_feat, dtype=np.float32)
    #             raw_feature_list.append(list(married_feat))

    #         # feature_dict[img_path]['feature_dict'] = extended_feature_dict.copy()

    return feature_dict, raw_feature_list

def feature_detection(imgpath, img, detetype, gpu=True, kmax=500):
    # try:
    if detetype == "SURF3" or detetype == 'MARRIAGE':
        st_t = time.time()
        # detects the SURF keypoints, with very low Hessian threshold

        if not gpu:
            surf = cv2.xfeatures2d.SURF_create(hessianThreshold=10, nOctaves=nOctaves, nOctaveLayers=nOctaveLayers,
                                                   extended=extended, upright=upright)
            keypoints = surf.detect(img)
        else:
            surf_gpu = cv2.cuda.SURF_CUDA_create(_hessianThreshold=10, _nOctaves=nOctaves, _nOctaveLayers=nOctaveLayers,
                                                   _extended=extended, _upright=upright)

            def upscale_image(img):
                return cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_AREA)

            upscales = 0
            detect_worked = False
            while not detect_worked:
                try:
                    gpu_img = cv2.cuda_GpuMat()
                    gpu_img.upload(img)
                    keypoints_gpu = surf_gpu.detect(gpu_img, None)
                except:
                    #tqdm.write('trying to upscale')
                    img = upscale_image(img)
                    upscales += 1
                    if upscales > 4:
                        tqdm.write('Failed upscales, skipping')
                        return [], -1
                else:
                    detect_worked = True

            keypoints = cv2.cuda_SURF_CUDA.downloadKeypoints(surf_gpu, keypoints_gpu)

        # sorts the keypoints according to their Hessian value
        keypoints = sorted(keypoints, key=lambda match: match.response, reverse=True)

        # obtains the positions of the keypoints within the described image
        positions = []
        for kp in keypoints:
            positions.append((kp.pt[0], kp.pt[1]))
        positions = np.array(positions).astype(np.float32)

        # selects the keypoints based on their positions and distances
        selectedKeypoints = []
        selectedPositions = []

        if len(keypoints) > 0:
            # keeps the top-n strongest keypoints
            for i in range(min(keepTopNCount,len(keypoints))):
                selectedKeypoints.append(keypoints[i])
                selectedPositions.append(positions[i])

                # if the amount of wanted keypoints was reached, quits the loop
                if len(selectedKeypoints) >= kmax:
                    break

            selectedPositions = np.array(selectedPositions)

            # adds the remaining keypoints according to the distance threshold,
            # if the amount of wanted keypoints was not reached yet
            # print('selected keypoints size: ', len(selectedKeypoints), ' kmax: ',kmax)
            if len(selectedKeypoints) < kmax:
                matcher = cv2.BFMatcher()
                for i in range(keepTopNCount, positions.shape[0]):
                    currentPosition = [positions[i]]
                    currentPosition = np.array(currentPosition)

                    match = matcher.match(currentPosition, selectedPositions)[0]
                    if match.distance > distanceThreshold:
                        selectedKeypoints.append(keypoints[i])
                        selectedPositions = np.vstack((selectedPositions, currentPosition))

                    # if the amount of wanted keypoints was reached, quits the loop
                    if len(selectedKeypoints) >= kmax:
                        break;
            keypoints = selectedKeypoints
        ed_t = time.time()

    elif detetype == "PHASH":
        st_t = time.time()
        h, w = img.shape
        x_center, y_center = int(w / 2), int(h / 2)
        keypoints = [cv2.KeyPoint(x=x_center, y=y_center, _size=1, _angle=0)]
        ed_t = time.time()

    elif detetype == "VGG":
        st_t = time.time()
        h, w = img.shape
        x_center, y_center = int(w / 2), int(h / 2)
        keypoints = [cv2.KeyPoint(x=x_center, y=y_center, _size=1, _angle=0)]
        ed_t = time.time()

    elif detetype == 'MOBILE':
        return [0], -1

    elif detetype == 'DINO':
        return [0], -1

    # except Exception as e:
    #     return [], -1

    # det_t = ed_t - st_t
    return keypoints, img

def feature_description(img, img_path, kp, desc_type, gpu=True, model=None):
    new_kp = []

    # try:
    if desc_type == "SURF3" or desc_type == 'MARRIAGE':
        st_t = time.time()
        if not gpu:
            surf = cv2.xfeatures2d.SURF_create(hessianThreshold=10, nOctaves=nOctaves, nOctaveLayers=nOctaveLayers,
                                                       extended=extended, upright=upright)
            __, features = surf.compute(img, kp)
        else:
            surf_gpu = cv2.cuda.SURF_CUDA_create(_hessianThreshold=10, _nOctaves=nOctaves, _nOctaveLayers=nOctaveLayers,
                                                   _extended=extended, _upright=upright)
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)

            try:
                features_gpu = surf_gpu.usefulDetectDescribe(gpu_img, mask=None, keypoints=kp, useProvidedKeypoints=True)
                # gpu_img.free()
                features = np.reshape(features_gpu, (-1, 64))
            except:
                pass

        ed_t = time.time()

    elif desc_type == "PHASH":
        st_t = time.time()
        features = imagehash.phash(Image.fromarray(img))
        features = [np.array([float(ord(c) * 0.001) for c in list(str(features))], dtype=np.float32)]
        ed_t = time.time()

    elif desc_type == "VGG":
        import tensorflow as tf
        from keras.preprocessing import image
        from keras.applications.vgg19 import preprocess_input
        from keras import backend as kBackend
        from keras.applications.vgg19 import VGG19

        model = VGG19(weights='imagenet', include_top=False, pooling='avg')

        core_config = tf.compat.v1.ConfigProto()
        core_config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=core_config)
        kBackend.set_session(session)
        st_t = time.time()

        i = image.load_img(img_path, target_size=(244, 244))
        i_data = image.img_to_array(i)
        i_data = np.expand_dims(i_data, axis=0)
        i_data = preprocess_input(i_data)

        features = model.predict(i_data)
        session.close()
        kBackend.clear_session()

        ed_t = time.time()


    elif desc_type == 'MOBILE':
        st_t = time.time()
        g_f = global_features(image_path=img_path, model=model)
        try:
            features = [g_f.extract_features().cpu().numpy().flatten()]
        except:
            features = []
        ed_t = time.time()

    elif desc_type == 'DINO':
        st_t = time.time()
        preprocess = transforms.Compose([
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        # features = model(preprocess(img).unsqueeze(0))

        try:
            with Image.open(img_path) as sneed:
                features = model(preprocess(sneed).unsqueeze(0))
            features = [features.cpu().detach().numpy().flatten()]
        except:
            features = []
        ed_t = time.time()
    # except:
    #     print("[ERROR]: Failure in describing keypoints...")
    #     return [], -1, 0,[]

    dsc_t = ed_t - st_t
    return features, dsc_t, 1, new_kp

def feature_detection_and_description(img_path, feature_type, kmax=500, img=None, gpu=True, model=None):
    keyps = None
    det_t=0

    if not img:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    keyps, img = feature_detection(img_path, img, feature_type, gpu=gpu, kmax=kmax)

    if not keyps:
        tqdm.write(f'[ERROR]: Failed to detect keypoints for {img_path}')
        return [], [], None, None

    feat, dsc_t, success, keyps2 = feature_description(img, img_path, keyps, feature_type, gpu=gpu, model=model)

    if keyps is None or len(keyps) == 0:
        keyps= keyps2

    if keyps == []:
        keyps = keyps2
    if feat == []:
        return keyps, [], None, None

    return keyps, feat, det_t, dsc_t
