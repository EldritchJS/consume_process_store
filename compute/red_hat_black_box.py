import random
import itertools
import ranking
import clustering_functions
import json

from index_class import Index
from datetime import datetime
from glob import glob
from tqdm import tqdm

import torch.multiprocessing as mp
import numpy as np
import networkx as nx



def cluster(G, cluster_type):
    output_file = 'meow.dat'

    if cluster_type == 'hierarchical':
        clustering_functions.hierarchical_clustering(G, output_file)
    elif cluster_type == 'spectral':
        clustering_functions.spectral_clustering(G, output_file)
    elif cluster_type == 'louvain':
        clustering_functions.louvain_clustering(G, output_file)
    elif cluster_type == 'hdbscan':
        clustering_functions.h_dbscan(G, output_file)
    elif cluster_type == 'markov':
        clustering_functions.markov_clustering(G, output_file)
    else:
        print('Unrecognized cluster type...')


def _query_result_to_image(ID_to_path, query_result_dict, simple_vote=True):
    if simple_vote:
        ranking_method = ranking.simple_vote
    else:
        ranking_method = ranking.simple_weighted_vote

    return ranking_method(
        ID_to_path=ID_to_path,
        query_results=query_result_dict,
    )


def features_from_batch(i, batch_image_path_list):
    batch_feature_list, batch_feature_dict = i.features_from_path_list(batch_image_path_list, ID=False)

    return batch_feature_dict


def create_graph(i):
    G = nx.MultiGraph()
    G.add_nodes_from(list(set(i.ID_to_path.values())))

    tqdm.write(f'Created empty multigraph with {len(G.nodes())} images')

    num_queries = 500
    recall = 50
    iterations = 0
    image_list = list(G.nodes())

    while len(list(nx.isolates(G))) > 0:
        tqdm.write(f'Currently {len(list(nx.isolates(G)))} unconnected nodes...')

        try:
            image_list = random.sample(list(nx.isolates(G)), num_queries)
        except ValueError as e:
            image_list = list(nx.isolates(G))

        query_feature_list = []

        image_list = [q.replace(':', '/') for q in image_list]
        query_feature_dict = features_from_batch(i, image_list)
        print(len(query_feature_dict))

        for img, feature_dict in query_feature_dict.items():
            query_feature_list.extend(list(zip(itertools.repeat(img), list(query_feature_dict[img]['feature_dict'].values()))))

        query_result_dict = i.query_index(None, query_feature_list = query_feature_list, gpu=False)

        feature_type = 'surf_mobile'
        batch_edge_tuples = []
        if 'surf' in feature_type:
            for k, v in tqdm(query_result_dict.items(), desc='Computing Edges'):
                for voted_image, votes in _query_result_to_image(i.ID_to_path, v).items():
                    batch_edge_tuples.append((k, voted_image, {'weight': votes}))

            G.add_edges_from(batch_edge_tuples)
        else:
            for query_image, results in query_result_dict.items():
                for feature, result_tuple in results.items():
                    for weight, ID in zip(*result_tuple[:recall]):
                        if ID != -1:
                            G.add_edge(query_image, i.ID_to_path[ID], weight=weight, edge_type='image')

        tqdm.write(f'After Iteration {iterations}, there are {len(list(nx.isolates(G)))} isolates left...')
        iterations += 1

    tqdm.write(f'The graph is connected? {nx.is_connected(G)}, it has {nx.number_connected_components(G)} components.')
    tqdm.write(f'The graph has {len(G.edges)} edges.')

    return G

def full_pipeline(start_date, end_date, root_data_path='./data', output_file=None):
    mp.set_start_method('spawn')
    data_tag = 'ukr'
    tag_size = 16
    feature = 'PHASH'

    root_post_path = root_data_path
    json_path_list = glob(root_post_path + '/**/*.json', recursive=True)

    timely_images = []
    for json_file in tqdm(json_path_list, desc='Filtering posts by date'):
        json_date_str = json_file.split('_')[-1].split('T')[0]
        json_date = datetime.strptime(json_date_str, '%Y-%m-%d')
        if json_date >= start_date and json_date <= end_date:
            image_list = glob('/'.join(json_file.split('/')[0:-1]) + '/*.jpg')
            if image_list:
                timely_images.append(image_list[0])

    i = Index(cache_dir=f'./{root_data_path}/mini_{data_tag}_{feature}_{start_date}_{end_date}_index/', feature_type=feature)
    feature_list, feature_dict = i.features_from_path_list(timely_images, ID=False)

    id_feature_dict = i.ID_features(feature_dict)
    i.feature_list = feature_list
    i.ID_list = list(id_feature_dict.keys())

    i.train_index(None, training_features=np.array(feature_list, dtype=np.float32), write=False)
    i.add_to_index(None, feature_list=np.array(i.feature_list, dtype=np.float32), ids=np.array(i.ID_list), write=False)
    g = create_graph(i)
    cluster_data = cluster(g, 'louvain')

    # if output_file:
    #     with open(output_file, 'w+') as f:
    #         json.dump(cluster_data, f)

    return cluster_data


if __name__ == '__main__':
    start_date = datetime.strptime('2016-05-01', '%Y-%m-%d')
    end_date = datetime.strptime('2023-05-05', '%Y-%m-%d')

    full_pipeline(start_date, end_date, 'clusters.dat')
