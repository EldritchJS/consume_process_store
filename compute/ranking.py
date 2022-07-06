import warnings

import collections
import numpy as np


def simple_vote(ID_to_path, query_results, distances=None):
    """
    ID_to_path: from index (dict)
    image_result_dict: map of hashed_feature -> result_tuple
    """
    if distances is not None:
        warnings.warn('distances provided to simple_vote will be ignored.')
    if isinstance(query_results, dict):
        # assume is a query_result_dict
        images = np.concatenate([result_tuple[1]
                                 for result_tuple in query_results.values()])
    elif isinstance(query_results, (list, tuple, np.ndarray)):
        # assume is a list of image IDs
        images = np.asarray(query_results)
    else:
        raise TypeError(f'Unknown type "{type(query_results)}"')

    images.sort()
    idxs_change = np.where(images[:-1] != images[1:])[0]

    voted_images = {}
    idx_prev = 0
    for idx in idxs_change:
        voted_images[images[idx]] = idx + 1 - idx_prev
        idx_prev = idx + 1
    voted_images[images[-1]] = len(images) - idx_prev

    if -1 in voted_images:
        del voted_images[-1]

    return collections.Counter({
        ID_to_path[i]: votes
        for i, votes in voted_images.items()
    })


def simple_weighted_vote(ID_to_path, query_results, distances=None,
                         weighting='tanh'):
    """
    ID_to_path: from index (dict)
    image_result_dict: map of hashed_feature -> result_tuple
    distances: distance of results from query feature
    """
    if isinstance(query_results, dict):
        assert distances is None, ('distances must be None if providing '
                                   'query_results as dict!')
        # assume is a query_result_dict
        images = np.concatenate([result_tuple[1]
                                 for result_tuple in query_results.values()])
        dists = np.concatenate([result_tuple[0]
                                for result_tuple in query_results.values()])
    elif isinstance(query_results, (list, tuple, np.ndarray)):
        assert distances is not None, ('distances must be provided if '
                                       'query_results is not a dict!')
        assert len(query_results) == len(distances)
        # assume is a list of image IDs
        images = np.asarray(query_results)
        dists = np.asarray(distances)
    else:
        raise TypeError(f'Unknown type "{type(query_results)}"')

    if weighting == 'min-max':
        dists -= dists.min()
        dists /= dists.max()
        dists = 1. - dists
    elif weighting == 'tanh':
        dists = 1 - np.tanh(dists)
    elif weighting == 'min-max-tanh':
        dists -= dists.min()
        dists /= dists.max()
        dists = 1 - np.tanh(dists)
    else:
        raise ValueError(f'Unknown weighting type "{weighting}"')

    idxs_sorted = np.argsort(images)
    images = images[idxs_sorted]
    dists = dists[idxs_sorted]

    idxs_change = np.where(images[:-1] != images[1:])[0]

    voted_images = collections.defaultdict(float)
    idx_prev = 0
    for idx in idxs_change:
        image_id = images[idx]
        if image_id == -1:
            continue
        voted_images[ID_to_path[image_id]] += dists[idx_prev:idx + 1].sum()
        idx_prev = idx + 1
    image_id = images[-1]
    if image_id != -1:
        voted_images[ID_to_path[image_id]] += dists[idx_prev:].sum()

    return collections.Counter(voted_images)


def simple_weighted_vote_original(ID_to_path, query_results, distances=None):
    """
    ID_to_path: from index (dict)
    image_result_dict: map of hashed_feature -> result_tuple
    distances: distance of results from query feature
    """
    from math import tanh

    if isinstance(query_results, dict):
        assert distances is None, ('distances must be None if providing '
                                   'query_results as dict!')
        # assume is a query_result_dict
        image_generator = (ID_to_path[i]
                           for result_tuple in query_results.values()
                           for i in result_tuple[1])
        dist_generator = (d
                          for result_tuple in query_results.values()
                          for d in result_tuple[0])
    elif isinstance(query_results, (list, tuple, np.ndarray)):
        assert distances is not None, ('distances must be provided if '
                                       'query_results is not a dict!')
        assert len(query_results) == len(distances)
        # assume is a list of image IDs
        image_generator = (ID_to_path[i] for i in query_results)
        dist_generator = iter(distances)
    else:
        raise TypeError(f'Unknown type "{type(query_results)}"')

    voted_images = collections.defaultdict(float)

    for i, d in zip(image_generator, dist_generator):
        voted_images[i] += 1 - tanh(d)

    return collections.Counter(voted_images)
