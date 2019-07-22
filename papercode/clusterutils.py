"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


def get_silhouette_scores(features: np.ndarray) -> Tuple[dict]:
    """Calculate the mean and min silhouette score of a range of possible cluster numbers

    For this paper we only consider possible numbers of clusters from 3 to 14.
    
    Parameters
    ----------
    features : np.ndarray
        Array containing the input features.
    
    Returns
    -------
    mean_scores : dict
        Dictionary containing the mean silhouette scores for each cluster number
    min_scores : dict
        Dictionary containing the min silhouette scores for each cluster number
    """
    mean_scores, min_scores = {}, {}
    min_scores = {}
    for n_clusters in range(3, 15, 1):
        clusterer = KMeans(n_clusters=n_clusters, random_state=0, init='k-means++')
        cluster_labels = clusterer.fit_predict(features)
        silhouette_scores = silhouette_samples(X=features,
                                               labels=cluster_labels,
                                               metric="euclidean")
        mean_scores[n_clusters] = np.mean(silhouette_scores)
        min_scores[n_clusters] = np.min(silhouette_scores)

    return mean_scores, min_scores


def get_clusters(lstm_features: Dict, raw_features: pd.DataFrame, ks: List,
                 basins: List[str]) -> Dict:
    """[summary]
    
    Parameters
    ----------
    lstm_features : Dict
        Dictionary containing the LSTM catchment embedding for each basin.
    raw_features : pd.DataFrame
        Dataframe, containing the normalized catchment attributes per basin.
    ks : List
        List of cluster numbers to run KMeans for
    basins : List[str]
        List of 8-digit USGS basin ids.
    
    Returns
    -------
    Dict
        Dictionary containing the cluster labels for each k in ks, both set of input features and
        for each basin.
    """
    clusters = {k: defaultdict(dict) for k in ks}
    for k in ks:
        for name in ['lstm', 'raw']:
            if name == "lstm":
                features = np.array(list(lstm_features.values()))[:, 0, :]
            else:
                features = raw_features.values
            clusterer = KMeans(n_clusters=k, random_state=0, init='k-means++',
                               n_init=200).fit(features)
            for basin in basins:
                if name == 'lstm':
                    emb = lstm_features[basin]
                else:
                    emb = raw_features.loc[raw_features.index == basin].values
                clusters[k][name][basin] = clusterer.predict(emb.reshape(1, -1))[0]
    return clusters


def get_label_2_color(lstm_clusters: Dict, raw_clusters: Dict) -> defaultdict:
    """Helper function to match colors between cluster results.

    This function tries to match the colors of different cluster results, by comparing the number of 
    shared basins between to clusters. This is not a bullet-proof algorithm but works for our plots.

    Basically what is does is to take the results of one clusterer and then compare a second one by
    finding the cluster label of the second clusterer that has the most basins in common. Then 
    assigning both labels the same color.
    
    Parameters
    ----------
    lstm_feats : Dict
        Cluster labels for the LSTM embeddings
    raw_feats : Dict
        Cluster labels for the raw catchment attributes
    
    Returns
    -------
    defaultdict
        Dictionary that contains a mapping from label number to color for both cluster results.
    """
    color_list = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#e6ab02', '#66a61e']
    label_2_color = defaultdict(dict)
    basin_in_cluster = {'lstm': defaultdict(list), 'raw': defaultdict(list)}
    for basin, label in lstm_clusters.items():
        basin_in_cluster["lstm"][label].append(basin)
    for basin, label in raw_clusters.items():
        basin_in_cluster["raw"][label].append(basin)

    for label, basins in basin_in_cluster["lstm"].items():
        label_2_color["lstm"][label] = color_list[label]

        max_count = -1
        color_label = None
        for label2, basins2 in basin_in_cluster["raw"].items():
            intersect = set(basins).intersection(basins2)
            if len(intersect) > max_count:
                max_count = len(intersect)
                color_label = label2

        label_2_color["raw"][color_label] = color_list[label]

    return label_2_color


def get_variance_reduction(lstm_clusters: Dict, raw_clusters: Dict,
                           df: pd.DataFrame) -> defaultdict:
    """Calculate per feature fraction variance reduction.
    
    Parameters
    ----------
    lstm_clusters : Dict
        Cluster labels for the LSTM embeddings
    raw_clusters : Dict
        Cluster labels for the raw catchment attributes
    df : pd.DataFrame
        Dataframe containing the attributes of interest. For each column in this DataFrame the 
        fractional variance reduction is calculated.
    
    Returns
    -------
    defaultdict
        Dictionary containing the per-cluster fractional variance reduction for the LSTM clusters 
        and the clusters from the raw catchment attributes.
    """
    results = defaultdict(dict)
    for n_class in list(set(lstm_clusters.values())):

        class_basins = []
        for basin, label in lstm_clusters.items():
            if label == n_class:
                class_basins.append(basin)
        drop_basins = [b for b in df.index if b not in class_basins]
        df_q_sub = df.drop(drop_basins, axis=0)
        results["lstm"][n_class] = df_q_sub.var() / df.var()

    for n_class in list(set(raw_clusters.values())):
        class_basins = []
        for basin, label in raw_clusters.items():
            if label == n_class:
                class_basins.append(basin)
        drop_basins = [b for b in df.index if b not in class_basins]
        df_q_sub = df.drop(drop_basins, axis=0)
        results["raw"][n_class] = df_q_sub.var() / df.var()
    return results
