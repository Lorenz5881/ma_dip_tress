import argparse
import itertools

import os
import traceback
from typing import Optional
import pandas as pd
import numpy as np
import distinctipy
import logging
import sys
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import silhouette_score
from skbio.stats.composition import clr
import json


sys.path.insert(0, os.path.join(".."))
from utils import get_ohes, read_json_lists
from utils import DATA_DIR, load_data, identify_candidates, log_and_norm, apply_cutoff, transform_meta_features, get_sequence_quicker, get_remaining_sequence
from utils import get_3_5_ratio, get_3_5_diff, get_DI_Length, get_direct_repeat_length, get_length_proportion, get_peptide_len, get_delta_G, calculate_features, get_standard_feature, CLUSTERING_DIR, UNPOOLED_DATA_DIR
from utils import _consensus_motif, _extract_junction_window, _get_motif_for_id, _identity_similarity, ensure_id_column, get_k_identities, get_kmer_jaccard_similiarities, _hamming_distance

#RESULT_PATH = os.path.join(os.getcwd(),'..','results')
MODELS = ['linear', 'ridge', 'lasso', 'adaboost', 'naive_bayes', 'logistic_regression', 'knn', 'random_forest', 'support_vector', 'gradient_boost']#, 'mlp']
MODEL_COLORS = distinctipy.get_colors(len(MODELS), pastel_factor=0.5, n_attempts=5000)
RESULT_PATH = os.path.abspath(os.path.join(os.getcwd(), "cluster featureplots"))
ALL_PUBS = ["Lui2019", "Kupke2020", "Penn2022", "Sheng2018", "Zhuravlev2020", "vdHoecke2015", "Boussier2020", "Southgate2019", "Valesano2020", "Mendes2021", "Alnaji2019", "Berry2021", "Alnaji2021", "Wang2020", "Wang2023", "Pelz2021"]
ALL_STRAINS = ["A_Anhui_1_2013", "A_California_07_2009", "A_Connecticut_Flu122_2013", "A_NewCaledonia_20-JY2_1999", "A_Perth_16_2009", "A_PuertoRico_8_1934", "A_turkey_Turkey_1_2005", "A_WSN_33", "B_Brisbane_60_2008", "B_Lee_1940", "B_Victoria_504_2000", "B_Yamagata_16_1988"]

STANDARD_FEATURES_DEFAULT = ['Segment', 'Start', 'End', 'Direct_repeat', 'Junction', 'remaining_length', 'deletion_length', '3_5_diff', '3_len', "5_len", 'length_proportion', 'Peptide_Length']
STRAIN_TO_PUBS = {'A_PuertoRico_8_1934': ['Alnaji2021', 'Pelz2021', 'Wang2023', 'Wang2020', 'Zhuravlev2020', 'Kupke2020', 'VdHoecke2015'],
                  'A_WSN_33': ['Boussier2020', 'Mendes2021'],
                  'B_Victoria_504_2000': ['Valesano2020', 'Berry2021'],
                  'B_Yamagata_16_1988': ['Southgate2019', 'Valesano2020', 'Berry2021']}
os.makedirs(RESULT_PATH, exist_ok=True)
DEFAULT_HDBSCAN_GRID = {"min_cluster_size": list(range(4, 101, 2)), "min_samples": list(range(4, 101, 2)), "cluster_selection_epsilon": [0.0, 0.001, 0.01, 0.1]}
DEFAULT_KMEANS_GRID = {"k": list(range(2, 16))}
import warnings


def load_scaffold(strain, cutoff=0, clustering=None, logger=logging):
    logger.debug(f"Loading scaffold for {strain} with cutoff {cutoff} and clustering {clustering}...")
    scaffold_dir = os.path.join(CLUSTERING_DIR, "scaffold", strain)
    if clustering is None:
        scaffold_path = os.path.join(scaffold_dir, f'{strain}_scaffold_kmeans_{cutoff}.csv')
    else:
        scaffold_path = os.path.join(scaffold_dir, f'{strain}_scaffold_{clustering}_{cutoff}.csv')

    if os.path.exists(scaffold_path):
        scaffold_df = pd.read_csv(scaffold_path)
        return scaffold_df
    else:
        logger.warning(f'Scaffold file not found for {strain} with cutoff {cutoff}. Expected at: {scaffold_path}')
        return None

def load_comb_umap(strain, cutoff=0, clustering=None, logger=logging):
    logger.debug(f"Loading combined UMAP for {strain} with cutoff {cutoff} and clustering {clustering}...")
    umap_dir = os.path.join(CLUSTERING_DIR, "comb", strain)
    if clustering is None:
        umap_path = os.path.join(umap_dir, f'{strain}_comb_kmeans_{cutoff}.csv')
    else:
        umap_path = os.path.join(umap_dir, f'{strain}_comb_{clustering}_{cutoff}.csv')

    if os.path.exists(umap_path):
        umap_df = pd.read_csv(umap_path)
        return umap_df
    else:
        logger.warning(f'UMAP file not found for {strain} with cutoff {cutoff}. Expected at: {umap_path}')
        return None

def load_feature_umap(strain, cutoff=0, clustering=None, logger=logging):
    logger.debug(f"Loading feature UMAP for {strain} with cutoff {cutoff} and clustering {clustering}...")
    umap_dir = os.path.join(CLUSTERING_DIR, "feature", strain)
    if clustering is None:
        umap_path = os.path.join(umap_dir, f'{strain}_feature_kmeans_{cutoff}.csv')
    else:
        umap_path = os.path.join(umap_dir, f'{strain}_feature_{clustering}_{cutoff}.csv')

    if os.path.exists(umap_path):
        umap_df = pd.read_csv(umap_path)
        return umap_df
    else:
        logger.warning(f'Feature UMAP file not found for {strain} with cutoff {cutoff}. Expected at: {umap_path}')
        return None

def load_intersecting_clusters(strain, cutoff=0, clustering=None, logger=logging):
    logger.debug(f"Loading intersecting clusters for {strain} with cutoff {cutoff} and clustering {clustering}...")
    intersect_dir = os.path.join(CLUSTERING_DIR, "intersecting", strain)
    #if clustering is None:
    #    intersect_path = os.path.join(intersect_dir, f'{strain}_intersecting_kmeans_{cutoff}.csv')
    #else:
    #    intersect_path = os.path.join(intersect_dir, f'{strain}_intersecting_{clustering}_{cutoff}.csv')
    intersect_path = os.path.join(intersect_dir, f'{strain}_intersecting_hdbscan_{cutoff}.csv')

    if os.path.exists(intersect_path):
        intersect_df = pd.read_csv(intersect_path)
        if "Cluster" not in intersect_df.columns:
            logger.warning(f'Intersecting clusters file for {strain} with cutoff {cutoff} does not contain a "Cluster" column. Expected at: {intersect_path}')
            potentials = [col for col in intersect_df.columns if "cluster" in col.lower()]
            if potentials:
                logger.info(f'Found potential cluster columns: {potentials}')
                intersect_df.rename(columns={potentials[0]: "Cluster"}, inplace=True)
        return intersect_df
    else:
        logger.warning(f'Intersecting clusters file not found for {strain} with cutoff {cutoff}. Expected at: {intersect_path}')
        return None

def hdbscan_labels(embedding: np.ndarray, min_cluster_size: Optional[int] = None, min_samples: Optional[int] = None, param_grid: Optional[dict] = None, logger=logging):

    if param_grid is not None:
        logger.info(f"Starting HDBSCAN grid search with param_grid: {param_grid}")
        best_model = None
        best_score = -2.0
        for epsilon in param_grid.get("cluster_selection_epsilon", [0.0]):
            for size in param_grid.get("min_cluster_size", [5]  if min_cluster_size is None else [min_cluster_size]):
                for samples in param_grid.get("min_samples", [min_samples]):
                    model = HDBSCAN(min_cluster_size=size, min_samples=samples, cluster_selection_epsilon=epsilon, store_centers="centroid", copy=True).fit(embedding)
                    labels = model.labels_
                    non_noise = labels >= 0
                    unique_non_noise = np.unique(labels[non_noise])
                    if unique_non_noise.size < 2:
                        score = -1.0
                    else:
                        score = silhouette_score(embedding, labels)#silhouette_score(embedding[non_noise], labels[non_noise])
                    if score > best_score:
                        logger.info(f"HDBSCAN grid search current best - min_cluster_size: {size}, min_samples: {samples}, cluster_selection_epsilon: {epsilon}, with #clusters: {unique_non_noise.size}, silhouette_score: {score} against previous {best_score}")
                        best_score = score
                        best_model = model
                        
                    logger.debug(f"HDBSCAN grid search - min_cluster_size: {size}, min_samples: {samples}, cluster_selection_epsilon: {epsilon}, #clusters: {unique_non_noise.size}, silhouette_score: {score}")
        if best_model is None:
            best_model = HDBSCAN(min_cluster_size=5, min_samples=min_samples, cluster_selection_epsilon=0.0, store_centers="centroid", copy=True).fit(embedding)
        logger.info(f"Best HDBSCAN model - min_cluster_size: {best_model.min_cluster_size}, min_samples: {best_model.min_samples}, cluster_selection_epsilon: {best_model.cluster_selection_epsilon}, #clusters: {len(np.unique(best_model.labels_[best_model.labels_ >= 0]))}, silhouette_score: {best_score}")
        return best_model.labels_, best_model

    if min_cluster_size is not None:
        model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=0.0, store_centers="centroid", copy=True).fit(embedding)
        logger.info(f"HDBSCAN - min_cluster_size: {model.min_cluster_size}, min_samples: {model.min_samples}, cluster_selection_epsilon: {model.cluster_selection_epsilon}, #clusters: {len(np.unique(model.labels_[model.labels_ >= 0]))}, silhouette_score: {silhouette_score(embedding[model.labels_ >= 0], model.labels_[model.labels_ >= 0]) if np.unique(model.labels_[model.labels_ >= 0]).size >= 2 else -1.0}")
        return model.labels_, model

    max_size = min(50, len(embedding) - 1)
    if max_size < 5:
        model = HDBSCAN(min_cluster_size=max(2, min(5, len(embedding))), min_samples=min_samples, cluster_selection_epsilon=0.0, store_centers="centroid", copy=True).fit(embedding)
        logger.info(f"Not enough points for grid search, using min_cluster_size={model.min_cluster_size}. Silhouette score: {silhouette_score(embedding[model.labels_ >= 0], model.labels_[model.labels_ >= 0]) if np.unique(model.labels_[model.labels_ >= 0]).size >= 2 else -1.0}")
        return model.labels_, model
    best_model = None
    best_score = -2.0
    
    # grid-search over min_cluster_size and min_samples with silhouette score to find best clustering
    for size in range(5, max_size + 1):
        model = HDBSCAN(min_cluster_size=size, min_samples=min_samples, cluster_selection_epsilon=0.0, store_centers="centroid", copy=True).fit(embedding)
        labels = model.labels_
        non_noise = labels >= 0
        unique_non_noise = np.unique(labels[non_noise])
        if unique_non_noise.size < 2:
            score = -1.0
        else:
            score = silhouette_score(embedding, labels)#silhouette_score(embedding[non_noise], labels[non_noise])
        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        best_model = HDBSCAN(min_cluster_size=5, min_samples=min_samples, cluster_selection_epsilon=0.0, store_centers="centroid", copy=True).fit(embedding)
    logger.info(f"Best HDBSCAN model - min_cluster_size: {best_model.min_cluster_size}, min_samples: {best_model.min_samples}, cluster_selection_epsilon: {best_model.cluster_selection_epsilon}, #clusters: {len(np.unique(best_model.labels_[best_model.labels_ >= 0]))}, silhouette_score: {best_score}")
    return best_model.labels_, best_model

def kmeans_labels(embedding: np.ndarray, k: Optional[int] = None, param_grid: Optional[dict] = None, logger=logging):
    if param_grid is not None and "k" in param_grid:
        logger.info(f"Starting KMeans grid search with param_grid: {param_grid}")
        best_model = None
        best_score = -2.0
        for cur_k in param_grid.get("k", [5]):
            model = KMeans(n_clusters=cur_k, max_iter=1000, random_state=42, copy_x=True).fit(embedding)
            score = silhouette_score(embedding, model.labels_)
            if score > best_score:
                logger.info(f"KMeans grid search current best - k: {cur_k}, silhouette_score: {score} against previous {best_score}")
                best_score = score
                best_model = model
            logger.debug(f"KMeans grid search - k: {cur_k}, silhouette_score: {score}")
        if best_model is None:
            best_model = KMeans(n_clusters=2, max_iter=1000, random_state=42, copy_x=True).fit(embedding)
            logger.info(f"Not enough points for grid search, using k={best_model.n_clusters}. Silhouette score: {score}")
        logger.info(f"Best KMeans model - k: {best_model.n_clusters}, silhouette_score: {best_score}")
        return best_model.labels_, best_model

    if k is not None:
        model = KMeans(n_clusters=k, max_iter=1000, random_state=42, copy_x=True).fit(embedding)
        logger.info(f"KMeans - k: {model.n_clusters}, silhouette_score: {silhouette_score(embedding, model.labels_)}")
        return model.labels_, model

    max_k = min(20, len(embedding) - 1)
    if max_k < 2:
        model = KMeans(n_clusters=1, max_iter=1000, random_state=42, copy_x=True).fit(embedding)
        logger.info(f"Not enough points for grid search, using k={model.n_clusters}. Silhouette score: {silhouette_score(embedding, model.labels_)}")
        return model.labels_, model

    best_model = None
    best_score = -2.0
    for cur_k in range(2, max_k + 1):
        model = KMeans(n_clusters=cur_k, max_iter=1000, random_state=42, copy_x=True).fit(embedding)
        score = silhouette_score(embedding, model.labels_)
        if score > best_score:
            best_score = score
            best_model = model

    logger.info(f"Best KMeans model - k: {best_model.n_clusters}, silhouette_score: {best_score}")
    return best_model.labels_, best_model

def add_centers(df: pd.DataFrame, labels: np.ndarray, model, algorithm: str, name_prefix: str, cluster_id_col="Cluster") -> pd.DataFrame:
    out = df.copy()
    #out["Cluster"] = labels

    if algorithm == "hdbscan":
        if hasattr(model, "centroids_") and model.centroids_ is not None:
            centroids = model.centroids_
            out[f"{name_prefix}centroid_x"] = [centroids[label][0] if label >= 0 else np.nan for label in labels]
            out[f"{name_prefix}centroid_y"] = [centroids[label][1] if label >= 0 else np.nan for label in labels]
        else:
            out[f"{name_prefix}centroid_x"] = np.nan
            out[f"{name_prefix}centroid_y"] = np.nan
            for label in sorted(out[cluster_id_col].unique()):
                if label < 0:
                    continue
                mask = out[cluster_id_col] == label
                out.loc[mask, f"{name_prefix}centroid_x"] = out.loc[mask, f"{name_prefix}UMAP1"].mean()
                out.loc[mask, f"{name_prefix}centroid_y"] = out.loc[mask, f"{name_prefix}UMAP2"].mean()
    else:
        centers = model.cluster_centers_
        out[f"{name_prefix}center_x"] = [centers[label][0] for label in labels]
        out[f"{name_prefix}center_y"] = [centers[label][1] for label in labels]

    return out

def find_closest_dvg(dataframe, strain=None, seg=None, start=None, end=None, id=None):
    '''
    Find the closest DVG on the same strain and segment to the given coordinates. Returns the ID of the closest DVG and the distance to it.
    '''
    if id is not None:
        strain, seg, start, end = id.rsplit('_', 3)
    elif None in [strain, seg, start, end]:
        raise ValueError("Either 'id' or all of 'strain', 'seg', 'start', and 'end' must be provided.")
    if f'{strain}_{seg}_{start}_{end}' in dataframe["ID"].values:
        print(f"Exact match found for ID {id}. Returning distance 0.")
        return f'{strain}_{seg}_{start}_{end}', 0
    subset = dataframe.copy()
    if any(param not in dataframe.columns for param in [strain, seg, start, end]):
        # get params from ID column if not provided
        if 'ID' not in dataframe.columns:
            raise ValueError("Dataframe must contain 'ID' column to extract parameters.")
        subset[["Strain", "Segment", "Start", "End"]] = subset["ID"].str.rsplit('_', n=3, expand=True)
    subset = subset[(subset["Strain"] == strain) & (subset["Segment"] == seg)]
    if subset.empty:
        return None, None
    # use start and end columns to calculate euclidean distance to the given coordinates and find the closest DVG
    subset["Distance_to_coords"] = subset.apply(lambda row: np.hypot(int(row["Start"]) - int(start), int(row["End"]) - int(end)), axis=1)
    #display(subset[["ID", "Strain", "Segment", "Start", "End", "Distance_to_coords"]].sort_values("Distance_to_coords").head(10))
    closest_row = subset.loc[subset["Distance_to_coords"].idxmin()]
    return closest_row["ID"], closest_row["Distance_to_coords"]

def cluster_intersecting_on_embedding(embedding_df, intersecting_ids, algorithm, logger=logging, **kwargs):
    intersecting_dataframe = embedding_df[embedding_df["ID"].isin(intersecting_ids)].copy()
    if intersecting_dataframe.empty:
        logger.warning('No intersecting IDs found in the embedding dataframe.')
        return None

    if algorithm == "hdbscan":
        param_grid = kwargs.get("param_grid", DEFAULT_HDBSCAN_GRID)
        labels, model = hdbscan_labels(intersecting_dataframe[['UMAP1', 'UMAP2']], param_grid=param_grid)
    elif algorithm == "kmeans":
        param_grid = kwargs.get("param_grid", DEFAULT_KMEANS_GRID)
        labels, model = kmeans_labels(intersecting_dataframe[['UMAP1', 'UMAP2']], param_grid=param_grid)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
    
    intersecting_dataframe["Intersecting Cluster"] = labels
    intersecting_dataframe = add_centers(intersecting_dataframe, labels, model, algorithm, name_prefix="Intersecting ", cluster_id_col="Intersecting Cluster")
    logger.info(f'Intersecting dataframe after reclustering: {intersecting_dataframe.shape[0]} rows and {intersecting_dataframe.shape[1]} columns. Intersecting Cluster column added with {intersecting_dataframe["Intersecting Cluster"].nunique()} unique clusters.\n{intersecting_dataframe.columns.tolist()}\n{intersecting_dataframe.head()}')
    centroid_cols = ("Intersecting centroid_x", "Intersecting centroid_y") if algorithm == "hdbscan" else ("Intersecting center_x", "Intersecting center_y")
    full_dataframe = embedding_df.copy()
    full_dataframe = full_dataframe.merge(intersecting_dataframe[["ID", "Intersecting Cluster", centroid_cols[0], centroid_cols[1]]], on="ID", how="left")
    logger.info(f'Full dataframe after reclustering: {full_dataframe.shape[0]} rows and {full_dataframe.shape[1]} columns. Intersecting cluster column added with {full_dataframe["Intersecting Cluster"].nunique()} unique clusters.\n{full_dataframe.columns.tolist()}\n{full_dataframe.head()}')
    return full_dataframe

def get_centroid_distance(dataframe, clustering_df, cluster_id_col="Cluster", centroid_cols=["centroid_x", "centroid_y"], coord_cols=["UMAP1", "UMAP2"], inplace=False, logger=logging, **kwargs):
    '''
    Calculate the euclidean distance of each point to each cluster centroid and add it as a new column in the dataframe.
    '''
    if inplace:
        df = dataframe
    else:
        df = dataframe.copy()
    logger.debug(f"Calculating centroid distances for {len(df)} points.")
    clust_name = kwargs.get("cluster_name", None)
    logger.debug(f'Clustering dataframe {clust_name} has {len(clustering_df)} points and {clustering_df["ID"].nunique()} unique IDs vs {df["ID"].nunique()} unique IDs in the main dataframe.')
    clustering_ids = set(clustering_df["ID"].unique().tolist())
    #id_to_coords = {id: (clustering_df.set_index("ID").loc[id, coord_cols[0]], clustering_df.set_index("ID").loc[id, coord_cols[1]]) for id in clustering_ids}
    df[["tmp_umap1", "tmp_umap2"]] = df["ID"].map(lambda id: clustering_df.set_index("ID").loc[id, coord_cols].values if id in clustering_ids else (np.nan, np.nan)).apply(pd.Series)
    # any rows with missing coordinates get the coordinates of the closest DVG
    missing_coords_mask = df["tmp_umap1"].isna() | df["tmp_umap2"].isna()
    if missing_coords_mask.any():
        missing_ids = df.loc[missing_coords_mask, "ID"].tolist()
        logger.debug(f"{missing_coords_mask.sum()} rows have missing coordinates ({len(missing_ids)} IDs). Attempting to fill with closest DVG coordinates.")
        for id in missing_ids:
            closest_id, distance = find_closest_dvg(clustering_df, id=id)
            if closest_id is not None:
                df.loc[df["ID"] == id, ["tmp_umap1", "tmp_umap2"]] = clustering_df.set_index("ID").loc[closest_id, coord_cols].values
                #logger.debug(f"Filled missing coordinates for ID {id} using closest DVG {closest_id} with distance {distance}.")
            else:
                logger.warning(f"Could not find a closest DVG for ID {id}. Leaving coordinates as NaN.")
        #for idx in df[missing_coords_mask].index:
        #    id = df.loc[idx, "ID"]
        #    closest_id, distance = find_closest_dvg(clustering_df, id=id)
        #    if closest_id is not None:
        #        df.loc[idx, ["tmp_umap1", "tmp_umap2"]] = clustering_df.set_index("ID").loc[closest_id, coord_cols].values
        #        logger.debug(f"Filled missing coordinates for ID {id} using closest DVG {closest_id} with distance {distance}.")
        #    else:
        #        logger.warning(f"Could not find a closest DVG for ID {id}. Leaving coordinates as NaN.")
        
    #logger.debug(f"Filled missing coordinates for ID {id} using closest DVG {closest_id} with distance {distance}.")
    #df["ID"].map(lambda id: id_to_coords[id] if id in clustering_ids else (np.nan, np.nan)).apply(pd.Series)
    #df.loc[df[df["ID"].isin(clustering_ids)], "tmp_umap1"] = clustering_df.set_index("ID").loc[df[df["ID"].isin(clustering_ids)]["ID"], coord_cols[0]].values
    #df.loc[df[df["ID"].isin(clustering_ids)], "tmp_umap2"] = clustering_df.set_index("ID").loc[df[df["ID"].isin(clustering_ids)]["ID"], coord_cols[1]].values
    if centroid_cols[0] not in clustering_df.columns or centroid_cols[1] not in clustering_df.columns:
        logger.warning(f"Centroid columns {centroid_cols} not found in clustering dataframe. Trying alternatives...")
        #df.drop(columns=["tmp_umap1", "tmp_umap2"], inplace=True, errors="ignore")
        #return df
        if "center_x" in clustering_df.columns and "center_y" in clustering_df.columns:
            centroid_cols = ["center_x", "center_y"]
        elif "centroid_x" in clustering_df.columns and "centroid_y" in clustering_df.columns:
            centroid_cols = ["centroid_x", "centroid_y"]
        else:
            logging.error(f"Could not find centroid columns in clustering dataframe. Expected one of: {centroid_cols}, ['center_x', 'center_y'], ['centroid_x', 'centroid_y']. Found columns: {clustering_df.columns.tolist()}")
            return df
    for cluster_label in clustering_df[cluster_id_col].unique():
        if cluster_label == -1 or cluster_label == "-1":
            continue
        cluster_points = clustering_df[clustering_df[cluster_id_col] == cluster_label]
        if cluster_points.empty:
            continue
        centroid_x = cluster_points[centroid_cols[0]].iloc[0]
        centroid_y = cluster_points[centroid_cols[1]].iloc[0]
        df[f"Distance_to{f'_{clust_name}' if clust_name else ''}_centroid_{cluster_label}"] = np.sqrt((df["tmp_umap1"] - centroid_x) ** 2 + (df["tmp_umap2"] - centroid_y) ** 2)
    df.drop(columns=["tmp_umap1", "tmp_umap2"], inplace=True, errors="ignore")
    return df

def get_cluster_motif_identities(dataframe, clustering_data, flank=5, cluster_id_col="Cluster", centroid_cols=["centroid_x", "centroid_y"], coord_cols=["UMAP1", "UMAP2"], inplace=False, logger=logging, **kwargs):
    '''
    Adds motif-identities comparing each row's junction window to the consensus
    motif of each cluster.
    '''
    clust_name = kwargs.get("cluster_name", None)
    if inplace:
        df = dataframe
    else:
        df = dataframe.copy()
    df = ensure_id_column(df)
    if "Full_Sequence" not in df.columns:
        df = get_sequence_quicker(df)
    for id, group in df.groupby("ID"):
        seq, start, end = group[["Full_Sequence", "Start", "End"]].iloc[0]
        df.loc[group.index, "_junction_window"] = _extract_junction_window(seq, start, end, flank=flank)
    
    for cid, group in clustering_data.groupby(cluster_id_col):
        if cid == -1 or cid == "-1":
            continue
        motifs = []
        for member_id in group["ID"].tolist():
            try:
                motifs.append(_get_motif_for_id(member_id, df, flank=flank))
            except Exception:
                continue
        if motifs:
            ref_motif = _consensus_motif(motifs)
            for id, group in df.groupby("ID"):
                id_window = group["_junction_window"].iloc[0]
                df.loc[group.index, f"junction_identity_motif{f'_{clust_name}' if clust_name else ''}_{cid}"] = _identity_similarity(id_window, ref_motif)
                df.loc[group.index, f"junction_hamming_motif{f'_{clust_name}' if clust_name else ''}_{cid}"] = _hamming_distance(id_window, ref_motif)
                #df["_junction_window"].map(lambda window: _identity_similarity(window, ref_motif))
    df.drop(columns=["_junction_window"], inplace=True)
    return None if inplace else df

def get_vip_features(dataframe, vips, inplace=False, **kwargs):
    '''
    Adds columns for each VIP feature, containing the value of that feature for each row.
    '''
    if inplace:
        df = dataframe
    else:
        df = dataframe.copy()
    df = get_kmer_jaccard_similiarities(df=df, candidates=vips, k_list=[4,5,6])
    df = get_k_identities(df=df, candidates=vips)
    return df

def get_current_vip_target(dataframe, vips, target_column, inplace=False, **kwargs):
    '''
    Adds columns for each VIP feature, containing the abundance of that VIP in the current dataframe.
    '''
    if inplace:
        df = dataframe
    else:
        df = dataframe.copy()
    
    for vip in vips:
        df[f'{vip}_{target_column}'] = 0

    for acc_num, group in df.groupby("ACC_num"):
        local_vips = np.intersect1d(group["ID"].unique(), vips)
        for vip in local_vips:
            df.loc[group.index, f'{vip}_{target_column}'] = group[group["ID"] == vip][target_column].values[0]
    return df

def get_multi_kmers(dataframe, k_list=[4,5,6], inplace=False, **kwargs):
    '''
    Adds columns for each k in k_list, containing the set of kmers of that length in the current dataframe.
    '''
    if inplace:
        df = dataframe
    else:
        df = dataframe.copy()
    remove_remaining = False
    if "Remaining" not in df.columns:
        df = get_remaining_sequence(df)
        remove_remaining = True
    def get_multi_kmers(seq):
        kmers = set()
        for k in k_list:
            kmers.update(seq[i:i+k] for i in range(len(seq) - k + 1))
        return kmers
    
    for id, group in df.groupby("ID"):
        seq = group["Remaining"].iloc[0]
        df.loc[group.index, "multi_kmers"] = get_multi_kmers(seq)
    
    if remove_remaining:
        df.drop(columns=["Remaining"], inplace=True)

    return df

def get_ngs_cluster_features(dataframe, ngs_clustering, inplace=False, name_prefix="", logger=logging, **kwargs):
    '''
    Add columns for each cluster in ngs_clustering, based on the NGS_read_count of each DVG in that cluster. The new columns will contain the proportion, mean proportion, and clr of NGS_read_count of each cluster in each ACC_num.
    '''
    if inplace:
        df = dataframe
    else:
        df = dataframe.copy()

    if ngs_clustering is None:
        logger.warning("No NGS clustering results supplied; returning dataframe unchanged.")
        return df
    
    for cluster_id, group in ngs_clustering.groupby("Cluster"):
        if cluster_id == -1 or cluster_id == "-1":
            continue
        cluster_name = f"{name_prefix}{cluster_id}"
        df[f'{cluster_name}_sum'] = 0
        df[f'{cluster_name}_mean_normed'] = 0
        df[f'{cluster_name}_proportion'] = 0
        df[f'{cluster_name}_CLR'] = 0
        #df[f'{cluster_name}_mean_proportion'] = 0
    for acc_num, acc_group in df.groupby("ACC_num"):
        for cluster_id, clust_group in ngs_clustering.groupby("Cluster"):
            if cluster_id == -1 or cluster_id == "-1":
                continue
            # calculate mean and sum of NGS_read_count for each cluster in the current ACC_num group
            cluster_name = f"{name_prefix}{cluster_id}"
            cluster_dvgs = clust_group["ID"].unique()
            df.loc[acc_group.index, f'{cluster_name}_mean_normed'] = acc_group[acc_group["ID"].isin(cluster_dvgs)]["NGS_read_count"].mean() / acc_group["NGS_read_count"].max() if acc_group["NGS_read_count"].max() != 0 and pd.notna(acc_group[acc_group["ID"].isin(cluster_dvgs)]["NGS_read_count"].mean()) else 0
            df.loc[acc_group.index, f'{cluster_name}_sum'] = acc_group[acc_group["ID"].isin(cluster_dvgs)]["NGS_read_count"].sum() if pd.notna(acc_group[acc_group["ID"].isin(cluster_dvgs)]["NGS_read_count"].sum()) else 0
        
        total_sum = acc_group[[f"{name_prefix}{cluster_id}_sum" for cluster_id in ngs_clustering["Cluster"].unique() if cluster_id != -1]].iloc[0].sum()
        geometric_mean = np.exp(np.log(acc_group[[f"{name_prefix}{cluster_id}_sum" for cluster_id in ngs_clustering["Cluster"].unique() if cluster_id != -1]].iloc[0].replace(0, 1e-6)).mean())
        
        for cluster_id in ngs_clustering["Cluster"].unique():
            if cluster_id == -1 or cluster_id == "-1":
                continue
            cluster_name = f"{name_prefix}{cluster_id}"
            df.loc[acc_group.index, f'{cluster_name}_proportion'] = df.loc[acc_group.index, f'{cluster_name}_sum'] / total_sum if total_sum != 0 else 0
            df.loc[acc_group.index, f'{cluster_name}_CLR'] = np.log(df.loc[acc_group.index, f'{cluster_name}_sum'].replace(0, 1e-6) / geometric_mean)

    df.drop(columns=[f"{name_prefix}{cluster_id}_sum" for cluster_id in ngs_clustering["Cluster"].unique()], inplace=True, errors="ignore")

    for cluster_id in ngs_clustering["Cluster"].unique():
        if cluster_id == -1 or cluster_id == "-1":
            continue
        cluster_name = f"{name_prefix}{cluster_id}"
        for col in [f"{cluster_name}_proportion", f"{cluster_name}_CLR", f"{cluster_name}_mean_normed"]:
            if df[col].isna().any():
                logger.error(f"After calculating cluster features, found NaN values in column {col}. This should not happen due to the filler rows. Sample rows with NaN values:\n{df[df[col].isna()][col].head()}")
    
    return df
    


def get_clr_encoded_vips(dataframe, vips, inplace=False, name_prefix="", logger=logging, **kwargs):
    '''
    Adds columns for each VIP feature, containing the CLR-encoded abundance of that VIP in the current dataframe.
    '''
    if inplace:
        df = dataframe
    else:
        df = dataframe.copy()

    if not vips:
        logger.warning("No VIPs supplied for CLR encoding; returning dataframe unchanged.")
        return df

    vip_list = list(set(vips))
    vip_df = df.loc[df["ID"].isin(vip_list), ["ACC_num", "ID", "NGS_read_count"]].copy()
    vip_df["NGS_read_count"] = vip_df["NGS_read_count"].replace(0, 1e-6)
    vip_df = vip_df.groupby(["ACC_num", "ID"], as_index=False)["NGS_read_count"].sum()

    # Build a complete ACC_num x VIP scaffold so experiments with zero VIP hits still get values.
    acc_nums = df["ACC_num"].dropna().unique().tolist()
    scaffold = pd.MultiIndex.from_product([acc_nums, vip_list], names=["ACC_num", "ID"]).to_frame(index=False)
    vip_df = scaffold.merge(vip_df, on=["ACC_num", "ID"], how="left")
    vip_df["NGS_read_count"] = vip_df["NGS_read_count"].fillna(1e-6)
    logger.debug(f'Built CLR scaffold with {len(acc_nums)} ACC_num groups and {len(vip_list)} VIPs. vip_df shape after scaffold merge: {vip_df.shape}\nColumns: {vip_df.columns.tolist()}\nSample rows:\n{vip_df.head()}')
    vip_df["vip_ratios"] = vip_df.groupby("ACC_num")["NGS_read_count"].transform(lambda x: x / x.sum())
    vip_df["CLR"] = vip_df.groupby("ACC_num")["vip_ratios"].transform(clr)
    
    # send CLR-encoded VIP values to the main dataframe
    for vip in vip_list:
        df[f'{name_prefix}{vip}_CLR'] = np.nan
    for acc_num, group in df.groupby("ACC_num"):
        sub_vip_df = vip_df[vip_df["ACC_num"] == acc_num]
        for vip in vip_list:
            vip_value = sub_vip_df[sub_vip_df["ID"] == vip]["CLR"].values
            if len(vip_value) > 0:
                df.loc[group.index, f'{name_prefix}{vip}_CLR'] = vip_value[0]
                if len(vip_value) > 1 and len(np.unique(vip_value)) > 1:
                    logger.warning(f"Multiple CLR values found for VIP {vip} in ACC_num {acc_num}. This should not happen due to the filler rows, but got: {vip_value}. Using the first value.")
            else:
                logger.error(f"VIP {vip} not found in group for ACC_num {acc_num}. This should not happen due to the filler rows, but got: {group['ID'].tolist()}")
                raise ValueError(f"VIP {vip} not found in group for ACC_num {acc_num}. This should not happen due to the filler rows, but got: {group['ID'].tolist()}")
    
    # checking if any CLR values are still NaN after the mapping, which should not happen due to the filler rows
    for vip in vip_list:
        if df[f'{name_prefix}{vip}_CLR'].isna().any():
            logger.error(f"After mapping CLR values, found NaN values in column {name_prefix}{vip}_CLR. This should not happen due to the filler rows. Sample rows with NaN values:\n{df[df[f'{name_prefix}{vip}_CLR'].isna()].head()}")
    return df

def save_preprocessed_data(df, strain, col_dict):
    output_dir = os.path.join(UNPOOLED_DATA_DIR, strain)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f"preprocessed_data.csv"), index=False)
    with open(os.path.join(output_dir, f"processed_column_dict.json"), "w") as f:
        json.dump(col_dict, f)

def load_preprocessed_data(strain, temp_id=0):
    if temp_id == 0:
        data_path = os.path.join(UNPOOLED_DATA_DIR, strain, "preprocessed_data.csv")
        col_dict_path = os.path.join(UNPOOLED_DATA_DIR, strain, "processed_column_dict.json")
    else:
        data_path = os.path.join(UNPOOLED_DATA_DIR, strain, f"temp_preprocessed_data_{temp_id}.csv")
        col_dict_path = os.path.join(UNPOOLED_DATA_DIR, strain, f"temp_processed_column_dict_{temp_id}.json")

    if not os.path.exists(data_path):
        logging.warning(f"Preprocessed data not found for strain {strain}. Expected at: {data_path}")
        return None, None
    if not os.path.exists(col_dict_path):
        logging.warning(f"Preprocessed column dictionary not found for strain {strain}. Expected at: {col_dict_path}")
        df = pd.read_csv(data_path)
        return df, None
    
    df = pd.read_csv(data_path)
    with open(col_dict_path, "r") as f:
        col_dict = json.load(f)
    
    return df, col_dict

def tmp_save(df, strain, col_dict, temp_id=0):
    temp_dir = os.path.join(UNPOOLED_DATA_DIR, strain)
    filename = f"temp_preprocessed_data_{temp_id}.csv"
    col_dict_filename = f"temp_processed_column_dict_{temp_id}.json"
    os.makedirs(temp_dir, exist_ok=True)
    df.to_csv(os.path.join(temp_dir, filename), index=False)
    with open(os.path.join(temp_dir, col_dict_filename), "w") as f:
        json.dump(col_dict, f)

def preprocess_strain_data(strain, logger=logging, **kwargs):
    precomputed = kwargs.get("precomputed", 0)
    if precomputed > 0 or kwargs.get("load_precomputed", False):
        logger.info(f"Preprocessing for strain {strain} with precomputed: {precomputed} and load_precomputed: {kwargs.get('load_precomputed', False)}. Attempting to load preprocessed data...")
        data, col_dict = load_preprocessed_data(strain, temp_id=precomputed)
        if data is None:
            raise ValueError(f"Preprocessed data not found for strain {strain}. Cannot load precomputed data.")
        logger.info(f"Loaded preprocessed data for strain {strain} with {data.shape[0]} rows and {data.shape[1]} columns and column dictionary:\n{col_dict}")
    else:
        data = load_data(STRAIN_TO_PUBS.get(strain, ALL_PUBS), unpooled=True)
        data = data[data["Strain"]==strain]
        if data.empty:
            logger.warning(f'No data found for strain {strain}. Skipping preprocessing.')
            return
        col_dict = {}
    
    if kwargs.get("get_sequence", False):
        data = get_sequence_quicker(data, inplace=True)

    if kwargs.get("skip_dvg_identification", False) or precomputed >= 1:
        init_columns = col_dict.get("init", [])
    else:
        data = identify_candidates(data)
        pub_ids = {pub_id: pub for pub_id, pub in enumerate(STRAIN_TO_PUBS[strain])}
        data["pub_id"] = data["Publication"].map({pub: pub_id for pub_id, pub in pub_ids.items()})
        # labeling intersecting candidates
        id_group_counts = data.groupby("ID")["Publication"].nunique()
        data["Num_Publications"] = data["ID"].map(id_group_counts)
        data["Intersecting"] = data["Num_Publications"] > max(2, data["Publication"].nunique() / 2)
        init_columns = data.columns.tolist()
        try:
            tmp_save(data.drop(columns=["Full_Sequence"], inplace=False, errors="ignore"), strain, col_dict={"init": init_columns}, temp_id=1)
        except Exception as e:
            logger.error(f"Error occurred while making tmp save after DVG identification for strain {strain}: {e}\n{traceback.format_exc()}")

    if kwargs.get("skip_meta_features", False) or precomputed >= 2:
        meta_columns = col_dict.get("meta", [])
    else:
        # getting encoded meta-features and their columns
        data, meta_columns = transform_meta_features(data, get_columns=True)
        try:
            tmp_save(data.drop(columns=["Full_Sequence"], inplace=False, errors="ignore"), strain, col_dict={"init": init_columns, "meta": meta_columns}, temp_id=2)
        except Exception as e:
            logger.error(f"Error occurred while making tmp save after meta features for strain {strain}: {e}\n{traceback.format_exc()}")


    # getting standard features
    if kwargs.get("skip_standard_features", False) or precomputed >= 3:
        standard_feature_columns = col_dict.get("standard", [])
    else:
        logger.info("Calculating standard features...")
        tmp_cols = set(data.columns.tolist())
        for feature in STANDARD_FEATURES_DEFAULT:
            data = get_standard_feature(data, feature, scale="none", normalize_by_length=False, inplace=False)
        data = get_ohes(data, STANDARD_FEATURES_DEFAULT)
        data.drop(columns=["Full_Sequence","Remaining"], errors="ignore", inplace=True)
        standard_feature_columns = [col for col in data.columns if col not in tmp_cols]
        standard_feature_columns = standard_feature_columns + [col for col in data.columns if col in STANDARD_FEATURES_DEFAULT and col not in standard_feature_columns and col != "Full_Sequence"]
        try:
            tmp_save(data.drop(columns=["Full_Sequence"], inplace=False, errors="ignore"), strain, col_dict={"init": init_columns, "meta": meta_columns, "standard": standard_feature_columns}, temp_id=3)
        except Exception as e:
            logger.error(f"Error occurred while making tmp save after standard features for strain {strain}: {e}\n{traceback.format_exc()}")
    
    # getting static vip features
    if kwargs.get("skip_vip_features", False) or precomputed >= 4:
        vip_feature_columns = col_dict.get("vip", [])
    else:
        logger.info("Calculating VIP features...")
        tmp_cols = set(data.columns.tolist())
        if kwargs.get("vip_lists_overwrite", None) is not None:
            vip_lists = read_json_lists(kwargs["vip_lists_overwrite"])
        else:
            vip_lists = read_json_lists("vips.json")
        all_vips = set()
        for vips in vip_lists.values():
            if [vip for vip in vips if strain in vip]: # only include VIPs relevant to the current strain
                all_vips.update(vips)

        data = get_vip_features(data, all_vips, inplace=True)
        vip_feature_columns = [col for col in data.columns if col not in tmp_cols and col != "Full_Sequence"]
        try:
            tmp_save(data.drop(columns=["Full_Sequence"], inplace=False, errors="ignore"), strain, col_dict={"init": init_columns, "meta": meta_columns, "standard": standard_feature_columns, "vip": vip_feature_columns}, temp_id=4)
        except Exception as e:
            logger.error(f"Error occurred while making tmp save after VIP features for strain {strain}: {e}\n{traceback.format_exc()}")
    # getting static clustering features
    if kwargs.get("skip_clustering_features", False) or precomputed >= 5:
        clustering_feature_columns = col_dict.get("clustering", [])
    else:
        logger.info("Calculating clustering features...")
        tmp_cols = set(data.columns.tolist())
        centroid_cols = {"hdbscan": ["centroid_x", "centroid_y"], "kmeans": ["center_x", "center_y"]}
        cutoff_grid = kwargs.get("cutoff_grid", [0,5,10,15])
        for algorithm in ["hdbscan", "kmeans"]:
            for cutoff in cutoff_grid:
                for umap_name, load_func in [("scaff", load_scaffold), ("comb", load_comb_umap)]:#, ("feature", load_feature_umap)]:
                    logger.debug(f"Processing {algorithm} clustering with cutoff {cutoff} on {umap_name} embedding...")
                    clustering_df = load_func(strain=strain, cutoff=cutoff, clustering=algorithm)
                    if kwargs.get("recluster",False) and clustering_df is None:
                        clustering_df = load_func(strain=strain, cutoff=cutoff, clustering="kmeans")
                    if clustering_df is not None:
                        if kwargs.get("recluster",False):
                            logger.info(f"Reclustering {algorithm} on {umap_name} embedding with cutoff {cutoff} for intersecting candidates...")
                            intersecting_ids = data[data["Intersecting"] == True]["ID"].unique()
                            clustering_df = cluster_intersecting_on_embedding(clustering_df, intersecting_ids, algorithm, logger=logger, kwargs=kwargs.get("clustering_kwargs", {}))
                            cluster_id_col = "Intersecting Cluster"
                            cluster_centroid_cols = [f"Intersecting centroid_x", f"Intersecting centroid_y"] if algorithm == "hdbscan" else [f"Intersecting center_x", f"Intersecting center_y"]
                            if cluster_centroid_cols[0] not in clustering_df.columns or cluster_centroid_cols[1] not in clustering_df.columns:
                                logger.warning(f"Expected centroid columns {cluster_centroid_cols} not found in reclustered dataframe. Found columns: {clustering_df.columns.tolist()}. This may cause issues with distance calculations.")
                            logging.info(f"Finished reclustering {algorithm} on {umap_name} embedding with cutoff {cutoff}. Clustering dataframe has {clustering_df.shape[0]} rows and {clustering_df.shape[1]} columns.")
                        else:
                            cluster_centroid_cols = centroid_cols[algorithm]
                            cluster_id_col = "Cluster"
                        data = get_centroid_distance(dataframe=data, clustering_df=clustering_df, cluster_id_col=cluster_id_col, algorithm=algorithm, cluster_name=f"{umap_name}{cutoff}_{algorithm}", centroid_cols=cluster_centroid_cols, logger=logger)
                        data = get_cluster_motif_identities(dataframe=data, clustering_data=clustering_df, cluster_id_col=cluster_id_col, algorithm=algorithm, cluster_name=f"{umap_name}{cutoff}_{algorithm}", centroid_cols=cluster_centroid_cols, logger=logger)
                if kwargs.get("debug", False):
                    logger.debug(f"After processing {algorithm} clustering with cutoff {cutoff} on {umap_name} embedding:")
                    break
            if kwargs.get("debug", False):
                break
        if kwargs.get("debug", False):
            logger.debug(data.head())
        clustering_feature_columns = [col for col in data.columns if col not in tmp_cols and col != "Full_Sequence"]
        if not kwargs.get("debug", False):
            try:
                tmp_save(data.drop(columns=["Full_Sequence"], inplace=False, errors="ignore"), strain, col_dict={"init": init_columns, "meta": meta_columns, "standard": standard_feature_columns, "vip": vip_feature_columns, "clustering": clustering_feature_columns}, temp_id=5)
            except Exception as e:
                logger.error(f"Error occurred while making tmp save after clustering features for strain {strain}: {e}\n{traceback.format_exc()}")
    
    # getting context features
    if kwargs.get("skip_context_features", False) or precomputed >= 6:
        context_feature_columns = col_dict.get("context", [])
    else:
        logger.info(f"Calculating context features with starting shape {data.shape}")
        tmp_cols = set(data.columns.tolist())

        # getting intersecting ngs cluster features for each intersecting cluster, with prefix indicating the cluster name
        intersecting_clusters = load_intersecting_clusters(strain=strain)
        data = get_ngs_cluster_features(data, intersecting_clusters, inplace=False, name_prefix="Intersecting Cluster ", logger=logger)

        if kwargs.get("vip_lists_overwrite", None) is not None:
            vip_lists = read_json_lists(kwargs["vip_lists_overwrite"])
        else:
            vip_lists = read_json_lists("vips.json")
        # getting CLR-encoded VIP features for each VIP list, with prefix indicating the list name
        for key in vip_lists.keys():
            if not any(strain in vip for vip in vip_lists[key]): # only include VIPs relevant to the current strain
                logger.debug(f"Skipping VIP list {key} as it does not contain any VIPs relevant to strain {strain}.")
                continue
            #if key.startswith("principal"):
            #    prefix = "PCA_"
            #else:
            try:
                prefix = "_".join(key.split("_")[:2])+ "_"
            except Exception:
                logger.warning(f"Could not parse VIP list key: {key}. Skipping this VIP list.")
                continue
            prefix = prefix.replace("_" + strain + "_", "_")
            data = get_clr_encoded_vips(data, vips=vip_lists[key], inplace=False, name_prefix=prefix, logger=logger)
            if key.startswith("context_") and key.endswith(strain):
                context_vips = vip_lists[key]
                data = get_clr_encoded_vips(data, vips=context_vips, inplace=True, name_prefix=f"{key}_", logger=logger)
        logger.info(f"Finished calculating context features. New shape is {data.shape}.")
        context_feature_columns = [col for col in data.columns if col not in tmp_cols and col != "Full_Sequence"]
        try:
            tmp_save(data.drop(columns=["Full_Sequence"], inplace=False, errors="ignore"), strain, col_dict={"init": init_columns, "meta": meta_columns, "standard": standard_feature_columns, "vip": vip_feature_columns, "clustering": clustering_feature_columns, "context": context_feature_columns}, temp_id=6)
        except Exception as e:
            logger.error(f"Error occurred while making tmp save after context features for strain {strain}: {e}\n{traceback.format_exc()}")

    return data.drop(columns=["Full_Sequence"], inplace=False, errors="ignore"), {"init": init_columns, "meta": meta_columns, "standard": standard_feature_columns, "vip": vip_feature_columns, "clustering": clustering_feature_columns, "context": context_feature_columns}

def setup_logging(verbose=False):
    fmt_debug = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s'
    fmt_info  = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(handlers=[logging.StreamHandler()],
                        format=fmt_debug if verbose else fmt_info,
                        force=True)
    logging.getLogger('shap').setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib\..*")
    logger = logging.getLogger("ModelTraining")
    logger.setLevel(logging.DEBUG if verbose else logging.DEBUG)
    return logger

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='Short model testing.')
    parser.add_argument('-d', '--strain', type=str, help='Strain to test on.', default='A_PuertoRico_8_1934')
    parser.add_argument('--debug', action='store_true', help='Whether to run in debug mode with limited data and extra logging.')
    parser.add_argument('--recluster', action='store_true', help="Whether to recluster the intersecting candidates on the embeddings.")
    parser.add_argument('--precomputed', type=int, help="Step ID to load precomputed temp data and skip finished steps.", default=0)
    parser.add_argument('--kwargs', type=json.loads, default={}, help='Additional keyword arguments for preprocessing as a JSON string. For example: \'{"skip_vip_features": true}\'')
    args = parser.parse_args()
    strain = args.strain
    logger = setup_logging(verbose=args.debug)
    logger.info(f"Starting preprocessing for strain {strain}...")
    data, col_dict = preprocess_strain_data(strain, debug=args.debug, logger=logger, recluster=args.recluster, precomputed=args.precomputed, **args.kwargs)
    logger.info(f"Preprocessing completed for strain {strain}. Data has {data.shape[0]} rows and {data.shape[1]} columns. Column dictionary:\n{col_dict}\n{data.head()}")
    if data is not None and col_dict is not None:
        save_preprocessed_data(data, strain, col_dict)
        logger.info(f"Finished preprocessing for strain {strain}. Data saved with {data.shape[0]} rows and {data.shape[1]} columns.")
    else:
        logger.warning(f"Preprocessing failed for strain {strain}. No data to save.")
