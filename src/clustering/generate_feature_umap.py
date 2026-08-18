# This script generates clusterings and UMAP projections for a given strain, based on any available datasets, using HDBSCAN or KNN.
# It is used for visualizing the feature space and understanding the relationships between features across different strains and datasets.
# The generated UMAP projections, together with the clusterings are stored in the "feature_umap" directory, organized by strain and dataset.
# The script can be run with the following command:
# python src/Clustering/generate_feature_umap.py --output_dir /path/to/output_dir --n_neighbors 15 --min_cluster_size 5
import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
import umap
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import distinctipy

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from utils import calculate_features, load_data, calculate_target, calculate_features, make_multiclass, apply_cutoff, cutoff_clean, drop_non_numeric, split_data, stratified_undersample, identify_candidates, transform_meta_features

ALL_PUBS = ["Lui2019", "Kupke2020", "Penn2022", "Sheng2018", "Zhuravlev2020", "vdHoecke2015", "Boussier2020", "Southgate2019", "Valesano2020", "Mendes2021", "Alnaji2019", "Berry2021", "Alnaji2021", "Wang2020", "Wang2023", "Pelz2021"]
ALL_STRAINS = ["A_Anhui_1_2013", "A_California_07_2009", "A_Connecticut_Flu122_2013", "A_NewCaledonia_20-JY2_1999", "A_Perth_16_2009", "A_PuertoRico_8_1934", "A_turkey_Turkey_1_2005", "A_WSN_33", "B_Brisbane_60_2008", "B_Lee_1940", "B_Victoria_504_2000", "B_Yamagata_16_1988"]
ALL_SEGMENTS = ["PB2", "PB1", "PA", "HA", "NP", "NA", "M", "NS"]

STANDARD_FEATURES_DEFAULT = ['Segment', 'Start', 'End', 'Direct_repeat', 'Junction', 'remaining_length', 'deletion_length', '3_5_diff', 'length_proportion', 'Peptide_Length']
strain_to_pubs = {'A_PuertoRico_8_1934': ['Kupke2020', 'Zhuravlev2020', 'VdHoecke2015', 'Alnaji2021', 'Wang2020', 'Wang2023', 'Pelz2021'],
                  'A_WSN_33': ['Boussier2020', 'Mendes2021'],
                  'B_Victoria_504_2000': ['Valesano2020', 'Berry2021'],
                  'B_Yamagata_16_1988': ['Southgate2019', 'Valesano2020', 'Berry2021']}

STRAIN_COLORS = distinctipy.get_colors(len(ALL_STRAINS),n_attempts=50000,rng=42)
SEGMENT_COLORS = distinctipy.get_colors(len(ALL_SEGMENTS),n_attempts=50000,rng=42)
CLUSTER_COLORS = distinctipy.get_colors(10,[(0,0,0),(1,1,1),(1,0,0)],n_attempts=50000,rng=42)

def fill_missing(df, abundance):
    """Fill missing NGS read counts for calculation of the abundance feature, if necessary for the chosen abundance measure of candidates."""
    if "ilr" in abundance or "clr" in abundance:
        all_ids = df["ID"].unique()
        missing_dfs = []
        for sample, group in df.groupby("ACC_num"):
            missing_ids = [id for id in all_ids if id not in group["ID"].values]
            if missing_ids:
                missing_dfs.append(pd.DataFrame({"ID": missing_ids, "ACC_num": sample, "NGS_read_count": 0}))
        df = pd.concat([df] + missing_dfs, ignore_index=True)
    return df

def extend_pivot(pivot, df, abundance):
    """
    Extend the pivot matrix with an abundance feature based on the chosen abundance measure of candidates.
    New columns will refer to samples within dataframe, and values will be the abundance measure for the candidate in those samples.
    """
    if abundance not in df.columns:
        filled_df = fill_missing(df, abundance)
        abundance_values = calculate_target(filled_df, abundance)[["ID","ACC_num",abundance]]
    else:
        abundance_values = df[["ID","ACC_num",abundance]]
    abundance_pivot = abundance_values.pivot_table(index="ID", columns="ACC_num", values=abundance, aggfunc="first").fillna(0)
    extended_pivot = pivot.join(abundance_pivot, how="left").fillna(0)
    return extended_pivot
    
def generate_clustering(pivot, method, **kwargs):
    if "param_grid" in kwargs:
        param_grid = kwargs.pop("param_grid")
        best_score = -np.inf
        best_params = None
        for params in ParameterGrid(param_grid):
            if method == "hdbscan":
                clusterer = HDBSCAN(**params)
                clusters = clusterer.fit_predict(pivot)
            elif method == "knn":
                knn = NearestNeighbors(**params)
                knn.fit(pivot)
                distances, indices = knn.kneighbors(pivot)
                clusters = KMeans(n_clusters=len(set(clusters))).fit_predict(pivot)
            else:
                raise ValueError("Invalid clustering method. Choose 'hdbscan' or 'knn'.")
            score = silhouette_score(pivot, clusters, metric='nan_euclidean')
            if score > best_score:
                best_score = score
                best_params = params
        print(f"Best parameters: {best_params} with silhouette score: {best_score}")
        return generate_clustering(pivot, method, **best_params)
    else:
        if method == "hdbscan":
            clusterer = HDBSCAN(**kwargs)
            clusters = clusterer.fit_predict(pivot)
        elif method == "knn":
            knn = NearestNeighbors(**kwargs)
            knn.fit(pivot)
            distances, indices = knn.kneighbors(pivot, **kwargs)
            clusters = KMeans(n_clusters=len(set(clusters))).fit_predict(pivot)
        else:
            raise ValueError("Invalid clustering method. Choose 'hdbscan' or 'knn'.")
    return clusters

def generate_umap(pivot, **kwargs):
    if "random_state" in kwargs:
        reducer = umap.UMAP(**kwargs)
    else:
        reducer = umap.UMAP(random_state=42, **kwargs)
    reducer.fit(pivot)
    return reducer, reducer.embedding_

def save_reducer(reducer, name, abundance, strain, result_path):
    """Save the UMAP reducer to a file for later use."""
    os.makedirs(result_path, exist_ok=True)
    output_path = os.path.join(result_path, f"{f'{name}_' if name != '' else ''}{f'{abundance}_' if abundance is not None else ''}{strain}_umap_reducer.pkl")
    joblib.dump(reducer, output_path)
    print(f"Saved UMAP reducer to {output_path}")

def load_reducer(name, abundance, strain, result_path):
    """Load a UMAP reducer from a file if it exists."""
    input_path = os.path.join(result_path, f"{f'{name}_' if name != '' else ''}{f'{abundance}_' if abundance is not None else ''}{strain}_umap_reducer.pkl")
    if os.path.exists(input_path):
        reducer = joblib.load(input_path)
        print(f"Loaded UMAP reducer from {input_path}")
        return reducer
    else:
        print(f"No UMAP reducer found at {input_path}")
        return None

def save_embedding_index(embedding, index, name, abundance, strain, result_path):
    """Save the UMAP embedding with the corresponding index to a file for later use."""
    os.makedirs(result_path, exist_ok=True)
    output_path = os.path.join(result_path, f"{f'{name}_' if name != '' else ''}{f'{abundance}_' if abundance is not None else ''}{strain}_umap_embedding.csv")
    pd.DataFrame(embedding, index=index, columns=["UMAP_1", "UMAP_2"]).to_csv(output_path)
    print(f"Saved UMAP embedding with index to {output_path}")

def save_clustering_index(clusters, index, name, abundance, strain, result_path):
    """Save the clustering assignments with the corresponding index to a file for later use."""
    os.makedirs(result_path, exist_ok=True)
    output_path = os.path.join(result_path, f"{f'{name}_' if name != '' else ''}{f'{abundance}_' if abundance is not None else ''}{strain}_clusters.csv")
    pd.DataFrame({"Cluster": clusters}, index=index).to_csv(output_path)
    print(f"Saved clustering assignments with index to {output_path}")

def save_full_index(embedding, clusters, index, name, abundance, strain, result_path):
    """Save the UMAP embedding and clustering assignments with the corresponding index to a file for later use."""
    os.makedirs(result_path, exist_ok=True)
    output_path = os.path.join(result_path, f"{f'{name}_' if name != '' else ''}{f'{abundance}_' if abundance is not None else ''}{strain}_umap_full.csv")
    pd.DataFrame({
        "UMAP_1": embedding[:, 0],
        "UMAP_2": embedding[:, 1],
        "Cluster": clusters
    }, index=index).to_csv(output_path)
    print(f"Saved full UMAP embedding and clustering assignments with index to {output_path}")

def generate_feature_clustering(strain:str="A_PuertoRico_8_1934", 
                                cutoff:int=0,
                                features:list=STANDARD_FEATURES_DEFAULT,
                                abundance:str|None=None,
                                name:str="Comb",
                                result_path="",
                                method="hdbscan",
                                target:str|None=None,
                                **kwargs):
    """
    Generate UMAP projections and feature clusterings for a given strain.

    Parameters:
    - strain: The viral strain for which data is found by the load_data function in utils.
    - cutoff: The cutoff value for filtering the data.
    - features: A list of features to include in the UMAP projection and clustering. Assumed to be standard features.
    - abundance: An optional feature to include in the UMAP projection and clustering, which will be extended to the pivot matrix by the chosen abundance measure of each candidate.
    - name: Name to use for the output files.
    - result_path: The path where the results will be saved.
    - method: The clustering method to use ("hdbscan" or "knn").
    - target: The target variable to include in the output (optional).
    - **kwargs: Additional keyword arguments for the clustering method.
    """
    # get the data for the given strain
    if strain in strain_to_pubs.keys():
        relevant_pubs = strain_to_pubs[strain]
    else:
        relevant_pubs = ALL_PUBS
    df = load_data(names=relevant_pubs, unpooled=True)
    df = df[df["Strain"] == strain].copy().reset_index(drop=True)
    tmp_cols = df.columns
    df = calculate_features(df, standard_features=features)
    pivot_columns = [col for col in df.columns if col not in tmp_cols or col in features]
    df = cutoff_clean(df, cutoff)
    if 'Start' in features:
        df["Start"] = StandardScaler().fit_transform(pd.DataFrame(df["Start"]))
    if 'End' in features:
        df["End"] = StandardScaler().fit_transform(pd.DataFrame(df["End"]))

    # Generate the pivot matrix for the features
    pivot = df[["ID"]+pivot_columns].drop_duplicates("ID").set_index("ID")[pivot_columns].copy()
    pivot = drop_non_numeric(pivot)

    if abundance is not None:
        pivot = extend_pivot(pivot, df, abundance)
    
    if n_neighbors := kwargs.get("n_neighbors", None):
        print(f"Using n_neighbors={n_neighbors} for clustering")
    clusters = generate_clustering(pivot, method, **kwargs)
    
    # Generate UMAP projection
    reducer, embedding = generate_umap(pivot, n_neighbors=kwargs.get("n_neighbors", 15))
    save_reducer(reducer, name, abundance, strain, result_path)
    save_embedding_index(embedding, pivot.index, name, abundance, strain, result_path)
    save_clustering_index(clusters, pivot.index, name, abundance, strain, result_path)
    save_full_index(embedding, clusters, pivot.index, name, abundance, strain, result_path)
    return embedding, clusters

def merge_embedding_clustering(embedding, clusters, index):
    """Merge the UMAP embedding and clustering assignments into a single DataFrame for easier plotting."""
    return pd.DataFrame({
        "UMAP_1": embedding[:, 0],
        "UMAP_2": embedding[:, 1],
        "Cluster": clusters
    }, index=index)

def plot_umap(embedding, clusters=None, colors:dict=None):
    """Plot the UMAP embedding with optional coloring."""
    if not isinstance(embedding, pd.DataFrame):
        embedding = pd.DataFrame(embedding, columns=["UMAP_1", "UMAP_2"])
    if clusters is not None:
        embedding["Cluster"] = clusters
        sns.scatterplot(embedding, x="UMAP_1", y="UMAP_2", hue="Cluster", palette=colors if colors else "tab10", legend="full", alpha=0.7)
    else:
        sns.scatterplot(embedding, x="UMAP_1", y="UMAP_2", alpha=0.7)
    plt.title("UMAP Projection")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(True)
    plt.show()

def plot_umap_df(df, color_by:str|None=None, colors:dict|None=None):
    if color_by and color_by not in df.columns:
        raise ValueError(f"color_by column '{color_by}' not found in DataFrame.")
    elif color_by:
        sns.scatterplot(df, x="UMAP_1", y="UMAP_2", hue=color_by, palette=colors if colors else "tab10", legend="full", alpha=0.7)
    else:
        sns.scatterplot(df, x="UMAP_1", y="UMAP_2", alpha=0.7)
    plt.title("UMAP Projection")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    emb, clust = generate_feature_clustering(strain="A_PuertoRico_8_1934",
                                cutoff=0,
                                features=STANDARD_FEATURES_DEFAULT,
                                abundance="NGS_clr",
                                name="Comb",
                                result_path="comb_umap",
                                method="hdbscan",
                                target=None,
                                param_grid={"min_cluster_size": [5, 10, 15], "min_samples": [1, 5, 10]})

    