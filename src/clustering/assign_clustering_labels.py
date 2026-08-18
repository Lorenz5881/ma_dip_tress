import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Optional
import warnings
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import distinctipy
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import silhouette_score
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from utils import calculate_features, load_data, calculate_target, calculate_features, make_multiclass, apply_cutoff, cutoff_clean, drop_non_numeric, split_data, stratified_undersample, identify_candidates, transform_meta_features
EMBEDDING_SUFFIXES = (".npz", ".npy", ".sav", ".joblib", ".pkl", ".csv")

strain_to_pubs = {'A_PuertoRico_8_1934': ['Kupke2020', 'Zhuravlev2020', 'VdHoecke2015', 'Alnaji2021', 'Wang2020', 'Wang2023', 'Pelz2021'],
                  'A_WSN_33': ['Boussier2020', 'Mendes2021'],
                  'B_Victoria_504_2000': ['Valesano2020', 'Berry2021'],
                  'B_Yamagata_16_1988': ['Southgate2019', 'Valesano2020', 'Berry2021']}

def setup_logging(path, name="results", verbose=True):
    os.makedirs(path, exist_ok=True)
    log_path = os.path.join(path, f'{name}.log')
    fmt_debug = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s'
    fmt_info  = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
        format=fmt_debug if verbose else fmt_info,
        level=logging.DEBUG if verbose else logging.INFO,
        force=True
    )
    logging.getLogger('shap').setLevel(logging.WARNING)
    
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib\..*")
    
def _score_file_candidate(path: Path, strain: str, cutoff: Optional[int], c_type: str="scaffold") -> int:
    path_l = str(path).lower()
    score = 0

    if not strain.lower() in path_l:
        return 0
    score += 50

    if cutoff is not None:
        cutoff_tokens = (
            f"c{cutoff}",
            f"_{cutoff}_",
            f"-{cutoff}-",
            f"cutoff {cutoff}",
            f"cutoff_{cutoff}",
        )
        if not any(token in path_l for token in cutoff_tokens):
            return 0
        score += 40

    # Prefer files that look like embeddings over output artifacts.
    if any(token in path_l for token in ("emb", "umap", "scaffold", "full", c_type)):
        score += 20

    if any(token in path_l for token in ("cluster", "clustering", "index", "plot", "grid")):
        score -= 30

    return score


def find_source_embedding(base_dir: Path, strain: str, cutoff: Optional[int], c_type: str="scaffold") -> Path:
    candidates = []
    for p in base_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in EMBEDDING_SUFFIXES:
            continue
        if strain.lower() in str(p).lower():
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"No embedding files found in {base_dir}")

    ranked = sorted(candidates, key=lambda p: _score_file_candidate(p, strain, cutoff, c_type), reverse=True)
    best = ranked[0]
    if _score_file_candidate(best, strain, cutoff, c_type) < 0:
        raise FileNotFoundError(
            f"Could not confidently identify an embedding source file. "
            f"Please ensure filename/path contains strain and, if used, cutoff info.\nCandidates found:\n" + "\n".join(f"{p} (score: {_score_file_candidate(p, strain, cutoff, c_type)})" for p in ranked)
        )
    return best, ranked


def load_embedding(path: Path) -> np.ndarray|pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".npz":
        npz = np.load(path)
        if "arr_0" in npz.files:
            emb = npz["arr_0"]
        else:
            emb = npz[npz.files[0]]
    elif suffix == ".npy":
        emb = np.load(path)
    elif suffix in (".sav", ".joblib", ".pkl"):
        emb = joblib.load(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
        if not {"UMAP1", "UMAP2"}.issubset(df.columns):
            raise ValueError(f"CSV source {path} must contain UMAP1 and UMAP2 columns")
        if "ID" in df.columns:
            # If an ID column exists, we assume this is a preprocessed CSV and return the full DataFrame for downstream processing.
            logging.debug(f'Found ID column in CSV source {path}, returning full DataFrame for downstream processing.')
            return df
        emb = df[["UMAP1", "UMAP2"]].to_numpy()
    else:
        raise ValueError(f"Unsupported embedding format: {path.suffix}")

    emb = np.asarray(emb)
    if emb.ndim != 2 or emb.shape[1] < 2:
        raise ValueError(f"Embedding must be a 2D array with at least 2 columns, got {emb.shape}")
    return emb[:, :2]


def find_index_files(source_file: Path, strain: str, cutoff: Optional[int]) -> Optional[Path]:
    local_candidates = list(source_file.parent.glob("*index*.csv"))
    if not local_candidates:
        logging.warning(f"No index file found for source {source_file}.")
        return None

    def score(p: Path) -> int:
        score_val = 0
        p_l = p.name.lower()
        if strain.lower() in p_l:
            score_val += 20
        else:
            return 0
        if cutoff is not None and (f"c{cutoff}" in p_l or f"{cutoff}" in p_l):
            score_val += 10
        else:
            return 0
        return score_val

    return sorted(local_candidates, key=score, reverse=True)


def load_metadata(index_file: Optional[Path], n_rows: int) -> pd.DataFrame:
    idx_df = pd.read_csv(index_file)

    if len(idx_df) == n_rows:
        if "ID" not in idx_df.columns:
            idx_df["ID"] = [f"row_{i}" for i in range(n_rows)]
        return idx_df.reset_index(drop=True)

    unnamed_cols = [col for col in idx_df.columns if col.lower().startswith("unnamed")]
    if unnamed_cols:
        idx_df = idx_df.drop(columns=unnamed_cols)

    if "index" in idx_df.columns:
        ordered = idx_df.sort_values("index").reset_index(drop=True)
        if len(ordered) == n_rows:
            if "ID" not in ordered.columns:
                ordered["ID"] = [f"row_{i}" for i in range(n_rows)]
            return ordered
    raise ValueError(f"Index file {index_file} has {len(idx_df)} rows but expected {n_rows}. Cannot load metadata.")

def get_metadata(source_file, strain, cutoff) -> tuple[Path, pd.DataFrame]:
    idx_file_candidates = find_index_files(source_file, strain, cutoff)
    for idx_file in idx_file_candidates:
        try:
            meta_df = load_metadata(idx_file, n_rows=0)  # We just want to check the IDs here, so we can pass n_rows=0 to
            if strain in meta_df["ID"].astype(str).str.lower().values:
                logging.debug(f"Successfully loaded metadata from {idx_file} for strain '{strain}'.")
                return idx_file, meta_df
        except Exception as e:
            pass

    raise ValueError(f"Metadata for strain '{strain}' not found in any of the provided index files.")

def load_scaffold_deprecated(strain: str, name=""): # taken from feature_scaffolds.py
    '''
    Loads precomputed umap scaffold and its respective index mapping.

    :param strain: Name of the strain for which to load a scaffold.

    :return: umap embedding, pivot index
    '''
    if name!="" and name!=strain:
        name=f'{name}_'
    scaff_path = os.path.abspath(os.path.join(os.getcwd(),'..','..','results','scaffolds',strain,f'{name}{strain}_scaffold.sav'))
    scaffold = joblib.load(scaff_path)
    try:
        index_path = os.path.abspath(os.path.join(os.getcwd(),'..','..','results','scaffolds',strain,f'{name}{strain}_index.parquet'))
        pivot_id = pq.read_table(index_path).to_pandas()
    except:
        index_path = os.path.abspath(os.path.join(os.getcwd(),'..','..','results','scaffolds',strain,f'{name}{strain}_index.csv'))
        pivot_id = pd.read_csv(index_path,index_col=0)
    dataframe = pivot_id.copy()
    dataframe[["Strain","Segment","Start","End"]] = dataframe["ID"].str.rsplit("_", n=3, expand=True)
    return dataframe, pivot_id, scaffold

def load_embedding_and_IDs(strain, cutoff, umap_type):
    if umap_type == "scaffold":
        try:
            scaff_path = os.path.abspath(os.path.join(umap_type,strain,f'{strain}_scaffold.npz'))
            id_path = os.path.abspath(os.path.join(umap_type,strain,f'{strain}_index.csv'))
            try:
                embedding = np.load(scaff_path)["arr_0"].astype(float)
            except FileNotFoundError as e:
                try:
                    embedding = joblib.load(scaff_path.split(".")[0]+".sav")
                except FileNotFoundError as e2:
                    raise FileNotFoundError(f"File not found for {strain}:\n{e}\n{e2}")

            if isinstance(embedding, list):
                embedding = np.vstack(embedding)
            dataframe = pd.read_csv(id_path, index_col=0)

        except Exception as e:
            try:
                depr_df, dataframe, embedding = load_scaffold_deprecated(strain)
            except Exception as e2:
                raise FileNotFoundError(f"Error occurred while loading embedding or index for {strain}: {e}, {e2}")
        
        dataframe["index"] = dataframe.index
        dataframe[["UMAP1","UMAP2"]] = embedding[dataframe["index"]]
        return dataframe
    elif umap_type == "comb":
        if cutoff is None:
            cutoff = 0
        embedding_path = os.path.abspath(os.path.join(umap_type,strain,f'{strain}_{cutoff}.sav'))
        index_path = os.path.abspath(os.path.join(umap_type,strain,f'{strain}_{cutoff}_index.csv'))
        try:
            dataframe = pd.read_csv(index_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Index file not found for {strain} at {index_path}: {e}")
        try:
            embedding = joblib.load(embedding_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Embedding file not found for {strain} at {embedding_path}: {e}")
        if unnamed_cols := [col for col in dataframe.columns if col.lower().startswith("unnamed")]:
            dataframe.rename(columns={unnamed_cols[0]: "index"}, inplace=True)
            #dataframe["index"] = dataframe[unnamed_cols[0]]
        else:
            dataframe["index"] = dataframe.index
        dataframe[["UMAP1","UMAP2"]] = embedding
        return dataframe
    raise ValueError(f"Unsupported umap_type: {umap_type}. Expected 'scaffold' or 'comb'.")

def direct_load(strain, cutoff, umap_type):
    base_dir = os.path.join(Path(__file__).resolve().parent, umap_type)
    if base_dir.glob("*").is_empty():
        raise FileNotFoundError(f"No files found in source directory: {base_dir}")
    
    # loading embedding and index pointers, then merging them into a single dataframe

    # trying .sav first
    if not base_dir.glob(f'{strain}.sav').is_empty():
        for candidate in base_dir.glob(f'{strain}.sav'):
            if cutoff is not None and f"c{cutoff}" not in str(candidate).lower():
                continue
            if cutoff is None and candidate.name.lower().startswith("c"):
                continue
            try:
                embedding = joblib.load(candidate)
                logging.debug(f"Successfully loaded embedding from {candidate}")
                break
            except Exception as e:
                logging.warning(f"Failed to load {candidate}: {e}")
    elif not base_dir.glob(f'{strain}*.npz').is_empty():
        for candidate in base_dir.glob(f'{strain}*.npz'):
            if cutoff is not None and f"c{cutoff}" not in str(candidate).lower():
                continue
            if cutoff is None and candidate.name.lower().startswith("c"):
                continue
            try:
                embedding = load_embedding(candidate)
                logging.debug(f"Successfully loaded embedding from {candidate}")
                break
            except Exception as e:
                logging.warning(f"Failed to load {candidate}: {e}")
    else:
        raise FileNotFoundError(f"No suitable embedding file found for strain '{strain}' with cutoff '{cutoff}' in {base_dir}")



def hdbscan_labels(embedding: np.ndarray, min_cluster_size: Optional[int] = None, min_samples: Optional[int] = None, param_grid: Optional[dict] = None):

    if param_grid is not None:
        logging.info(f"Starting HDBSCAN grid search with param_grid: {param_grid}")
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
                        best_score = score
                        best_model = model
                    logging.info(f"HDBSCAN grid search - min_cluster_size: {size}, min_samples: {samples}, cluster_selection_epsilon: {epsilon}, #clusters: {unique_non_noise.size}, silhouette_score: {score}")
        if best_model is None:
            best_model = HDBSCAN(min_cluster_size=5, min_samples=min_samples, cluster_selection_epsilon=0.0, store_centers="centroid", copy=True).fit(embedding)
        logging.info(f"Best HDBSCAN model - min_cluster_size: {best_model.min_cluster_size}, min_samples: {best_model.min_samples}, cluster_selection_epsilon: {best_model.cluster_selection_epsilon}, #clusters: {len(np.unique(best_model.labels_[best_model.labels_ >= 0]))}, silhouette_score: {best_score}")
        return best_model.labels_, best_model

    if min_cluster_size is not None:
        model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=0.0, store_centers="centroid", copy=True).fit(embedding)
        logging.info(f"HDBSCAN - min_cluster_size: {model.min_cluster_size}, min_samples: {model.min_samples}, cluster_selection_epsilon: {model.cluster_selection_epsilon}, #clusters: {len(np.unique(model.labels_[model.labels_ >= 0]))}, silhouette_score: {silhouette_score(embedding[model.labels_ >= 0], model.labels_[model.labels_ >= 0]) if np.unique(model.labels_[model.labels_ >= 0]).size >= 2 else -1.0}")
        return model.labels_, model

    max_size = min(50, len(embedding) - 1)
    if max_size < 5:
        model = HDBSCAN(min_cluster_size=max(2, min(5, len(embedding))), min_samples=min_samples, cluster_selection_epsilon=0.0, store_centers="centroid", copy=True).fit(embedding)
        logging.info(f"Not enough points for grid search, using min_cluster_size={model.min_cluster_size}. Silhouette score: {silhouette_score(embedding[model.labels_ >= 0], model.labels_[model.labels_ >= 0]) if np.unique(model.labels_[model.labels_ >= 0]).size >= 2 else -1.0}")
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
    logging.info(f"Best HDBSCAN model - min_cluster_size: {best_model.min_cluster_size}, min_samples: {best_model.min_samples}, cluster_selection_epsilon: {best_model.cluster_selection_epsilon}, #clusters: {len(np.unique(best_model.labels_[best_model.labels_ >= 0]))}, silhouette_score: {best_score}")
    return best_model.labels_, best_model


def kmeans_labels(embedding: np.ndarray, k: Optional[int] = None, param_grid: Optional[dict] = None):
    if param_grid is not None and "k" in param_grid:
        logging.info(f"Starting KMeans grid search with param_grid: {param_grid}")
        best_model = None
        best_score = -2.0
        for cur_k in param_grid.get("k", [5]):
            model = KMeans(n_clusters=cur_k, max_iter=1000, random_state=42, copy_x=True).fit(embedding)
            score = silhouette_score(embedding, model.labels_)
            if score > best_score:
                best_score = score
                best_model = model
            logging.info(f"KMeans grid search - k: {cur_k}, silhouette_score: {score}")
        if best_model is None:
            best_model = KMeans(n_clusters=2, max_iter=1000, random_state=42, copy_x=True).fit(embedding)
            logging.info(f"Not enough points for grid search, using k={best_model.n_clusters}. Silhouette score: {score}")
        logging.info(f"Best KMeans model - k: {best_model.n_clusters}, silhouette_score: {best_score}")
        return best_model.labels_, best_model

    if k is not None:
        model = KMeans(n_clusters=k, max_iter=1000, random_state=42, copy_x=True).fit(embedding)
        logging.info(f"KMeans - k: {model.n_clusters}, silhouette_score: {silhouette_score(embedding, model.labels_)}")
        return model.labels_, model

    max_k = min(20, len(embedding) - 1)
    if max_k < 2:
        model = KMeans(n_clusters=1, max_iter=1000, random_state=42, copy_x=True).fit(embedding)
        logging.info(f"Not enough points for grid search, using k={model.n_clusters}. Silhouette score: {silhouette_score(embedding, model.labels_)}")
        return model.labels_, model

    best_model = None
    best_score = -2.0
    for cur_k in range(2, max_k + 1):
        model = KMeans(n_clusters=cur_k, max_iter=1000, random_state=42, copy_x=True).fit(embedding)
        score = silhouette_score(embedding, model.labels_)
        if score > best_score:
            best_score = score
            best_model = model

    logging.info(f"Best KMeans model - k: {best_model.n_clusters}, silhouette_score: {best_score}")
    return best_model.labels_, best_model


def add_centers(df: pd.DataFrame, labels: np.ndarray, model, algorithm: str) -> pd.DataFrame:
    out = df.copy()
    out["Cluster"] = labels

    if algorithm == "hdbscan":
        if hasattr(model, "centroids_") and model.centroids_ is not None:
            centroids = model.centroids_
            out["centroid_x"] = [centroids[label][0] if label >= 0 else np.nan for label in labels]
            out["centroid_y"] = [centroids[label][1] if label >= 0 else np.nan for label in labels]
        else:
            out["centroid_x"] = np.nan
            out["centroid_y"] = np.nan
            for label in sorted(out["Cluster"].unique()):
                if label < 0:
                    continue
                mask = out["Cluster"] == label
                out.loc[mask, "centroid_x"] = out.loc[mask, "UMAP1"].mean()
                out.loc[mask, "centroid_y"] = out.loc[mask, "UMAP2"].mean()
    else:
        centers = model.cluster_centers_
        out["center_x"] = [centers[label][0] for label in labels]
        out["center_y"] = [centers[label][1] for label in labels]

    return out

def get_cluster_coloring(df: pd.DataFrame) -> dict:
    if "Cluster" not in df.columns:
        logging.warning("DataFrame must contain 'Cluster' column to generate color mapping.")
        return None
    if df["Cluster"].nunique() - (1 if -1 in df["Cluster"].unique() else 0) > 20:
        logging.warning("More than 20 clusters detected. Colors may be indistinguishable.")
    color_dict = {cid: color for cid, color in
                  zip(df[df["Cluster"]!=-1]["Cluster"].unique(),
                      distinctipy.get_colors(df[df["Cluster"]!=-1]["Cluster"].nunique(),
                                             exclude_colors=[(0,0,0),(1,1,1),(1,0,0)],
                                             n_attempts=1000,
                                             pastel_factor=0.5,
                                             rng=42))}
    color_dict[-1] = "#FF0000"  # fixed color for noise
    return color_dict

def quick_scatter(df: pd.DataFrame, title: str, out_png: Path=""):
    plt.figure(figsize=(8, 8))
    labels = sorted(df["Cluster"].unique())
    colors = get_cluster_coloring(df)
    for label in labels:
        mask = df["Cluster"] == label
        color = colors.get(label, "#000000")  # default to black if something goes wrong with color assignment
        if label == -1:
            name = "noise"
        else:
            name = str(label)
        plt.scatter(df.loc[mask, "UMAP1"], df.loc[mask, "UMAP2"], s=5, linewidth=0.5, color=color, label=name)
    plt.title(title)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend(title="Cluster", fontsize=8, markerscale=2)
    plt.tight_layout()
    if out_png != "":
        plt.savefig(out_png, dpi=250)
    else:
        plt.show()
    plt.close()

def slow_scatter(df: pd.DataFrame, title: str, out_png: Path=""):
    plt.figure(figsize=(8, 8))
    #labels = sorted(df["Cluster"].unique())
    #for label in labels:
    #    mask = df["Cluster"] == label
    #    if label == -1:
    #        color = "#999999"
    #        name = "noise"
    #    else:
    #        color = None
    #        name = str(label)
    sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue="Cluster", palette=get_cluster_coloring(df), legend="full", s=5, linewidth=0.5)

    plt.title(title)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend(title="Cluster", fontsize=8, markerscale=2)
    plt.tight_layout()
    if out_png != "":
        plt.savefig(out_png, dpi=250)
    else:
        plt.show()
    plt.close()


def build_output_stem(source_file: Path, clustering_algorithm: str, cutoff: Optional[int]) -> str:
    stem = f"{source_file.stem}_{clustering_algorithm.lower()}"
    if cutoff is not None:
        stem += f"_c{cutoff}"
    return stem


def run(strain: str, umap_type: str, clustering_algorithm: str, cutoff: Optional[int], k: Optional[int], min_cluster_size: Optional[int], param_grid: Optional[dict], debug=False):
    dataframe = load_embedding_and_IDs(strain, cutoff, umap_type)
    if debug:
        logging.debug(f"Loaded embedding and metadata for strain '{strain}' with cutoff '{cutoff}' from umap type '{umap_type}'.")
        logging.debug(f"Dataframe head:\n{dataframe.head()}")
        logging.debug(f"Dataframe describe:\n{dataframe.describe()}")
        logging.debug(f"Dataframe columns: {dataframe.columns}")
        logging.debug(f"Embedding shape: {dataframe[['UMAP1', 'UMAP2']].shape}")

    if cutoff is not None:
        logging.info(f"Applying cutoff {cutoff} to specify relevant IDs.")
        exp_data = load_data(strain_to_pubs.get(strain, []), unpooled=True)
        exp_data = exp_data[exp_data["Strain"] == strain].copy()
        exp_data = cutoff_clean(exp_data, cutoff)
        if exp_data.empty:
            logging.warning(f"No experimental data found for strain '{strain}' after applying cutoff {cutoff}. Proceeding with full embedding.")
        else:
            exp_data = identify_candidates(exp_data)["ID"].unique()
            dataframe = dataframe[dataframe["ID"].isin(exp_data)].reset_index(drop=True)
            if len(dataframe) == len(exp_data):
                logging.info(f"Cutoff {cutoff} successfully applied, all {len(dataframe)} IDs in exp data are found in the embedding.")
            else:
                logging.warning(f"After applying cutoff {cutoff}, {len(dataframe)} points remain in the embedding for clustering. This does not match the {len(exp_data)} IDs identified in the experimental data. Please check embedding generation.")

    algo = clustering_algorithm.lower()
    if algo == "hdbscan":
        labels, model = hdbscan_labels(dataframe[['UMAP1', 'UMAP2']], min_cluster_size=min_cluster_size, param_grid=param_grid)
    elif algo == "kmeans":
        labels, model = kmeans_labels(dataframe[['UMAP1', 'UMAP2']], k=k, param_grid=param_grid)
    else:
        raise ValueError("clustering_algorithm must be one of: hdbscan, kmeans")

    labeled_df = add_centers(dataframe, labels, model, "hdbscan" if algo == "hdbscan" else "kmeans")

    output_stem = os.path.join(Path(__file__).resolve().parent, umap_type, strain, f'{strain}_{umap_type}_{algo}{"_"+str(cutoff) if cutoff is not None else ""}')
    out_csv = f"{output_stem}.csv"
    out_png = f"{output_stem}.png"
    if not debug:
        labeled_df.to_csv(out_csv, index=False)
        #quick_scatter(labeled_df, f"{algo.upper()} clustering: {strain} {umap_type} cutoff: {cutoff if cutoff is not None else '0'}", out_png)
        slow_scatter(labeled_df, f"{algo.upper()} clustering: {strain} {umap_type} cutoff: {cutoff if cutoff is not None else '0'}", out_png)
    else:
        logging.debug(f'Debug mode enabled - not saving outputs, but showing what would be saved:\n{labeled_df.head()}')
        #quick_scatter(labeled_df, f"{algo.upper()} clustering: {strain} {umap_type} cutoff: {cutoff if cutoff is not None else '0'}", "")
        slow_scatter(labeled_df, f"{algo.upper()} clustering: {strain} {umap_type} cutoff: {cutoff if cutoff is not None else '0'}", "")

    logging.info(f"Saved CSV: {out_csv}")
    logging.info(f"Saved plot: {out_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign clustering labels to UMAP embeddings and save a CSV plus a quick sanity-check scatter plot."
        )
    )
    parser.add_argument("-s", "--strain", type=str, help="Strain name, used to select matching source files.")
    parser.add_argument("-t", "--umap_type", type=str, help="Embedding type folder under src/Clustering, e.g. scaffold or comb.")
    parser.add_argument("-a", "--clustering_algorithm", type=str, choices=["hdbscan", "kmeans"], help="Clustering algorithm.")
    parser.add_argument("-o", "--cutoff", type=int, default=None, help="Optional cutoff value used to refine source selection and output naming.")
    parser.add_argument("-k", "--k", type=int, default=None, help="Optional number of clusters for kmeans mode.")
    parser.add_argument("-m", "--min-cluster-size", type=int, default=None, help="Optional min_cluster_size for HDBSCAN.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("-g", "--grid-search", action="store_true", help="Set to use preset grid-search parameters.")
    parser.add_argument("-p", "--param_grid", nargs="?", default=None, help="Optional JSON string to specify grid search parameters for HDBSCAN, e.g. '{\"min_cluster_size\": [5, 10], \"min_samples\": [5, 10]}'")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(os.path.join(Path(__file__).resolve().parent, args.umap_type), name=f'{args.strain}_{args.clustering_algorithm}{"_"+str(args.cutoff) if args.cutoff is not None else ""}', verbose=args.debug)
    logging.info(f"Starting clustering label assignment for strain '{args.strain}' with umap type '{args.umap_type}', clustering algorithm '{args.clustering_algorithm}', cutoff '{args.cutoff}', k '{args.k}', min_cluster_size '{args.min_cluster_size}', grid_search '{args.grid_search}', param_grid '{args.param_grid}'.")
    param_grid = None
    if args.grid_search or args.param_grid is not None:
        if args.param_grid is not None:
            param_grid = args.param_grid
        else:
            if args.clustering_algorithm.lower() == "hdbscan":
                param_grid = {"min_cluster_size": list(range(4, 101, 2)), "min_samples": list(range(4, 101, 2)), "cluster_selection_epsilon": [0.0, 0.001, 0.01, 0.1]}
            elif args.clustering_algorithm.lower() == "kmeans":
                param_grid = {"k": list(range(2, 16))}
    run(
        strain=args.strain,
        umap_type=args.umap_type,
        clustering_algorithm=args.clustering_algorithm,
        cutoff=args.cutoff,
        k=args.k,
        min_cluster_size=args.min_cluster_size,
        param_grid=param_grid,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
