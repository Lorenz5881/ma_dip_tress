import argparse
import inspect
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
import warnings

import igraph as ig
import leidenalg as la
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import distinctipy
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import cutoff_clean, identify_candidates, load_data

ALL_PUBS = ["Lui2019", "Kupke2020", "Penn2022", "Sheng2018", "Zhuravlev2020", "vdHoecke2015", "Boussier2020", "Southgate2019", "Valesano2020", "Mendes2021", "Alnaji2019", "Berry2021", "Alnaji2021", "Wang2020", "Wang2023", "Pelz2021"]
ALL_STRAINS = ["A_Anhui_1_2013", "A_California_07_2009", "A_Connecticut_Flu122_2013", "A_NewCaledonia_20-JY2_1999", "A_Perth_16_2009", "A_PuertoRico_8_1934", "A_turkey_Turkey_1_2005", "A_WSN_33", "B_Brisbane_60_2008", "B_Lee_1940", "B_Victoria_504_2000", "B_Yamagata_16_1988"]

STRAIN_TO_PUBS = {
    "A_PuertoRico_8_1934": ["Kupke2020", "Zhuravlev2020", "VdHoecke2015", "Alnaji2021", "Wang2020", "Wang2023", "Pelz2021"],
    "A_WSN_33": ["Boussier2020", "Mendes2021"],
    "B_Victoria_504_2000": ["Valesano2020", "Berry2021"],
    "B_Yamagata_16_1988": ["Southgate2019", "Valesano2020", "Berry2021"],
}


def setup_logging(verbose: bool = False) -> None:
    warnings.filterwarnings("ignore", module="PIL.*")
    warnings.filterwarnings("ignore", module="matplotlib.*")

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def resolve_publications(strain: str, publications: list[str] | None) -> list[str]:
    if publications:
        return publications
    pubs = STRAIN_TO_PUBS.get(strain)
    if pubs is None:
        return ALL_PUBS
    return pubs


def prepare_strain_dataframe(strain: str, publications: list[str], cutoff: int, unpooled: bool, sample_col: str, min_publications: int) -> pd.DataFrame:
    df = load_data(publications, unpooled=unpooled)
    df = identify_candidates(df)
    df = df[df["Strain"] == strain].copy()
    if df.empty:
        raise ValueError(f"No data left after filtering for strain {strain}")
    if min_publications > 1:
        df = df[df.groupby("ID")["Publication"].transform("nunique") >= min_publications]
    if df.empty:
        raise ValueError(
            "No data left after min-publications filtering. Try lowering --min-publications."
        )

    if cutoff > 0:
        df = cutoff_clean(df, threshold=cutoff)
    if df.empty:
        raise ValueError(f"No data left after applying cutoff of {cutoff} on column {sample_col}")
    if sample_col not in df.columns:
        raise ValueError(f"Missing required sample column: {sample_col}")
    if "NGS_read_count" not in df.columns:
        raise ValueError("Missing required column: NGS_read_count")
    if "Publication" not in df.columns:
        raise ValueError("Missing required column: Publication")
    return df


def build_candidate_matrix(df: pd.DataFrame, sample_col: str, value_col: str, ) -> pd.DataFrame:
    matrix = df.pivot_table(index=sample_col, columns="ID", values=value_col, aggfunc="sum")
    matrix = matrix.sort_index(axis=0).sort_index(axis=1)
    return matrix


def compute_shared_sample_counts(matrix: pd.DataFrame) -> pd.DataFrame:
    presence = matrix.notna().astype(int)
    shared = presence.T.dot(presence)
    shared.index.name = "source"
    shared.columns.name = "target"
    return shared


def edge_counts_by_shared_threshold(shared_counts: pd.DataFrame, min_threshold: int = 2) -> pd.DataFrame:
    if shared_counts.empty:
        return pd.DataFrame(columns=["min_shared_samples", "num_edges"])
    arr = shared_counts.to_numpy(dtype=int, copy=False)
    upper_idx = np.triu_indices_from(arr, k=1)
    values = arr[upper_idx]
    if values.size == 0:
        return pd.DataFrame(columns=["min_shared_samples", "num_edges"])
    max_shared = int(values.max())
    rows = []
    for threshold in range(min_threshold, max_shared + 1):
        rows.append({"min_shared_samples": threshold, "num_edges": int(np.sum(values >= threshold))})
    return pd.DataFrame(rows)


def detect_elbow_if_applicable(edge_count_df: pd.DataFrame, min_knee_strength: float = 0.08) -> int | None:
    if edge_count_df.shape[0] < 3:
        return None
    x = edge_count_df["min_shared_samples"].to_numpy(dtype=float)
    y = edge_count_df["num_edges"].to_numpy(dtype=float)
    if np.all(y == y[0]):
        return None

    start = np.array([x[0], y[0]], dtype=float)
    end = np.array([x[-1], y[-1]], dtype=float)
    line_vec = end - start
    line_norm = np.linalg.norm(line_vec)
    if line_norm == 0:
        return None

    distances = []
    for xi, yi in zip(x, y):
        p = np.array([xi, yi], dtype=float)
        distance = abs(line_vec[0] * (p[1] - start[1]) - line_vec[1] * (p[0] - start[0])) / line_norm
        distances.append(float(distance))

    max_idx = int(np.argmax(distances))
    max_distance = distances[max_idx]
    y_span = max(1.0, float(y.max() - y.min()))
    knee_strength = max_distance / y_span
    #if knee_strength < min_knee_strength:
    #    return None
    return int(x[max_idx])


def plot_edge_count_curve(edge_count_df: pd.DataFrame, elbow_threshold: int | None, output_path: Path, strain: str, ) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    if edge_count_df.empty:
        ax.text(0.5, 0.5, "No candidate pairs available", ha="center", va="center")
    else:
        ax.plot(
            edge_count_df["min_shared_samples"],
            edge_count_df["num_edges"],
            marker="o",
            linewidth=1.5,
            label="edge count",
        )
        if elbow_threshold is not None:
            elbow_row = edge_count_df[edge_count_df["min_shared_samples"] == elbow_threshold].iloc[0]
            ax.scatter(
                [elbow_row["min_shared_samples"]],
                [elbow_row["num_edges"]],
                color="crimson",
                s=90,
                zorder=3,
                label=f"elbow at {elbow_threshold}",
            )
            ax.axvline(elbow_threshold, color="crimson", linestyle="--", linewidth=1)
    ax.set_xlabel("Minimum shared samples")
    ax.set_ylabel("Number of resulting edges")
    ax.set_title(f"Edge count vs shared-sample threshold ({strain.replace('_', '/')})")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_weighted_edges(matrix: pd.DataFrame, shared_counts: pd.DataFrame, min_shared_samples: int, corr_method: str, abs_weights: bool, min_abs_corr: float | None) -> pd.DataFrame:
    corr_df = matrix.corr(method=corr_method, min_periods=min_shared_samples)
    rows = []
    ids = list(corr_df.columns)
    for i, source in enumerate(ids):
        for target in ids[i + 1 :]:
            shared = int(shared_counts.at[source, target])
            if shared < min_shared_samples:
                continue
            corr = corr_df.at[source, target]
            if pd.isna(corr):
                continue
            corr = float(corr)
            weight = abs(corr) if abs_weights else corr
            if min_abs_corr is not None and abs(corr) < min_abs_corr:
                continue
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "shared_samples": shared,
                    "corr_signed": corr,
                    "weight": float(weight),
                }
            )
    edges = pd.DataFrame(rows)
    if edges.empty:
        return pd.DataFrame(columns=["source", "target", "shared_samples", "corr_signed", "weight"])
    return edges.sort_values(["weight", "shared_samples"], ascending=[False, False]).reset_index(drop=True)

def compute_node_weights_publication_normalized(df: pd.DataFrame, publication_sample_counts: dict[str, int]) -> dict[str, float]:
    '''
    Compute node weights normalized by publication sample counts.
    For each ID, sum over all publications: (|samples_in_pub_with_ID| / |total_samples_in_pub|)
    '''
    node_weights = {}
    for node_id in df["ID"].unique():
        id_df = df[df["ID"] == node_id]
        weight = 0.0
        for pub, samples_count in publication_sample_counts.items():
            if samples_count > 0:
                pub_id_samples = id_df[id_df["Publication"] == pub]["ACC_num"].nunique()
                if pub_id_samples > 0:
                    weight += pub_id_samples / samples_count
        node_weights[node_id] = float(weight)
    return node_weights


def build_nodes(df: pd.DataFrame, keep_ids: set[str] | None = None, node_weights: dict[str, float] | None = None) -> pd.DataFrame:
    node_stats = (
        df.groupby("ID")
        .agg(
            strain=("Strain", "first"),
            segment=("Segment", "first"),
            start=("Start", "first"),
            end=("End", "first"),
            num_samples=("ACC_num", "nunique"),
            num_publications=("Publication", "nunique"),
        )
        .reset_index()
        .rename(columns={"ID": "name"})
    )
    if keep_ids is not None:
        node_stats = node_stats[node_stats["name"].isin(keep_ids)].copy()
    if node_weights is not None:
        node_stats["node_weight"] = node_stats["name"].map(node_weights).fillna(0.0)
    return node_stats


def build_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> ig.Graph:
    if nodes_df.empty or edges_df.empty:
        return ig.Graph()

    # Enforce edge-driven node inclusion: every vertex must participate in >=1 edge.
    connected_ids = set(edges_df["source"]).union(edges_df["target"])
    nodes_df = nodes_df[nodes_df["name"].isin(connected_ids)].copy()
    if nodes_df.empty:
        return ig.Graph()

    valid_ids = set(nodes_df["name"])
    edges_df = edges_df[
        edges_df["source"].isin(valid_ids) & edges_df["target"].isin(valid_ids)
    ].copy()
    if edges_df.empty:
        return ig.Graph()

    graph = ig.Graph.DataFrame(
        edges_df[["source", "target", "weight", "shared_samples", "corr_signed"]],
        directed=False,
        vertices=nodes_df,
        use_vids=False,
    )
    return graph


def compute_graph_statistics(graph: ig.Graph) -> dict:
    num_nodes = graph.vcount()
    num_edges = graph.ecount()
    weights = np.array(graph.es["weight"], dtype=float) if num_edges > 0 else np.array([], dtype=float)
    degrees = np.array(graph.degree(), dtype=float) if num_nodes > 0 else np.array([], dtype=float)
    components = graph.connected_components(mode="weak") if num_nodes > 0 else []
    largest_component_size = max((len(comp) for comp in components), default=0)
    stats = {
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "num_connected_components": int(len(components) if num_nodes > 0 else 0),
        "largest_component_size": int(largest_component_size),
        "density": float(graph.density() if num_nodes > 1 else 0.0),
        "mean_degree": float(degrees.mean() if degrees.size > 0 else 0.0),
        "median_degree": float(np.median(degrees) if degrees.size > 0 else 0.0),
        "mean_edge_weight": float(weights.mean() if weights.size > 0 else 0.0),
        "median_edge_weight": float(np.median(weights) if weights.size > 0 else 0.0),
    }
    return stats


def scale_values(values: np.ndarray, out_min: float, out_max: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    if np.allclose(values.max(), values.min()):
        return np.full(values.shape, (out_min + out_max) / 2.0, dtype=float)
    return out_min + (values - values.min()) * ((out_max - out_min) / (values.max() - values.min()))


def get_node_size_range(num_nodes: int) -> tuple[float, float]:
    if num_nodes <= 60:
        return (12.0, 32.0)
    if num_nodes <= 250:
        return (8.0, 22.0)
    if num_nodes <= 1000:
        return (5.0, 14.0)
    return (3.0, 9.0)

def get_weight_mode_label(abs_weights: bool) -> str:
    return "abs" if abs_weights else "signed"


def with_weight_mode(name: str, abs_weights: bool) -> str:
    label = get_weight_mode_label(abs_weights)
    path = Path(name)
    if path.suffix:
        return str(path.with_name(f"{path.stem}_{label}{path.suffix}"))
    return f"{name}_{label}"


def get_edge_width_range(num_edges: int) -> tuple[float, float]:
    if num_edges <= 200:
        return (1.0, 3.0)
    if num_edges <= 2000:
        return (0.5, 1.5)
    return (0.1, 1.0)

def normalize_membership(membership: list[int]) -> tuple[list[int], dict[int, int]]:
    if len(membership) == 0:
        return [], {}
    unique_ids = sorted(set(membership))
    mapping = {cid: idx for idx, cid in enumerate(unique_ids)}
    return [mapping[cid] for cid in membership], mapping


def get_reduced_label_min_size(values: list[int]) -> float:
    if len(values) == 0:
        return 18.0
    max_digits = max(len(str(int(v))) for v in values)
    return float(12 + 4 * max_digits)


def get_distinctipy_palette(num_colors: int) -> ig.Palette:
    if num_colors <= 0:
        num_colors = 1
    base_colors = distinctipy.get_colors(num_colors, n_attempts=3000, rng=42, pastel_factor=0.4)
    colors = [(r, g, b, 1.0) for (r, g, b) in base_colors]
    return ig.PrecalculatedPalette(colors)


def plot_graph(graph: ig.Graph, output_path: Path, title: str, show_labels: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    if graph.ecount() == 0:
        ax.text(0.5, 0.5, "Graph has no edges", ha="center", va="center")
        ax.set_axis_off()
    else:
        edge_width_min, edge_width_max = get_edge_width_range(graph.ecount())
        widths = scale_values(np.array(graph.es["weight"], dtype=float), edge_width_min, edge_width_max)
        node_size_min, node_size_max = get_node_size_range(graph.vcount())
        node_sizes = scale_values(np.array(graph.vs["num_samples"], dtype=float), node_size_min, node_size_max)
        graph_copy = graph.copy()
        graph_copy.es["width"] = list(widths)
        graph_copy.vs["size"] = list(node_sizes)
        if show_labels:
            graph_copy.vs["label"] = graph_copy.vs["name"]
        ig.plot(graph_copy, target=ax, layout="fr", vertex_label_size=7)
    ax.set_title(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_graph_statistics(stats: dict, output_path: Path, title: str) -> None:
    names = [
        "num_nodes",
        "num_edges",
        "num_connected_components",
        "largest_component_size",
        "density",
        "mean_degree",
        "mean_edge_weight",
    ]
    values = [stats[key] for key in names]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(names)), values, color="#4C72B0")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def resolve_leiden_partition(partition_name: str) -> tuple[str, type]:
    mapping = {
        "modularity": la.ModularityVertexPartition,
        "cpm": la.CPMVertexPartition,
        "rbconfiguration": la.RBConfigurationVertexPartition,
        "rber": la.RBERVertexPartition,
        "surprise": la.SurpriseVertexPartition,
        "significance": la.SignificanceVertexPartition,
    }
    normalized = partition_name.lower().replace("-", "").replace("_", "")
    if normalized not in mapping:
        valid = ", ".join(sorted(mapping.keys()))
        raise ValueError(f"Unknown Leiden partition '{partition_name}'. Choose from: {valid}")
    return normalized, mapping[normalized]


def partition_supports_resolution(partition_cls: type) -> bool:
    params = inspect.signature(partition_cls.__init__).parameters
    return "resolution_parameter" in params


def partition_supports_weights(partition_cls: type) -> bool:
    params = inspect.signature(partition_cls.__init__).parameters
    return "weights" in params


def partition_supports_node_sizes(partition_cls: type) -> bool:
    params = inspect.signature(partition_cls.__init__).parameters
    return "node_sizes" in params


def build_leiden_partition_kwargs(
    partition_cls: type,
    leiden_resolution: float | None,
    node_sizes_available: bool = False,
) -> tuple[dict[str, float | str], bool]:
    kwargs: dict[str, float | str] = {}
    if partition_supports_weights(partition_cls):
        kwargs["weights"] = "weight"
    if partition_supports_node_sizes(partition_cls) and node_sizes_available:
        kwargs["node_sizes"] = "node_weight"
    supports_resolution = partition_supports_resolution(partition_cls)
    if supports_resolution and leiden_resolution is not None:
        kwargs["resolution_parameter"] = float(leiden_resolution)
    return kwargs, supports_resolution


def is_resolution_partition(partition_key: str) -> bool:
    _, partition_cls = resolve_leiden_partition(partition_key)
    return partition_supports_resolution(partition_cls)


def get_partition_quality_name(partition_key: str) -> str:
    names = {
        "modularity": "modularity",
        "cpm": "CPM",
        "rbconfiguration": "RBConfiguration",
        "rber": "RBER",
        "surprise": "surprise",
        "significance": "significance",
    }
    return names.get(partition_key, partition_key)

def supports_negative_weights(partition_cls: type) -> bool:
    # Based on Leidenalg documentation and source code.
    return issubclass(partition_cls, la.CPMVertexPartition)

def run_communities(graph: ig.Graph, methods: list[str], leiden_partitions: list[str], leiden_resolution: float | None, leiden_n_iterations: int, allow_negative_weights: bool) -> tuple[dict[str, list[int]], dict[str, tuple[str, float]]]:
    partitions: dict[str, list[int]] = {}
    partition_quality: dict[str, tuple[str, float]] = {}
    resolved_leiden: list[tuple[str, type]] = [resolve_leiden_partition(name) for name in leiden_partitions]
    
    # Check if graph has node_weight attribute (publication-normalized node weights)
    has_node_weights = graph.vcount() > 0 and "node_weight" in graph.vs.attributes()

    for method in methods:
        lower = method.lower()
        if lower == "leiden":
            if graph.ecount() > 0:
                min_weight = float(min(graph.es["weight"]))
                if min_weight < 0 and not allow_negative_weights:
                    raise ValueError(
                        "Leiden received negative edge weights. "
                        "Use --abs-weights or ensure non-negative weights before Leiden community detection."
                    )
            for leiden_partition_key, leiden_partition_cls in resolved_leiden:
                leiden_kwargs, supports_resolution = build_leiden_partition_kwargs(
                    leiden_partition_cls,
                    leiden_resolution,
                    node_sizes_available=has_node_weights,
                )
                if leiden_resolution is not None and not supports_resolution:
                    logging.info(
                        "Ignoring --leiden-resolution for partition '%s' (not applicable).",
                        leiden_partition_key,
                    )
                if min_weight < 0 and allow_negative_weights and not supports_negative_weights(leiden_partition_cls):
                    logging.warning(
                        "Partition '%s' does not support negative weights. Skipping...",
                        leiden_partition_key,
                    )
                    continue
                logging.info("Running Leiden community detection with partition '%s'...", leiden_partition_key)
                if graph.ecount() == 0:
                    membership = list(range(graph.vcount()))
                    quality_label = leiden_partition_key
                    quality_value = 0.0
                else:
                    partition = la.find_partition(
                        graph,
                        leiden_partition_cls,
                        n_iterations=leiden_n_iterations,
                        **leiden_kwargs,
                    )
                    membership = list(partition.membership)
                    quality_label = leiden_partition_key
                    quality_value = float(partition.quality())
                partitions[f"leiden_{leiden_partition_key}"] = membership
                partition_quality[f"leiden_{leiden_partition_key}"] = (quality_label, quality_value)
        elif lower == "louvain":
            min_weight = float(min(graph.es["weight"]))
            if min_weight < 0:
                logging.warning("Louvain does not allow negative edge weights. Skipping Louvain...")
                continue
            if graph.ecount() == 0:
                membership = list(range(graph.vcount()))
                quality_label = "modularity"
                quality_value = 0.0
            else:
                logging.info("Running Louvain community detection...")
                partition = graph.community_multilevel(weights="weight")
                membership = list(partition.membership)
                quality_label = "modularity"
                quality_value = float(partition.modularity)
            partitions["louvain_default"] = membership
            partition_quality["louvain_default"] = (quality_label, quality_value)
        else:
            logging.warning("Unknown community method skipped: %s", method)
    return partitions, partition_quality


def parse_partition_result_key(result_key: str) -> tuple[str, str]:
    if "_" not in result_key:
        return result_key, "default"
    method, suffix = result_key.split("_", 1)
    return method, suffix


def plot_leiden_resolution_profile(graph: ig.Graph, leiden_partition: str, resolution_min: float, resolution_max: float, number_iterations: int, output_path: Path) -> pd.DataFrame:
    partition_key, partition_cls = resolve_leiden_partition(leiden_partition)
    if not is_resolution_partition(partition_key):
        return pd.DataFrame(columns=["resolution", "quality", "num_communities"])
    if graph.ecount() == 0:
        return pd.DataFrame(columns=["resolution", "quality", "num_communities"])
    if resolution_min <= 0 or resolution_max <= resolution_min:
        raise ValueError(
            "Invalid resolution profile range. Ensure 0 < --leiden-profile-min < --leiden-profile-max."
        )

    sns.set_theme(style="whitegrid", context="paper")
    optimiser = la.Optimiser()
    profile_kwargs, _ = build_leiden_partition_kwargs(partition_cls, leiden_resolution=None)
    weights_arg = profile_kwargs.pop("weights", None)
    profile_partitions = optimiser.resolution_profile(
        graph,
        partition_cls,
        resolution_range=(resolution_min, resolution_max),
        weights=weights_arg,
        number_iterations=number_iterations,
        **profile_kwargs,
    )
    rows = []
    for partition in profile_partitions:
        rows.append(
            {
                "resolution": float(partition.resolution_parameter),
                "quality": float(partition.quality()),
                "num_communities": int(len(set(partition.membership))),
            }
        )
    profile_df = pd.DataFrame(rows).sort_values("resolution").reset_index(drop=True)

    quality_label = get_partition_quality_name(partition_key)

    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    sns.lineplot(
        data=profile_df,
        x="resolution",
        y="quality",
        ax=ax1,
        color="#2A6F97",
        marker="o",
        linewidth=1.4,
        markersize=4,
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("Resolution parameter")
    #ax1.set_ylabel(quality_label, color="#2A6F97")
    ax1.set_ylabel("Quality", color="#2A6F97")
    ax1.tick_params(axis="y", labelcolor="#2A6F97")

    ax2 = ax1.twinx()
    sns.lineplot(
        data=profile_df,
        x="resolution",
        y="num_communities",
        ax=ax2,
        color="#C75B39",
        linewidth=1.2,
    )
    ax2.set_ylabel("Number of communities", color="#C75B39")
    ax2.tick_params(axis="y", labelcolor="#C75B39")

    ax1.set_title(f"Leiden resolution profile ({partition_key})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return profile_df


def save_community_tables(graph: ig.Graph, partitions: dict[str, list[int]], output_dir: Path, abs_weights: bool) -> pd.DataFrame:
    frames = []
    for result_key, membership in partitions.items():
        method, partition_suffix = parse_partition_result_key(result_key)
        frame = pd.DataFrame(
            {
                "ID": graph.vs["name"],
                "method": method,
                "partition": partition_suffix,
                "edge_weights": get_weight_mode_label(abs_weights),
                "community": membership,
            }
        )
        frames.append(frame)
        frame.to_csv(output_dir / f"communities_{with_weight_mode(result_key, abs_weights)}.csv", index=False)
    if not frames:
        return pd.DataFrame(columns=["ID", "method", "partition", "edge_weights", "community"])
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(output_dir / with_weight_mode("communities_all.csv", abs_weights), index=False)
    return merged


def plot_community_graph(graph: ig.Graph, membership: list[int], method: str, output_path: Path, show_labels: bool = False, annotation: str | None = None, max_nodes_to_plot: int | None = None, force_plot: bool = False) -> None:
    if not force_plot and max_nodes_to_plot is not None and graph.vcount() > max_nodes_to_plot:
        logging.info(
            "Skipping community graph plot for %s: %d nodes exceeds threshold of %d. Use --force-community-plot to override.",
            method,
            graph.vcount(),
            max_nodes_to_plot,
        )
        return
    logging.info("Plotting community graph for %s...", method)
    fig, ax = plt.subplots(figsize=(12, 10))
    if graph.vcount() == 0:
        ax.text(0.5, 0.5, "Graph has no nodes", ha="center", va="center")
        ax.set_axis_off()
    else:
        graph_copy = graph.copy()
        edge_width_min, edge_width_max = get_edge_width_range(graph_copy.ecount())
        edge_widths = scale_values(np.array(graph_copy.es["weight"], dtype=float), edge_width_min, edge_width_max)
        graph_copy.es["width"] = list(edge_widths)
        node_size_min, node_size_max = get_node_size_range(graph_copy.vcount())
        node_sizes = scale_values(np.array(graph_copy.vs["num_samples"], dtype=float), node_size_min, node_size_max)
        graph_copy.vs["size"] = list(node_sizes)
        if show_labels:
            graph_copy.vs["label"] = graph_copy.vs["name"]
        membership_idx, _ = normalize_membership(membership)
        communities = ig.VertexClustering(graph_copy, membership=membership_idx)
        palette = get_distinctipy_palette(max(membership_idx) + 1 if membership_idx else 1)
        ig.plot(
            communities,
            target=ax,
            layout="fr",
            mark_groups=True,
            palette=palette,
            vertex_label_size=7,
        )
    ax.set_title(f"Community structure ({method})")
    if annotation:
        ax.text(
            0.02,
            0.02,
            annotation,
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_reduced_community_graph(graph: ig.Graph, membership: list[int]) -> tuple[ig.Graph, pd.DataFrame, pd.DataFrame]:
    if graph.vcount() == 0 or len(membership) == 0:
        empty_nodes = pd.DataFrame(columns=["name", "num_included"])
        empty_edges = pd.DataFrame(columns=["source", "target", "weight"])
        return ig.Graph(), empty_nodes, empty_edges

    membership_idx, community_index = normalize_membership(membership)
    reverse_index = {idx: cid for cid, idx in community_index.items()}

    counts = pd.Series(membership_idx).value_counts().sort_index()
    nodes_df = pd.DataFrame(
        {
            "name": [f"C{i}" for i in counts.index],
            "num_included": counts.values.astype(int),
            "community_id": counts.index.astype(int),
            "original_community_id": [int(reverse_index[i]) for i in counts.index],
        }
    )

    weight_map: dict[tuple[int, int], list[float]] = {}
    for edge in graph.es:
        s, t = edge.tuple
        cs = membership_idx[s]
        ct = membership_idx[t]
        if cs == ct:
            continue
        key = (cs, ct) if cs < ct else (ct, cs)
        weight_map.setdefault(key, []).append(float(edge["weight"]))

    edge_rows = []
    for (cs, ct), values in weight_map.items():
        if len(values) == 0:
            continue
        edge_rows.append(
            {
                "source": f"C{cs}",
                "target": f"C{ct}",
                "weight": float(np.mean(values)),
                "num_inter_edges": int(len(values)),
            }
        )
    edges_df = pd.DataFrame(edge_rows)
    if edges_df.empty:
        reduced_graph = ig.Graph.DataFrame(
            pd.DataFrame(columns=["source", "target", "weight"]),
            directed=False,
            vertices=nodes_df[["name", "num_included", "community_id"]],
            use_vids=False,
        )
    else:
        reduced_graph = ig.Graph.DataFrame(
            edges_df[["source", "target", "weight", "num_inter_edges"]],
            directed=False,
            vertices=nodes_df[["name", "num_included", "community_id"]],
            use_vids=False,
        )
    return reduced_graph, nodes_df, edges_df


def plot_reduced_community_graph(graph: ig.Graph, membership: list[int], method: str, output_path: Path, max_node_size: float = 120.0, min_node_size: float = 40.0, annotation: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    reduced_graph, nodes_df, edges_df = build_reduced_community_graph(graph, membership)

    fig, ax = plt.subplots(figsize=(10, 8))
    if reduced_graph.vcount() == 0:
        ax.text(0.5, 0.5, "No reduced community graph available", ha="center", va="center")
        ax.set_axis_off()
    else:
        membership_idx, _ = normalize_membership(membership)
        palette = get_distinctipy_palette(max(membership_idx) + 1 if membership_idx else 1)
        label_safe_min_size = max(min_node_size, get_reduced_label_min_size(reduced_graph.vs["num_included"]))
        node_sizes = scale_values(
            np.array(reduced_graph.vs["num_included"], dtype=float),
            label_safe_min_size,
            max_node_size,
        )
        reduced_copy = reduced_graph.copy()
        reduced_copy.vs["size"] = list(node_sizes)
        reduced_copy.vs["label"] = [str(int(x)) for x in reduced_copy.vs["num_included"]]
        reduced_copy.vs["color"] = [palette.get(int(i)) for i in reduced_copy.vs["community_id"]]
        edge_labels = None
        if len(reduced_copy.es) > 0:
            ew_min, ew_max = get_edge_width_range(reduced_copy.ecount())
            reduced_copy.es["width"] = list(scale_values(np.array(reduced_copy.es["weight"], dtype=float), ew_min, ew_max))
            if reduced_copy.vcount() < 10:
                edge_labels = [f"{w:.2f}" for w in reduced_copy.es["weight"]]
        ig.plot(
            reduced_copy,
            target=ax,
            layout="fr",
            vertex_label_size=10,
            edge_label=edge_labels,
            edge_label_size=8,
        )
    ax.set_title(f"Reduced community graph ({method})")
    if annotation:
        ax.text(
            0.02,
            0.02,
            annotation,
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return nodes_df, edges_df


def build_run_output_dir(root: Path, strain: str, run_name: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = run_name if run_name else f"shared_corr_{stamp}"
    out = root / strain / suffix
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_pipeline_variant(args: argparse.Namespace, abs_weights: bool, leiden_partitions: list[str], run_name_suffix: str | None = None, allow_negative_weights: bool = False) -> dict:
    publications = resolve_publications(args.strain, args.publications)
    logging.info("Using publications: %s", publications)
    start_time = datetime.now()
    weight_mode_label = get_weight_mode_label(abs_weights)

    df = prepare_strain_dataframe(
        strain=args.strain,
        publications=publications,
        cutoff=args.cutoff,
        unpooled=args.unpooled,
        sample_col=args.sample_col,
        min_publications=args.min_publications,
    )

    if args.min_publications > 1:
        logging.info(
            "Filtered IDs by min_publications=%d, remaining IDs=%d",
            args.min_publications,
            df["ID"].nunique(),
        )

    logging.info("Prepared dataframe with %d rows and %d unique IDs", len(df), df["ID"].nunique())
    if df.empty:
        raise ValueError(
            "No data left after min-publications filtering. Try lowering --min-publications."
        )

    matrix = build_candidate_matrix(df, sample_col=args.sample_col, value_col=args.value_col)
    shared_counts = compute_shared_sample_counts(matrix)
    logging.info("Computed shared sample counts for %d ID pairs", (shared_counts.values > 0).sum() // 2)

    edge_count_df = edge_counts_by_shared_threshold(shared_counts, min_threshold=args.sweep_start)
    elbow = detect_elbow_if_applicable(edge_count_df)

    output_root = Path(args.output_dir)
    output_run_name = args.run_name if run_name_suffix is None else f"{args.run_name}_{run_name_suffix}" if args.run_name else run_name_suffix
    output_dir = build_run_output_dir(output_root, args.strain, with_weight_mode(output_run_name or "shared_corr", abs_weights))

    edge_count_df.to_csv(output_dir / with_weight_mode("edge_count_by_min_shared_samples.csv", abs_weights), index=False)
    plot_edge_count_curve(
        edge_count_df=edge_count_df,
        elbow_threshold=elbow,
        output_path=output_dir / with_weight_mode("edge_count_curve.png", abs_weights),
        strain=args.strain,
    )
    if args.force_min_shared_samples or elbow is None:
        min_shared_samples = args.min_shared_samples
    else:
        logging.info("Detected elbow threshold at %d shared samples with edgecount %d", elbow, edge_count_df.loc[edge_count_df["min_shared_samples"] == elbow, "num_edges"].iloc[0])
        min_shared_samples = elbow
    logging.info("Using min_shared_samples=%d for edge construction", min_shared_samples)
    edges_df = build_weighted_edges(
        matrix=matrix,
        shared_counts=shared_counts,
        min_shared_samples=min_shared_samples,
        corr_method=args.corr_method,
        abs_weights=abs_weights,
        min_abs_corr=args.min_abs_corr,
    )
    connected_ids = set(edges_df["source"].unique().tolist()).union(set(edges_df["target"].unique().tolist())) if not edges_df.empty else set()
    logging.info("Constructed graph edges with %d edges and %d unique connected IDs", len(edges_df), len(connected_ids))
    
    # Compute publication-normalized node weights
    pub_sample_counts = df.groupby("Publication")["ACC_num"].nunique().to_dict()
    node_weights = compute_node_weights_publication_normalized(df, pub_sample_counts)
    
    nodes_df = build_nodes(df, keep_ids=connected_ids, node_weights=node_weights)
    edges_df.to_csv(output_dir / with_weight_mode("graph_edges.csv", abs_weights), index=False)
    nodes_df.to_csv(output_dir / with_weight_mode("graph_nodes.csv", abs_weights), index=False)

    graph = build_graph(nodes_df=nodes_df, edges_df=edges_df)
    graph.write_graphml(str(output_dir / with_weight_mode("graph.graphml", abs_weights)))

    stats = compute_graph_statistics(graph)
    stats["strain"] = args.strain
    stats["corr_method"] = args.corr_method
    stats["edge_weights"] = weight_mode_label
    stats["min_shared_samples"] = min_shared_samples
    stats["min_publications"] = args.min_publications
    stats["elbow_threshold"] = elbow
    stats["leiden_partitions"] = leiden_partitions
    stats["leiden_resolution"] = args.leiden_resolution
    stats["leiden_n_iterations"] = args.leiden_n_iterations
    with open(output_dir / with_weight_mode("graph_stats.json", abs_weights), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    plot_graph(
        graph,
        output_dir / with_weight_mode("graph.png", abs_weights),
        title=f"NGS correlation graph ({args.strain.replace('_', '/')}, {weight_mode_label})",
        show_labels=args.node_labels,
    )
    plot_graph_statistics(stats, output_dir / with_weight_mode("graph_stats.png", abs_weights), title="Graph statistics")

    community_dir = output_dir / "communities"
    community_dir.mkdir(parents=True, exist_ok=True)

    if "leiden" in [m.lower() for m in args.community_methods]:
        for leiden_partition in leiden_partitions:
            if not is_resolution_partition(leiden_partition):
                continue
            logging.info("Plotting Leiden resolution profile for partition '%s'...", leiden_partition)
            profile_df = plot_leiden_resolution_profile(
                graph=graph,
                leiden_partition=leiden_partition,
                resolution_min=args.leiden_profile_min,
                resolution_max=args.leiden_profile_max,
                number_iterations=args.leiden_profile_iterations,
                output_path=community_dir / with_weight_mode(f"leiden_resolution_profile_{leiden_partition}.png", abs_weights),
            )
            profile_df.to_csv(
                community_dir / with_weight_mode(f"leiden_resolution_profile_{leiden_partition}.csv", abs_weights),
                index=False,
            )

    partitions, partition_quality = run_communities(
        graph,
        args.community_methods,
        leiden_partitions=leiden_partitions,
        leiden_resolution=args.leiden_resolution,
        leiden_n_iterations=args.leiden_n_iterations,
        allow_negative_weights=allow_negative_weights,
    )
    save_community_tables(graph, partitions, community_dir, abs_weights=abs_weights)
    for result_key, membership in partitions.items():
        method, partition_suffix = parse_partition_result_key(result_key)
        quality_label, quality_value = partition_quality.get(result_key, (partition_suffix, 0.0))
        annotation = f"{quality_label} = {quality_value:.3f}"
        plot_community_graph(
            graph,
            membership,
            f"{method}_{partition_suffix}",
            community_dir / with_weight_mode(f"community_{result_key}.png", abs_weights),
            show_labels=args.node_labels,
            annotation=annotation,
            max_nodes_to_plot=args.community_plot_max_nodes,
            force_plot=args.force_community_plot,
        )
        reduced_nodes_df, reduced_edges_df = plot_reduced_community_graph(
            graph,
            membership,
            f"{method}_{partition_suffix}",
            community_dir / with_weight_mode(f"reduced_community_{result_key}.png", abs_weights),
            max_node_size=args.reduced_max_node_size,
            annotation=annotation,
        )
        reduced_nodes_df.to_csv(
            community_dir / with_weight_mode(f"reduced_community_nodes_{result_key}.csv", abs_weights),
            index=False,
        )
        reduced_edges_df.to_csv(
            community_dir / with_weight_mode(f"reduced_community_edges_{result_key}.csv", abs_weights),
            index=False,
        )

    with open(output_dir / with_weight_mode("run_config.json", abs_weights), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    logging.info("Saved run outputs to %s\nFinished Pipeline in %s", output_dir, datetime.now() - start_time)
    return {"output_dir": str(output_dir), "stats": stats}

def run_pipeline(args: argparse.Namespace) -> dict:
    primary_result = run_pipeline_variant(
        args,
        abs_weights=args.abs_weights,
        leiden_partitions=args.leiden_partition,
        allow_negative_weights=False,
    )

    extra_partitions = args.leiden_signed_extra_partitions
    if extra_partitions:
        if args.corr_method != "kendall":
            raise ValueError("--leiden-signed-extra-partitions requires --corr-method kendall.")
        logging.info(
            "Running signed Kendall extra pass for partitions: %s",
            ", ".join(extra_partitions),
        )
        run_pipeline_variant(
            args,
            abs_weights=False,
            leiden_partitions=extra_partitions,
            run_name_suffix="signed_extra",
            allow_negative_weights=True,
        )

    return primary_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a correlation network from NGS read-counts and run community detection.")
    parser.add_argument("--strain", required=True, choices=ALL_STRAINS)
    parser.add_argument("--publications", nargs="+", default=None, help="Optional publication filter.")
    parser.add_argument("--sample-col", default="ACC_num", help="Sample column used to define shared samples.")
    parser.add_argument("--value-col", default="NGS_read_count", help="Read-count column for correlation.")
    parser.add_argument("--cutoff", type=int, default=0, help="Optional read-count cutoff via apply_cutoff.")
    parser.add_argument("--unpooled", action="store_true", default=True, help="Load unpooled data.")
    parser.add_argument("--pooled", dest="unpooled", action="store_false", help="Load pooled data instead.")

    parser.add_argument("--min-shared-samples", type=int, default=3, help="Minimum shared ACC_num samples for an ID-ID edge.")
    parser.add_argument("--force-min-shared-samples", action="store_true", help="Force minimum shared ACC_num samples for an ID-ID edge. Otherwise uses elbow detection on edge count curve to suggest a threshold.")
    parser.add_argument("--min-publications", type=int, default=2, help="Minimum number of unique publications an ID must occur in to be considered.")
    parser.add_argument("--sweep-start", type=int, default=2, help="Start threshold for edge-count sweep plot.")
    parser.add_argument("--corr-method", choices=["pearson", "spearman", "kendall"], default="kendall")
    parser.add_argument("--abs-weights", action="store_true", default=True, help="Use absolute correlation for edge weights.")
    parser.add_argument("--signed-weights", dest="abs_weights", action="store_false", help="Use signed correlation as edge weights.")
    parser.add_argument("--min-abs-corr", type=float, default=None, help="Optional minimum absolute correlation filter.")

    parser.add_argument("--community-methods", nargs="+", default=["leiden"], choices=["leiden", "louvain"], help="Community detection methods to run.")
    parser.add_argument("--leiden-partition", nargs="+",default=["modularity"], choices=["modularity", "cpm", "rbconfiguration", "rber", "surprise", "significance"], help="Leiden objective function / partition class (accepts one or more values).")
    parser.add_argument("--leiden-signed-extra-partitions", nargs="+", default=[], choices=["modularity", "cpm", "rbconfiguration", "rber", "surprise", "significance"], help="Optional Leiden partitions to rerun after the main pipeline using pure Kendall (signed) edge weights.")
    parser.add_argument("--leiden-resolution", type=float, default=None, help="Optional Leiden resolution parameter (used by cpm, rbconfiguration, rber).")
    parser.add_argument("--leiden-n-iterations", type=int, default=-1, help="Leiden optimization iterations (-1 runs until no improvement).")
    parser.add_argument("--leiden-profile-min", type=float, default=0.001, help="Minimum resolution value for Leiden resolution-profile plotting.")
    parser.add_argument("--leiden-profile-max", type=float, default=1.0, help="Maximum resolution value for Leiden resolution-profile plotting.")
    parser.add_argument("--leiden-profile-iterations", type=int, default=1, help="Iterations per step when computing the Leiden resolution profile.")
    parser.add_argument("--output-dir", default=str(Path("results") / "ngs_correlation_graph"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--node-labels", action="store_true", help="Show node labels in graph visualizations.")
    parser.add_argument("--reduced-max-node-size", type=float, default=120.0, help="Maximum node size for reduced community graph normalization.")
    parser.add_argument("--community-plot-max-nodes", type=int, default=200, help="Maximum number of graph nodes allowed before skipping the full community plot.")
    parser.add_argument("--force-community-plot", action="store_true", help="Force the full community plot even when the graph exceeds --community-plot-max-nodes.")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    try:
        run_pipeline(args)
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc)
        raise
