import os
import logging
import argparse
import warnings
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR.parent
REPO_ROOT = BASE_DIR.parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, os.path.join(".."))
sys.path.insert(0, os.path.join(os.getcwd(), ".."))

try:
    from utils import load_data, identify_candidates, log_and_norm, apply_cutoff
except ImportError:
    import glob

    logging.error("Failed to import from utils.py. Please ensure that utils.py is in the same directory as model_check.py or in the Python path.")
    paths_to_check = [str(BASE_DIR), str(SRC_DIR), os.path.join(os.getcwd(), "..")]
    logging.error(f"Checked the following paths for utils.py:\n{paths_to_check}")
    py_files = glob.glob(os.path.join(BASE_DIR, "*.py")) + glob.glob(os.path.join(SRC_DIR, "*.py")) + glob.glob(os.path.join(os.getcwd(), "..", "*.py"))
    logging.error(f"Python files found in those directories:\n{py_files}")
    logging.error("Current working directory: " + os.getcwd())
    logging.error(f"Python paths: {sys.path}")
    raise exit(1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import igraph as ig
import leidenalg as la
import networkx as nx
from community import community_louvain
import json
import traceback
from itertools import combinations

import distinctipy
ALL_STRAINS = ["A_Anhui_1_2013", "A_California_07_2009", "A_Connecticut_Flu122_2013", "A_NewCaledonia_20-JY2_1999", "A_Perth_16_2009", "A_PuertoRico_8_1934", "A_turkey_Turkey_1_2005", "A_WSN_33", "B_Brisbane_60_2008", "B_Lee_1940", "B_Victoria_504_2000", "B_Yamagata_16_1988"]
ALL_SEGMENTS = ["PB2", "PB1", "PA", "HA", "NP", "NA", "M", "NS"]
ALL_PUBS = ["Lui2019", "Kupke2020", "Penn2022", "Sheng2018", "Zhuravlev2020", "vdHoecke2015", "Boussier2020", "Southgate2019", "Valesano2020", "Mendes2021", "Alnaji2019", "Berry2021", "Alnaji2021", "Wang2020", "Wang2023", "Pelz2021"]
PUBS_TO_USE = ['Alnaji2021', 'Kupke2020', 'Pelz2021', 'vdHoecke2015', 'Wang2020', 'Wang2023', 'Zhuravlev2020']
STRAIN_COLORS = distinctipy.get_colors(len(ALL_STRAINS),n_attempts=5000,rng=42)
SEGMENT_COLORS = distinctipy.get_colors(len(ALL_SEGMENTS),n_attempts=5000,rng=42)

EDGE_COLORS = {"dvg": "cornflower blue", "exp": "chocolate", "com": "light green", "dvg-com": "cadet blue"}
VERTEX_COLORS = {"dvg": "royal blue", "exp": "coral", "com": "sea green"}
VERTEX_COLORS_RGB = {"dvg": (0.25,0.41,0.88), "exp": (1.0,0.5,0.31), "com": (0.18,0.55,0.34)}
VERTEX_SHAPES = {"dvg": "circle", "exp": "diamond", "com": "square"}
LEGEND_INFO = {"dvg_marker": "o", "dvg_label": "Intersecting DVG",
               "exp_marker": "D", "exp_label": "Experiment",
               "com_marker": "s", "com_label": "Community"}

legend_handles = [mlines.Line2D([], [], color=VERTEX_COLORS_RGB["dvg"], marker="o", linestyle="None", markersize=10, label="Intersecting DVG"),
                  mlines.Line2D([], [], color=VERTEX_COLORS_RGB["exp"], marker="D", linestyle="None", markersize=10, label="Experiment")]
legend_handles_partitions = [mlines.Line2D([], [], color=(0.66,0.66,0.66), marker="o", linestyle="None", markersize=10, label="Intersecting DVG"),
                             mlines.Line2D([], [], color=(0.66,0.66,0.66), marker="D", linestyle="None", markersize=10, label="Experiment")]
legend_handles_community = [mlines.Line2D([], [], color=(0.66,0.66,0.66), marker="o", linestyle="None", markersize=10, label="Lone DVG"),
                             mlines.Line2D([], [], color=(0.66,0.66,0.66), marker="D", linestyle="None", markersize=10, label="Experiment"),
                             mlines.Line2D([], [], color=(0.66,0.66,0.66), marker="s", linestyle="None", markersize=10, label="Community")]
legend_handles_community_color = [mlines.Line2D([], [], color=VERTEX_COLORS_RGB["dvg"], marker="o", linestyle="None", markersize=10, label="Lone DVG"),
                             mlines.Line2D([], [], color=VERTEX_COLORS_RGB["exp"], marker="D", linestyle="None", markersize=10, label="Experiment"),
                             mlines.Line2D([], [], color=VERTEX_COLORS_RGB["com"], marker="s", linestyle="None", markersize=10, label="Community")]

def setup_logging(result_path, verbose = False):
    os.makedirs(result_path, exist_ok=True)
    log_path = os.path.join(result_path, 'results.log')
    if verbose:
        logging.basicConfig(handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
                            format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s', level=logging.DEBUG, force=True)
    else:
        logging.basicConfig(handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO, force=True)

    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.debug("Logging works.")


def distance_matrix_to_edge_list(dist_df, min_threshold=None, max_threshold=None, round=None):
    '''
    Convert a distance matrix DataFrame into an edge list DataFrame.
    
    :param dist_df: square pandas DataFrame (symmetric distance matrix)
    :param min_threshold: float or None, optional min distance to include
    :param max_threshold: float or None, optional max distance to include
    :param round: int or None, optional number of decimal places to round weights

    :return edge_df: pandas DataFrame with columns [source, target, weight]
    '''
    # Not sure why but need to reset underlying reference to prevent columns=index issue
    dist_df = dist_df.copy()
    
    # Stack the matrix into long form
    dist_df.index.name = "source"
    dist_df.columns.name = "target"
    edge_df = dist_df.stack().reset_index()
    edge_df.columns = ["source", "target", "weight"]
    
    # Remove self-loops (distance=0 on diagonal)
    edge_df = edge_df[edge_df["source"] != edge_df["target"]]

    # Apply threshold if given
    if min_threshold is not None:
        edge_df = edge_df[edge_df["weight"] >= min_threshold]
    if max_threshold is not None:
        edge_df = edge_df[edge_df["weight"] <= max_threshold]

    # Avoid duplicate edges (since matrix is symmetric)
    edge_df = edge_df.loc[edge_df["source"] < edge_df["target"]]
    
    if round is not None:
        edge_df = edge_df.round(round)

    return edge_df.reset_index(drop=True)

def generate_acc_edges(dataframe):
    '''
    Returns dataframe with all edges between DVGs and Experiments (ACC_num)
    '''
    acc_edges = []
    for acc, group in dataframe.groupby("ACC_num"):
        sub = group[group["num_pubs"]>1]
        acc_edges.append(pd.DataFrame({"source": sub["ACC_num"].values, "target": sub["ID"].values, "weight": sub["NGS_log_norm"].values}))
    
    return pd.concat(acc_edges, ignore_index=True)

def scale_values(values, pixel_range=(1, 5)):
    vals = np.array(values, dtype=float)
    if vals.max() == vals.min():  # Avoid divide-by-zero
        return np.full_like(vals, np.mean(pixel_range))
    lo, hi = pixel_range
    return lo + (vals - vals.min()) / (vals.max() - vals.min()) * (hi - lo)

def set_attributes(graph,
                   edge_colors = EDGE_COLORS,
                   vertex_colors = VERTEX_COLORS,
                   vertex_colors_rgb = VERTEX_COLORS_RGB,
                   vertex_shapes = VERTEX_SHAPES,
                   scale_edges=False):
    if graph is None:
        return graph
    graph.vs["shape"] = [vertex_shapes[node] for node in graph.vs["type"]]
    graph.vs["size"] = [num+4 if t=="dvg" else 10 for (num, t) in zip(graph.vs["num_pubs"], graph.vs["type"])]
    graph.vs["weight"] = graph.vs["num_pubs"]
    graph.vs["norm_weight"] = [x/max(graph.vs["weight"]) for x in graph.vs["weight"]]
    graph.vs["color"]=[vertex_colors[node] for node in graph.vs["type"]]
    graph.es["color"]=[edge_colors[edge] for edge in graph.es["type"]]
    if scale_edges:
        graph.es["width"] = scale_values(graph.es["weight"])
    else:
        graph.es["width"] = graph.es["weight"]
    return graph

def edge_count_plotting(edge_df, abs, step_size=0.001, title="Edge cound by minimum threshold", name="edge count", path=""):
    edge_count = {}
    if abs:
        name = f'abs {name}'
        title = f'absolute {title}'
    for threshold in np.arange(1.0,step=step_size):
        logging.info(f'Using threshold {threshold}')
        edge_count[threshold] = len(edge_df[edge_df["weight"]>=threshold])
    plot_edge_count(edge_count, title=title, name=name, path=path)
            

def make_graph(strain:str="A_PuertoRico_8_1934", source_publications:list=ALL_PUBS, cutoff:int=0, target_col="NGS_log_norm",
               abs:bool=True, double_output:bool=False, edge_weight_calc:str="correlation",
               include_exp_nodes:bool=True, drop_lonely:bool=False,
               min_correlation:float=None, max_correlation:float=None,
               min_edge_weight:float=None, max_edge_weight:float=None,
               sample=0, na_value=None, plot_edge_counts_dict:dict=None):
    
    def compute_inner(distancematrix_dataframe, outer_edges):
        '''
        Computes inner edges and nodes, referring to DVG-nodes and edges connecting them to each other.
        '''
        inner_edges = distance_matrix_to_edge_list(distancematrix_dataframe, min_threshold=min_correlation, max_threshold=max_correlation)
        inner_edges["type"] = "dvg"
        if drop_lonely:
            inner_nodes = dataframe[dataframe["ID"].isin(set(inner_edges["source"].unique()).union(inner_edges["target"].unique()))][["ID","num_pubs"]].drop_duplicates("ID")
        else:
            inner_nodes = dataframe[dataframe["ID"].isin(set(inner_edges["source"].unique()).union(inner_edges["target"].unique()).union(outer_edges["target"].unique()))][["ID","num_pubs"]].drop_duplicates("ID")
        inner_nodes["type"] = "dvg"
        return inner_edges, inner_nodes
    
    def compute_totals(inner_edges, inner_nodes, outer_edges, outer_nodes):
        '''
        Returns the resulting edge and node dataframes to be used for the graph.
        '''
        if include_exp_nodes:
            total_edges = pd.concat([inner_edges,outer_edges], ignore_index=True)
            total_nodes = pd.concat([inner_nodes,outer_nodes],ignore_index=True)
        else:
            total_edges = inner_edges
            total_nodes = inner_nodes
        return total_edges, total_nodes
    
    def filter_edge_weights(current_edges):
        '''
        Returns the edges after filtering based on min_edge_weight and max_edge_weight
        '''
        if min_edge_weight is not None:
            current_edges = current_edges[current_edges["weight"] >= min_edge_weight]
        if max_edge_weight is not None:
            current_edges = current_edges[current_edges["weight"] <= max_edge_weight]
        return current_edges.reset_index(drop=True)

    # get experimental results to base graph on
    dataframe = load_data(source_publications, unpooled=True)
    dataframe = identify_candidates(dataframe)
    dataframe = dataframe[dataframe["Strain"]==strain]
    dataframe = log_and_norm(dataframe,norm="NGS_log_norm",experiment_col="ACC_num",drop_read_count=False)
    dataframe["num_pubs"] = dataframe.groupby("ID")["Publication"].transform("nunique").astype(int)
    dataframe = apply_cutoff(dataframe, cutoff=cutoff, exp_col="ACC_num")
    dataframe = dataframe[dataframe["num_pubs"]>1]
    logging.info(f'Found {dataframe["ID"].nunique()} intersecting DVGs for Graph')
    if sample > 0:
        logging.info(f'Sampling {sample} random DVGs')
        dataframe = dataframe[dataframe["ID"].isin(dataframe["ID"].drop_duplicates().sample(sample))]
    
    if include_exp_nodes:
        # summarize non-intersecting by adding one node per accession number
        exp_edges = generate_acc_edges(dataframe)
        exp_edges["type"] = "exp"
        exp_nodes = exp_edges["source"].unique()
        exp_nodes = pd.DataFrame({"ID": exp_nodes, "type":["exp"]*len(exp_nodes), "num_pubs":[1]*len(exp_nodes)}).drop_duplicates("ID")
        logging.info(f'Found {exp_nodes["ID"].nunique()} separate experiments connected to DVGs')
    else:
        exp_edges, exp_nodes = pd.DataFrame(columns=["source","target","ID","num_pubs"]), pd.DataFrame()
    
    match edge_weight_calc.lower():
        case "correlation":
            # edge weights based on correlation
            pivot_df = dataframe.pivot_table(index="ACC_num", columns="ID", values=target_col)
            if na_value is not None:
                pivot_df.fillna(na_value, inplace=True)
            distances_df = pivot_df.corr()
            if abs and not double_output:
                distances_df = distances_df.abs()
        case "num_pubs":
            # edge weights based on number of intersections
            pivot_df = dataframe.pivot_table(index="ID", columns="Publication", values=target_col)
            mask = pivot_df.notna().astype(int)
            distances_df = mask.dot(mask.T)
        case _:
            logging.warning(f'Unknown edge weight calculation method: {edge_weight_calc}')
    
    dvg_edges, dvg_nodes = compute_inner(distances_df,exp_edges)
    edges_df, nodes_df = compute_totals(dvg_edges,dvg_nodes,exp_edges,exp_nodes)
    if plot_edge_counts_dict is not None:
        edge_count_plotting(edges_df, **plot_edge_counts_dict)
        return None, None
    edges_df = filter_edge_weights(edges_df)
    graph = ig.Graph.DataFrame(edges_df, directed=False, vertices=nodes_df, use_vids=False)
    
    logging.info(f'{graph.summary(verbosity=-1)}')
    if not double_output:
        if abs:
            return None, graph
        return graph, None
    
    distances_df = distances_df.abs()
    dvg_edges, dvg_nodes = compute_inner(distances_df,exp_edges)
    edges_df, nodes_df = compute_totals(dvg_edges,dvg_nodes,exp_edges,exp_nodes)
    edges_df = filter_edge_weights(edges_df)
    abs_graph = ig.Graph.DataFrame(edges_df, directed=False, vertices=nodes_df, use_vids=False)
    
    logging.info(f'{abs_graph.summary(verbosity=-1)}')
    return graph, abs_graph

def get_partitions(graph, partition_functions, optimiser=la.Optimiser(), split_opt=False):
    logging.debug(f'Getting partitions {partition_functions}')
    new_partitions = {}
    if len(partition_functions)==0:
        return new_partitions
    for method in partition_functions:
        try:
            if isinstance(method, str):
                if "louvain" in method:
                    logging.info(f'Found call for Louvain -> generating networkx graph')
                    nxG = graph.to_networkx()
                    logging.info(f'Getting Louvain communitys')
                    part = community_louvain.best_partition(nxG)
                    part = ig.VertexClustering(graph=graph, membership=list(part.values()))
                    new_partitions[f'{method}'] = part
                elif "integrated" in method:
                    core_method = method.split("_")[1]
                    logging.info(f'Getting ig integrated {core_method}{" weighted" if "weighted" in method else ""}')
                    part = graph.community_leiden(objective_function=core_method,weights="weight",n_iterations=1000,
                                                node_weights="norm_weight" if "weighted" in method else None)
                    new_partitions[f'Integrated {core_method}{" (weighted)" if "weighted" in method else ""}'] = part
                else:
                    logging.warning(f'Unknown partition function: {method}')
            else:
                meth_args = get_partition_args(method, "weight", None)
                logging.info(f'Getting {method.__name__}')
                part = la.find_partition(graph, method, **meth_args)
                if split_opt:
                    new_partitions[f'init {method.__name__}'] = part
                    logging.info(f'Optimising {method.__name__}')
                    tmp = 1
                    while tmp > 0:
                        tmp = optimiser.optimise_partition(part, n_iterations=10)
                    new_partitions[f'opt {method.__name__}'] = part
                else:
                    logging.info(f'Optimising {method.__name__}')
                    tmp = 1
                    while tmp > 0:
                        tmp = optimiser.optimise_partition(part, n_iterations=10)
                    new_partitions[f'{method.__name__}'] = part
        except Exception as e:
            logging.error(f'Issue with partition method {method}:\n{e}\n{traceback.format_exc()}')
    for key in new_partitions.keys():
        logging.info(f'Partition {key}: {new_partitions[key].summary(verbosity=-1)}')
    return new_partitions

def get_partition_args(partition_func, edge_weight=None, node_weight=None):
    match partition_func.__name__:
        case "CPMVertexPartition":
            return {"weights": edge_weight, "node_sizes": node_weight}
        case "SurpriseVertexPartition":
            return {"weights": edge_weight, "node_sizes": node_weight}
        case "ModularityVertexPartition":
            return {"weights": edge_weight}
        case "RBERVertexPartition":
            return {"weights": edge_weight, "node_sizes": node_weight}
        case "RBConfigurationVertexPartition":
            return {"weights": edge_weight}
        case "SignificanceVertexPartition":
            return {}
        case _:
            logging.warning(f'Unknown partition function: {partition_func}')
            return {}

def plot_graph(graph, edge_labels=False, node_labels=False,
               legend_handles=None, legend_contents=("color",["dvg","exp","com"]),
               layout="graphopt", calculated_layout=None,
               figsize=(12,12), title="", name="graph", path=""):
    if legend_handles is None and legend_contents is not None:
        legend_handles = []
        for label in legend_contents[1]:
            legend_handles.append(mlines.Line2D([], [], color=VERTEX_COLORS_RGB[label] if legend_contents[0]=="color" else (0.66,0.66,0.66),
                                                marker=LEGEND_INFO[f'{label}_marker'], linestyle="None", markersize=10,
                                                label=LEGEND_INFO[f'{label}_label']))
    if isinstance(layout, list):
        for lay in layout:
            try:
                plot_graph(graph=graph, edge_labels=edge_labels, node_labels=node_labels,
                        legend_handles=legend_handles, legend_contents=legend_contents,
                        layout=lay, calculated_layout=calculated_layout,
                        figsize=figsize, title=f'{title} ({lay})', name=f'{name} ({lay})', path=path)
            except Exception as e:
                logging.error(f'Issue plotting with layout {lay}:\n{e}\n{traceback.format_exc()}')
                plt.close()
    else:
        if len(graph.es["weight"])<1:
            logging.warning(f'Graph {name} has no edges\n{graph.summary(verbosity=-1)}')
            return
        graph_cp = graph.copy()
        if edge_labels:
            graph_cp.es["label"] = graph_cp.es["weight"]
        if node_labels:
            try:
                graph_cp.vs["label"] = [x["num_included"] if x["type"]=="com" else x["num_pubs"] if x["type"]=="dvg" else x["size"] for x in graph_cp.vs]
            except Exception as e:
                logging.error(f'Error with node labels:\n{e}\n{traceback.format_exc()}')
        try:
            fig, ax = plt.subplots(figsize=figsize)
            ig.plot(graph_cp, layout=calculated_layout if calculated_layout is not None else layout, target=ax)
            if legend_handles is not None:
                ax.legend(handles=legend_handles)

            plt.title(title)
            if path != "":
                os.makedirs(path, exist_ok=True)
                plt.savefig(os.path.join(path,f'{name}.png'))
            else:
                plt.show()
            plt.close()
        except Exception as e:
            logging.error(f'Issue with graph plotting for {name}:\n{e}\n{traceback.format_exc()}\nInput:\nig.plot({graph_cp.summary(verbosity=-1)}, layout={calculated_layout if calculated_layout is not None else layout}, target={ax})\n{f"ax.legend(handles={legend_handles})" if legend_handles is not None else ""}')
            plt.close()

def plot_community(graph, partition, metric:dict=None, figsize_overwrite=None, path="", reset_color=True, name="community", title="Communities"):
    logging.debug(f'Using special plot function for full communities.')
    if isinstance(partition, dict):
        communities = ig.VertexClustering(graph=graph, membership=list(partition.values()))
    else:
        communities = partition
    # setting colors
    num_communities = len(communities)
    palette1 = ig.RainbowPalette(n=num_communities)
    if reset_color:
        logging.debug(f'Adjusting colors')
        for i, community in enumerate(communities):
            graph.vs[community]["color"] = i
            community_edges = graph.es.select(_within=community)
            community_edges["color"] = i
    
    # plotting graph
    logging.debug(f'Plotting')
    fig, ax = plt.subplots(figsize=(20,20) if figsize_overwrite is None else figsize_overwrite)
    ig.plot(
        communities,
        target=ax,
        mark_groups=True,
        palette=palette1,
        vertex_size=15,
        edge_width=0.5,
    )
    plt.title(title, fontsize=26)

    if metric is not None:
        plt.annotate(f'{metric["name"]}: {metric["value"]:.4f}', xy=(10,10), xycoords='figure points', fontsize=18)

    if path != "":
        logging.debug(f'Saving in {path} as {name}.png')
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path,f'{name}.png'))
    else:
        plt.show()
    plt.close()

def plot_reduced_community(graph, partition, metric:dict=None, reset_color=True, reset_shape=True, figsize_overwrite=None, path="", name="reduced community", title="Reduced Communities"):
    logging.debug(f'Using special plot function for reduced communities')
    if isinstance(partition, dict):
        communities = ig.VertexClustering(graph=graph, membership=partition.values())
    else:
        communities = partition
    num_communities = len(communities)
    cluster_graph = prepare_partition_graph(communities)
    palette1 = ig.RainbowPalette(n=num_communities)
    palette2 = ig.GradientPalette("gainsboro", "black")
    graph.es["color"] = [
        palette2.get(int(i))
        for i in ig.rescale(cluster_graph.es["weight"], (0, 255))
    ]
    if reset_color:
        logging.debug(f'Adjusting colors')
        cluster_graph.vs["color"] = [palette1.get(int(i)) for i, col in enumerate(palette1)]
    if reset_shape:
        logging.debug(f'Adjusting shapes')
        cluster_graph.vs["shape"] = "o"

    logging.debug(f'Plotting')
    fig, ax = plt.subplots(figsize=(10,10) if figsize_overwrite is None else figsize_overwrite)
    ig.plot(
        cluster_graph,
        target=ax,
        palette=palette1,
        # set a minimum size on vertex_size, otherwise vertices are too small
        vertex_size=[i for i in ig.rescale(cluster_graph.vs["num_included"], out_range=(5,250), in_range=(1,max(max(cluster_graph.vs["num_included"]),1000)))],
        edge_color="gray",
        edge_width=0.8,
        edge_label=[f'{w:.2f}' for w in cluster_graph.es["weight"]]
    )

    # Add a legend
    legend_handles = []
    for i in range(num_communities):
        handle = ax.scatter(
            [],
            [],
            s=100,
            facecolor=palette1.get(i),
            edgecolor="k",
            label=f'i (contains {int(cluster_graph.vs[i]["num_included"])})',
        )
        legend_handles.append(handle)

    ax.legend(
        handles=legend_handles,
        title="Communities:",
        #bbox_to_anchor=(0, 1.0),
        #bbox_transform=ax.transAxes,
    )

    plt.title(title, fontsize=18)
    if metric is not None:
        plt.annotate(f'{metric["name"]}: {metric["value"]:.4f}', xy=(10,10), xycoords='figure points', fontsize=14)

    if path != "":
        logging.debug(f'Saving in {path} as {name}.png')
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path,f'{name}.png'))
    else:
        plt.show()
    plt.close()

def plot_each_community(graph, partition, reset_color=False, reset_shape=False, figsize_overwrite=None, path="", name="reduced community", title="Subgraph of"):
    logging.debug(f'Using special plot function for community subgraphs')
    if isinstance(partition, dict):
        communities = ig.VertexClustering(graph=graph, membership=partition.values())
    else:
        communities = partition
    num_communities = len(communities)
    for i, community in enumerate(communities):
        sub_graph = graph.subgraph(community)

        palette1 = ig.RainbowPalette(n=num_communities)
        palette2 = ig.GradientPalette("gainsboro", "black")
        sub_graph.es["color"] = [
            palette2.get(int(i))
            for i in ig.rescale(sub_graph.es["weight"], (0, 255))
        ]
        if reset_color:
            logging.debug(f'Adjusting colors')
            sub_graph.vs["color"] = [palette1.get(int(i)) for i, col in enumerate(palette1)]
        if reset_shape:
            logging.debug(f'Adjusting shapes')
            sub_graph.vs["shape"] = "o"

        logging.debug(f'Plotting')
        fig, ax = plt.subplots(figsize=(10,10) if figsize_overwrite is None else figsize_overwrite)
        ig.plot(
            sub_graph,
            target=ax,
            palette=palette1,
            # set a minimum size on vertex_size, otherwise vertices are too small
            vertex_size=[i for i in ig.rescale(sub_graph.vs["num_pubs"], out_range=(5,20), in_range=(1,max(max(sub_graph.vs["num_pubs"]),7)))],
            edge_color=sub_graph.es["color"],
            edge_width=0.8,
        )

        plt.title(f'{title} Community {i}')
        if path != "":
            logging.debug(f'Saving in {path} as {name} sub {i}.png')
            os.makedirs(path, exist_ok=True)
            plt.savefig(os.path.join(path,f'{name} sub {i}.png'))
        else:
            plt.show()
        plt.close()

def plot_graph_partitions(graph=None,
                          name="graph",
                          path="community_detection",
                          layout="lgl",
                          partitions=None,
                          partition_functions=[la.SurpriseVertexPartition, la.ModularityVertexPartition,
                                               la.RBConfigurationVertexPartition,la.RBERVertexPartition,
                                               la.CPMVertexPartition,la.SignificanceVertexPartition,
                                               la.SurpriseVertexPartition]):
    if graph is None:
        logging.warning(f'{name} is None.')
        return
    
    os.makedirs(path,exist_ok=True)
    if layout=="test":
        logging.info(f'Testing all known layouts') # "auto","circle","large","drl" #"kk3d","random_3d","fr3d","sphere"
        for i, test_layout in enumerate(["fr","graphopt","kk","mds","random","rt","dh","grid","rt_circular"]):
            try:
                partitions = plot_graph_partitions(graph=graph, name=name, path=os.path.join(path,test_layout),
                                                   layout=test_layout, partitions=partitions, partition_functions=partition_functions)
            except Exception as e:
                logging.error(f'Issue with {test_layout} test:\n{e}\n{traceback.format_exc()}')
        return
    
    logging.info(f'Plotting {name} with partitions using {layout} layout')

    # setting up style choice dictionaries
    vertex_colors_rgb = {"dvg": (0.25,0.41,0.88), "exp": (1.0,0.5,0.31)}
    legend_handles = [mlines.Line2D([], [], color=vertex_colors_rgb["dvg"], marker="o", linestyle="None", markersize=10, label="Intersecting DVG"),
                      mlines.Line2D([], [], color=vertex_colors_rgb["exp"], marker="D", linestyle="None", markersize=10, label="Experiment")]
    legend_handles_partitions = [mlines.Line2D([], [], color=(0.66,0.66,0.66), marker="o", linestyle="None", markersize=10, label="Intersecting DVG"),
                                 mlines.Line2D([], [], color=(0.66,0.66,0.66), marker="D", linestyle="None", markersize=10, label="Experiment")]
    optimiser = la.Optimiser()
    figsize = (12,12) if len(graph.vs["type"])+len(graph.es["type"])<10000 else (18,18)

    if partitions is None:
        partitions = get_partitions(graph, partition_functions, optimiser)

    logging.info("Getting Base Layout")
    calculated_layout = graph.layout(layout)
    json.dump(calculated_layout.coords, open(os.path.join(path,f'{name} layout.json'), "w"))
    
    logging.info("Plotting base graph")
    plot_graph(graph=graph, legend_handles=legend_handles,
               calculated_layout=calculated_layout, figsize=figsize,
               title=f'Base Graph ({layout} layout)', name=name,
               path=path)
    
    for method in partitions.keys():
        partition = partitions[method]
        logging.info(f'Plotting {method}')
        plot_graph(graph=partition, legend_handles=legend_handles_partitions,
                   calculated_layout=calculated_layout, figsize=figsize,
                   title=f'{method} ({layout} layout)', name=f'{name} {method}',
                   path=path)
    
    return partitions

def partition_to_df(partition):
    id_dict = {i: {"name": i,
                   "weight": len(x),
                   "indexes": x,
                   "ids": abs_g.vs[x]["name"],
                   "types": abs_g.vs[x]["type"],
                   "nums_pubs": abs_g.vs[x]["num_pubs"]} for i, x in enumerate(partition)}
    try: 
        return pd.DataFrame([id_dict[key] for key in id_dict.keys()])
    except Exception as e:
        logging.error(f'Issue with partition dataframe generation:\n{e}\n{traceback.format_exc()}')
        return pd.DataFrame(columns=["name","weight","indexes","ids","types","nums_pubs"])

def get_partition_edges(partition_df, graph):
    edges = []
    graph_dict_dict = graph.to_dict_dict()
    for key in partition_df["name"].unique():
        locals = partition_df[partition_df["name"]==key]["indexes"].values[0]
        local_edges = []
        for source in locals:
            if source in graph_dict_dict.keys():
                local_edges = local_edges + [{"source": source,
                                  "target": target,
                                  "weight": graph_dict_dict[source][target]["weight"],
                                  "type": graph_dict_dict[source][target]["type"]}
                                  for target in graph_dict_dict[source].keys() if target not in locals]
        out_edges = []
        for local_edge in local_edges:
            out_edges.append({"source": key,
                          "target": partition_df[partition_df["indexes"].apply(lambda x: local_edge["target"] in x)]["name"].values[0],
                          "weight": local_edge["weight"],
                          "type": "com"})
        if len(out_edges)>0:
            out_df = pd.DataFrame(out_edges)
            out_df["weight"] = out_df.groupby(["source","target"])["weight"].transform("mean")
            edges.append(out_df.drop_duplicates(["source","target"]))
    try:
        return pd.concat(edges, ignore_index=True)
    except Exception as e:
        logging.info(f'Issue with creation of edge dataframe:\n{e}')
        return pd.DataFrame(columns=["source","target","weight","type"])

def prepare_partition_graph(partition, 
                            edge_colors = EDGE_COLORS,
                            vertex_colors = VERTEX_COLORS,
                            vertex_shapes = VERTEX_SHAPES,
                            min_size=3, max_size=10):
    def fix_edge_type(edge, node_df):
        source = node_df["type"].iat[int(edge["source"])]
        target = node_df["type"].iat[int(edge["target"])]
        if source=="dvg":
            if target=="dvg":
                return "dvg"
            return "dvg-com"
        if target=="dvg":
            return "dvg-com"
        return "com"

    logging.info("Making cluster graph")
    try:
        com_graph = partition.cluster_graph(combine_vertices={"num_pubs": "max", "name": list, "size": "sum", "type": list},
                                            combine_edges={"weight": "mean"})
    except:
        logging.error(f'Error occured during combining of vertices and edges:\n{traceback.format_exc()}')
        com_graph = partition.cluster_graph(combine_vertices={"num_pubs": "max", "name": list, "size": "sum", "type": list},
                                            combine_edges="mean")
    com_graph.vs["num_included"] = [len(x) for x in com_graph.vs["type"]]
    com_graph.vs["type"] = ["com" if len(x)>1 else "dvg" if x[0]=="dvg" else "exp" for x in com_graph.vs["type"]]
    com_graph.vs["size"] = [4+len(x["name"]) if x["type"]=="com" else x["size"] for x in com_graph.vs]
    com_graph.vs["size"] = scale_values(com_graph.vs["size"], (min_size, max_size))
    com_graph.vs["shape"] = [vertex_shapes[x] for x in com_graph.vs["type"]]
    com_graph.vs["color"] = [vertex_colors[x] for x in com_graph.vs["type"]]
    try:
        edge_df = com_graph.get_edge_dataframe()
        node_df = com_graph.get_vertex_dataframe()
    except Exception as e:
        logging.error(f'Error occured during use of edge and/or vertex dataframes:\n{e}\n{traceback.format_exc()}')
    try:
        edge_df["type"] = edge_df.apply(lambda row: fix_edge_type(row, node_df), axis=1)
        com_graph.es["type"] = edge_df["type"]
    except Exception as e:
        logging.error(f'Error occured while trying to fix edge_type:\n{e}\n{traceback.format_exc()}')
        edge_df["type"] = "com"
    com_graph.es["color"] = [edge_colors[x] for x in com_graph.es["type"]]
    com_graph.es["width"] = scale_values(com_graph.es["weight"])
    return com_graph

def make_partition_graph(original_graph, partition, old=False, min_size=3, max_size=10):
    if not old:
        return prepare_partition_graph(partition)
    else:
        nodes_df = partition_to_df(partition=partition)
        edges_df = get_partition_edges(partition_df=nodes_df, graph=original_graph)
        return ig.Graph.DataFrame(edges=edges_df, directed=False, vertices=nodes_df, min_size=min_size, max_size=max_size)

def plot_subgraphs(vertex_clustering:ig.VertexClustering, figsize=(12,12),
                   title=None, name="partition", path="", layout="graphopt",
                   min_size=2, save_graphs=False):
    logging.info(f'Plotting subgraphs for {name}\nSaving in {path}')
    for i, sub_graph in enumerate(vertex_clustering.subgraphs()):
        if len(sub_graph.vs["type"]) < min_size:
            logging.warning(f'Not plotting Subgraph {i} as it is below minimum size {min_size}')
            continue
        try:
            plot_graph(graph=sub_graph,
                    edge_labels=len(sub_graph.es["type"])<100, node_labels=len(sub_graph.vs["type"])<100,
                    layout=layout, figsize=figsize, legend_contents=("color", ["dvg", "exp"]),
                    name=f'{name} Cluster {i}', title=f'{title} Cluster {i}', path=path)
            if save_graphs:
                save_graph(graph=sub_graph, path=path, name=f'{name} Cluster {i}')
        except Exception as e:
            logging.error(f'Issue with subplot {name} Cluster {i}:\n{e}\n{traceback.format_exc()}\nInput: plot_graph(graph={sub_graph.summary(verbosity=-1)}, edge_labels={len(sub_graph.es["type"])<100}, node_labels={len(sub_graph.vs["type"])<100}, layout={layout}, figsize={figsize}, legend_contents=("color", ["dvg", "exp"]), name={name} Cluster {i}, title={title} Cluster {i}, path=path)')

def plot_partition_graphs(original_graph:ig.Graph, partitions:dict,
                          figsize:tuple=(12,12), calculated_layout=None, layout:str="graphopt",
                          name="partition", path="", save_graphs:bool=False,
                          subgraphs:bool=False, subgraphs_at:int=0, min_size=2):
    sub_path = os.path.join(path,f'{name} Subgraphs')
    logging.info(f'Plotting partition graphs for {name}\nSaving in {path}')
    for partition_name in partitions.keys():
        try:
            partition_graph = make_partition_graph(original_graph=original_graph, partition=partitions[partition_name])
            try:
                plot_graph(graph=partition_graph,
                        edge_labels=len(partition_graph.es["type"])<100, node_labels=len(partition_graph.vs["type"])<100,
                        calculated_layout=calculated_layout if calculated_layout is not None else layout,
                        figsize=figsize, title=f'{partition_name} ({layout} layout)',
                        name=f'{name} {partition_name}', path=path)
            except Exception as e:
                logging.error(f'Issue with plotting partition graph for {partition_name}:\n{e}\n{traceback.format_exc()}')
                try:
                    logging.info(f'Trying alternative plot functions')
                    plot_community(graph=original_graph, partition=partitions[partition_name], path=path, name=f'{name} {partition_name}', title=f'{partition_name}')
                    plot_reduced_community(graph=original_graph, partition=partitions[partition_name], path=path, name=f'{name} {partition_name} reduced', title=f'{partition_name} (reduced)')
                except Exception as e:
                    logging.error(f'Issue with alternive plotting of partition graph for {partition_name}:\n{e}\n{traceback.format_exc()}')
            if save_graphs:
                save_graph(graph=partition_graph, path=path, name=f'{name} {partition_name}')
            if subgraphs or subgraphs_at>=len([x for x in partitions[partition_name].sizes() if x>=min_size]):
                try:
                    os.makedirs(sub_path, exist_ok=True)
                    plot_subgraphs(vertex_clustering=partitions[partition_name], figsize=figsize, title=f'{partition_name}', name=f'{name} {partition_name}', path=sub_path, min_size=min_size)
                except Exception as e:
                    logging.error(f'Issue with subgraphs for {partition_name}:\n{e}\n{traceback.format_exc()}')
        except Exception as e:
            logging.error(f'Issue with partition graph for {partition_name}:\n{e}\n{traceback.format_exc()}')

def save_graph(graph, path, name, format="picklez"):
    if graph is not None:
        try:
            match format:
                case "picklez":
                    graph.write_picklez(os.path.join(path,f'{name}.gzip'))
                case "svg":
                    graph.write_svg(os.path.join(path,f'{name}.svg'))
                case _:
                    logging.error(f'Unknown format {format} to save graph {name}, using picklez instead.')
                    graph.write_picklez(os.path.join(path,f'{name}.gzip'))
        except Exception as e:
            logging.error(f'Issue with saving graph {name} in {format}:\n{e}\n{traceback.format_exc()}')

def plot_edge_count(data_dict, title="edge counts by weight threshold", name="edge_counts", path=""):
    logging.info(f'Plotting edge counts and saving in {path} as {name}')
    x = sorted(data_dict.keys())
    y = [data_dict[k] for k in x]

    plt.plot(x, y, marker='o')
    plt.title(f'{title}')
    plt.xlabel("Minimum Edge Weight")
    plt.ylabel("Edge Count")
    plt.xlim(0, 1)
    if path != "":
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path,f"{name}.png"))
    else:
        plt.show()

def plot_step_metrics(leid_mods, louv_mods, metric="Modularity", path="", name="test", title="Quality progresssion"):
    plt.figure(figsize=(8, 5))

    if len(leid_mods)>0:
        plt.plot(list(range(1,len(leid_mods)+1)), leid_mods, label="Leiden", marker="o", color="tab:red")
    if len(louv_mods)>0:    
        plt.plot(list(range(1,len(louv_mods)+1)), louv_mods, label="Louvain", marker="o", color="tab:blue")
    plt.xticks(range(1,max(len(leid_mods)+1, len(louv_mods)+1)))

    plt.xlabel("Iteration")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    if path != "":
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path,f"{name}.png"))
    else:
        plt.show()

def conductance(g, membership):
    comms = {}
    for c in set(membership):
        nodes = [i for i, m in enumerate(membership) if m == c]
        edges_inside = 0
        edges_cut = 0
        
        for e in g.es:
            u, v = e.tuple
            if u in nodes and v in nodes:
                edges_inside += 1
            elif (u in nodes) ^ (v in nodes):
                edges_cut += 1
        
        denom = edges_inside + edges_cut
        comms[c] = edges_cut / denom if denom > 0 else 0.0

    return comms

def intracluster_density(g, membership):
    densities = {}
    for c in set(membership):
        nodes = [i for i, m in enumerate(membership) if m == c]
        k = len(nodes)
        if k < 2:
            densities[c] = 0.0
            continue

        possible = k * (k - 1) / 2
        inside = g.induced_subgraph(nodes).ecount()
        densities[c] = inside / possible

    return densities

def intercluster_density(g, membership):
    densities = {}
    groups = {c: [i for i, m in enumerate(membership) if m == c]
              for c in set(membership)}
    
    for c1, c2 in combinations(groups.keys(), 2):
        A, B = groups[c1], groups[c2]
        possible = len(A) * len(B)

        count = 0
        for e in g.es:
            u, v = e.tuple
            if (u in A and v in B) or (u in B and v in A):
                count += 1

        densities[(c1, c2)] = count / possible if possible > 0 else 0.0

    return densities

def cut_ratio(g, membership):
    n = g.vcount()
    results = {}
    
    for c in set(membership):
        nodes = [i for i,m in enumerate(membership) if m == c]
        k = len(nodes)

        if k == 0 or k == n:
            results[c] = 0.0
            continue
        
        cut = 0
        for e in g.es:
            u, v = e.tuple
            if (u in nodes) ^ (v in nodes):
                cut += 1

        denom = k * (n - k)
        results[c] = cut / denom

    return results

def compare_partitions(graph, name_a, partition_a, name_b, partition_b):
    logging.info(f'Comparing Partitions {name_a} and {name_b}')
    metric_dict = dict()
    try:
        metric_dict["vi"] = ig.compare_communities(partition_a, partition_b, method="vi")
        metric_dict["nmi"] = ig.compare_communities(partition_a, partition_b, method="nmi")
        metric_dict["split-join"] = ig.compare_communities(partition_a, partition_b, method="split-join")
        metric_dict["rand"] = ig.compare_communities(partition_a, partition_b, method="rand")
        metric_dict["adjusted rand"] = ig.compare_communities(partition_a, partition_b, method="adjusted_rand")
    except:
        logging.error(f'Issue with metric dict:\n{traceback.format_exc()}')
    logging.info(pd.Series(metric_dict))
    logging.info(ig.split_join_distance(partition_a, partition_b))
    logging.info(f'{name_a}: {graph.modularity(partition_a)}\t\t{name_b}: {graph.modularity(partition_b)}')
    logging.info(f'Conductance:\n{name_a}\n{pd.Series(conductance(graph, partition_a))}\n{name_b}\n{pd.Series(conductance(graph, partition_b))}')
    logging.info(f'Intracluster Density:\n{name_a}\n{pd.Series(intracluster_density(graph, partition_a))}\n{name_b}\n{pd.Series(intracluster_density(graph, partition_b))}')
    logging.info(f'Intercluster Density:\n{name_a}\n{pd.Series(intercluster_density(graph, partition_a))}\n{name_b}\n{pd.Series(intercluster_density(graph, partition_b))}')
    logging.info(f'Cut Ratio:\n{name_a}\n{pd.Series(cut_ratio(graph, partition_a))}\n{name_b}\n{pd.Series(cut_ratio(graph, partition_b))}\n')

def louv_vs_leid(graph, path="", name="test", seed=42, debug=False):
    # leiden
    leid_opt = la.Optimiser()
    leid_part = la.find_partition(graph, la.ModularityVertexPartition, n_iterations=1, weights="weight", seed=seed)
    leid_mods = [graph.modularity(leid_part,weights="weight")]
    tmp, xtra = 1, 1
    while xtra > 0:
        while tmp > 0:
            tmp = leid_opt.optimise_partition(leid_part, n_iterations=1)
            leid_mods.append(graph.modularity(leid_part,weights="weight"))
        xtra -= 0.5
    logging.info(f'Step-wise modularities of leiden partitions:\n{leid_mods}')
    
    # plot leiden partition
    if not debug:
        save_part_membership(graph=graph, partition=leid_part, name=f'{name} leiden', path=path)
        plot_community(graph=graph, partition=leid_part, metric={"name": "Modularity", "value": leid_mods[-1]}, path=path, name=f'{name} leiden', title=f'Leiden Community Graph')
        plot_reduced_community(graph=graph, partition=leid_part, metric={"name": "Modularity", "value": leid_mods[-1]}, path=path, name=f'{name} leiden reduced', title=f'Leiden Community Graph (reduced)')
        plot_each_community(graph=graph, partition=leid_part, path=path, name=f'{name} leiden subgraphs', title=f'Leiden')

    # louvain
    logging.info(f'Getting networkx graph')
    nx_graph = graph.to_networkx()
    logging.info(f'Getting Louvain communitys')
    try:
        # need dendrogram to plot step-wise modularities
        louv_dend = community_louvain.generate_dendrogram(graph=nx_graph, random_state=seed)
        louv_part = community_louvain.partition_at_level(louv_dend, len(louv_dend)-1)
        logging.info(f'{louv_dend}')
        louv_mods = [community_louvain.modularity(community_louvain.partition_at_level(louv_dend, i), nx_graph) for i in range(len(louv_dend))]
        logging.info(f'Step-wise modularities of louvain partitions:\n{louv_mods}')
        plot_step_metrics(leid_mods=leid_mods, louv_mods=louv_mods, path=path, name=f'{name} modularity progression', title="Step-wise Modularity of Partitions")
    except Exception as e:
        logging.error(f'Issue with dendrogram version:\n{traceback.format_exc()}')
        logging.info(f'Skipping dendrogram version and just using best partition for louvain')
        louv_part = community_louvain.best_partition(nx_graph)
    try:
        compare_partitions(graph=graph, name_a="Leiden", partition_a=leid_part.membership, name_b="Louvain", partition_b=louv_part.values())
    except:
        logging.error(f'Issue with comparison function:\n{traceback.format_exc()}')
    # plot louvain partition
    if not debug:
        save_part_membership(graph=graph, partition=louv_part, name=f'{name} louvain', path=path)
        plot_community(graph=graph, partition=louv_part, metric={"name": "Modularity", "value": louv_mods[-1]}, path=path, name=f'{name} louvain', title=f'Louvain Community Graph')
        plot_reduced_community(graph=graph, partition=louv_part, metric={"name": "Modularity", "value": louv_mods[-1]}, path=path, name=f'{name} louvain reduced', title=f'Louvain Community Graph (reduced)')
        plot_each_community(graph=graph, partition=louv_part, path=path, name=f'{name} louvain subgraphs', title=f'Louvain')

    return leid_part, louv_part

def quick_leid(graph, path="", name="test", seed=42, debug=False):
    partition_funcs = {la.SurpriseVertexPartition: "Surprise",# la.ModularityVertexPartition: "Modularity",
                  la.RBConfigurationVertexPartition: "RBConf", la.RBERVertexPartition: "RBER",
                  la.CPMVertexPartition: "CPM", la.SignificanceVertexPartition: "Significance"}
    result_partitions = {}
    for partition_func in partition_funcs.keys():
        try:
            cur_metric = partition_funcs[partition_func]
            leid_opt = la.Optimiser()
            leid_part = la.find_partition(graph, partition_func, n_iterations=1, seed=seed,
                                        **get_partition_args(partition_func=partition_func, edge_weight="weight", node_weight="num_pubs"))
            leid_mods = [graph.modularity(leid_part,weights="weight")]
            tmp, xtra = 1, 1
            while xtra > 0:
                while tmp > 0:
                    tmp = leid_opt.optimise_partition(leid_part, n_iterations=1)
                    leid_mods.append(graph.modularity(leid_part,weights="weight"))
                xtra -= 0.5
            logging.info(f'Step-wise modularities of leiden partitions:\n{leid_mods}')
            result_partitions[cur_metric] = leid_part
            # plot leiden partition
            if not debug:
                plot_community(graph=graph, partition=leid_part, metric={"name": partition_func.__name__, "value": leid_mods[-1]}, path=path, name=f'{name} {cur_metric} leiden', title=f'Leiden Community Graph ({cur_metric})')
                plot_reduced_community(graph=graph, partition=leid_part, metric={"name": partition_func.__name__, "value": leid_mods[-1]}, path=path, name=f'{name} {cur_metric} leiden reduced', title=f'Reduced Leiden Community Graph ({cur_metric})')
                plot_each_community(graph=graph, partition=leid_part, path=path, name=f'{name} {cur_metric} leiden subgraphs', title=f'Leiden {cur_metric}')
                plot_step_metrics(leid_mods=leid_mods, louv_mods=[], metric=cur_metric, path=path, name=f'{name} {cur_metric} Modularity progression', title="Step-wise Modularity of Partitions")
                logging.info(f'Conductance:\n{cur_metric}\n{pd.Series(conductance(graph, leid_part))}\n')
                logging.info(f'Intracluster Density:\n{cur_metric}\n{pd.Series(intracluster_density(graph, leid_part))}\n')
                logging.info(f'Intercluster Density:\n{cur_metric}\n{pd.Series(intercluster_density(graph, leid_part))}\n')
                logging.info(f'Cut Ratio:\n{cur_metric}\n{pd.Series(cut_ratio(graph, leid_part))}\n')
        except:
            logging.error(f'Issue while trying to use {cur_metric}:\n{traceback.format_exc()}')
    return result_partitions

def save_part_membership(graph, partition, name, path):
    if isinstance(partition, dict):
        communities = ig.VertexClustering(graph=graph, membership=list(partition.values()))
    else:
        communities = partition
    part_df = pd.DataFrame({"ID":graph.vs["name"],"Membership":communities.membership})
    part_df.to_csv(os.path.join(path, f'{name} membership.csv'))

def make_histogram(dataframe, name, feature, thresholds, bin_overwrite=None, iqr=True, result_path="") -> None:
    '''
    Creates a histogram for the given feature. Marks the median and, for normed log of NGS count, any other given
    thresholds in the plot.

    :param df: Dataframe containing the feature to plot and the NGS_read_count column
    :param name: Name for the files to save each plot
    :param feature: Feature to plot
    :param thresholds: List of thresholds to mark in the plot
    :param result_path: Path to save the plots

    :return: None
    '''
    logging.info(f'Creating histogram for {name}: {feature}')
    plt.rcParams.update({'font.size': 14})
    df_c = dataframe.copy()
    uniques = df_c[feature].nunique()
    if bin_overwrite is not None:
        bins = bin_overwrite
    else:
        if uniques < 20:
            bins = 2*uniques
        elif uniques < 150:
            bins = 40
        else:
            bins = 100
        logging.debug(f'bins are set to {bins}, based on number of uniques {uniques}')
        if feature not in ["Intersections", "Inter_norm"]:
            bins = int(uniques/3)
        else:
            bins = uniques
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df_c[feature], bins=bins, color='b', alpha=0.4)
    if "length_proportion" in dataframe.columns:
        df_long_del = df_c[df_c["length_proportion"] >= 0.85].copy()
        ax.hist(df_long_del[feature], bins=bins, color='r', alpha=0.3, label='Long DelVGs')

    median = np.median(dataframe[feature])
    if iqr:
        q1, q3 = np.percentile(df_c[feature], [25, 75])
        ax.axvspan(
            q1, q3,
            color='teal',
            alpha=0.15,
            label='IQR (25–75%)'
        )
        ax.axvline(q1, color='teal', linestyle=':', linewidth=2)
        ax.axvline(q3, color='teal', linestyle=':', linewidth=2)
    ax.axvline(median, color='teal', linestyle='dashed', linewidth=3, label='Median')
    ax.text(median-.01, ax.get_ylim()[1]*0.95, f'{median:.2f}', color='teal', ha='right')


    lo = df_c[feature].min()
    hi = df_c[feature].max()
    xlim = ax.get_xlim()

    # Adjusting plots with negative x-values, to center around 0
    if lo < 0:
        ax.set_xlim(-max(abs(hi),abs(lo)), max(abs(hi),abs(lo)))
    else:
        ax.set_xlim(0, xlim[1])
    
    if max(df_c[feature]) == 1:
        ax.set_xlim(ax.get_xlim()[0],1)

    # Marking thresholds for ngs_log_norm
    if feature == 'NGS_log_norm':
        for i, threshold in enumerate(thresholds):
            ax.axvline(threshold, color='r', linestyle='dashed', linewidth=2, label=i)
            if threshold == median:
                continue
            # leaving out Text for threshold, if it is too far to the left
            if threshold <= lo:
                continue
            ax.text(threshold-.01, ax.get_ylim()[1]*0.8, f'{threshold:.2f}', color='r', ha='right')


    ax.set_xlabel(f'{feature.replace("_", " ")}')
    ax.set_ylabel('Count')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_title(f'{name}: {feature.replace("_", " ")} Histogram')
    plt.tight_layout()
    if result_path != "":
        os.makedirs(result_path, exist_ok=True)
        path = os.path.join(result_path, f'{name}_{feature}_histogram.png')
        plt.savefig(path)
    else:
        plt.show()
    plt.close(fig)

def make_barplot(dataframe, name, feature, y_column, result_path="", groups_overwrite=None, heatmap=False) -> None:
    ''' Creates a bar plot for the given feature. If heatmap=True, the function additionally creates a plot with each bar
    colored as a heatmap, according to the normed logarithm of the data's NGS count.

    :param df: Dataframe that includes the NGS_log_norm column and all columns that belong to the feature
    :param name: Name for the files to save each plot
    :param result_path: Path to save the plots
    :param feature: Feature to plot
    :param heatmap: Boolean to decide if a heatmap-barplot should be created

    :return: None
    '''
    plt.rcParams.update({'font.size': 14})
    df_c = dataframe
    if groups_overwrite is not None:
        all_groups = groups_overwrite
    else:
        match feature:
            case 'Strain':
                all_groups = ALL_STRAINS
            case 'Segment':
                all_groups = ALL_SEGMENTS
            case 'Publication':
                all_groups = ALL_PUBS
            case _:
                logging.error(f'Feature {feature} not recognized for barplots.')

    sums = {}
    for group in all_groups:
        sums[group] = 0
        try:
            sums[group] = len(df_c[df_c[feature]==group])
        except:
            logging.error(traceback.format_exc())
    keys = sums.keys()
    values = sums.values()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.barh(keys, values)
    ax.set_title(f'{name}: {feature} distribution')
    ax.set_xlabel('Number of candidates')
    plt.tight_layout()
    if result_path != "":
        os.makedirs(result_path, exist_ok=True)
        path = os.path.join(result_path, f'{name}_{feature}_barplot.png')
        plt.savefig(path)
    else:
        plt.show()
    plt.close()

def community_dvg_analysis(graph:ig.Graph, dataframe:pd.DataFrame=None, partitions:dict=None, source_publications:list=None, path="", name=""):
    if dataframe is None:    
        dataframe = load_data(ALL_PUBS, unpooled=True)
        dataframe = identify_candidates(dataframe)
        dataframe = log_and_norm(dataframe, experiment_col="ACC_num", drop_read_count=False)
        dataframe = dataframe[dataframe["ID"].isin(graph.vs["name"])]
    for key in partitions.keys():
        try:
            logging.info(f'Analysis of {key} DVGs')
            partition = partitions[key]
            if isinstance(partition, dict):
                communities = ig.VertexClustering(graph=graph, membership=partition.values())
            else:
                communities = partition

            for i, community in enumerate(communities):
                com_df = dataframe[dataframe["ID"].isin(graph.vs[community]["name"])]

                # Plot NGS read counts
                make_histogram(dataframe=com_df, name=f'{name} {key} {i}', feature="NGS_read_count", bin_overwrite=20, thresholds=[], result_path=path)
                make_histogram(dataframe=com_df, name=f'{name} {key} {i}', feature="NGS_log_norm", bin_overwrite=20, thresholds=[], result_path=path)
                
                # Plot Segment distribution
                make_barplot(dataframe=com_df.drop_duplicates("ID"), name=f'{name} {key} {i}', feature="Segment", y_column="NGS_log_norm", result_path=path)
                make_barplot(dataframe=com_df.drop_duplicates(["ID","Publication"]), name=f'{name} {key} {i}', feature="Publication", y_column="NGS_log_norm", groups_overwrite=source_publications, result_path=path)

        except:
            logging.error(f'Issue with analyzing partition {key}:\n{traceback.format_exc()}')

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='Run classifiers on DI-RNA data')
    parser.add_argument('-l', '--layout', type=str, help='Algorithm to use for layout of graphs. Enter <test> to try out each possibility.', default='auto')
    parser.add_argument('-n', '--name', type=str, help='Name of the test run. Affects result directory path.', default='test')
    parser.add_argument('-s', '--strain', type=str, help='Name of the strain for which to build the graph.', default="A_PuertoRico_8_1934")
    parser.add_argument('-c', '--cutoff', type=int, help='Minimum NGS count to be considered.', default=0)
    parser.add_argument('-v', '--verbose', action="store_true", help='Whether to use verbose logging.')
    parser.add_argument('-a', '--absolute_correlations', action="store_true", help='Whether to use absolute correlations.')
    parser.add_argument('-b', '--both_correlations', action="store_true", help='Whether to use both correlations.')
    parser.add_argument('-z', '--na_to_value', type=int, help='Set NaN values for correlation calc to specified value.', default=None)
    parser.add_argument('-q', '--min_correlation', type=float, help='Minimum correlation value to be considered.', default=None)
    parser.add_argument('-w', '--max_correlation', type=float, help='Maximum correlation value to be considered.', default=None)
    parser.add_argument('-e', '--min_edge_weight', type=float, help='Minimum edge weight to be considered.', default=None)
    parser.add_argument('-r', '--max_edge_weight', type=float, help='Maximum edge weight to be considered.', default=None)
    parser.add_argument('-m', '--scale', action="store_true", help="Use to scale edge widths up.")
    parser.add_argument('-p', '--partitions', type=str, help='Partition functions to test', default='standard')
    parser.add_argument('-d', '--mode', type=int, help='Decide on which test script to run', default=0)
    parser.add_argument('-t', '--target_col', type=str, help='Column to use for correlations', default="NGS_log_norm")
    parser.add_argument('-x', '--seed', type=int, help='Seed for random number generators', default=42)
    parser.add_argument('-f', '--source_publications', nargs='+', default=ALL_PUBS)
    
    args = parser.parse_args()
    part_funcs = args.partitions
    na_value = args.na_to_value
    mode = args.mode

    if part_funcs == "standard":
        part_funcs = {"base":[la.CPMVertexPartition],
                      "abs":[la.SurpriseVertexPartition, la.ModularityVertexPartition,
                             la.RBConfigurationVertexPartition,la.RBERVertexPartition,
                             la.CPMVertexPartition,la.SignificanceVertexPartition,
                             "integrated_CPM_weighted", "integrated_CPM",
                             "integrated_modularity_weighted", "integrated_modularity"]}
    elif part_funcs == "red":
        part_funcs = {"base": [la.CPMVertexPartition],
                      "abs": [la.SurpriseVertexPartition, la.ModularityVertexPartition,
                              la.RBConfigurationVertexPartition,la.RBERVertexPartition,
                              la.CPMVertexPartition,la.SignificanceVertexPartition]}
    elif part_funcs == "none":
        part_funcs = {"base":[], "abs":[]}
    else:
        part_funcs = {"base": part_funcs, "abs": part_funcs}

    RESULT_PATH=os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'results',args.name))
    setup_logging(RESULT_PATH, args.verbose)
    logging.debug(f'Result path set as {RESULT_PATH}')

    if mode == 0:
        logging.info(f'Running test script to plot base graphs and partitions')
        logging.info(f'Beginning with full graph')
        g, abs_g = make_graph(strain=args.strain, cutoff=args.cutoff,
                            abs=args.absolute_correlations, double_output=args.both_correlations,
                            include_exp_nodes=True, drop_lonely=False,
                            min_correlation=args.min_correlation, max_correlation=args.max_correlation,
                            min_edge_weight=args.min_edge_weight, max_edge_weight=args.max_edge_weight,
                            na_value=na_value)
        
        g = set_attributes(graph=g, scale_edges=args.scale)
        save_graph(graph=g, path=RESULT_PATH, name="full graph")
        base_partitions = plot_graph_partitions(graph=g, name="full graph", path=RESULT_PATH, layout=args.layout, partition_functions=part_funcs["base"])
        abs_g = set_attributes(graph=abs_g, scale_edges=args.scale)
        save_graph(graph=abs_g, path=RESULT_PATH, name="absolute full graph")
        abs_partitions = plot_graph_partitions(graph=abs_g, name="absolute full graph", path=RESULT_PATH, layout=args.layout, partition_functions=part_funcs["abs"])

        logging.info(f'Beginning with dvg graph')
        g, abs_g = make_graph(strain=args.strain, cutoff=args.cutoff,
                            abs=args.absolute_correlations, double_output=args.both_correlations,
                            include_exp_nodes=False, drop_lonely=False,
                            min_correlation=args.min_correlation, max_correlation=args.max_correlation,
                            min_edge_weight=args.min_edge_weight, max_edge_weight=args.max_edge_weight,
                            na_value=na_value)
        
        g = set_attributes(graph=g, scale_edges=args.scale)
        save_graph(graph=g, path=RESULT_PATH, name="dvg graph")
        plot_graph_partitions(graph=g, name="dvg graph", path=RESULT_PATH, layout=args.layout, partition_functions=part_funcs["base"])
        abs_g = set_attributes(graph=abs_g, scale_edges=args.scale)
        save_graph(graph=abs_g, path=RESULT_PATH, name="absolute dvg graph")
        plot_graph_partitions(graph=abs_g, name="absolute dvg graph", path=RESULT_PATH, layout=args.layout, partition_functions=part_funcs["abs"])

        logging.info(f'Beginning with connected dvg graph')
        g, abs_g = make_graph(strain=args.strain, cutoff=args.cutoff,
                            abs=args.absolute_correlations, double_output=args.both_correlations,
                            include_exp_nodes=False, drop_lonely=True,
                            min_correlation=args.min_correlation, max_correlation=args.max_correlation,
                            min_edge_weight=args.min_edge_weight, max_edge_weight=args.max_edge_weight,
                            na_value=na_value)
        
        g = set_attributes(graph=g, scale_edges=args.scale)
        save_graph(graph=g, path=RESULT_PATH, name="dvg-connected graph")
        plot_graph_partitions(graph=g, name="dvg-connected graph", path=RESULT_PATH, layout=args.layout, partition_functions=part_funcs["base"])
        abs_g = set_attributes(graph=abs_g, scale_edges=args.scale)
        save_graph(graph=abs_g, path=RESULT_PATH, name="absolute dvg-connected graph")
        plot_graph_partitions(graph=abs_g, name="absolute dvg-connected graph", path=RESULT_PATH, layout=args.layout, partition_functions=part_funcs["abs"])
    elif mode == 1:
        logging.info(f'Running test to plot partition graphs')
        
        param_dict = {"full graph": {"include_exp_nodes": True, "drop_lonely": False},
                      "dvg graph": {"include_exp_nodes": False, "drop_lonely": False},
                      "dvg-connected graph": {"include_exp_nodes": False, "drop_lonely": True}}
        args_dict = {"strain": args.strain, "cutoff": args.cutoff,
                     "abs": args.absolute_correlations, "double_output": args.both_correlations,
                     "min_correlation": args.min_correlation, "max_correlation": args.max_correlation,
                     "min_edge_weight": args.min_edge_weight, "max_edge_weight": args.max_edge_weight,
                     "na_value": na_value}
        if args.layout=="test":
            chosen_layout = ["auto","fr","graphopt","kk","mds","rt","dh","grid","rt_circular"]
        else:
            chosen_layout = args.layout
        
        for cur_version in ["dvg graph","dvg-connected graph","full graph"]:
            logging.info(f'Beginning with {cur_version}')
            g, abs_g = make_graph(**args_dict, **param_dict[cur_version])
            if g is not None:
                g = set_attributes(graph=g, scale_edges=args.scale)
                save_graph(graph=g, path=RESULT_PATH, name=cur_version)
                cur_partitions = get_partitions(graph=g,partition_functions=part_funcs["base"])
                plot_partition_graphs(original_graph=g, partitions=cur_partitions,
                                      figsize=(12,12) if max([max(p.sizes()) for p in cur_partitions.values()])<1000 else (18,18),
                                      layout=chosen_layout, name=cur_version, path=RESULT_PATH, subgraphs_at=20)

            if abs_g is not None:
                abs_g = set_attributes(graph=abs_g, scale_edges=args.scale)
                save_graph(graph=abs_g, path=RESULT_PATH, name=f"absolute {cur_version}")
                cur_partitions = get_partitions(graph=abs_g,partition_functions=part_funcs["abs"])
                plot_partition_graphs(original_graph=abs_g, partitions=cur_partitions,
                                      figsize=(12,12) if max([max(p.sizes()) for p in cur_partitions.values()])<1000 else (18,18),
                                      layout=chosen_layout, name=cur_version, path=RESULT_PATH, subgraphs_at=20)
    elif mode == 2:
        param_dict = {"full graph": {"include_exp_nodes": True, "drop_lonely": False},
                      "dvg graph": {"include_exp_nodes": False, "drop_lonely": False},
                      "dvg-connected graph": {"include_exp_nodes": False, "drop_lonely": True}}
        if args.layout=="test":
            chosen_layout = ["auto","fr","graphopt","kk","mds","rt","dh","grid","rt_circular"]
        else:
            chosen_layout = args.layout
        for cur_version in ["dvg graph","dvg-connected graph","full graph"]:
            args_dict = {"strain": args.strain, "cutoff": args.cutoff,
                        "abs": args.absolute_correlations, "double_output": args.both_correlations,
                        "min_correlation": args.min_correlation, "max_correlation": args.max_correlation,
                        "min_edge_weight": args.min_edge_weight, "max_edge_weight": args.max_edge_weight,
                        "na_value": na_value}
            logging.info(f'Beginning with {cur_version}')
            edge_counts_dict = {"abs": args.absolute_correlations, "step_size": 0.001, "title": f'{cur_version} edge counts by weight threshold', "name": f'{cur_version} edge counts', "path": RESULT_PATH}
            g, abs_g = make_graph(plot_edge_counts_dict=edge_counts_dict, **args_dict, **param_dict[cur_version])
    elif mode == 3:
        part_funcs = {"base": [la.CPMVertexPartition],
                      "abs": [la.ModularityVertexPartition,
                              "louvain"]}
        logging.info(f'Running test to plot partition graphs')
        
        param_dict = {"full graph": {"include_exp_nodes": True, "drop_lonely": False},
                      "dvg graph": {"include_exp_nodes": False, "drop_lonely": False},
                      "dvg-connected graph": {"include_exp_nodes": False, "drop_lonely": True},
                      "simplified graph": {"include_exp_nodes": False, "drop_lonely": True, "edge_weight_calc": "num_pubs"}}
        args_dict = {"strain": args.strain, "cutoff": args.cutoff,
                     "abs": args.absolute_correlations, "double_output": args.both_correlations,
                     "min_correlation": 3, "max_correlation": args.max_correlation,
                     "min_edge_weight": args.min_edge_weight, "max_edge_weight": args.max_edge_weight,
                     "na_value": na_value}
        if args.layout=="test":
            chosen_layout = ["auto","fr","graphopt","kk","mds","rt","dh","grid","rt_circular"]
        else:
            chosen_layout = args.layout
        
        for cur_version in ["simplified graph", "dvg-connected graph", "dvg graph", "full graph"]:
            logging.info(f'Beginning with {cur_version}')
            g, abs_g = make_graph(**args_dict, **param_dict[cur_version])
            if g is not None:
                g = set_attributes(graph=g, scale_edges=args.scale)
                save_graph(graph=g, path=RESULT_PATH, name=cur_version)
                leid_part, louv_part = louv_vs_leid(graph=g, path=RESULT_PATH, name=cur_version)
                community_dvg_analysis(graph=g, dataframe=None, partitions={"Leiden": leid_part, "Louvain": louv_part}, source_publications=PUBS_TO_USE, path=RESULT_PATH, name=cur_version)

            if abs_g is not None:
                abs_g = set_attributes(graph=abs_g, scale_edges=args.scale)
                save_graph(graph=abs_g, path=RESULT_PATH, name=f"absolute {cur_version}")
                leid_part, louv_part = louv_vs_leid(graph=abs_g, path=RESULT_PATH, name=f"absolute {cur_version}", seed=args.seed)
                community_dvg_analysis(graph=abs_g, dataframe=None, partitions={"Leiden": leid_part, "Louvain": louv_part}, source_publications=PUBS_TO_USE, path=RESULT_PATH, name=f'absoluts {cur_version}')
    elif mode == 4:
        part_funcs = {"base": [la.CPMVertexPartition],
                      "abs": [la.ModularityVertexPartition,
                              "louvain"]}
        logging.info(f'Running test to plot partition graphs')
        
        param_dict = {"full graph": {"include_exp_nodes": True, "drop_lonely": False},
                      "dvg graph": {"include_exp_nodes": False, "drop_lonely": False},
                      "dvg-connected graph": {"include_exp_nodes": False, "drop_lonely": True},
                      "simplified graph": {"include_exp_nodes": False, "drop_lonely": True, "edge_weight_calc": "num_pubs"}}
        args_dict = {"strain": args.strain, "cutoff": args.cutoff,
                     "abs": args.absolute_correlations, "double_output": args.both_correlations,
                     "min_correlation": 3, "max_correlation": args.max_correlation,
                     "min_edge_weight": args.min_edge_weight, "max_edge_weight": args.max_edge_weight,
                     "na_value": na_value}
        if args.layout=="test":
            chosen_layout = ["auto","fr","graphopt","kk","mds","rt","dh","grid","rt_circular"]
        else:
            chosen_layout = args.layout
        
        for cur_version in ["simplified graph", "dvg-connected graph", "dvg graph", "full graph"]:
            logging.info(f'Beginning with {cur_version}')
            g, abs_g = make_graph(**args_dict, **param_dict[cur_version])
            if g is not None:
                g = set_attributes(graph=g, scale_edges=args.scale)
                more_leids = quick_leid(graph=g, path=RESULT_PATH, name=cur_version)
                community_dvg_analysis(graph=g, dataframe=None, partitions=more_leids, source_publications=PUBS_TO_USE, path=RESULT_PATH, name=cur_version)

            if abs_g is not None:
                abs_g = set_attributes(graph=abs_g, scale_edges=args.scale)
                more_leids = quick_leid(graph=abs_g, path=RESULT_PATH, name=f"absolute {cur_version}", seed=args.seed)
                community_dvg_analysis(graph=abs_g, dataframe=None, partitions=more_leids, source_publications=PUBS_TO_USE, path=RESULT_PATH, name=f"absolute {cur_version}")