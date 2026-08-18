import argparse
import os.path
import warnings
import datetime
import logging
import sys
import numpy as np
import time
import umap.umap_ as umap
import sklearn.cluster as cluster
import distinctipy
import joblib

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, RocCurveDisplay, silhouette_samples
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

sys.path.insert(0, "..")
from utils import calculate_standard_features, load_data, log_and_norm, apply_cutoff, calculate_target, identify_candidates, DATA_DIR, SEGMENTS, STRAINS, get_sequence, drop_non_numeric, get_sequence_quicker, transform_meta_features

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.cluster import DBSCAN, HDBSCAN
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
import matplotlib.colors as mc
#import colorsys
import warnings
warnings.filterwarnings("ignore")
from sklearn.neighbors import NearestNeighbors
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.metrics import silhouette_score as shs
import itertools
from funcy import log_durations

ALL_SEGMENTS = ["PB2", "PB1", "PA", "HA", "NP", "NA", "M", "NS"]
ALL_PUBS = ["Lui2019", "Kupke2020", "Penn2022", "Sheng2018", "Zhuravlev2020", "Wang2020", "vdHoecke2015", "Boussier2020", "Southgate2019", "Valesano2020", "Mendes2021", "Alnaji2019", "Berry2021", "Alnaji2021", "Wang2023", "Pelz2021"]
ALL_STRAINS = ["A_Anhui_1_2013", "A_California_07_2009", "A_Connecticut_Flu122_2013", "A_NewCaledonia_20-JY2_1999", "A_Perth_16_2009", "A_PuertoRico_8_1934", "A_turkey_Turkey_1_2005", "A_WSN_33", "B_Brisbane_60_2008", "B_Lee_1940", "B_Victoria_504_2000", "B_Yamagata_16_1988"]
exp_col = "ACC_num"
STRAIN_COLORS = distinctipy.get_colors(len(ALL_STRAINS),n_attempts=50000,rng=42)
SEGMENT_COLORS = distinctipy.get_colors(len(ALL_SEGMENTS),n_attempts=50000,rng=42)
CLUSTER_COLORS = distinctipy.get_colors(10,[(0,0,0),(1,1,1),(1,0,0)],n_attempts=50000,rng=42)

# take logger from main
logger = logging.getLogger(__name__)

def get_data(strain="", pubs=ALL_PUBS, cutoff=0, exp_col="ACC_num", unpooled=True, drop_read_count=True):
    '''
    Gets test data with applied cutoff, normalization and identification.
    '''
    if strain == 0:
        strain = "A_PuertoRico_8_1934"
    data = load_data(pubs,unpooled=unpooled)
    if isinstance(strain, list):
        data = data[data["Strain"].isin(strain)]
    else:
        if strain != "":
            data = data[data["Strain"]==strain]
    data = apply_cutoff(data, cutoff=cutoff, method='quick_alternative', exp_col=exp_col)
    data = identify_candidates(data)
    data = log_and_norm(data,norm="NGS_log_min_max_norm",experiment_col=exp_col,drop_read_count=drop_read_count)
    logging.info(f'Found {data["ID"].nunique()} IDs after cutoff {cutoff}.')
    return data

def make_artificials_file(output_file="artificials.parquet", strains=STRAINS, step_size=1, chunk_size=10000):
    if isinstance(strains,str):
        strains=[strains]
    all_chunks = []

    for strain in strains:
        logging.info(f"Working on strain {strain}")
        for seg in ALL_SEGMENTS:
            logging.info(f"Segment {seg}")
            max_len = len(get_sequence(strain, seg))
            start, end = [], []

            for i in range(15, max_len - 15 - step_size, step_size):
                for j in range(i + 20, max_len - 15, step_size):
                    start.append(i)
                    end.append(j)

                    if len(start) >= chunk_size:
                        all_chunks.append(pd.DataFrame({
                            "Strain": [strain] * len(start),
                            "Segment": [seg] * len(start),
                            "Start": start,
                            "End": end
                        }))
                        start, end = [], []  # Reset lists

            # Save any remaining data
            if start:
                all_chunks.append(pd.DataFrame({
                    "Strain": [strain] * len(start),
                    "Segment": [seg] * len(start),
                    "Start": start,
                    "End": end
                }))
    logging.info("Concatenating and saving.")
    # Concatenate all chunks and write to a single Parquet file
    final_df = pd.concat(all_chunks, ignore_index=True)
    logging.info(f"{final_df.head()}\n{final_df.describe()}")
    if "parquet" in output_file:
        final_df.to_parquet(output_file, index=False)
    else:
        final_df.to_csv(output_file, index=False)

    logging.info(f"Data saved to {output_file}")

def prepare_dataframe(strain="A_PuertoRico_8_1934", pubs=['Alnaji2021', 'Kupke2020', 'Pelz2021', 'vdHoecke2015', 'Wang2023', 'Zhuravlev2020'], cutoff=5, exp_col="ACC_num", exp_as_col=True):
    data = get_data(strain, pubs, cutoff)
    data = data[["ID",exp_col,"NGS_log_min_max_norm"]]
    intersection_counts = data["ID"].value_counts()
    maximal = max(intersection_counts)
    filtered_data = data[data["ID"].isin(intersection_counts[intersection_counts >= maximal].index)].reset_index(drop=True)
    if filtered_data["ID"].nunique() < 5:
        logging.info(f'Too few candidates ({filtered_data["ID"].nunique()}) intersect across all experiments, so all candidates with any intersections were chosen instead.')
        filtered_data = data[data["ID"].isin(intersection_counts[intersection_counts >= 2].index)].reset_index(drop=True)
        logging.info(f'Found {filtered_data["ID"].nunique()} IDs for analysis, intersecting in at least 2 of {data[exp_col].nunique()} experiments.')
    if exp_as_col:
        filtered_data = filtered_data.pivot(index="ID",columns=exp_col,values="NGS_log_min_max_norm")
    else:
        filtered_data = filtered_data.pivot(index=exp_col,columns="ID",values="NGS_log_min_max_norm")
    return filtered_data

def prepare_incomplete(strain="A_PuertoRico_8_1934", pubs=['Alnaji2021', 'Kupke2020', 'Pelz2021', 'vdHoecke2015', 'Wang2023', 'Zhuravlev2020'], cutoff=5, exp_col="ACC_num", intersect_ratio=0):
    data = get_data(strain, pubs, cutoff)
    data = data[["ID",exp_col,"NGS_log_min_max_norm"]]
    minimal_intersect = intersect_ratio*data[exp_col].nunique()
    intersection_counts = data["ID"].value_counts()
    filtered_data = data[data["ID"].isin(intersection_counts[intersection_counts >= minimal_intersect].index)]
    print(f'Found {filtered_data["ID"].nunique()} IDs for analysis, appearing in at least {minimal_intersect} of {data[exp_col].nunique()} experiments.')
    return filtered_data

# Creating Matrix of pairwise correlations for maximal intersecting candidates
def basic_corr_visual(df):
    '''
    Visualizes pearson correlations (standard and absolute) of the dataframe in a matrix
    '''
    logging.info("plotting absolute correlation matrix")
    fig1 = plt.figure(figsize = (12,12))
    plt.matshow(df.corr().abs(),fignum=fig1.number,cmap="Purples",vmin=0,vmax=1)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix (absolute)', fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.close

    logging.info("plotting correlation matrix")
    fig2 = plt.figure(figsize = (12,12))
    plt.matshow(df.corr(),fignum=fig2.number,cmap="PiYG",vmin=-1,vmax=1)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.close()

def try_hierarchical(corr_matrix):
    '''
    Sorts correlation matrix via hierarchical clustering algorithm
    '''
    logging.info("Sorting correlation matrix hierarchical and plotting")
    # Perform hierarchical clustering
    linkage = sch.linkage(corr_matrix, method='ward')
    dendro_order = sch.leaves_list(linkage)

    # Reorder the correlation matrix
    sorted_corr = corr_matrix.iloc[dendro_order, dendro_order]

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(sorted_corr, cmap="PiYG", annot=False, fmt=".2f", vmin=-1, vmax=1)
    plt.title("Hierarchically Sorted Correlation Matrix")
    plt.show()
    plt.close()

def try_pca(corr_matrix):
    '''
    Sorts correlation matrix by one component from a pca
    '''
    logging.info("Sorting correlation matrix by pca and plotting")
    # Perform PCA on the correlation matrix
    pca = PCA(n_components=1)
    pca_values = pca.fit_transform(corr_matrix)  # Transform to 1D representation

    # Get sorting order based on PCA projection
    pca_order = np.argsort(pca_values.flatten())  # Sort indices by first component

    # Reorder the correlation matrix
    sorted_corr_pca = corr_matrix.iloc[pca_order, pca_order]

    # Plot the sorted heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(sorted_corr_pca, cmap="PiYG", annot=False, fmt=".2f", vmin=-1, vmax=1)
    plt.title("PCA-Sorted Correlation Matrix")
    plt.show()
    plt.close()

def try_spectral(corr_matrix):
    '''
    Sorts correlation matrix via spectral sort algorithm
    '''
    logging.info("Sorting correlation matrix with spectral sort and plotting")
    # Convert correlation matrix to a NumPy array
    corr_array = corr_matrix.to_numpy()

    # Compute the Laplacian matrix (1 - correlation to turn it into a distance-like measure)
    L = laplacian(1 - corr_array, normed=True)

    # Compute eigenvalues and eigenvectors
    _, eigvecs = eigh(L)

    # Sort indices based on the second smallest eigenvector
    spectral_order = np.argsort(eigvecs[:, 1])  # Second smallest eigenvector gives meaningful order

    # Reorder the correlation matrix based on spectral sorting
    sorted_corr_spectral = corr_matrix.iloc[spectral_order, spectral_order]

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(sorted_corr_spectral, cmap="PiYG", annot=False, fmt=".2f", vmin=-1, vmax=1)
    plt.title("Spectrally-Sorted Correlation Matrix")
    plt.show()
    plt.close()

def get_exp_to_pub(strain="A_PuertoRico_8_1934", pubs=['Alnaji2021', 'Kupke2020', 'Pelz2021', 'vdHoecke2015', 'Wang2023', 'Zhuravlev2020'], cutoff=10):
    '''
    Returns a dictionary, which takes an exp_col entry as key and returns the respective publication as value
    '''
    logging.info("Getting exp to pub dictionary")
    data = load_data(pubs,unpooled=True)
    data = data[data["Strain"]==strain]
    data = apply_cutoff(data, cutoff=cutoff, method='quick_alternative', exp_col=exp_col)
    data = identify_candidates(data)
    if data["Strain"].nunique() == 1:
        data["ID"] =data["ID"].str.replace(str(data["Strain"].unique()[0])+"_","")
    data = log_and_norm(data,norm="NGS_log_min_max_norm",experiment_col=exp_col)
    data = data[[exp_col,"Publication"]].drop_duplicates().set_index(exp_col)
    return data["Publication"].to_dict()

def umap_by_pub(data, exp_to_pub, title=""):
    '''
    
    '''
    logging.info("Plotting UMAP by pubs")
    plt.figure(figsize=(12,12))
    num_pubs = len(set(exp_to_pub.values()))
    to_col = {pub: i/(num_pubs-1) for i, pub in enumerate(set(exp_to_pub.values()))}
    data = data.fillna(0)
    standard_embedding = umap.UMAP(random_state=42).fit_transform(data)
    scatter = plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=data.index.map(exp_to_pub).map(to_col), s=10, cmap='Accent')
    plt.title(title)
    plt.legend(handles=[Line2D([0], [0], label=x, marker='s', markersize=10, markerfacecolor=plt.colormaps["Accent"](to_col[x]), linestyle='') for x in to_col.keys()])
    plt.show()
    plt.close()

def test_umap_by_pub():
    cut = 5
    ratio = 0.5
    exp_dict = get_exp_to_pub(cutoff=cut)

    raw_df = get_data(cutoff=cut)
    raw_df = raw_df.pivot(index=exp_col,columns="ID",values="NGS_log_min_max_norm")

    full_df = prepare_incomplete(cutoff=cut, intersect_ratio=ratio)
    full_df = full_df.pivot(index=exp_col,columns="ID",values="NGS_log_min_max_norm")

    intersect_df = prepare_dataframe(exp_as_col=False, cutoff=cut)

# UMAP projection of all candidates

def make_umap_DelVG_plot(pivot, coloring=[], alpha=1.0, standard_embedding=[], title="UMAP projection of DelVGs, based on normalized NGS read counts",path="test",name="test"):
    '''
    Plots UMAP results of provided embedding or creates new embedding, if none is provided.
    Should coloring be provided, only the included ids will be plotted.
    Will set both axes based on embedding, independed of id exclusions.
    '''
    # Preparing Dataframe for umap embedding
    logging.info(f"Creating UMAP plot ({title})\nSaving in {path}")
    logging.debug(f"Pivot head:\n{pivot.head()}")
    # Calculating embedding
    if len(standard_embedding)==0:
        standard_embedding = umap.UMAP(random_state=42).fit_transform(pivot)
    
    if not "ID" in pivot.columns:
        pivot.reset_index(inplace=True)
        logging.debug(f"New Pivot head:\n{pivot.head()}")
    
    if len(coloring)==0:
        coloring = [("DelVG","blue",[id for id in pivot["ID"].unique()])]
    
    # Making sure axes in graphic don't change after application of filters
    plt.figure(figsize=(16,16))
    plt.scatter(max(standard_embedding[:, 0]), max(standard_embedding[:, 1]), c="white", s=10, alpha=0.0)
    plt.scatter(min(standard_embedding[:, 0]), min(standard_embedding[:, 1]), c="white", s=10, alpha=0.0)

    # Plotting each group separately
    for group in coloring:
        mask = pivot[pivot["ID"].isin(group[2])].index
        plt.scatter(standard_embedding[mask, 0], standard_embedding[mask, 1], c=group[1], s=10, label=f'{group[0]} ({len(group[2])})', alpha=alpha)
    
    ax = plt.gca()
    plt.title(title)
    if coloring[0][0]=="":
        cbar=plt.colorbar(boundaries=range(1,len(coloring)+2),values=[group[1] for group in coloring],aspect=50)
        cbar.set_ticks(np.linspace(1,len(coloring),len(coloring))+0.5)
        cbar.ax.set_yticklabels(range(1,len(coloring)+1))
        cbar.set_label("Number Of Occurences")
    else:
        plt.legend()
    plt.savefig(os.path.join(path,name+".png"))
    plt.close()
    return standard_embedding


# Setting up coloring and filtering functions for UMAPs

def get_coloring(data, group_by="intersections", accepted_ids=[]):
    '''
    Returns a grouping of labels, colors and respective ids. Grouping is done based on chosen option.
    Possible options: intersections, intersections_extra, strain, segment, type, num_publications
    '''

    logging.info(f"Calculating coloring for data by {group_by}, {'using '+str(len(accepted_ids))+' accepted ids' if len(accepted_ids)>0 else 'without filter'}")
    # get labels, colors and ids for coloring masks, based on chosen option
    match group_by.lower():
        case "intersections":
            assert "Publication" in data.columns, "Publication column is missing for highlighting by intersection!"
            assert "ID" in data.columns, "ID column is missing for highlighting by intersection!"
            assert "ACC_num" in data.columns, "ACC_num column is missing for highlighting by intersection!"
            pub_counts = data[["ID","Publication"]].groupby(["ID"])["Publication"].nunique()
            cross_pub_ids = [id for id in pub_counts[pub_counts>1].index] # IDs that are shared between publications
            acc_counts = data[["ID","ACC_num"]].groupby(["ID"])["ACC_num"].nunique()
            multi_acc_ids = [id for id in acc_counts[acc_counts>1].index] # IDs that are shared between accession numbers
            multi_pub_intersecting_ids = [id for id in cross_pub_ids if id in multi_acc_ids and acc_counts[id]>pub_counts[id]] # IDs that are shared within and between publications
            single_pub_intersecting_ids = [id for id in multi_acc_ids if id not in cross_pub_ids] # IDs that are shared between accession numbers, but only show up in one single publication
            only_pub_intersecting_ids = [id for id in cross_pub_ids if id not in multi_pub_intersecting_ids] # IDs that are shared only between publications but not between samples of the same publication
            non_intersecting_ids = [id for id in acc_counts[acc_counts==1].index] # unshared IDs, as in they only show up under one specific accession number and within a single publication

            if len(accepted_ids)>0: # Applying filter to calculated groups.
                cross_pub_ids = [id for id in cross_pub_ids if id in accepted_ids]
                single_pub_intersecting_ids = [id for id in single_pub_intersecting_ids if id in accepted_ids]
                non_intersecting_ids = [id for id in non_intersecting_ids if id in accepted_ids]
            highlights = [("Non-Intersecting", "gray", non_intersecting_ids),
                          ("Intersecting within single Publication", "teal", single_pub_intersecting_ids),
                          ("Intersecting between multiple Publications", "purple", cross_pub_ids)]
            highlights = [group for group in highlights if len(group[2])>0]
            return highlights, 0.5
        case "intersections_extra":
            assert "Publication" in data.columns, "Publication column is missing for highlighting by intersection!"
            assert "ID" in data.columns, "ID column is missing for highlighting by intersection!"
            assert "ACC_num" in data.columns, "ACC_num column is missing for highlighting by intersection!"
            pub_counts = data[["ID","Publication"]].groupby(["ID"])["Publication"].nunique()
            cross_pub_ids = [id for id in pub_counts[pub_counts>1].index] # IDs that are shared between publications
            acc_counts = data[["ID","ACC_num"]].groupby(["ID"])["ACC_num"].nunique()
            multi_acc_ids = [id for id in acc_counts[acc_counts>1].index] # IDs that are shared between accession numbers
            multi_pub_intersecting_ids = [id for id in cross_pub_ids if id in multi_acc_ids and acc_counts[id]>pub_counts[id]] # IDs that are shared within and between publications
            value_counts = data[["ID"]].value_counts()
            single_pub_intersecting_ids = [id[0] for id in value_counts[value_counts>=2].index if id[0] not in cross_pub_ids] # IDs that are shared between accession numbers, but only show up in one single publication
            only_pub_intersecting_ids = [id for id in cross_pub_ids if id not in multi_pub_intersecting_ids] # IDs that are shared only between publications but not between samples of the same publication
            non_intersecting_ids = [id[0] for id in value_counts[value_counts==1].index] # IDs that are unshared, as in they only show up under one specific accession number and within a single publication

            if len(accepted_ids)>0: # Applying filter to calculated groups.
                only_pub_intersecting_ids = [id for id in only_pub_intersecting_ids if id in accepted_ids]
                multi_pub_intersecting_ids = [id for id in multi_pub_intersecting_ids if id in accepted_ids]
                single_pub_intersecting_ids = [id for id in single_pub_intersecting_ids if id in accepted_ids]
                non_intersecting_ids = [id for id in non_intersecting_ids if id in accepted_ids]
            highlights = [("Non-Intersecting", "gray", non_intersecting_ids),
                          ("Intersecting within single Publication", "teal", single_pub_intersecting_ids),
                          ("Intersecting between multiple Publications", "purple", multi_pub_intersecting_ids),
                          ("Only Intersecting between Publications", "orange", only_pub_intersecting_ids)]
            highlights = [group for group in highlights if len(group[2])>0]
            return highlights, 0.5
        case "strain":
            assert "Strain" in data.columns, "Strain column is missing for highlighting by strain!"
            if isinstance(STRAIN_COLORS, list):
                colors = STRAIN_COLORS
            else:
                colors = distinctipy.get_colors(len(ALL_STRAINS),n_attempts=10000,rng=42)
            highlights = []
            for num, strain in enumerate(ALL_STRAINS):
                if strain in data["Strain"].unique() if len(accepted_ids)==0 else strain in data[data["ID"].isin(accepted_ids)]["Strain"].unique():
                    strain_ids = [id for id in data[data["Strain"]==strain]["ID"]]
                    if len(accepted_ids)>0: # Applying filter to calculated groups.
                        strain_ids = [id for id in strain_ids if id in accepted_ids]
                    highlights.append((strain.replace("_"," "), colors[num], strain_ids))
            highlights = [group for group in highlights if len(group[2])>0]
            return highlights, 0.9
        case "segment":
            assert "Segment" in data.columns, "Segment column is missing for highlighting by segment!"
            if isinstance(SEGMENT_COLORS, list):
                colors = SEGMENT_COLORS
            else:
                colors = distinctipy.get_colors(len(ALL_SEGMENTS),n_attempts=50000,rng=42)
            highlights = []
            for num, segment in enumerate(ALL_SEGMENTS):
                if segment in data["Segment"].unique() if len(accepted_ids)==0 else segment in data[data["ID"].isin(accepted_ids)]["Segment"].unique():
                    segment_ids = [id for id in data[data["Segment"]==segment]["ID"]]
                    if len(accepted_ids)>0: # Applying filter to calculated groups.
                        segment_ids = [id for id in segment_ids if id in accepted_ids]
                    highlights.append((segment, colors[num], segment_ids))
            highlights = [group for group in highlights if len(group[2])>0]
            return highlights, 0.9
        case "type":
            assert "Strain" in data.columns, "Strain column is missing for highlighting by strain type!"
            strain_types = ["A","B","C"]
            colors = plt.colormaps["brg"](np.linspace(0, 1, len(strain_types)))
            highlights = []
            data["Type"] = data["ID"].str[0]
            for num, strain_type in enumerate(strain_types):
                if strain_type in data["Type"].unique() if len(accepted_ids)==0 else strain_type in data[data["ID"].isin(accepted_ids)]["Type"].unique():
                    type_ids = [id for id in data[data["Type"] == strain_type]["ID"]]
                    if len(accepted_ids)>0: # Applying filter to calculated groups.
                        type_ids = [id for id in type_ids if id in accepted_ids]
                    highlights.append((f"Influenza Virus Type {strain_type}", colors[num], type_ids))
            highlights = [group for group in highlights if len(group[2])>0]
            return highlights, 0.75
        case "num_publications":
            assert "Publication" in data.columns, "Publication column is missing for highlighting by publication density!"
            if data["Publication"].nunique()==1:
                return [("","red",[id for id in data["ID"].unique()])], 0.75
            colors = plt.colormaps["viridis"](np.linspace(0,1,data["Publication"].nunique()))
            highlights = []
            pub_counts = data[["ID","Publication"]].groupby(["ID"])["Publication"].nunique()
            for num in range(data["Publication"].nunique()):
                ids = [id for id in pub_counts[pub_counts==num+1].index]
                if len(accepted_ids) > 0:
                    ids = [id for id in ids if id in accepted_ids]
                highlights.append(("",colors[num],ids))
            while len(highlights[-1][2])==0:
                highlights.pop()
            return highlights, 0.75
        case _:
            logging.error(f"Unknown coloring requested: {group_by}")
            ids = accepted_ids if len(accepted_ids)>0 else data["ID"]
            highlights = [("DelVG","blue",ids)]
            return highlights, 0.5


def filter_ids(data, by, keep=None, drop=None, accepted_ids=""): # drop > keep
    '''
    Returns a list of ids from provided data that survive specified filters.
    by is used to pick attributes by which to filter and keep/drop are used to choose the acceptable/unacceptable
    values of said attributes.
    Drop is more powerful than keep, meaning that an id that passes keep will be dropped if it does not pass respective criteria.
    '''
    if isinstance(accepted_ids,list):
        data_remaining = data[data["ID"].isin(accepted_ids)].copy()
    else:
        data_remaining = data.copy()
    ids = [id for id in data_remaining["ID"].unique()] # in case only exclusion criteria are used
    to_drop = [] # in case no exclusion criteria are used
    if isinstance(by,list):
        logging.info(f"Filtering by list: {by}\nkeep - {keep}\ndrop - {drop}")
        logging.debug(by)
        for criterion in by: # each filter runs over the leftovers of the previous one
            kept = filter_ids(data,criterion,keep,None,ids)
            dropped = filter_ids(data,criterion,None,drop,ids)
            ids = [id for id in kept if id in dropped]
            logging.debug(f'Criterion: {criterion}\tkeep: {keep}\tdrop: {drop}\n keep: {len(kept)}\t{set(kept)}\n drop: {len(dropped)}\t{set(dropped)}\n gets: {len(ids)}\t{set(ids)}')
        return ids
    elif by in data_remaining.columns:
        if isinstance(keep,list):
            ids = [id for id in data_remaining[data_remaining[by].isin(keep)]["ID"]]
        elif isinstance(keep,str):
            ids = [id for id in data_remaining[data_remaining[by]==keep]["ID"]]
        if isinstance(drop,list):
            to_drop = [id for id in data_remaining[data_remaining[by].isin(drop)]["ID"]]
        elif isinstance(drop,str):
            to_drop = [id for id in data_remaining[data_remaining[by]==drop]["ID"]]
        return list(set([id for id in ids if id not in to_drop]))
    match by.lower():
        case "type":
            data_remaining["Type"] = data_remaining["ID"].str[0]
            if any([t not in ["A","B","C"] for t in data_remaining["Type"]]):
                logging.error(f'ERROR: UNKOWN TYPE READ IN ID\n{data_remaining["Type"].describe()}\n{data_remaining["Type"].unique()}')
            if isinstance(keep,list):
                if any([k in ["A","B","C"] for k in keep]):
                    ids = [id for id in data_remaining[data_remaining["Type"].isin(keep)]["ID"]]
            elif isinstance(keep,str):
                if keep in ["A","B","C"]:
                    ids = [id for id in data_remaining[data_remaining["Type"]==keep]["ID"]]
            if isinstance(drop,list):
                if any([d in ["A","B","C"] for d in drop]):
                    to_drop = [id for id in data_remaining[data_remaining["Type"].isin(drop)]["ID"]]
            elif isinstance(drop,str):
                if drop in ["A","B","C"]:
                    to_drop = [id for id in data_remaining[data_remaining["Type"]==drop]["ID"]]
            return list(set([id for id in ids if id not in to_drop]))
        case "intersections":
            pub_counts = data_remaining[["ID","Publication"]].groupby(["ID"])["Publication"].nunique()
            acc_counts = data_remaining[["ID","ACC_num"]].groupby(["ID"])["ACC_num"].nunique()
            def get_intersection_filter(k):
                match k:
                        case "non-intersecting":
                            return [id for id in acc_counts[acc_counts==1].index] # unshared IDs, as in they only show up under one specific accession number and within a single publication
                        case "cross_pubs":
                            return [id for id in pub_counts[pub_counts>1].index] # IDs that are shared between publications
                        case "single_pub":
                            multi_acc_ids = [id for id in acc_counts[acc_counts>1].index]
                            return [id for id in multi_acc_ids if id not in cross_pub_ids] # IDs that are shared between accession numbers, but only show up in one single publication
                        case "multi_pub":
                            multi_acc_ids = [id for id in acc_counts[acc_counts>1].index]
                            cross_pub_ids = [id for id in pub_counts[pub_counts>1].index]
                            return [id for id in cross_pub_ids if id in multi_acc_ids and acc_counts[id]>pub_counts[id]] # IDs that are shared within and between publications
                        case "only_pub":
                            multi_acc_ids = [id for id in acc_counts[acc_counts>1].index]
                            cross_pub_ids = [id for id in pub_counts[pub_counts>1].index]
                            multi_pub_intersecting_ids = [id for id in cross_pub_ids if id in multi_acc_ids and acc_counts[id]>pub_counts[id]]
                            return [id for id in cross_pub_ids if id not in multi_pub_intersecting_ids] # IDs that are shared only between publications but not within the same publication
                        case "any":
                            return [id for id in acc_counts[acc_counts>1].index] # IDs that intersect between accession numbers
                        case _:
                            return "unknown"
            if isinstance(keep,list):
                ids = []
                for crit in keep:
                    newly_kept = get_intersection_filter(crit)
                    if newly_kept != "unknown":
                        ids.extend(newly_kept)
            elif isinstance(keep,str):
                newly_kept = get_intersection_filter(keep)
                if newly_kept != "unknown":
                    ids = newly_kept
            if isinstance(drop,list):
                to_drop = []
                for crit in drop:
                    newly_dropped = get_intersection_filter(crit)
                    if newly_dropped != "unknown":
                        to_drop.extend(newly_dropped)
            elif isinstance(drop,str):
                newly_dropped = get_intersection_filter(drop)
                if newly_dropped != "unknown":
                    to_drop = newly_dropped
            ids = list(set([id for id in ids if id not in to_drop]))
            return ids

        case _:
            logging.error(f"Unknown filtering option: {by}")

# UMAPs based on standard sequence-derived features features
def get_combined_umap_embedding(strains=STRAINS, features=['Strain', 'Segment', 'Start', 'End', 'Direct_repeat', 'Junction', '3_5_diff', 'length_proportion'], cutoff=0):
    '''
    Calculates umap embedding based on given features and ngs read counts of given dataframe.
    '''
    logging.info(f"Getting combined embedding for {strains}")
    dataframe = get_data(strain=strains,pubs=ALL_PUBS, unpooled=True, exp_col="ACC_num", cutoff=cutoff)
    ngs_pivot = dataframe.pivot(index="ID",columns=exp_col,values="NGS_log_min_max_norm").fillna(0)
    # Preparing artificial dataset
    logging.debug(f"Preparing feature calculation for {strains}")
    if isinstance(strains,str):
        strains=[strains]
    
    # Calculating features
    dataframe = get_sequence_quicker(dataframe)
    dataframe["s_len"] = dataframe.apply(lambda row: len(row["Full_Sequence"]), axis=1)
    if dataframe["s_len"].min() <= 0:
        logging.error(f'Found seq length of 0 or lower in dataframe:\n{dataframe[dataframe["s_len"]<=0]}')
    original_columns = dataframe.columns.tolist()
    feature_pivot = calculate_standard_features(dataframe,features)
    try:
        feature_pivot.drop([col for col in original_columns if col != "ID"],inplace=True,axis=1)
        feature_pivot.drop_duplicates(["ID"],inplace=True)
        feature_pivot.set_index(feature_pivot["ID"],inplace=True)
        feature_pivot = drop_non_numeric(feature_pivot)
        feature_pivot.drop(["Start","End","s_len","Full_Sequence"],inplace=True,axis=1)
    except Exception as e:
        logging.warning(f'Exception when trying to remove leftover columns in feature pivot:\n{e}')
    
    logging.debug("Joining ngs data and calculated features.")
    try:
        full_pivot = pd.concat([ngs_pivot, feature_pivot],axis=1)
    except Exception as e:
        logging.error(f'Exception when trying to merge pivots:\n{e}\n{ngs_pivot.head()}\n{feature_pivot.head()}\n')

    logging.info("Creating Embedding with ngs counts and calculated features.")
    embedding = umap.UMAP(random_state=42).fit_transform(full_pivot)
    full_pivot.reset_index(inplace=True)
    logging.debug(f'pivot data:\n{full_pivot.head()}\n{full_pivot.describe()}\ndataframe:\n{dataframe.head()}\n{dataframe.describe()}\nembedding:\n{embedding}\n')
    return dataframe, full_pivot, embedding

def get_combined_pivot(publications:list, strain:str, features:list=['Strain', 'Segment', 'Start', 'End', 'Direct_repeat', 'Junction', '3_5_diff', 'length_proportion'], cutoff:int=0):
    logging.info(f"Getting combined pivot for {strain} in {publications}")
    dataframe = get_data(strain=[strain],pubs=publications, unpooled=True, exp_col="ACC_num", cutoff=cutoff)
    if dataframe.empty:
        return dataframe, None
    ngs_pivot = dataframe.pivot(index="ID",columns="ACC_num",values="NGS_log_min_max_norm").fillna(0)
    
    # Calculating features
    dataframe = get_sequence_quicker(dataframe)
    dataframe["s_len"] = dataframe["Full_Sequence"].transform(lambda x: len(x))
    if dataframe["s_len"].min() <= 0:
        logging.error(f'Found seq length of 0 or lower in dataframe:\n{dataframe[dataframe["s_len"]<=0]}')
    original_columns = dataframe.columns.tolist()
    feature_pivot = calculate_standard_features(dataframe,features)
    try:
        feature_pivot.drop([col for col in original_columns if col != "ID"],inplace=True,axis=1)
        feature_pivot.drop_duplicates(["ID"],inplace=True)
        feature_pivot.set_index(feature_pivot["ID"],inplace=True)
        feature_pivot = drop_non_numeric(feature_pivot)
        feature_pivot.drop(["Start","End","s_len","Full_Sequence"],inplace=True,axis=1)
    except Exception as e:
        logging.warning(f'Exception when trying to remove leftover columns in feature pivot:\n{e}')
    
    logging.debug("Joining ngs data and calculated features.")
    try:
        full_pivot = pd.concat([ngs_pivot, feature_pivot],axis=1)
    except Exception as e:
        logging.error(f'Exception when trying to merge pivots:\n{e}\n{ngs_pivot.head()}\n{feature_pivot.head()}\n')
    return dataframe[["ID","Publication","ACC_num"]], full_pivot

def get_feature_umap_embedding(strains=STRAINS, features=['Strain', 'Segment', 'Start', 'End', 'Direct_repeat', 'Junction', '3_5_diff', 'length_proportion'], step_size=5, chosen_ids=[], just_data=False):
    '''
    Creates artificial dataset and calculates features. Finally calculates umap embedding based on those features.
    '''
    # Preparing artificial dataset
    if isinstance(strains,str):
        strains=[strains]
    if len(chosen_ids)>0:
        logging.info(f'Getting dataframe with {len(chosen_ids)} rows from chosen ids.')
        strain_col, seg_col, start_col, end_col = zip(*[(strain, seg, int(start), int(end)) for strain, seg, start, end in (id.rsplit("_", 3) for id in chosen_ids)])
        dataframe = pd.DataFrame.from_dict({"Strain": strain_col, "Segment": seg_col, "Start": start_col, "End": end_col})
        dataframe = dataframe[dataframe["Strain"].isin(strains)]
        logging.debug(f'\n{dataframe.head()}')
    else:
        logging.info(f"Creating artificial dataset with step size {step_size} for {strains}")
        dataframe = pd.DataFrame()
        for strain in strains:
            for seg in ALL_SEGMENTS:
                max_len = len(get_sequence(strain, seg))
                start, end = [], []
                for i in range(25, max_len-15-step_size, step_size):
                    for j in range(i+20, max_len-15, step_size):
                        start.append(i)
                        end.append(j)
                sub_frame = pd.DataFrame.from_dict({"Strain": [strain]*len(start), "Segment": [seg]*len(start), "Start": start, "End": end})
                dataframe = pd.concat([dataframe, sub_frame], ignore_index=True)
        dataframe = get_sequence_quicker(dataframe)
        dataframe["Remaining"] = dataframe.apply(lambda row: row["Full_Sequence"][:row["Start"]-1]+row["Full_Sequence"][row["End"]:], axis=1)
        dataframe.drop_duplicates(["Strain","Segment","Remaining"],inplace=True)
        dataframe.drop("Remaining",axis=1)
        if just_data:
            return dataframe
    if dataframe["Segment"].nunique() < 2:
        logging.warning(f"Weirdness going on: Only one type of segment found?!\n{dataframe.head()}")
    # Calculating features
    dataframe = get_sequence_quicker(dataframe)
    dataframe["s_len"] = dataframe.apply(lambda row: len(row["Full_Sequence"]), axis=1)
    dataframe = identify_candidates(dataframe)
    pivot_data = calculate_standard_features(dataframe,features)
    pivot_data.drop_duplicates(["ID"],inplace=True)
    pivot_data.set_index(pivot_data["ID"],inplace=True)
    
    
    pivot_data = drop_non_numeric(pivot_data)
    try:
        pivot_data.drop(["Start","End","s_len","Full_Sequence"],inplace=True,axis=1)
    except Exception as e:
        logging.warning(f'Exception when trying to remove leftover columns in pivot data:\n{e}\n')
    logging.debug("Creating Embedding for calculated features.")
    embedding = umap.UMAP(random_state=42).fit_transform(pivot_data)
    pivot_data.reset_index(inplace=True)
    logging.debug(f'pivot data:\n{pivot_data.head()}\n{pivot_data.describe()}\ndataframe:\n{dataframe.head()}\n{dataframe.describe()}\nembedding:\n{embedding}')
    return dataframe, pivot_data, embedding

@log_durations(logging.info)
def grid_search_HDBSCAN(min_sizes, X, save_iterations=True):
    if save_iterations:
        scores = []
        all_label = []
        all_centroids = []
    else:
        best_score = -10
    logging.info(f'Beginning grid search for HDBSCAN min cluster size. Total iterations: {len(min_sizes)}\tdata shape: {X.shape}')
    for i in min_sizes:
        model = HDBSCAN(min_cluster_size=i, store_centers="centroid", copy=True).fit(X)
        labels = model.labels_
        classes = set(labels)
        centroids = model.centroids_
        num_clusters = len(classes)
        if -1 in classes:
            num_clusters-=1
        if (num_clusters<2) or (num_clusters>50):
            if save_iterations:
                scores.append(-20)
                all_label.append("Poor")
                logging.debug(f'Bad result at iteration {i}: min size={i}\t\tnumber of clusters={num_clusters} ...  Moving on')
            continue
        if save_iterations:
            scores.append(shs(X,labels))
        else:
            new_score = shs(X,labels)
            if new_score > best_score:
                best_score = new_score
                best_parameters = i
                best_labels = labels
                best_centroids = centroids
                best_num_clusters = num_clusters
                logging.debug(f'At iteration {i}: score={best_score}\t\tnumber of clusters={best_num_clusters}') 
        if save_iterations:
            all_label.append(labels)
            all_centroids.append(centroids)
            logging.debug(f'At iteration {i}: score={scores[-1]}\t\tnumber of clusters={num_clusters}')
    if save_iterations:
        best_index = np.argmax(scores)
        best_parameters = min_sizes[best_index]
        best_labels = all_label[best_index]
        best_score = scores[best_index]
        best_centroids = all_centroids[best_index]
        classes = set(best_labels)
        num_clusters = len(classes)
    if -1 in classes:
        num_clusters -= 1
    if save_iterations:
        return {'best_min_cluster_size': best_parameters, 'best_labels': best_labels, 'best_score': best_score, "centroids": best_centroids, 'num_clusters': num_clusters}, (min_sizes, scores)
    return {'best_min_cluster_size': best_parameters, 'best_labels': best_labels, 'best_score': best_score, "centroids": best_centroids, 'num_clusters': num_clusters}, (min_sizes, best_score)

@log_durations(logging.info)
def grid_search_shs(combinations, X, save_iterations=True):
    if save_iterations:
        scores = []
        all_label = []
    else:
        best_score = -10
    logging.info(f'Beginning grid search for DBSCAN parameters. Total iterations: {len(combinations)}\tdata shape: {X.shape}')
    for i, (epsilon, num_samples) in enumerate(combinations):
        model = DBSCAN(eps=epsilon, min_samples=num_samples).fit(X)
        labels = model.labels_
        classes = set(labels)
        num_clusters = len(classes)
        if -1 in classes:
            num_clusters-=1
        if (num_clusters<2) or (num_clusters>50):
            if save_iterations:
                scores.append(-20)
                all_label.append("Poor")
                logging.debug(f'Bad result at iteration {i}: eps={epsilon}\tmin_samples={num_samples}\t\tnumber of clusters={num_clusters} ...  Moving on')
            continue
        if save_iterations:
            scores.append(shs(X,labels))
        else:
            new_score = shs(X,labels)
            if new_score > best_score:
                best_score = new_score
                best_parameters = (epsilon, num_samples)
                best_labels = labels
                best_num_clusters = num_clusters
                logging.debug(f'At iteration {i}: score={best_score}\t\tnumber of clusters={best_num_clusters}') 
        if save_iterations:
            all_label.append(labels)
            logging.debug(f'At iteration {i}: score={scores[-1]}\t\tnumber of clusters={num_clusters}')
    if save_iterations:
        best_index = np.argmax(scores)
        best_parameters = combinations[best_index]
        best_labels = all_label[best_index]
        best_score = scores[best_index]
        classes = set(best_labels)
        num_clusters = len(classes)
    if -1 in classes:
        num_clusters -= 1
    if save_iterations:
        return {'best_epsilon': best_parameters[0], 'best_min_samples': best_parameters[1], 'best_labels': best_labels, 'best_score': best_score, 'num_clusters': num_clusters}, (combinations,scores)
    return {'best_epsilon': best_parameters[0], 'best_min_samples': best_parameters[1], 'best_labels': best_labels, 'best_score': best_score, 'num_clusters': num_clusters}, (combinations, best_score)

def plot_silhouette(base_data, clustering, cmap=None, path="", name="", title="", save=True):
    '''
    Creates silhouette plot based on given base_data and respective clustering labels.
    '''
    cluster_titles = set(clustering.labels_)
    num_clusters = len(cluster_titles)
    if -1 in cluster_titles:
        num_clusters -= 1
    sample_scores = silhouette_samples(base_data, clustering.labels_)

    y_lower=10
    y_height = {}
    plt.figure(figsize=(10,10))
    if cmap: # Use the given colormap
        distinct_cmap = cmap
    else: # or create new distinct colormap if none was given
        distinct_cmap = distinctipy.get_colormap(distinctipy.get_colors(num_clusters,n_attempts=5000,rng=42,exclude_colors=[(0,0,0),(1,1,1),(1,0,0)]))
    
    for cluster in cluster_titles: # plot each clusters samples, sorted by score
        if cluster == -1: # skip noise
            continue
        samp_vals = sample_scores[clustering.labels_ == cluster]
        samp_vals.sort()
        cluster_size = samp_vals.shape[0]
        y_upper = y_lower + cluster_size

        color = distinct_cmap(cluster / num_clusters)
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            samp_vals,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        y_height[cluster] = y_lower+0.5*cluster_size # Labels and height for x axis
        y_lower = y_upper + 10 # Compute the new y_lower for next plot adding 10 for the 0 samples
    sil_avg = shs(base_data,clustering.labels_)
    plt.vlines(sil_avg,-1,y_lower, color="red", linestyle="--", label=f'Average Silhouette Score {sil_avg:.4f}')

    plt.xlabel("silhouette coefficient values")
    plt.xlim(-0.25,1)
    plt.yticks(ticks=list(y_height.values()),labels=list(y_height.keys()))
    plt.ylabel("cluster label")
    plt.ylim(-10,y_lower)

    plt.legend(loc=1)
    plt.title(title)
    if save:
        plt.savefig(os.path.join(path,name+".png"))
    else:
        plt.show()
    plt.close()

def plot_cluster(plot_base_data, plot_embedding, plot_clustering, cmap, plot_path, plot_name, plot_title, save=True):
    try:
        if "ID" in plot_base_data.columns:
            logging.info("Resetting Index to ID")
            base_data = plot_base_data.set_index("ID")
            logging.debug(f'\n{base_data.head()}')
        else:
            logging.info("Not changing Index")
            base_data = plot_base_data
    except AttributeError:
        pass
    except Exception as e:
        logging.error(f'Problem when trying to fix base data up: (ignore if base data is not pivot)\n{e}')

    current_clusters = set(plot_clustering.labels_)
    plt.figure(figsize=(16,16))
    legend_handles = []
    for label in current_clusters:
        if label == -1:
            continue  # Skip noise since it's already added
        mask = plot_clustering.labels_ == label
        color = cmap(label)
        plt.scatter(plot_embedding[mask, 0], plot_embedding[mask, 1], color=color, s=10, alpha=0.6)
        legend_handles.append(mpatches.Patch(color=color, label=f"Cluster {label}"))
    noise_mask = plot_clustering.labels_ == -1
    plt.scatter(plot_embedding[noise_mask, 0], plot_embedding[noise_mask, 1], color='red', s=10, label="Noise", edgecolors='k', alpha=0.5)
    legend_handles.append(mpatches.Patch(color='red', label="Noise"))
    plt.legend(handles=legend_handles, title="Clusters")
    plt.title(plot_title)
    plt.annotate(f'Silhouette Score: {shs(plot_base_data,plot_clustering.labels_):.4f}',(min(plot_embedding[:,0]),min(plot_embedding[:,1])))
    if save:
        plt.savefig(os.path.join(plot_path,plot_name+".png"))
    else:
        plt.show()
    plt.close()

def get_clustering(source_data, set_name, path, name, grid_search=None, set_epsilon=None, set_n_samples=None, results=None, grid=[]):
    if grid_search:
        try:
            if len(grid) == 0:
                eps = np.linspace(0.1,2,100)
                min_samples = np.arange(5,100,step=1)
                params = list(itertools.product(eps, min_samples))
            else:
                params = list(itertools.product(grid[0],grid[1]))
            grid_results = grid_search_shs(params, source_data)
            best_results = grid_results[0]
            results.append((set_name, best_results))
            epsilon = best_results['best_epsilon']
            n_samples = best_results['best_min_samples']
            logging.info(f"Grid search results:\n{best_results}")
            
            # plotting grid
            x, y = zip(*grid_results[1][0])
            grid_df = pd.DataFrame({"x":x,"y":y,"score":grid_results[1][1]})
            grid_df = grid_df[grid_df["score"]>=0]
            plt.figure(figsize=(6,6))
            plt.scatter(grid_df["x"],grid_df["y"],c=grid_df["score"],s=10,cmap='jet',vmin=0,vmax=1)
            plt.title("Scatter plot of grid-search results")
            plt.xlabel("eps")
            plt.ylabel("min samples")
            plt.colorbar(label="Silhouette Score",fraction=0.1)
            plt.savefig(os.path.join(path,name+f"_{set_name}_grid.png"))
            plt.close()
        except Exception as e:
            logging.error(f'Issue with grid search:\n{e}')
            results.append((set_name, {"best_results":"No grid search done."}))
            epsilon = set_epsilon if set_epsilon else 1.1
            n_samples = set_n_samples if set_n_samples else 40
    else:
        results.append((set_name, {"best_results":"No grid search done."}))
        epsilon = set_epsilon if set_epsilon else 1.1
        n_samples = set_n_samples if set_n_samples else 40

    clustering = DBSCAN(eps=epsilon,min_samples=n_samples).fit(source_data)
    save_path = os.path.join(path,f'{name}_{set_name}.sav')
    joblib.dump(clustering.labels_, save_path)
    return clustering, results

def get_cluster_plots(embedding, pivot_data, title, path, name, grid_search=False, set_epsilon=None, set_n_samples=None):
    logging.info(f'Beginning Cluster Function')
    if "ID" in pivot_data.columns:
        logging.info("Resetting Index to ID")
        base_data = pivot_data.set_index("ID")
        logging.debug(f'\n{base_data.head()}')
    else:
        logging.info("Not changing Index")
        base_data = pivot_data
    results = []

    for (source_data, set_name) in [(embedding, "emb"), (base_data, "base")]:
        logging.info(f'Making plots with {set_name} data')        
        clustering, results = get_clustering(source_data, set_name, path, name, grid_search, set_epsilon, set_n_samples, results)
        num_clusters = set(clustering.labels_)
        distinct_cmap = distinctipy.get_colormap(distinctipy.get_colors(len(num_clusters)-1,n_attempts=5000,rng=42,exclude_colors=[(0,0,0),(1,1,1),(1,0,0)]))
        
        plot_cluster(source_data, embedding, clustering, distinct_cmap, path, name+f"_{set_name}_DBSCAN", title)
        plot_silhouette(source_data,clustering,distinct_cmap,path=path,name=name+f"_{set_name}_silhouette",title="Silhouette Plot of DBSCAN Clustering")

        n_samples = results[-1]["best_min_samples"] if results else 5
        misc_plots(source_data,n_samples,path,name+f"_{set_name}")
    return results

def misc_plots(base_data,n_samples,path,name,save=True):
    # Elbow plot
    logging.debug("Plotting Elbow Graphic")
    min_size = max(n_samples,5)
    neighbors = NearestNeighbors(n_neighbors=min_size)
    neighbors.fit(base_data)
    distances, _ = neighbors.kneighbors(base_data)
    distances = np.sort(distances[:, -1])
    plt.figure(figsize=(10,10))
    plt.plot(distances)
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{min_size}th Nearest Neighbor Distance")
    plt.yticks(np.linspace(0,int(max(distances)+1),int(max(distances))+2).astype(int))
    plt.title("Elbow Plot")
    if save:
        plt.savefig(os.path.join(path,name+"_elbow.png"))
    else:
        plt.show()
    plt.close()

    # other visualizers
    logging.debug("Plotting other visualizer plots")
    km = cluster.KMeans(random_state=42)
    fig, axes = plt.subplots(2,2,figsize=(16,16),squeeze=False,layout='constrained')
    visualizers = [KElbowVisualizer(km, k=(2,max(n_samples+1,10)), ax=axes[0][0], metric="distortion"),
                   KElbowVisualizer(km, k=(2,max(n_samples+1,10)), ax=axes[0][1], metric="silhouette"),
                   KElbowVisualizer(km, k=(2,max(n_samples+1,10)), ax=axes[1][0], metric="calinski_harabasz"),
                   SilhouetteVisualizer(km, axes[1][1], colors=distinctipy.get_colors(10,n_attempts=5000,rng=42))] 

    for vis in visualizers:
        vis.fit(base_data)
        vis.finalize()
    if save:
        plt.savefig(os.path.join(path,name+"_vis.png"))
    else:
        plt.show()

def set_plot_background(scaff, ax):
    # drawing scaffold background
    ax.scatter(scaff[:,0], scaff[:,1], s=10, c="lightgrey", label=f'Background ({len(scaff)})')
    return ax
    
def plot_seg_mosaic(scaffold, scaff_ids, data_df=None, path="", name="seg mosaic", title=""):
    if isinstance(data_df,pd.DataFrame):
        ref = data_df
    else:
        ref = scaff_ids
        if "index" not in ref.columns:
            ref["index"] = ref.index
    if "Segment" not in ref.columns:
        ref["Segment"] = ref["ID"].str.split("_").str[-3]#.transform(lambda x: x.split('_')[-3])
    fig, axes = plt.subplot_mosaic([['Full','PB1','PB2'],
                                    ['PA','HA','NP'],
                                    ['NA','M','NS']
                                    ],figsize=(16,16),layout="constrained",dpi=500)
    ax = axes['Full']
    if isinstance(data_df,pd.DataFrame):
        ax = set_plot_background(scaffold, ax)
        
    # Setting plot boundaries
    if isinstance(scaffold, pd.DataFrame):
        ax.scatter(max(scaffold["UMAP1"]), max(scaffold["UMAP2"]), c="white", s=3, alpha=0.0)
        ax.scatter(min(scaffold["UMAP2"]), min(scaffold["UMAP2"]), c="white", s=3, alpha=0.0)
        ax.set_title("Feature Scaffold")
    else:
        ax.scatter(max(scaffold[:, 0]), max(scaffold[:, 1]), c="white", s=3, alpha=0.0)
        ax.scatter(min(scaffold[:, 0]), min(scaffold[:, 1]), c="white", s=3, alpha=0.0)
        ax.set_title("Feature Scaffold")

    # Coloring each segment
    for seg_num, seg in enumerate(SEGMENTS):
        if isinstance(scaffold, pd.DataFrame):
            mask = scaffold[scaffold["Segment"]==seg]
            
            seg_ax = axes[seg]
            seg_ax.set_title(f'{seg} Segment')
            if len(mask)==0:
                continue
            ax.scatter(mask["UMAP1"], mask["UMAP2"], c=SEGMENT_COLORS[seg_num], s=3, label=f'{seg} ({len(mask)})', alpha=0.25)
            if isinstance(data_df,pd.DataFrame):
                seg_ax = set_plot_background(scaffold, seg_ax)
                
            # Setting plot boundaries
            seg_ax.scatter(max(scaffold["UMAP1"]), max(scaffold["UMAP2"]), c="white", s=3, alpha=0.0)
            seg_ax.scatter(min(scaffold["UMAP2"]), min(scaffold["UMAP2"]), c="white", s=3, alpha=0.0)
            seg_ax.scatter(mask["UMAP1"], mask["UMAP2"], c=SEGMENT_COLORS[seg_num], s=3, label=f'{seg} ({len(mask)})', alpha=0.25)
        
        else:
            mask = ref[ref["Segment"]==seg]["index"].unique()
            
            seg_ax = axes[seg]
            seg_ax.set_title(f'{seg} Segment')
            if not mask.any():
                continue
            ax.scatter(scaffold[mask, 0], scaffold[mask, 1], c=SEGMENT_COLORS[seg_num], s=3, label=f'{seg} ({len(mask)})', alpha=0.25)
            if isinstance(data_df,pd.DataFrame):
                seg_ax = set_plot_background(scaffold, seg_ax)
                
            # Setting plot boundaries
            seg_ax.scatter(max(scaffold[:, 0]), max(scaffold[:, 1]), c="white", s=3, alpha=0.0)
            seg_ax.scatter(min(scaffold[:, 0]), min(scaffold[:, 1]), c="white", s=3, alpha=0.0)
            seg_ax.scatter(scaffold[mask, 0], scaffold[mask, 1], c=SEGMENT_COLORS[seg_num], s=3, label=f'{seg} ({len(mask)})', alpha=0.25)
    ax.legend()
    fig.suptitle(title)
    if path=="":
        plt.show()
    else:
        plt.savefig(os.path.join(path,name+".png"))
    plt.close()

def plot_clusters_vip(dataframe, cluster_col, umap1_col, umap2_col, path="", name="", title="", centroids="No", vips=[]):
    logging.info(f"Creating UMAP plot ({title})\nSaving in {path}")

    # Making sure axes in graphic don't change after application of filters
    plt.figure(figsize=(8,8),dpi=300)
    plt.scatter(dataframe[umap1_col].max(), dataframe[umap2_col].max(), c="white", s=0, alpha=0.0)
    plt.scatter(dataframe[umap1_col].min(), dataframe[umap2_col].min(), c="white", s=0, alpha=0.0)

    # Plotting each group separately
    if len(CLUSTER_COLORS)<dataframe[dataframe[cluster_col]>=0][cluster_col].nunique():
        if -1 in dataframe[cluster_col].unique():    
            colors = distinctipy.get_colors(dataframe[cluster_col].nunique()-1,[(0,0,0),(1,1,1),(1,0,0)])
            colors.append((1,0,0))
        else:
            colors = distinctipy.get_colors(dataframe[cluster_col].nunique())
    else:
        colors = CLUSTER_COLORS
        if -1 in dataframe[cluster_col].unique():
            colors.append((1,0,0))
    cluster_color = {cluster: col for cluster, col in zip(sorted(dataframe[cluster_col].unique()),colors) if cluster>=0}
    cluster_color[-1] = colors[-1]

    legend_handles = []
    for cluster in sorted(dataframe[cluster_col].unique()):
        mask = dataframe[dataframe[cluster_col] == cluster]
        label = f"{cluster}" if cluster != -1 else "noise"
        color = cluster_color[cluster]
        alpha = 0.5 if cluster != -1 else 1

        plt.scatter(mask[umap1_col], mask[umap2_col], c=color, s=5, label=label, alpha=alpha)

        # Custom legend entry with larger marker
        legend_handles.append(
            mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                          markersize=8, label=label)
        )
        
        if centroids!="No":

            if isinstance(centroids,list) and len(centroids) > 0 and cluster >= 0:  # direct list of centroids
                cent_x, cent_y = centroids[cluster]
            elif isinstance(centroids,bool) and centroids:
                if "centroid_x" in mask.columns and "centroid_y" in mask.columns:   # try to find in columns
                    cent_x, cent_y = mask["centroid_x"].iat[0], mask["centroid_y"].iat[0]
            elif isinstance(centroids, tuple):                                      # tuple of column names
                if centroids[0] in mask.columns and centroids[1] in mask.columns:
                    cent_x, cent_y = mask[centroids[0]].iat[0], mask[centroids[1]].iat[0]
            if not (cent_x and cent_y):                                             # Otherwise use own mean
                cent_x, cent_y = mask[umap1_col].mean(), mask[umap2_col].mean()

            plt.scatter(cent_x, cent_y, 
                    c=[cluster_color[cluster]], 
                    marker='X', 
                    s=50, 
                    edgecolor='black',
                    linewidth=1,
                    label=None, 
                    zorder=3)
            
    if len(vips)>0:
        plt.scatter(dataframe.iloc[vips][umap1_col],dataframe.iloc[vips][umap2_col],s=50,marker="*",edgecolor='black',linewidth=1,c="black",zorder=4)


    plt.annotate(f'Silhouette Score: {shs(dataframe[[umap1_col,umap2_col]],dataframe[cluster_col]):.4f}',(min(dataframe[umap1_col]),min(dataframe[umap2_col])))

    plt.legend(handles=legend_handles, title="Clusters", fontsize='small', title_fontsize='medium')
    
    plt.title(title)
    if path == "":
        plt.show()
    else:
        plt.savefig(os.path.join(path,name+".png"))
    plt.close()