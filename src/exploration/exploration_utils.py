# imports
import os
import warnings
import datetime
import logging
import sys
from pathlib import Path
import numpy as np
import shap

import traceback
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, RocCurveDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
import joblib
import json
import seaborn as sns
import glob
import scipy.stats as stats
from typing import Tuple
from matplotlib import rcParams
from matplotlib.patches import Patch
from matplotlib import patches
rcParams.update({'figure.autolayout': True})

from statsmodels.stats.contingency_tables import mcnemar

BASE_DIR = Path(os.getcwd())
SRC_DIR = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import calculate_features, load_data, calculate_target, make_multiclass, apply_cutoff, cutoff_clean, drop_non_numeric, split_data, stratified_undersample, identify_candidates, transform_meta_features, get_sequence_quicker, get_sequence, make_multiclass, _extract_junction_window, DATA_DIR, SEGMENTS, STRAIN_WISE_PUBLICATIONS, STRAIN_WISE_PUB_COLORS, STRAIN_COLORS, SEGMENT_COLORS, PUBLICATIONS
RESULT_PATH = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'results',"standard_feature_exploration"))

ALL_PUBS = ["Alnaji2021", "Pelz2021", "Wang2023", "Wang2020", "Zhuravlev2020", "Kupke2020", "vdHoecke2015", "Alnaji2019", "Mendes2021", "Boussier2020", "Berry2021", "Penn2022", "Lui2019", "Valesano2020", "Sheng2018", "Southgate2019"]
ALL_STRAINS = ["A_PuertoRico_8_1934", "A_California_07_2009", "A_NewCaledonia_20-JY2_1999", "A_WSN_33", "A_Perth_16_2009", "A_Connecticut_Flu122_2013", "A_turkey_Turkey_1_2005", "A_Anhui_1_2013", "B_Lee_1940", "B_Victoria_504_2000", "B_Brisbane_60_2008", "B_Yamagata_16_1988"]

strain_to_pubs = {'A_PuertoRico_8_1934': ['Alnaji2021', 'Pelz2021', 'Wang2023', 'Wang2020', 'Zhuravlev2020', 'Kupke2020', 'VdHoecke2015'],
                  'A_WSN_33': ['Boussier2020', 'Mendes2021'],
                  'B_Victoria_504_2000': ['Valesano2020', 'Berry2021'],
                  'B_Yamagata_16_1988': ['Southgate2019', 'Valesano2020', 'Berry2021']}
STRAIN_TO_PUBS = strain_to_pubs
STRAIN = 'A_PuertoRico_8_1934'
PUBS_TO_USE = ['Alnaji2021', 'Kupke2020', 'Pelz2021', 'vdHoecke2015', 'Wang2020', 'Wang2023', 'Zhuravlev2020']
ALL_SEGMENTS = ["PB2", "PB1", "PA", "HA", "NP", "NA", "M", "NS"]
STANDARD_FEATURES_DEFAULT = ['Segment', 'Start', 'End', 'Direct_repeat', 'Junction', 'remaining_length', 'deletion_length', '3_5_diff', 'length_proportion', 'Peptide_Length', "3_len", "5_len"]
NUCLEOTIDES = {'U': 'Uracil', 'G': 'Guanine', 'C': 'Cytosine', 'A': 'Adenine'}
FEATURES_TO_USE = ['Start', 'End', 'Direct_repeat', 'remaining_length', 'deletion_length', '3_5_diff', 'length_proportion', 'Peptide_Length', "3_len", "5_len"]

import warnings

def setup_logging(path, verbose=False, ignore_warnings=False):
    if ignore_warnings:
        warnings.filterwarnings('ignore')
    os.makedirs(path, exist_ok=True)
    log_path = os.path.join(path, 'results.log')
    fmt_debug = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s'
    fmt_info  = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
        format=fmt_debug if verbose else fmt_info,
        level=logging.DEBUG if verbose else logging.INFO,
        force=True
    )

def load_strain_data(strain: str, cutoff: int=15, features_to_use: list=FEATURES_TO_USE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Loads data for a given strain, applies cutoff, identifies candidates, and calculates features.

    :param strain: Strain to load data for
    :param cutoff: Cutoff to apply to data
    :param features_to_use: List of features to calculate

    :return: Tuple of (data, feature_data, features_cols, intersecting_ids)
    '''
    # getting experimental data for given strain
    data = load_data(STRAIN_TO_PUBS[strain], unpooled=True)
    data = data[data["Strain"]==strain]
    data = identify_candidates(data)

    # identifying intersecting IDs before RSC application
    id_group_counts = data.groupby('ID')['Publication'].nunique()
    intersecting_ids = id_group_counts[id_group_counts >= max(2, data["Publication"].nunique() / 2)].index.tolist()

    # applying RSC cutoff and calculating features
    data = cutoff_clean(data, cutoff, minimum_dataset_size=0).reset_index(drop=True)
    feature_data = calculate_features(data.drop_duplicates("ID").reset_index(drop=True), standard_features=features_to_use, inplace=False, scale="none", normalize_by_length=False)
    feature_data["Start_normalized"] = feature_data["Start"]/feature_data["Full_Sequence"].transform(len)
    feature_data["End_normalized"] = feature_data["End"]/feature_data["Full_Sequence"].transform(len)
    for col in ["remaining_length", "deletion_length", "Peptide_Length", "3_len", "5_len"]:
        if col in feature_data.columns:
            feature_data[f'{col}_normalized'] = feature_data[col]/feature_data["Full_Sequence"].transform(len)
    feature_data["3_5_diff_normalized"] = feature_data["3_len_normalized"] - feature_data["5_len_normalized"]
    features_cols = [col for col in feature_data.columns if col not in data.columns or col in FEATURES_TO_USE]
    
    return data, feature_data, features_cols, intersecting_ids

def load_synthetic(names: list, include_publication: bool = True, unpooled = False) -> pd.DataFrame:
    '''
    Loads data from any csv files, corresponding to the given list of publication names.

    :param names: List of publication names to load data from
    :param include_publication: Whether or not to include Publciation column in dataframe
    :param unpooled: Whether or not to use unpooled data

    :return: Dataframe containing all data from the given publications
    '''
    data_dir = DATA_DIR
    if isinstance(names, str):
        names = [names]
    csv_paths = []
    pubs = []
    for publication in names:
        publication_paths = glob.glob(os.path.join(data_dir, '**', f'{publication}*.tsv'), recursive=True)
        if len(publication_paths) < 1 and publication == "VdHoecke2015": # temporary fix for capitalization issue
            publication_paths = glob.glob(os.path.join(data_dir, '**', f'vdHoecke2015*.tsv'), recursive=True)
        csv_paths.extend(publication_paths)
        pubs.extend(publication)
    assert len(csv_paths) > 0, f'No data found for {names}'
    logging.debug(f'Found {len(csv_paths)} files for {names}:\n{csv_paths}')
    logging.info(f'Loading data for {names}')

    def load_and_label(file_path):
        '''
        Loads a csv file and adds a column 'Strain', based on the file's path.

        :param file_path: location of the csv file

        :return: Dataframe including all data from the csv file and a column 'Strain'
        '''
        df = pd.read_csv(file_path, sep='\t', low_memory=False, keep_default_na=False)
        df['Strain'] = os.path.basename(os.path.dirname(file_path))
        if include_publication:
            pub = os.path.basename(file_path).split('/')[-1].split('.')[0]
            if "_" in pub:
                pub = pub.split('_')[0]
            if pub == "vdHoecke2015":
                pub = "VdHoecke2015"
            #index = csv_paths.index(file_path)
            df['Publication'] = pub
        return df

    dfs = [load_and_label(file_path) for file_path in csv_paths]
    final_df = pd.concat(dfs, ignore_index=True)
    logging.debug(f'Loaded data for {names} with shape {final_df.shape}\nColumns: {list(final_df.columns)}')
    final_df["Start"] = final_df["Start"].astype(int)
    final_df["End"] = final_df["End"].astype(int)
    final_df["NGS_read_count"] = final_df["NGS_read_count"].astype(int)
    logging.debug(f'Datatypes:\n{final_df.dtypes}')
    #get_duplicates(final_df)

    return final_df

def create_sampling_space(seq: str, s: Tuple[int, int], e: Tuple[int, int])-> pd.DataFrame:
    '''
    Creates all possible candidates that would be expected.
    :param seq: RNA sequence
    :param s: tuple with start and end point of the range for the artifical
                start point of the deletion sites
    :param e: tuple with start and end point of the range for the artifical
                end point of the deletion sites
    
    :return: dataframe with possible DelVG candidates
    '''
    # create all combinations of start and end positions that are possible
    combinations = [(x, y) for x in range(s[0], s[1]+1) for y in range(e[0], e[1]+1)]

    # create for each the DelVG Sequence
    sequences = [seq[:start] + seq[end-1:] for (start, end) in combinations]

    # filter out duplicate DelVG sequences while keeping the ones with highest start number
    start, end = zip(*combinations)
    temp_df = pd.DataFrame(data=dict({"Start": start, "End": end, "Sequence": sequences}))

    # Find the index of the row with the maximum value in the "Start" column for each "Sequence"
    max_start_index = temp_df.groupby("Sequence")["Start"].idxmax()
    result_df = temp_df.loc[max_start_index]
    # Replicate each row by the number of times it was found in the group
    result_df = result_df.loc[result_df.index.repeat(temp_df.groupby("Sequence").size())]
    df_no_duplicates = result_df.reset_index(drop=True).drop("Sequence", axis=1)

    return df_no_duplicates

def generate_sampling_data(seq: str, s: Tuple[int, int], e: Tuple[int, int],  n: int)-> pd.DataFrame:
    '''
        Generates sampling data by creating random start and end points for
        artificial deletion sites. Generated data is used to calculate the
        expected values.
        :param seq: RNA sequence
        :param s: tuple with start and end point of the range for the artifical
                  start point of the deletion sites
        :param e: tuple with start and end point of the range for the artifical
                  end point of the deletion sites
        :param n: number of samples to generate

        :return: Pandas DataFrame of the artifical data set
    '''
    df_no_duplicates = create_sampling_space(seq, s, e)
    return df_no_duplicates.sample(n)

def generate_expected_data(strain: str, df: pd.DataFrame, cutoff: int=15, seg_sample_size: int=35000, force_recreate: bool=False, multi_source: bool=True)-> pd.DataFrame:
    '''
        Randomly samples deletion sites for a given dataset which can be used
        to compare the results of the real dataset.
        :param strain: name of the strain
        :param df: DelVG dataset
        :param cutoff: minimum number of deletions required for a site to be considered
        :param seg_sample_size: number of samples to generate per segment
        :param force_recreate: whether to force recreation of the synthetic dataset

        :return: artifical dataset that includes random deletion sites
    '''

    def get_all_segment_samples(sub_data):
        samp_df = None
        for seg in ALL_SEGMENTS:
            df_s = sub_data.loc[sub_data["Segment"] == seg]
            if len(df_s) == 0:
                continue
            seq = get_sequence(strain, seg)
            start = int(df_s["Start"].mean())
            end = int(df_s["End"].mean())
            s = (max(start-200, 50), start+200)
            e = (end-200, min(end+200, len(seq)-50))
            
            # skip if there is no range given this would lead to oversampling of a single position
            if s[0] == s[1] or e[0] == e[1]:
                continue
            if samp_df is not None:
                temp_df = generate_sampling_data(seq, s, e, n=seg_sample_size)
                temp_df["Segment"] = seg
                samp_df = pd.concat([samp_df, temp_df], ignore_index=True)
            else:
                samp_df = generate_sampling_data(seq, s, e, n=seg_sample_size)
                samp_df["Segment"] = seg
        return samp_df

    synth_name = "_".join([pub for pub in df["Publication"].unique()])
    synth_name = f"{strain}_{synth_name}_{cutoff}{f'_multisource' if multi_source and df["Publication"].nunique() > 1 else ''}_synthetic.csv"
    if os.path.exists(os.path.join(RESULT_PATH, synth_name)) and not force_recreate:
        logging.info(f"Loading existing synthetic dataset for {strain} from {synth_name}")
        return pd.read_csv(os.path.join(RESULT_PATH, synth_name))
    cut_df = cutoff_clean(df, cutoff, minimum_dataset_size=0, inplace=False).reset_index(drop=True)
    sample_df = None
    logging.info(f"Generating synthetic dataset for {strain} with cutoff {cutoff}{f' from multiple sources' if multi_source and df["Publication"].nunique() > 1 else ''}")
    if multi_source and df["Publication"].nunique() > 1:
        for pub in df["Publication"].unique():
            pub_data = cut_df[cut_df["Publication"]==pub]
            pub_sample_df = get_all_segment_samples(pub_data)
            if sample_df is not None:
                sample_df = pd.concat([sample_df, pub_sample_df], ignore_index=True)
            else:
                sample_df = pub_sample_df
    else:
        sample_df = get_all_segment_samples(cut_df)
    sample_df["Strain"] = strain
    sample_df["NGS_read_count"] = 1
    os.makedirs(RESULT_PATH, exist_ok=True)
    sample_df.to_csv(os.path.join(RESULT_PATH, synth_name), index=False)
    return sample_df.reset_index()


########    Functions for plotting the effect of different cutoff values on the number of intersecting and total DelVGs, as well as segment-wise intersecting counts and publication-wise intersecting counts    ########
def plot_intercount_cutoff_effect(cutoff_results_df, log=True, title="Effect of RSC threshold on remaining DelVGs", name_prefix="", path="", ax=None, get_ax=False, plot=True):
    '''
    Plots the effect of different cutoff values on the number of intersecting DelVGs and total DelVG IDs.
    :param cutoff_results_df: DataFrame containing the results for different cutoff values, with columns "Cutoff", "Intersecting_IDs", and "Total_IDs"
    :param log: Whether to use logarithmic scale for y-axis
    :param title: Title of the plot
    :param path: Path to save the plot (if empty, the plot will be shown instead)
    :param ax: Matplotlib axis to plot on (if None, a new figure will be created)
    :param get_ax: Whether to return the axis object after plotting
    :param plot: Whether to display or save the plot
    :return: Matplotlib axis object if get_ax is True, otherwise None
    '''
    sns.set_theme(style="darkgrid", context="notebook", palette="colorblind")
    if ax is None:
        plt.figure(figsize=(6, 4))
    ax = sns.lineplot(data=cutoff_results_df, x="Cutoff", y="Intersecting_IDs", marker='o', label="Intersecting DelVGs", ax=ax)
    if log:
        ax.set_yscale("log")
    sns.lineplot(data=cutoff_results_df, x="Cutoff", y="Total_IDs", marker='o', label="Total DelVGs", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("RSC threshold")
    ax.set_ylabel("Number of unique DelVGs")
    ax.set_xticks([5,10,15,20])
    ax.grid()
    if path != "" and plot:
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f"{name_prefix}cutoff_effect_on_delvg_counts.png"), dpi=300)
    elif plot:
        plt.show()
    if get_ax:
        return ax
    plt.close()

def plot_segwise_intercount_cutoff_effect(segwise_results_df, log=True, name_prefix="", title="Effect of RSC threshold on intersecting DelVGs\nper Segment", path="", ax=None, get_ax=False, plot=True):
    '''
    Plots the effect of different cutoff values on the number of remaining intersecting DelVGs for each segment.
    '''
    sns.set_theme(style="darkgrid", context="notebook", palette="colorblind")
    if ax is None:
        plt.figure(figsize=(6, 4))
    seg_colors = SEGMENT_COLORS
    for seg in  ALL_SEGMENTS:
            seg_data = segwise_results_df[segwise_results_df["Segment"]==seg]
            ax = sns.lineplot(data=seg_data, x="Cutoff", y="Intersecting_IDs", marker='o', label=f"{seg}", color=seg_colors.get(seg, "grey"), ax=ax)
            #sns.lineplot(data=seg_data, x="Cutoff", y="Total_IDs", marker='o', label=f"Total unique DelVGs - {seg}")
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("RSC threshold")
    ax.set_ylabel("Number of unique DelVGs")
    ax.set_xticks([5,10,15,20])
    ax.grid()
    if path != "" and plot:
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f"{name_prefix}cutoff_effect_on_intersecting_counts_segwise.png"), dpi=300)
    elif plot:
        plt.show()
    if get_ax:
        return ax
    plt.close()

def plot_segwise_combined_cutoff_effect(cutoff_results_df, segwise_results_df, log=True, name_prefix="", title="Number of unique DelVGs depending on Cutoff", path=""):
    '''
    Combines the plots for overall intersecting and total DelVG counts with the segment-wise intersecting counts for different cutoff values into a single figure with two subplots.
    '''
    sns.set_theme(style="darkgrid", context="notebook", palette="colorblind")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    # Overall plot
    sns.lineplot(data=cutoff_results_df, x="Cutoff", y="Intersecting_IDs", marker='o', label="Intersecting DelVGs", ax=axes[0])
    sns.lineplot(data=cutoff_results_df, x="Cutoff", y="Total_IDs", marker='o', label="Total DelVGs", ax=axes[0])
    if log:
        axes[0].set_yscale("log")
    axes[0].set_title("Overall DelVG Counts")
    axes[0].set_xlabel("RSC threshold")
    axes[0].set_ylabel("Number of observed DelVGs")
    axes[0].set_xticks([5, 10, 15, 20])

    seg_colors = SEGMENT_COLORS
    for seg in ALL_SEGMENTS:
        seg_data = segwise_results_df[segwise_results_df["Segment"]==seg]
        sns.lineplot(data=seg_data, x="Cutoff", y="Intersecting_IDs", marker='o', label=f"{seg}", ax=axes[1], color=seg_colors.get(seg, "grey"))
    if log:
        axes[1].set_yscale("log")
    axes[1].set_title("Intersecting DelVG Counts by Segment")
    axes[1].set_xlabel("RSC threshold")
    axes[1].set_ylabel("Number of intersecting DelVGs")
    axes[1].set_xticks([5, 10, 15, 20])
    plt.suptitle(title)
    #plt.xticks(cutoff_results_df["Cutoff"].unique())
    if path != "":
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f"{name_prefix}cutoff_effect_on_intersecting_counts_segwise_combined.png"), dpi=300)
    else:
        plt.show()
    plt.close()

def plot_pubwise_intersecting_counts(pub_wise_intersecting_counts_df, title="Intersecting DelVGs remaining after RSC application", name_prefix="", path="", log=True, ax=None, get_ax=False, plot=True):
    '''
    Plots the number of intersecting DelVGs in each publication by cutoff value.
    '''
    sns.set_theme(style="darkgrid", context="notebook", palette="colorblind")
    if ax is None:
        plt.figure(figsize=(6, 4))
    colors = STRAIN_WISE_PUB_COLORS.get(STRAIN, sns.color_palette("colorblind"))
    #colors = {pub: color for pub, color in zip(STRAIN_TO_PUBS.get(STRAIN, []), sns.color_palette("Set2", len(STRAIN_TO_PUBS.get(STRAIN, []))))}
    for pub in pub_wise_intersecting_counts_df.columns:
        ax = sns.lineplot(data=pub_wise_intersecting_counts_df, x=pub_wise_intersecting_counts_df.index, y=pub, marker='o', label=f"{pub}", color=colors.get(pub, "grey"), ax=ax)
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("RSC threshold")
    ax.set_ylabel("Number of intersecting DelVGs")
    ax.set_xticks([5,10,15,20])
    ax.grid()
    if path != "" and plot:
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f"{name_prefix}intersecting_delvgs_by_pub_and_cutoff.png"), dpi=300)
    elif plot:
        plt.show()
    if get_ax:
        return ax
    plt.close()

def plot_cutoff_pub_coverage(cutoff_results_df, log=True, title="Dataset coverage of intersecting DelVGs\nby RSC threshold", name_prefix="", path="", get_ax=False, ax=None, plot=True):
    sns.set_theme(style="darkgrid", context="notebook", palette="colorblind")
    if ax is None:
        plt.figure(figsize=(6, 4))
        ax = plt.gca()
    pub_counts_cols = [col for col in cutoff_results_df.columns if col not in ["Cutoff", "Total_IDs"] and col > 0]
    #colors = {cover: colors for cover, colors in zip(sorted(pub_counts_cols), sns.color_palette("Spectral", len(pub_counts_cols)))}
    colors = {cover: colors for cover, colors in zip(sorted(pub_counts_cols), sns.diverging_palette(10, 220, s=100, l=75, n=len(pub_counts_cols), center="dark"))}
    
    for col in sorted(pub_counts_cols):
        if col not in ["Cutoff", "Total_IDs"]:
            sns.lineplot(data=cutoff_results_df, x="Cutoff", y=col, marker='.', label=f"{int(col)} sets", color=colors.get(col, "grey"), ax=ax)
    ax = plt.gca() if ax is None else ax
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("RSC threshold")
    ax.set_ylabel("Number of intersecting DelVGs")
    ax.set_xticks([5,10,15,20])
    ax.grid()
    if path != "" and plot:
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f"{name_prefix}dataset_coverage.png"), dpi=300)
    elif plot:
        plt.show()
    if get_ax:
        return ax
    plt.close()

def cutoff_inter_pipeline(strain, cutoffgrid, ver="corrected", save_plots=False):
    '''
    Processes the data for a given strain and cutoff values, identifying intersecting DelVGs and calculating the number of intersecting DelVGs overall, by segment, and by publication for each cutoff value. The results are returned as DataFrames for overall counts, segment-wise counts, and publication-wise counts.
    :param strain: Strain to process
    :param cutoffgrid: List of cutoff values to apply
    :param ver: Version of the thresholding method to use ("a" for max(2, int(data["Publication"].nunique() / 2)), "b" for max(2, data["Publication"].nunique() / 2))
    :return: Tuple of DataFrames (pre_cutoff_results_df, post_cutoff_results_df, pre_seg_wise_results_df, post_seg_wise_results_df, pub_wise_intersecting_counts_df)
    '''
    data = load_data(strain_to_pubs[strain], unpooled=True)
    data = data[data["Strain"]==strain]
    data = identify_candidates(data)
    id_group_counts = data.groupby('ID')['Publication'].nunique()
    id_cutoff_counts = pd.DataFrame(data.groupby('ID')['Publication'].nunique())
    if ver=="a":
        all_intersecting_ids = id_group_counts[id_group_counts >= max(2,int(data["Publication"].nunique() / 2))].index.tolist()
    else:
        all_intersecting_ids = id_group_counts[id_group_counts >= max(2, data["Publication"].nunique() / 2)].index.tolist()
    pre_cutoff_results = []
    post_cutoff_results = []
    pre_seg_wise_results = []
    post_seg_wise_results = []
    pub_wise_intersecting_counts = {}
    #prog_display = display(f"Processing strain: {strain}", display_id=True)
    for cutoff in cutoffgrid:
        #prog_display.update(f"Processing strain: {strain} | Cutoff: {cutoff}")
        cutoff_data = cutoff_clean(data, cutoff, minimum_dataset_size=0).reset_index(drop=True)
        pre_cutoff_results.append({"Cutoff": cutoff, "Total_IDs": len(cutoff_data["ID"].unique()), "Intersecting_IDs": cutoff_data[cutoff_data["ID"].isin(all_intersecting_ids)]["ID"].nunique()})
        id_cutoff_counts[f'Cutoff {cutoff}'] = 0
        val_counts = cutoff_data.drop_duplicates(subset=['ID', 'Publication'])["ID"].value_counts()
        id_cutoff_counts.loc[val_counts.index, f'Cutoff {cutoff}'] = val_counts.values
        id_group_counts = cutoff_data.groupby('ID')['Publication'].nunique()
        if ver=="a":
            intersecting_ids = id_group_counts[id_group_counts >= max(2,int(data["Publication"].nunique() / 2))].index.tolist()
        else:
            intersecting_ids = id_group_counts[id_group_counts >= max(2, data["Publication"].nunique() / 2)].index.tolist()
        post_cutoff_results.append({"Cutoff": cutoff, "Total_IDs": len(cutoff_data["ID"].unique()), "Intersecting_IDs": len(intersecting_ids)})
        pub_wise_intersecting_counts[cutoff] = {}
        #prog_display.update(f"Processing strain: {strain} | Cutoff: {cutoff} | Calculating publication-wise intersecting counts")
        for pub in STRAIN_TO_PUBS[strain]:
            pub_wise_intersecting_counts[cutoff][pub] = cutoff_data[(cutoff_data["Publication"]==pub) & (cutoff_data["ID"].isin(all_intersecting_ids))]["ID"].nunique()
        for seg in ALL_SEGMENTS:
            #prog_display.update(f"Processing strain: {strain} | Cutoff: {cutoff} | Segment: {seg}")
            seg_data = cutoff_data[cutoff_data["Segment"]==seg]
            seg_id_counts = seg_data.groupby('ID')['Publication'].nunique()
            if ver=="a":
                seg_intersecting_ids = seg_id_counts[seg_id_counts >= max(2,int(data["Publication"].nunique() / 2))].index.tolist()
            else:
                seg_intersecting_ids = seg_id_counts[seg_id_counts >= max(2, data["Publication"].nunique() / 2)].index.tolist()
            post_seg_wise_results.append({"Cutoff": cutoff, "Segment": seg, "Total_IDs": len(seg_data["ID"].unique()), "Intersecting_IDs": len(seg_intersecting_ids)})
            pre_seg_wise_results.append({"Cutoff": cutoff, "Segment": seg, "Total_IDs": len(seg_data["ID"].unique()), "Intersecting_IDs": seg_data[seg_data["ID"].isin(all_intersecting_ids)]["ID"].nunique()})  # len([id for id in all_intersecting_ids if id in seg_data["ID"].values])})
    
    #prog_display.update(f"Processing strain: {strain} | Calculating publication-coverages")
    pub_coverage = []
    for cutoff in cutoffgrid:
        pub_counts = id_cutoff_counts.loc[all_intersecting_ids, f'Cutoff {cutoff}']
        cur_res = {num_pubs: count for num_pubs, count in pub_counts.value_counts().items()}
        cur_res.update({"Cutoff": cutoff, "Total_IDs": len(pub_counts[pub_counts>0])})
        pub_coverage.append(cur_res)

    #prog_display.update(f"Completed processing for strain: {strain}")
    pre_cutoff_results_df = pd.DataFrame(pre_cutoff_results)
    post_cutoff_results_df = pd.DataFrame(post_cutoff_results)
    post_seg_wise_results_df = pd.DataFrame(post_seg_wise_results)
    pre_seg_wise_results_df = pd.DataFrame(pre_seg_wise_results)
    pub_wise_intersecting_counts_df = pd.DataFrame.from_dict(pub_wise_intersecting_counts, orient="index")
    pub_coverage_df = pd.DataFrame(pub_coverage)
    cutoff_pipeline_plots(pre_cutoff_results_df, post_cutoff_results_df, pre_seg_wise_results_df, post_seg_wise_results_df, pub_wise_intersecting_counts_df, pub_coverage_df, strain, save_plots=save_plots)
    return pre_cutoff_results_df, post_cutoff_results_df, pre_seg_wise_results_df, post_seg_wise_results_df, pub_wise_intersecting_counts_df, pub_coverage_df

def cutoff_pipeline_plots(pre_cutoff_results_df, post_cutoff_results_df, pre_seg_wise_results_df, post_seg_wise_results_df, pub_wise_intersecting_counts_df, pub_coverage_df, strain, save_plots=False):
    if save_plots:
        strain_dir = os.path.join(RESULT_PATH, strain, "rsc plots")
        os.makedirs(strain_dir, exist_ok=True)
        logging.info(f"Saving plots to {strain_dir}")
    else:
        strain_dir = ""
    for prefix_add, log_tag in zip(["linear", "log"], [False, True]):
        plot_intercount_cutoff_effect(pre_cutoff_results_df, name_prefix=prefix_add, log=log_tag, path=strain_dir)
        plot_segwise_intercount_cutoff_effect(pre_seg_wise_results_df, name_prefix=prefix_add, log=log_tag, path=strain_dir)
        plot_intercount_cutoff_effect(post_cutoff_results_df, name_prefix=f"postRSC_{prefix_add}", log=log_tag, path=strain_dir)
        plot_segwise_intercount_cutoff_effect(post_seg_wise_results_df, name_prefix=f"postRSC_{prefix_add}", log=log_tag, path=strain_dir)
        plot_segwise_combined_cutoff_effect(post_cutoff_results_df, post_seg_wise_results_df, name_prefix=f"postRSC_{prefix_add}", log=log_tag, title=f"Post-RSC", path=strain_dir)
        plot_segwise_combined_cutoff_effect(pre_cutoff_results_df, pre_seg_wise_results_df, name_prefix=f"preRSC_{prefix_add}", log=log_tag, title=f"Pre-RSC", path=strain_dir)
        plot_pubwise_intersecting_counts(pub_wise_intersecting_counts_df, name_prefix=prefix_add, log=log_tag, path=strain_dir)
        plot_cutoff_pub_coverage(pub_coverage_df, name_prefix=prefix_add, log=log_tag, path=strain_dir)

def execute_rsc_intersecting_pipeline_per_strain():
    logging.info("Starting rsc pipeline execution...")
    for strain in STRAIN_TO_PUBS.keys():
        pre_cutoff_results_df, post_cutoff_results_df, pre_seg_wise_results_df, post_seg_wise_results_df, pub_wise_intersecting_counts_df, pub_coverage_df = cutoff_inter_pipeline(strain, cutoffgrid=list(range(1, 21)), ver="correct", save_plots=True)


########    Functions for comparing intersecting vs non-intersecting IDs feature-wise, including multiple-testing correction and effect size calculation    ########
def benjamini_hochberg(p_values):
    '''
    Benjamini-Hochberg FDR correction (returns adjusted p-values in original order).
    '''
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]

    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = prev

    adj = np.clip(adj, 0, 1)
    out = np.empty(n, dtype=float)
    out[order] = adj
    return out

def cliffs_delta(x, y):
    '''
    Cliff's delta effect size.
    '''
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan

    # Exact pairwise comparison; suitable for moderate sample sizes
    gt = np.sum(x[:, None] > y[None, :])
    lt = np.sum(x[:, None] < y[None, :])
    return (gt - lt) / (len(x) * len(y))

def aggregate_id_features(feature_df, feature_cols=None, id_col="ID", agg="median"):
    '''
    Aggregate feature values per ID to avoid repeated-observation bias.
    '''
    if feature_cols is None:
        feature_cols = FEATURES_TO_USE

    numeric_cols = [c for c in feature_cols if c in feature_df.columns and pd.api.types.is_numeric_dtype(feature_df[c])]
    agg_df = feature_df.groupby(id_col, as_index=False)[numeric_cols].agg(agg)
    return agg_df, numeric_cols

def compare_intersecting_vs_nonintersecting(feature_df, intersecting_ids, feature_cols=None, id_col="ID", agg="median", test="mannwhitney", alternative="two-sided", alpha=0.05):
    '''
    Compare intersecting vs non-intersecting IDs feature-wise.
    Returns a results DataFrame with raw and FDR-adjusted p-values.
    '''
    agg_df, numeric_cols = aggregate_id_features(
        feature_df=feature_df,
        feature_cols=feature_cols,
        id_col=id_col,
        agg=agg
    )

    inter_set = set(intersecting_ids)
    agg_df["is_intersecting"] = agg_df[id_col].isin(inter_set)

    results = []
    for feat in numeric_cols:
        x = agg_df.loc[agg_df["is_intersecting"], feat].dropna().values
        y = agg_df.loc[~agg_df["is_intersecting"], feat].dropna().values

        if len(x) < 3 or len(y) < 3:
            stat, p = np.nan, np.nan
        else:
            if test == "ttest":
                stat, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
            else:  # default: non-parametric
                stat, p = stats.mannwhitneyu(x, y, alternative=alternative)

        results.append(
            {
                "feature": feat,
                "n_intersecting": len(x),
                "n_non_intersecting": len(y),
                "mean_intersecting": np.nanmean(x) if len(x) else np.nan,
                "mean_non_intersecting": np.nanmean(y) if len(y) else np.nan,
                "median_intersecting": np.nanmedian(x) if len(x) else np.nan,
                "median_non_intersecting": np.nanmedian(y) if len(y) else np.nan,
                "effect_cliffs_delta": cliffs_delta(x, y),
                "statistic": stat,
                "p_value": p,
            }
        )

    res = pd.DataFrame(results)
    res["p_adj_bh"] = benjamini_hochberg(res["p_value"].fillna(1.0).values)
    res["significant"] = res["p_adj_bh"] < alpha
    res = res.sort_values(["p_adj_bh", "p_value"], ascending=True).reset_index(drop=True)
    return res

def run_feature_cutoff_pipeline(feature, strain, cutoffs, pubs_by_strain=None, agg="median", publication_threshold_ratio=0.5, min_group_n=5, seg_wise=False):
    '''
    Takes a feature and a strain, and runs the intersecting vs non-intersecting comparison for each cutoff value. Applies Wilcoxon-Mann-Whitney U test and Welch's t-test, returning a summary DataFrame with one row per cutoff value. Also calculates Cliff's delta effect size for each comparison.
    '''
    if pubs_by_strain is None:
        pubs_by_strain = strain_to_pubs

    if strain not in pubs_by_strain:
        raise ValueError(f"Strain '{strain}' not found in pubs_by_strain.")

    rows = []
    for c in cutoffs:
        # load + filter + cutoff
        d = load_data(pubs_by_strain[strain], unpooled=True)
        d = d[d["Strain"] == strain]
        d = identify_candidates(d)
        n_pubs = d["Publication"].nunique()
        id_pub_counts = d.groupby("ID")["Publication"].nunique()
        inter_ids = id_pub_counts[id_pub_counts >= n_pubs * publication_threshold_ratio].index.tolist()
        d = cutoff_clean(d, c, minimum_dataset_size=0).reset_index(drop=True)

        if d.empty:
            rows.append({
                "category": "overall",
                "cutoff": c,
                "n_ids": 0,
                "n_intersecting_ids": 0,
                "n_non_intersecting_ids": 0,
                "mannwhitney_stat": np.nan,
                "mannwhitney_p": np.nan,
                "ttest_stat": np.nan,
                "ttest_p": np.nan,
                "cliffs_delta": np.nan,
                "status": "no_data_after_cutoff"
            })
            continue

        # features per unique ID
        f = calculate_features(
            d.drop_duplicates("ID").reset_index(drop=True),
            standard_features=FEATURES_TO_USE,
            inplace=False,
            scale="none",
            normalize_by_length=False
        )

        # normalized features (same logic used earlier in notebook)
        seq_len = f["Full_Sequence"].str.len()
        f["Start_normalized"] = f["Start"] / seq_len
        f["End_normalized"] = f["End"] / seq_len
        for col in ["remaining_length", "deletion_length", "Peptide_Length", "3_len", "5_len"]:
            if col in f.columns:
                f[f"{col}_normalized"] = f[col] / seq_len
        if "3_len_normalized" in f.columns and "5_len_normalized" in f.columns:
            f["3_5_diff_normalized"] = f["3_len_normalized"] - f["5_len_normalized"]

        if feature not in f.columns or not pd.api.types.is_numeric_dtype(f[feature]):
            rows.append({
                "category": "overall",
                "cutoff": c,
                "n_ids": f["ID"].nunique(),
                "n_intersecting_ids": np.nan,
                "n_non_intersecting_ids": np.nan,
                "mannwhitney_stat": np.nan,
                "mannwhitney_p": np.nan,
                "ttest_stat": np.nan,
                "ttest_p": np.nan,
                "cliffs_delta": np.nan,
                "status": f"feature_not_numeric_or_missing: {feature}"
            })
            continue

        # get aggregated values to check group sizes
        agg_df, _ = aggregate_id_features(f, feature_cols=[feature], id_col="ID", agg=agg)
        agg_df["is_intersecting"] = agg_df["ID"].isin(set(inter_ids))
        x = agg_df.loc[agg_df["is_intersecting"], feature].dropna().values
        y = agg_df.loc[~agg_df["is_intersecting"], feature].dropna().values

        if len(x) < min_group_n or len(y) < min_group_n:
            rows.append({
                "category": "overall",
                "cutoff": c,
                "n_ids": agg_df["ID"].nunique(),
                "n_intersecting_ids": len(x),
                "n_non_intersecting_ids": len(y),
                "mannwhitney_stat": np.nan,
                "mannwhitney_p": np.nan,
                "ttest_stat": np.nan,
                "ttest_p": np.nan,
                "cliffs_delta": cliffs_delta(x, y),
                "status": "insufficient_group_size"
            })
            continue

        # run both tests via existing helper
        res_mw = compare_intersecting_vs_nonintersecting(
            feature_df=f,
            intersecting_ids=inter_ids,
            feature_cols=[feature],
            id_col="ID",
            agg=agg,
            test="mannwhitney",
            alpha=0.05
        ).iloc[0]

        res_tt = compare_intersecting_vs_nonintersecting(
            feature_df=f,
            intersecting_ids=inter_ids,
            feature_cols=[feature],
            id_col="ID",
            agg=agg,
            test="ttest",
            alpha=0.05
        ).iloc[0]

        rows.append({
            "category": "overall",
            "cutoff": c,
            "n_ids": agg_df["ID"].nunique(),
            "n_intersecting_ids": int(res_mw["n_intersecting"]),
            "n_non_intersecting_ids": int(res_mw["n_non_intersecting"]),
            "mean_intersecting": res_mw["mean_intersecting"],
            "mean_non_intersecting": res_mw["mean_non_intersecting"],
            "median_intersecting": res_mw["median_intersecting"],
            "median_non_intersecting": res_mw["median_non_intersecting"],
            "cliffs_delta": res_mw["effect_cliffs_delta"],
            "mannwhitney_stat": res_mw["statistic"],
            "mannwhitney_p": res_mw["p_value"],
            "ttest_stat": res_tt["statistic"],
            "ttest_p": res_tt["p_value"],
            "status": "ok"
        })

        if seg_wise:
            for seg in ALL_SEGMENTS:#f["Segment"].unique():
                seg_df = f[f["Segment"] == seg]
                seg_intersecting_ids = [id_ for id_ in inter_ids if id_ in seg_df["ID"].values]
                #if len(seg_intersecting_ids) < min_group_n:
                    #print(f"Segment {seg} has only {len(seg_intersecting_ids)} intersecting IDs at cutoff {c}; skipping statistical test.")
                    #continue
                # run both tests via existing helper
                if seg_df.empty:
                    rows.append({
                        "category": f"Segment {seg}",
                        "cutoff": c,
                        "n_ids": 0,
                        "n_intersecting_ids": 0,
                        "n_non_intersecting_ids": 0,
                        "mannwhitney_stat": np.nan,
                        "mannwhitney_p": np.nan,
                        "ttest_stat": np.nan,
                        "ttest_p": np.nan,
                        "cliffs_delta": np.nan,
                        "status": "no_data_after_cutoff"
                    })
                    continue
                if len(seg_intersecting_ids) <= 1:
                    rows.append({
                        "category": f"Segment {seg}",
                        "cutoff": c,
                        "n_ids": seg_df["ID"].nunique(),
                        "n_intersecting_ids": len(seg_intersecting_ids),
                        "n_non_intersecting_ids": seg_df["ID"].nunique() - len(seg_intersecting_ids),
                        "mannwhitney_stat": np.nan,
                        "mannwhitney_p": np.nan,
                        "ttest_stat": np.nan,
                        "ttest_p": np.nan,
                        "cliffs_delta": np.nan,
                        "status": "no_intersecting_ids"
                    })
                    continue
                res_mw = compare_intersecting_vs_nonintersecting(
                    feature_df=seg_df,
                    intersecting_ids=seg_intersecting_ids,
                    feature_cols=[feature],
                    id_col="ID",
                    agg=agg,
                    test="mannwhitney",
                    alpha=0.05
                ).iloc[0]

                res_tt = compare_intersecting_vs_nonintersecting(
                    feature_df=seg_df,
                    intersecting_ids=seg_intersecting_ids,
                    feature_cols=[feature],
                    id_col="ID",
                    agg=agg,
                    test="ttest",
                    alpha=0.05
                ).iloc[0]

                rows.append({
                    "category": f"Segment {seg}",
                    "cutoff": c,
                    "n_ids": seg_df["ID"].nunique(),
                    "n_intersecting_ids": int(res_mw["n_intersecting"]),
                    "n_non_intersecting_ids": int(res_mw["n_non_intersecting"]),
                    "mean_intersecting": res_mw["mean_intersecting"],
                    "mean_non_intersecting": res_mw["mean_non_intersecting"],
                    "median_intersecting": res_mw["median_intersecting"],
                    "median_non_intersecting": res_mw["median_non_intersecting"],
                    "cliffs_delta": res_mw["effect_cliffs_delta"],
                    "mannwhitney_stat": res_mw["statistic"],
                    "mannwhitney_p": res_mw["p_value"],
                    "ttest_stat": res_tt["statistic"],
                    "ttest_p": res_tt["p_value"],
                    "status": "ok"
                })

    out = pd.DataFrame(rows).sort_values("cutoff").reset_index(drop=True)

    # BH correction across cutoffs
    if "mannwhitney_p" in out.columns:
        out["mannwhitney_p_adj_bh"] = benjamini_hochberg(out["mannwhitney_p"].fillna(1.0).values)
    if "ttest_p" in out.columns:
        out["ttest_p_adj_bh"] = benjamini_hochberg(out["ttest_p"].fillna(1.0).values)

    return out

def execute_numeric_feature_cutoff_pipeline_per_strain(cutoff_grid=[0,5,10,15], features_cols=FEATURES_TO_USE):
    logging.info("Starting numeric feature pipeline...")
    significant_combinations = []
    significant_results = {}
    for strain in strain_to_pubs.keys():
        logging.info(f"Running feature cutoff pipeline for strain: {strain}")
        cutoff_grid = [0, 5, 10, 15]
        #feature_name = "Direct_repeat"#"length_proportion"  # change to any numeric feature in feature table
        for feature_name in features_cols:
            #logging.info(f"Processing feature: {feature_name}")
            feature_cutoff_results = run_feature_cutoff_pipeline(
                feature=feature_name,
                strain=strain,
                cutoffs=cutoff_grid,
                min_group_n=10,
                seg_wise=True)
            if feature_cutoff_results.empty:
                logging.info(f"No results for feature '{feature_name}' in strain '{strain}'; skipping save.")
                continue
            if not feature_cutoff_results["status"].eq("ok").any():
                logging.info(f"All cutoffs for feature '{feature_name}' in strain '{strain}' had issues; skipping save.")
                continue
            if len(feature_cutoff_results[feature_cutoff_results["mannwhitney_p_adj_bh"]<0.05]) >= 2:
                #logging.info(f"Significant results for feature '{feature_name}' in strain '{strain}':")
                #logging.info(feature_cutoff_results[feature_cutoff_results["mannwhitney_p_adj_bh"]<0.05])
                significant_combinations.append((strain, feature_name, feature_cutoff_results[feature_cutoff_results["mannwhitney_p_adj_bh"]<0.05].shape[0]))
                significant_results[(strain, feature_name)] = feature_cutoff_results[feature_cutoff_results["mannwhitney_p_adj_bh"]<0.05]
            try:
                os.makedirs(RESULT_PATH, exist_ok=True)
                feature_cutoff_results.to_csv(os.path.join(RESULT_PATH, strain, f"{feature_name}_precutoff_analysis.csv"), index=False)
            except Exception as e:
                logging.info(f"Error saving results for feature '{feature_name}': {e}")
    logging.info("Significant strain-feature combinations (Mann-Whitney U test, BH-adjusted p < 0.05 in at least 2 cutoffs):")
    logging.info(significant_combinations)
    logging.info("Significant results per combination:")
    for combo in significant_combinations:
        strain, feature_name, n_significant_cutoffs = combo
        logging.info(f"Strain: {strain}, Feature: {feature_name}, Significant Cutoffs: {n_significant_cutoffs}")
        logging.info(significant_results[(strain, feature_name)])



########    Segment-wise boxplots for each feature    ########
def prepare_intersection_plot_data(feature_df, intersecting_ids, feature_cols=None, id_col="ID", agg="median"):
    '''
    Aggregate features per ID and add an intersecting/non-intersecting label.
    '''
    if feature_cols is None:
        feature_cols = FEATURES_TO_USE

    # Reuse existing helper if available
    if "aggregate_id_features" in globals():
        agg_df, numeric_cols = aggregate_id_features(
            feature_df=feature_df,
            feature_cols=feature_cols,
            id_col=id_col,
            agg=agg
        )
    else:
        numeric_cols = [c for c in feature_cols if c in feature_df.columns and pd.api.types.is_numeric_dtype(feature_df[c])]
        agg_df = feature_df.groupby(id_col, as_index=False)[numeric_cols].agg(agg)

    agg_df["Group"] = np.where(agg_df[id_col].isin(set(intersecting_ids)), "Intersecting", "Non-intersecting")
    return agg_df, numeric_cols

def plot_feature_boxplots(plot_df, features, n_cols=3, figsize_per_panel=(4.2, 3.8), suptitle="", title_prefix="", path=""):
    '''
    Boxplots per feature: intersecting vs non-intersecting.
    '''
    features = [f for f in features if f in plot_df.columns and pd.api.types.is_numeric_dtype(plot_df[f])]
    if not features:
        print("No plottable numeric features found.")
        return

    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False
    )
    axes = axes.ravel()

    for i, feat in enumerate(features):
        ax = axes[i]
        x1 = plot_df.loc[plot_df["Group"] == "Intersecting", feat].dropna().values
        x0 = plot_df.loc[plot_df["Group"] == "Non-intersecting", feat].dropna().values

        ax.boxplot([x1, x0], tick_labels=[f"Intersecting\n{len(x1)}", f"Non-intersecting\n{len(x0)}"], showfliers=False)
        ax.set_title(f"{title_prefix}{feat}")
        ax.tick_params(axis="x", rotation=20)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    if path != "":
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f"{title_prefix.strip()}_boxplot.png"), dpi=300)
    else:
        plt.show()
    plt.close()

def plot_feature_histograms(plot_df, features, bins=40, n_cols=3, figsize_per_panel=(4.2, 3.8), suptitle="", title_prefix="", path=""):
    '''
    Overlaid density histograms per feature.
    '''
    features = [f for f in features if f in plot_df.columns and pd.api.types.is_numeric_dtype(plot_df[f])]
    if not features:
        print("No plottable numeric features found.")
        return

    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False
    )
    axes = axes.ravel()

    for i, feat in enumerate(features):
        ax = axes[i]
        x1 = plot_df.loc[plot_df["Group"] == "Intersecting", feat].dropna().values
        x0 = plot_df.loc[plot_df["Group"] == "Non-intersecting", feat].dropna().values

        ax.hist(x0, bins=bins, alpha=0.55, density=True, label=f"Non-intersecting (n={len(x0)})")
        ax.hist(x1, bins=bins, alpha=0.55, density=True, label=f"Intersecting (n={len(x1)})")
        ax.set_title(f"{title_prefix}{feat}")
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    if path != "":
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f"{title_prefix.strip()}_histogram.png"), dpi=300)
    else:
        plt.show()
    plt.close()

def get_boxplot_ax(df, feature, ax=None, title=None, y_label=None):
    ax = sns.boxplot(x="Group", y=feature, data=df, ax=ax, showfliers=False, order=["Intersecting", "Non-intersecting"])
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel(y_label)
    return ax

def plot_segment_boxplots(plot_df, intersecting_ids, feature, n_cols=3, y_label=None, figsize_per_panel=(4, 3), segments=ALL_SEGMENTS, suptitle="", title_prefix="", path="", **kwargs):
    '''
    Boxplots per segment, comparing intersecting vs non-intersecting IDs for each feature.
    '''
    sns.set_theme(style="darkgrid", context="notebook", palette="colorblind")
    n_rows = int(np.ceil(len(segments) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False
    )
    axes = axes.ravel()
    if y_label is None:
        y_label = feature
    for i, seg in enumerate(segments):
        if seg == "All":
            seg_df = plot_df
        else:
            seg_df = plot_df[plot_df["Segment"] == seg]
        axes[i] = get_boxplot_ax(seg_df, feature, ax=axes[i], title=f"{title_prefix}{seg}", y_label=y_label)
        #sns.boxplot(x="Group", y=feature, data=seg_df, ax=ax, showfliers=False, order=["Intersecting", "Non-intersecting"])
        #ax.set_title(f"{title_prefix}{seg}")
        #ax.tick_params(axis="x", rotation=0)
        #ax.set_xlabel("")
        #ax.set_ylabel(y_label)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    if path != "":
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f"{feature.strip()}_segwise_boxplot.png"), dpi=300)
    else:
        plt.show()
    plt.close()

def get_y_label(feature):
    '''
    Returns a more descriptive y-axis label for known features.
    '''
    mapping = {
        "Start": "Start Position",
        "End": "End Position",
        "Direct_repeat": "Direct Repeat Length",
        "remaining_length": "Remaining Length",
        "deletion_length": "Deletion Length",
        "3_5_diff": "3' - 5' Length Difference",
        "length_proportion": "Deletion Length / Full Length",
        "Peptide_Length": "Peptide Length",
        "3_len": "3' Length",
        "5_len": "5' Length"
    }
    return mapping.get(feature, feature)

def plot_segwise_distributions(feature_data, intersecting_ids, feature_cols, segments=ALL_SEGMENTS, add_all=False, path="", **kwargs):
    seg_wise_results = {}
    seg_wise_plots = {}
    if add_all:
        seg_intersecting_ids = [id_ for id_ in intersecting_ids if id_ in feature_data["ID"].values]
        test_results = compare_intersecting_vs_nonintersecting(
                feature_df=feature_data,
                intersecting_ids=seg_intersecting_ids,
                feature_cols=feature_cols,   # includes non-numeric cols; function filters automatically
                id_col="ID",
                agg="median",
                test="mannwhitney",
                alpha=0.05)
        seg_wise_results["All"] = test_results
        plot_df, numeric_cols = prepare_intersection_plot_data(
            feature_df=feature_data,
            intersecting_ids=seg_intersecting_ids,
            feature_cols=feature_cols,   # already available
            id_col="ID",
            agg="median"
        )
        seg_wise_plots["All"] = plot_df
    for seg in segments:
        seg_df = feature_data[feature_data["Segment"] == seg]
        seg_intersecting_ids = [id_ for id_ in intersecting_ids if id_ in seg_df["ID"].values]
        #if len(seg_intersecting_ids) < 3:
        #    print(f"Segment {seg} has only {len(seg_intersecting_ids)} intersecting IDs; skipping statistical test.")
        #    continue
        test_results = compare_intersecting_vs_nonintersecting(
                feature_df=seg_df,
                intersecting_ids=seg_intersecting_ids,
                feature_cols=feature_cols,   # includes non-numeric cols; function filters automatically
                id_col="ID",
                agg="median",
                test="mannwhitney",
                alpha=0.05)
        seg_wise_results[seg] = test_results
        plot_df, numeric_cols = prepare_intersection_plot_data(
            feature_df=seg_df,
            intersecting_ids=seg_intersecting_ids,
            feature_cols=feature_cols,   # already available
            id_col="ID",
            agg="median"
        )
        
        seg_wise_plots[seg] = plot_df
    
    for feature in feature_cols:
        logging.info(f"Feature: {feature}")
        if feature not in feature_data.columns or not pd.api.types.is_numeric_dtype(feature_data[feature]):
            logging.info(f"  - Skipping non-numeric or missing feature '{feature}'")
            continue
        for seg, res in seg_wise_results.items():
            if seg == "All":
                if f'{feature}_normalized' in feature_data.columns:
                    cur_feature = f'{feature}_normalized'
                else:
                    cur_feature = feature
            if cur_feature in res["feature"].values:
                row = res[res["feature"] == cur_feature].iloc[0]
                logging.info(f"  - Segment {seg}: p_adj_bh={row['p_adj_bh']:.4f}, significant={row['significant']}, cliffs_delta={row['effect_cliffs_delta']:.3f}")
            else:
                logging.info(f"  - Segment {seg}: feature not found in results")
        # generate plot_df for this feature across segments
        plot_df = pd.concat([seg_wise_plots[seg][["ID", "Group", feature if seg!="All" else cur_feature]].assign(Segment=seg) for seg in seg_wise_plots], ignore_index=True)
        #display(plot_df)
        plot_segment_boxplots(plot_df, intersecting_ids, feature, y_label=get_y_label(feature), segments=["All"]+segments if add_all else segments, path=path, **kwargs)
        #break  # remove this break to plot all features; currently just plotting the first one for demonstration
        
def execute_boxplotting_per_strain(cutoff_grid=[0, 5, 10, 15], segments=ALL_SEGMENTS, add_all=True, **kwargs):
    logging.info("Making boxplots for numeric features...")
    for strain in strain_to_pubs.keys():
        logging.info(f"Processing strain: {strain}")
        data = load_data(strain_to_pubs[strain], unpooled=True)
        data = data[data["Strain"] == strain]
        data = identify_candidates(data)
        id_group_counts = data.groupby('ID')['Publication'].nunique()
        intersecting_ids = id_group_counts[id_group_counts >= max(2, data["Publication"].nunique() / 2)].index.tolist()
        for cutoff in cutoff_grid:
            data = cutoff_clean(data, cutoff, minimum_dataset_size=0).drop_duplicates("ID").reset_index(drop=True)
            remaining_intersecting_ids = [id_ for id_ in intersecting_ids if id_ in data["ID"].unique()]
            feature_data = calculate_features(data.drop_duplicates("ID").reset_index(drop=True), standard_features=FEATURES_TO_USE, inplace=False, scale="none", normalize_by_length=False)
            if add_all: # normalize features by sequence length for better cross-segment comparison
                feature_data["Start_normalized"] = feature_data["Start"] / feature_data["Full_Sequence"].str.len()
                feature_data["End_normalized"] = feature_data["End"] / feature_data["Full_Sequence"].str.len()
                for col in ["remaining_length", "deletion_length", "Peptide_Length", "3_len", "5_len"]:
                    if col in feature_data.columns:
                        feature_data[f"{col}_normalized"] = feature_data[col] / feature_data["Full_Sequence"].str.len()
                feature_data["3_5_diff_normalized"] = feature_data["3_len_normalized"] - feature_data["5_len_normalized"]
            features_cols = [col for col in feature_data.columns if col not in data.columns or col in FEATURES_TO_USE]
            logging.info(f'Processing {strain} with cutoff {cutoff}: {len(feature_data)} total IDs, {len(remaining_intersecting_ids)} intersecting IDs')
            strain_path = os.path.join(RESULT_PATH, strain, f'Cutoff {cutoff}', "segwise_feature_distributions")
            plot_segwise_distributions(feature_data, remaining_intersecting_ids, features_cols, path=strain_path, segments=segments, add_all=add_all, **kwargs)


########   Functions for nucleotide enrichment analysis and plotting    ########
def create_nucleotide_ratio_matrix(df: pd.DataFrame, col: str)-> pd.DataFrame:
    '''
        Counts nucleotides around the deletion site. Used to create heatmaps.
        :param df: Pandas DataFrame that was created using sequence_df()
        :param col: column name which sequence to use

        :return: Pandas DataFrame with probabilites for the nucleotides
    '''
    probability_matrix = pd.DataFrame(columns=NUCLEOTIDES.keys())
    seq_matrix = df.filter([col], axis=1)
    seq_matrix = seq_matrix[col].str.split("", expand=True)
    # drop first and last column
    seq_matrix = seq_matrix.drop([0, len(seq_matrix.columns)-1], axis=1)
    
    for n in NUCLEOTIDES.keys():
        probability_matrix[n] = seq_matrix.apply(lambda x: dict(x.value_counts()).get(n,0)/len(x), axis=0)

    return probability_matrix

def plot_heatmap(y: list, x: list, vals: list, ax: object,
                 format=".2f", cmap="coolwarm", vmin=0, vmax=1, cbar=False, cbar_ax=None, cbar_kws=None)-> object:
    '''
        Helper function to plot heatmap.
        :param y: columns of heatmap
        :param x: rows of heatmap
        :param vals: values for heatmap
        :param ax: matplotlib.axes object
        :param: additional parameters check sns.heatmap() for more information
        
        :return: generated heatmap on matplotlib.axes object
    '''
    #sns.set_theme(style="darkgrid", context="notebook", palette="colorblind")
    df = pd.DataFrame({"x":x,"y":y,"vals":vals})
    df = pd.pivot_table(df, index="x", columns="y", values="vals", sort=False)
    ax = sns.heatmap(df, fmt=format, annot=True, vmin=vmin, vmax=vmax, ax=ax, cbar=cbar, cmap=cmap, cbar_ax=cbar_ax, cbar_kws=cbar_kws, square=True)
    return ax

def get_eta_squared(H, k, n):
    '''
    Computes the effect size eta squared from the Kruskal-Wallis H statistic.
    '''
    eta = (H - k + 1)/(n - k)
    return eta

def get_epsilon_squared(H, k, n):
    '''
    Computes the effect size epsilon squared from the Kruskal-Wallis H statistic.
    '''
    epsilon = H/((n**2 - 1)/(n+1))
    return epsilon

def get_cliffs_delta(U, n1, n2):
    '''
    Computes Cliff's delta effect size from the Wilcoxon-Mann-Whitney U statistic.
    '''
    delta = (2 * U) / (n1 * n2) - 1
    return delta

def get_r(U, n1, n2):
    '''
    Calculates effect size r from the Wilcoxon-Mann-Whitney U statistic.
    :param U: Mann-Whitney U statistic
    :param n1: number of observations in group 1
    :param n2: number of observations in group 2
    :return: Pearson's r (rank-biserial correlation)
    '''
    mu_U = n1 * n2 / 2
    sigma_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (U - mu_U) / sigma_U
    return z / np.sqrt(n1 + n2)

def get_r2(U, n1, n2):
    '''
    Calculates effect size r squared from the Wilcoxon-Mann-Whitney U statistic.
    '''
    r = get_r(U, n1, n2)
    return r**2

def get_carmers_v(chi2, n, min_dim):
    '''
    Calculates Cramér's V from the Chi-squared statistic.
    :param chi2: Chi-squared statistic from the contingency table
    :param n: Total number of observations
    :return: Cramér's V
    '''
    return np.sqrt(chi2 / (n * (min_dim - 1)))

def _extract_strain_label(df: pd.DataFrame, fallback: str) -> str:
    if "Strain" in df.columns and not df["Strain"].empty:
        return str(df["Strain"].iloc[0])
    return str(fallback)

def _contiguous_groups(labels: list) -> list:
    if not labels:
        return []
    groups = []
    start = 0
    cur_label = labels[0]
    for idx, label in enumerate(labels[1:], start=1):
        if label != cur_label:
            groups.append((cur_label, start, idx - 1))
            start = idx
            cur_label = label
    groups.append((cur_label, start, len(labels) - 1))
    return groups

def _draw_strain_brackets(ax: object, strain_groups: list, x_bracket: float=-0.42, tick_len: float=0.04, text_offset: float=0.08) -> None:
    trans = ax.get_yaxis_transform()
    for strain, start_idx, end_idx in strain_groups:
        y_start = start_idx + 0.5
        y_end = end_idx + 0.5
        y_mid = (y_start + y_end) / 2
        ax.plot([x_bracket, x_bracket], [y_start, y_end], color="black", lw=1.0, transform=trans, clip_on=False)
        ax.plot([x_bracket, x_bracket + tick_len], [y_start, y_start], color="black", lw=1.0, transform=trans, clip_on=False)
        ax.plot([x_bracket, x_bracket + tick_len], [y_end, y_end], color="black", lw=1.0, transform=trans, clip_on=False)
        ax.text(x_bracket - text_offset, y_mid, strain.replace("_", "/"), ha="center", va="center", fontsize=6, rotation=90, transform=trans, clip_on=False)

def get_stat_result(test_array, test_array2, test: str="kruskal"):
    
    match test:
        case "kruskal":
            res = stats.kruskal(test_array, test_array2)
        case "mannwhitney":
            res = stats.mannwhitneyu(test_array, test_array2)
        case "fisherexact":
            #contingency_table = get_contingency_table(test_array, test_array2)
            table = np.array([[len(test_array == 0), len((test_array == 1))],
                              [len(test_array2 == 0), len((test_array2 == 1))]])
            res = stats.fisher_exact(table)
        case "chisquare":
            #contingency_table = get_contingency_table(test_array, test_array2)
            table = np.array([[len(test_array == 0), len((test_array == 1))],
                              [len(test_array2 == 0), len((test_array2 == 1))]])
            res = stats.chi2_contingency(table)
        case _:
            logging.warning(f"Unknown test '{test}'; defaulting to 'kruskal'")
            res = stats.kruskal(test_array, test_array2)
    return res

def get_effect_size(statistic, effect_size: str="eta2", n_samples: int=0, n_samples2: int=0):
    '''
    Returns the effect size based on the specified type. Available effect sizes, depending on the test statistic and sample sizes:
     - eta squared: (H - k + 1) / (n - k)
        - meta squared: (H - k + 1) / (n - k) where n is the number of observations in the first group only (for meta-analysis version)
     - epsilon squared: H / ((n^2 - 1) / (n + 1))
     - r: z / sqrt(n1 + n2) where z is the standardized U statistic
     - r squared: (Pearson's r)^2
     - Cliff's delta: (2U / (n1 * n2)) - 1
    '''
    match effect_size:
        case "eta2":
            return get_eta_squared(statistic, 2, n_samples + n_samples2)
        case "meta2":
            return get_eta_squared(statistic, 2, n_samples)
        case "epsilon2":
            return (statistic - 2 + 1) / (n_samples + n_samples2 - 2 + 1)
        case "r":
            return get_r(statistic, n_samples, n_samples2)
        case "r2":
            return get_r2(statistic, n_samples, n_samples2)
        case "cliff"|"cliffs"|"cliff_delta"|"cliffs_delta"|"delta":
            return get_cliffs_delta(statistic, n_samples, n_samples2)
        case "cramer"|"cramers"|"cramer_v"|"cramers_v"|"v":
            return get_carmers_v(statistic, n_samples + n_samples2, 2)  # assuming 2x2 contingency table for Cramér's V
        case _:
            logging.error(f"Unknown effect size type '{effect_size}'; defaulting to 'eta2'")
            return get_eta_squared(statistic, 2, n_samples)

def get_effect_size_text(val: float|None=None, effect_size: str|None=None, pval: float|None=None, max_pval: float=0.05, **kwargs):#, statistic: float|None=None, n_samples: int=0, n_samples2: int=0):
    '''
    Returns the effect size as a formatted string if the p-value is below the specified threshold and the effect size exceeds commonly used thresholds for small/moderate/large effects. Otherwise, returns an empty string. If no value is provided, the function will attempt to calculate it from the test statistic and sample sizes, assuming the necessary parameters are provided as keyword arguments. Thresholds for a small or greater effect are based on common conventions in the literature:
     - eta squared >= 0.06 (moderate or greater effect)
     - meta squared >= 0.06 (moderate effect or greater) | refers to the version of eta squared used in meta-analysis by Lohmann et al.
     - epsilon squared >= 0.06 (moderate effect or greater)
     - Pearson's r absolute >= 0.10 (small effect or greater)
     - r squared >= 0.06 (moderate effect or greater)
     - Cliff's delta absolute >= 0.15 (small effect or greater)
     - Cramér's V >= 0.10 (small effect or greater)
    '''
    text = ""
    if pval >= max_pval:
        return text
    if val is None:
        val = get_effect_size(statistic=kwargs.get("statistic"),
                              effect_size=effect_size,
                              n_samples=kwargs.get("n_samples"),
                              n_samples2=kwargs.get("n_samples2"))
    if kwargs.get("threshold_overwrite", None) is not None:
        threshold = kwargs["threshold_overwrite"]
        if abs(val) >= threshold:
            text = f"{val:.2f}"
            text = text[1:] if val > 0 else "-" + text[2:]
        return text
    match effect_size:
        case "eta2":
            if val >= 0.06:
                text = f"{val:.2f}"
                text = text[1:]
        case "meta2":
            if val >= 0.06:
                text = f"{val:.2f}"
                text = text[1:]
        case "epsilon2":
            if val >= 0.06:
                text = f"{val:.2f}"
                text = text[1:]
        case "r":
            if abs(val) >= 0.1:
                text = f"{val:.2f}"
                text = text[1:] if val > 0 else "-" + text[2:]
        case "r2":
            if val >= 0.06:
                text = f"{val:.2f}"
                text = text[1:]
        case "cliff"|"cliffs"|"cliff_delta"|"cliffs_delta"|"delta":
            if abs(val) >= 0.1:
                text = f"{val:.2f}"
                text = text[1:] if val > 0 else "-" + text[2:]
        case "cramer"|"cramers"|"cramer_v"|"cramers_v"|"v":
            if val >= 0.1:
                text = f"{val:.2f}"
                text = text[1:]
        case _:
            logging.error(f"Unknown effect size type '{effect_size}'; cannot determine threshold for formatting.")
    return text

def nuc_enrich_stats(test_array, test_array2, test: str="kruskal", effect_size: str="jens", scale: float=1.0, n_samples: int=0, n_samples2: int=0):
    text = ""
    match test:
        case "kruskal":
            res = stats.kruskal(test_array, test_array2)
            if res.count < 0.05:
                match effect_size:
                    case "jens":
                        eta = get_eta_squared(res.statistic, 2, n_samples)
                    case "eta2":
                        eta = get_eta_squared(res.statistic, 2, n_samples + n_samples2)
                    case "epsilon2":
                        eta = (res.statistic - 2 + 1) / (n_samples + n_samples2 - 2 + 1)
                    case _:
                        logging.warning(f"Unknown effect size version '{effect_size}'; defaulting to 'jens'")
                        eta = get_eta_squared(res.statistic, 2, n_samples)
                if eta > 0.06:
                    text = f"{eta:.2f}"
                    text = text[1:]
                else:
                    text = ""
        case "mannwhitney":
            res = stats.mannwhitneyu(test_array, test_array2)
            if res.pvalue < 0.05:
                match effect_size:
                    case "r":
                        r = get_r(res.statistic, n_samples, n_samples2)
                        if abs(r) > 0.15:
                            if scale > 1.0:
                                text = f"{r:.2f}"
                                text = text[1:] if r > 0 else "-" + text[2:]
                            else:
                                text = f"{abs(r):.2f}"
                                text = text[1:]
                        else:
                            text = ""
                    case "r2":
                        r2 = get_r2(res.statistic, n_samples, n_samples2)
                        if abs(r2) > 0.06:
                            text = f"{r2:.2f}"
                            text = text[1:]
                        else:
                            text = ""
                    case "delta":
                        delta = get_cliffs_delta(res.statistic, n_samples, n_samples2)
                        if abs(delta) > 0.14:
                            if scale > 1.0:
                                text = f"{delta:.2f}"
                                text = text[1:] if delta > 0 else "-" + text[2:]
                            else:
                                text = f"{abs(delta):.2f}"
                                text = text[1:]# if delta > 0 else text[2:]
                    case _:
                        logging.warning(f"Unknown effect size version '{effect_size}'; defaulting to 'delta'")
                        delta = get_cliffs_delta(res.statistic, n_samples, n_samples2)
                        if abs(delta) > 0.14:
                            if scale > 1.0:
                                text = f"{delta:.2f}"
                                text = text[1:] if delta > 0 else "-" + text[2:]
                            else:
                                text = f"{abs(delta):.2f}"
                                text = text[1:]# if delta > 0 else text[2:]
    return text

def plot_expected_vs_observed_nucleotide_enrichment_heatmaps(dfs: list, dfnames: list, expected_dfs: list, compared: str, test: str="kruskal", name_prefix: str="", path: str="", effect_size: str="eta2", scale: float=1.0, show_strain_brackets: bool=False)-> None:
    '''
        plot difference of expected vs observed nucleotide enrichment around
        deletion junctions as heatmap.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        :param expected_dfs: The list of DataFrames containing the expected
            data, preprocessed with sequence_df(df)
        :param compared: defines in title what data is compared
        :param test: defines the statistical test to use
        :param name: defines the name of the plot
        :param path: defines where to save the results
        :param version: defines the version of the analysis
        :param scale: defines the scaling factor of the plot
        :param show_strain_brackets: if True, draw strain grouping brackets left of y labels
        :param correct_p: if True, correct p-values for multiple comparisons (currently only implemented with Benjamini-Hochberg correction)
        :return: None
    '''
    width = 10*scale
    height = (1.2+7/20*len(dfnames))*scale
    sns.set_theme(style="darkgrid", context="notebook", palette="colorblind")
    fig, axs = plt.subplots(figsize=(width, height), nrows=2, ncols=2)
    axs = axs.flatten()
    #res_display = display("Statistical test results:", display_id=True)
    strain_groups = []
    if show_strain_brackets:
        strain_labels = [_extract_strain_label(df, dfname) for dfname, df in zip(dfnames, dfs)]
        strain_groups = _contiguous_groups(strain_labels)
    for i, nuc in enumerate(sorted(NUCLEOTIDES.keys())):
        x = list()
        y = list()
        vals = list()
        val_labels = list()
        for dfname, df, expected_df in zip(dfnames, dfs, expected_dfs):
            df = df.reset_index()
            probability_matrix = create_nucleotide_ratio_matrix(df, "junction_window")
            n_samples = len(df)
            expected_probability_matrix = create_nucleotide_ratio_matrix(expected_df, "junction_window")
            n_samples2 = len(expected_df)
            for j in probability_matrix.index:
                x.append(j)
                y.append(dfname)

                p1 = probability_matrix.loc[j,nuc]
                p2 = expected_probability_matrix.loc[j,nuc]
                vals.append(p1 - p2)

                test_array = np.concatenate((np.ones(int(n_samples * p1)), np.zeros(int(n_samples - n_samples * p1))))
                test_array2 = np.concatenate((np.ones(int(n_samples2 * p2)), np.zeros(int(n_samples2 - n_samples2 * p2))))

                stat_res = get_stat_result(test_array, test_array2, test=test)
                effect_size_res = get_effect_size(statistic=stat_res.statistic, effect_size=effect_size, n_samples=n_samples, n_samples2=n_samples2)
                text = get_effect_size_text(val=effect_size_res, effect_size=effect_size, pval=stat_res.pvalue, threshold_overwrite=0)# if effect_size=="delta" else None)
                val_labels.append(text)

        if len(vals) != 0:        
            m = abs(min(vals)) if abs(min(vals)) > max(vals) else max(vals)
        else:
            m = 0
        axs[i] = plot_heatmap(x,y,vals, axs[i], format=".1e", cbar=True, vmin=-m, vmax=m, cbar_kws={"pad": 0.01})
        thres = 0.2 if i in [0, 2] else 0.15
        for v_idx, val_label in enumerate(axs[i].texts):
            val_label.set_text(val_labels[v_idx])
            val_label.set_size(9)
            if abs(vals[v_idx]) > abs(thres):
                val_label.set_color("white")
            else:
                val_label.set_color("black")
        axs[i].set_title(f"{NUCLEOTIDES[nuc]}")
        axs[i].set_ylabel("")
        axs[i].set_yticks([ytick + 0.5 for ytick in range(len(dfnames))])
        axs[i].set_xlabel("")  
        axs[i].set_xticks([xtick - 0.5 for xtick in probability_matrix.index])
        
        quarter = len(probability_matrix.index) // 4
        indexes = [pos for pos in range(1, quarter * 2 + 1)]
        if i % 2 == 0:
            if show_strain_brackets:
                axs[i].set_yticklabels([f"{dfname.split('_')[0]} ({len(df)})" for dfname,df in zip(dfnames,dfs)])#, fontsize=8)
            else:
                axs[i].set_yticklabels([f"{dfname} ({len(df)})" for dfname,df in zip(dfnames,dfs)])#, fontsize=8)
            axs[i].tick_params(axis="y", pad=8)
            if show_strain_brackets:
                _draw_strain_brackets(axs[i], strain_groups, text_offset=0.02)
        else:
            axs[i].set_yticklabels([])
        if i < 2:
            axs[i].xaxis.set_ticks_position("top")
            axs[i].xaxis.set_label_position("top")
        axs[i].tick_params(left=False, top=False, bottom=False)
        axs[i].set_xticklabels(indexes + indexes, rotation=0)
        xlabels = axs[i].get_xticklabels()
        for x_idx, xlabel in enumerate(xlabels):
            if x_idx < quarter or x_idx >= quarter * 3:
                xlabel.set_color("black")
                xlabel.set_fontweight("bold")
            else:
                xlabel.set_color("grey")   

    #fig.suptitle("Enriched (red) and depleted (blue) nucleotides")
    strain_labels = set([_extract_strain_label(df, dfname) for dfname, df in zip(dfnames, dfs)])
    fig.suptitle(f"{list(strain_labels)[0].replace('_', '/')} - Nucleotide Enrichment\n({compared.replace("_"," ")})")
    #ax.set_title(f'({compared})')
    if show_strain_brackets:
        fig.tight_layout(rect=[0.26, 0.0, 1.0, 0.96])
    else:
        fig.tight_layout(rect=[0.06, 0.0, 1.0, 0.96])
    if path != "":
        #save_path = os.path.join(path, f"nucleotide_enrichment_{compared}")
        #if not os.path.exists(save_path):
        #    os.makedirs(save_path)
        #plt.savefig(os.path.join(save_path, f"nuc_occ_diff.png"), dpi=300)
        plt.savefig(os.path.join(path, f"{name_prefix}_ds_nuc_occ_diff.png"), dpi=300)
    else:
        plt.show()
    plt.close()

def prepare_nucleotide_enrichment_heatmap_data(strain, cutoff, pub_overwrite=None, own_synthetic=False, force_recreate=False):
    if pub_overwrite is not None:
        pubs = pub_overwrite
    else:
        pubs = strain_to_pubs.get(strain, [])
    if not pubs:
        raise ValueError(f"No publications found for strain '{strain}' in strain_to_pubs.")
    dfs = []
    dfnames = []
    expected_dfs = []
    for pub in pubs:
        experimental_data = load_data([pub], unpooled=False)
        experimental_data = cutoff_clean(experimental_data, threshold=cutoff, minimum_dataset_size=40)
        if experimental_data.empty:
            print(f"Warning: No data left after cutoff cleaning for publication '{pub}' and strain '{strain}'. Skipping.")
            continue
        experimental_data = experimental_data[experimental_data["Strain"]==strain].reset_index(drop=True)
        experimental_data = identify_candidates(experimental_data).drop_duplicates("ID").reset_index(drop=True)
        experimental_data = get_sequence_quicker(experimental_data)
        experimental_data["junction_window"] = experimental_data.apply(lambda row: _extract_junction_window(row["Full_Sequence"], row["Start"], row["End"]), axis=1).transform(lambda x: x.replace("|", ""))
        dfs.append(experimental_data)
        dfnames.append(pub)
        if own_synthetic:
            synthetic_data = generate_expected_data(strain, experimental_data, cutoff=cutoff, seg_sample_size=10000, force_recreate=force_recreate)
        else:
            synthetic_data = load_synthetic([pub])
            synthetic_data = synthetic_data[synthetic_data["Strain"]==strain].reset_index(drop=True)
        synthetic_data = identify_candidates(synthetic_data)
        synthetic_data = get_sequence_quicker(synthetic_data)
        synthetic_data["junction_window"] = synthetic_data.apply(lambda row: _extract_junction_window(row["Full_Sequence"], row["Start"], row["End"]), axis=1).transform(lambda x: x.replace("|", ""))
        expected_dfs.append(synthetic_data)
    return dfs, dfnames, expected_dfs

def compare_jens_vs_me():    
    dfs, dfnames, expected_dfs = prepare_nucleotide_enrichment_heatmap_data("A_PuertoRico_8_1934", cutoff=15, own_synthetic=True)
    #dfs, dfnames, expected_dfs = prepare_nucleotide_enrichment_heatmap_data("A_PuertoRico_8_1934", cutoff=15)
    plot_expected_vs_observed_nucleotide_enrichment_heatmaps(dfs, dfnames, expected_dfs, compared="synthetic_vs_experimental", path="", version="jens")
    plot_expected_vs_observed_nucleotide_enrichment_heatmaps(dfs, dfnames, expected_dfs, compared="synthetic_vs_experimental", path="", version="me")
    plot_expected_vs_observed_nucleotide_enrichment_heatmaps(dfs, dfnames, expected_dfs, compared="synthetic_vs_experimental", path="", test="mannwhitney", version="r")
    plot_expected_vs_observed_nucleotide_enrichment_heatmaps(dfs, dfnames, expected_dfs, compared="synthetic_vs_experimental", path="", test="mannwhitney", version="r2")
    plot_expected_vs_observed_nucleotide_enrichment_heatmaps(dfs, dfnames, expected_dfs, compared="synthetic_vs_experimental", path="", test="mannwhitney", version="delta")

def prepare_inter_nucleotide_enrichment_heatmap_data(strain, cutoff, pub_overwrite=None, own_synthetic=False, force_recreate=False, **kwargs):
    if pub_overwrite is not None:
        pubs = pub_overwrite
    else:
        pubs = strain_to_pubs.get(strain, [])
    if not pubs:
        logging.warning(f"No publications found for strain '{strain}' in strain_to_pubs. Using all publications as fallback.")
        pubs = ALL_PUBS  # fallback to all publications if none found for the strain
    logging.info(kwargs)
    non_intersecting_dfs = []
    dfnames = []
    intersecting_dfs = []
    synthetic_dfs = []
    df = load_data(pubs, unpooled=False)
    df = df[df["Strain"]==strain].reset_index(drop=True)
    df = identify_candidates(df)
    intersecting_ids = df.groupby('ID')['Publication'].nunique()
    intersecting_ids = intersecting_ids[intersecting_ids >= max(2, df["Publication"].nunique() / 2)].index.tolist()
    if kwargs.get("only_cleaned_intersecting", True): # get only intersecting DVG that survive the cutoff
        df = cutoff_clean(df, threshold=cutoff, minimum_dataset_size=0)
    else: # get all intersecting DVG in pub, independent of cutoff
        df = cutoff_clean(df, threshold=cutoff, minimum_dataset_size=0, left_out_ids=intersecting_ids)
    df = get_sequence_quicker(df)
    df["junction_window"] = df.apply(lambda row: _extract_junction_window(row["Full_Sequence"], row["Start"], row["End"]), axis=1).transform(lambda x: x.replace("|", ""))
    logging.info(f'Processing {strain} with cutoff {cutoff}: {len(df)} total IDs, {len(intersecting_ids)} intersecting IDs, {df[df["ID"].isin(intersecting_ids)].drop_duplicates("ID").shape[0]} intersecting IDs after cutoff cleaning')
    if kwargs.get("only_cleaned_intersecting", False):
        df = cutoff_clean(df, threshold=cutoff, minimum_dataset_size=40).reset_index(drop=True)
    else:
        df = cutoff_clean(df, threshold=cutoff, minimum_dataset_size=40, left_out_ids=intersecting_ids).reset_index(drop=True)
    for pub in pubs:
        experimental_data = df[df["Publication"] == pub].drop_duplicates("ID").reset_index(drop=True)
        if experimental_data.empty:
            logging.warning(f"Warning: No data left after cutoff cleaning for publication '{pub}' and strain '{strain}'. Skipping.")
            continue
        if own_synthetic:
            synthetic_data = generate_expected_data(strain, experimental_data, cutoff=cutoff, seg_sample_size=kwargs.get("seg_sample_size", 35000), force_recreate=force_recreate)
        else:
            synthetic_data = load_synthetic([pub])
            synthetic_data = synthetic_data[synthetic_data["Strain"]==strain].reset_index(drop=True)
        synthetic_data = identify_candidates(synthetic_data)
        synthetic_data = get_sequence_quicker(synthetic_data)
        synthetic_data["junction_window"] = synthetic_data.apply(lambda row: _extract_junction_window(row["Full_Sequence"], row["Start"], row["End"]), axis=1).transform(lambda x: x.replace("|", ""))
        synthetic_dfs.append(synthetic_data)
        non_intersecting_dfs.append(experimental_data[~experimental_data["ID"].isin(intersecting_ids)].reset_index(drop=True))
        dfnames.append(get_datasetname(strain, pub))
        #if kwargs.get("only_cleaned_intersecting", True): # get only intesecting DVG that survive cutoff
        intersecting_dfs.append(experimental_data[experimental_data["ID"].isin(intersecting_ids)].reset_index(drop=True))
    return non_intersecting_dfs, dfnames, intersecting_dfs, synthetic_dfs

def get_datasetname(strain, pub):
    if pub in ["Alnaji2021", "Pelz2021", "Wang2023", "Wang2020", "Zhuravlev2020", "Mendes2021", "Boussier2020", "Penn2022", "Lui2019", "Sheng2018", "Southgate2019", "Kupke2020", "VdHoecke2015"]:
        return pub
    strain_suffixes = {"A_California_07_2009": "_Cal07", "A_NewCaledonia_20-JY2_1999": "_NC", "A_Perth_16_2009": "_Perth", "A_Connecticut_Flu122_2013": "_A", "B_Lee_1940": "_BLEE"}
    suffix = strain_suffixes.get(strain, "")
    if suffix == "":
        match pub:
            case "Valesano2020":
                suffix = "_Yam" if "Yamagata" in strain else "_Vic"
            case "Berry2021":
                suffix = "_B_Yam" if "Yamagata" in strain else "_B"
            case _:
                logging.warning(f"Unknown publication {pub} for strain {strain}.")
    return f"{pub}{suffix}"

def prepare_multi_strain_for_nucleotide_enrichment_heatmaps(strains=ALL_STRAINS, cutoff=15, own_synthetic=False):
    non_intersecting_dfs, dfnames, intersecting_dfs, synthetic_dfs = [], [], [], []
    for strain in strains:
        try:
            strain_non_intersecting_df, strain_dfnames, strain_intersecting_df, strain_synthetic_df = prepare_inter_nucleotide_enrichment_heatmap_data(strain, cutoff, own_synthetic=own_synthetic)
            non_intersecting_dfs = non_intersecting_dfs + strain_non_intersecting_df
            dfnames = dfnames + strain_dfnames
            intersecting_dfs = intersecting_dfs + strain_intersecting_df
            synthetic_dfs = synthetic_dfs + strain_synthetic_df
        except Exception as e:
            logging.error(f"Error processing strain '{strain}': {e}")
    return non_intersecting_dfs, dfnames, intersecting_dfs, synthetic_dfs

def per_dataset_nucleotide_plots(non_intersecting_dfs, dfnames, intersecting_dfs, synthetic_dfs, strain_path="", test_combs=None, debug=False, cutoff=15, show_strain_brackets: bool=False, scale: float=1.0):
    if test_combs is None or debug:
        test_combs = [("mannwhitney", "delta")]
    for test, effect_size in test_combs:
        logging.info(f"Plotting for test '{test}' and version '{effect_size}'")
        prefix = f'{test}_{effect_size}_'
        if effect_size == "jens":
            effect_size = "meta2"
        try:
            plot_expected_vs_observed_nucleotide_enrichment_heatmaps([pd.concat([non,inter], ignore_index=True) for non, inter in zip(non_intersecting_dfs, intersecting_dfs)], dfnames, synthetic_dfs, compared="observed_vs_synthetic", name_prefix=f"{prefix}_observed_vs_synthetic_", path=strain_path, effect_size=effect_size, test=test, show_strain_brackets=show_strain_brackets, scale=scale)
        except Exception as e:
            logging.error(f"Error plotting observed vs synthetic for test '{test}' and version '{effect_size}':\n{traceback.format_exc()}")
        if debug:
            break

        try:
            plot_expected_vs_observed_nucleotide_enrichment_heatmaps(non_intersecting_dfs, dfnames, synthetic_dfs, compared="non-intersecting_vs_synthetic", name_prefix=f"{prefix}_non_intersecting_vs_synthetic_", path=strain_path, effect_size=effect_size, test=test, show_strain_brackets=show_strain_brackets, scale=scale)
        except Exception as e:
            logging.error(f"Error plotting non-intersecting vs synthetic for test '{test}' and version '{effect_size}':\n{traceback.format_exc()}")

        try:
            plot_expected_vs_observed_nucleotide_enrichment_heatmaps(intersecting_dfs, dfnames, non_intersecting_dfs, compared="intersecting_vs_non-intersecting", name_prefix=f"{prefix}_intersecting_vs_non_intersecting_", path=strain_path, effect_size=effect_size, test=test, show_strain_brackets=show_strain_brackets, scale=scale)
        except Exception as e:
            logging.error(f"Error plotting intersecting vs non-intersecting for test '{test}' and version '{effect_size}':\n{traceback.format_exc()}")

        try:
            plot_expected_vs_observed_nucleotide_enrichment_heatmaps(intersecting_dfs, dfnames, synthetic_dfs, compared="intersecting_vs_synthetic", name_prefix=f"{prefix}_intersecting_vs_synthetic_", path=strain_path, effect_size=effect_size, test=test, show_strain_brackets=show_strain_brackets, scale=scale)
        except Exception as e:
            logging.error(f"Error plotting intersecting vs synthetic for test '{test}' and version '{effect_size}':\n{traceback.format_exc()}")

def single_strain_nucleotide_plots(non_intersecting_df, intersecting_df, background_df, strain, strain_path="", test_combs=None, scale: float=1.0, correct_p: bool=False, debug=False):
    if debug or test_combs is None:
        test_combs = [("mannwhitney", "delta")]  # Replace with actual test combinations'
    for test, effect_size in test_combs:
        logging.info(f"Plotting for strain '{strain}', test '{test}' and effect size '{effect_size}'")
        prefix = f'{test}_{effect_size}_'
        if effect_size == "jens":
            effect_size = "meta2"
        try:
            plot_strain_nucleotide_enrichment_heatmap(pd.concat([non_intersecting_df, intersecting_df], ignore_index=True), dfname=strain.replace("_","/"), background_df=background_df, compared="observed vs synthetic", test=test, effect_size=effect_size, path=strain_path, scale=scale, name_prefix=f"{prefix}observed_vs_synthetic_", correct_for_multiple_comparisons=correct_p)
        except Exception as e:
            logging.error(f"Error plotting observed vs synthetic for strain '{strain}', test '{test}' and effect size '{effect_size}':\n{traceback.format_exc()}")

        try:
            plot_strain_nucleotide_enrichment_heatmap(non_intersecting_df, dfname=strain.replace("_","/"), background_df=background_df, compared="non-intersecting vs synthetic", test=test, effect_size=effect_size, path=strain_path, scale=scale, name_prefix=f"{prefix}non_intersecting_vs_synthetic_", correct_for_multiple_comparisons=correct_p)
        except Exception as e:
            logging.error(f"Error plotting non-intersecting vs synthetic for strain '{strain}', test '{test}' and effect size '{effect_size}':\n{traceback.format_exc()}")

        try:
            plot_strain_nucleotide_enrichment_heatmap(intersecting_df, dfname=strain.replace("_","/"), background_df=non_intersecting_df, compared="intersecting vs non-intersecting", test=test, effect_size=effect_size, path=strain_path, scale=scale, name_prefix=f"{prefix}intersecting_vs_non_intersecting_", correct_for_multiple_comparisons=correct_p)
        except Exception as e:
            logging.error(f"Error plotting intersecting vs non-intersecting for strain '{strain}', test '{test}' and effect size '{effect_size}':\n{traceback.format_exc()}")

        try:
            plot_strain_nucleotide_enrichment_heatmap(intersecting_df, dfname=strain.replace("_","/"), background_df=background_df, compared="intersecting vs synthetic", test=test, effect_size=effect_size, path=strain_path, scale=scale, name_prefix=f"{prefix}intersecting_vs_synthetic_", correct_for_multiple_comparisons=correct_p)
        except Exception as e:
            logging.error(f"Error plotting intersecting vs synthetic for strain '{strain}', test '{test}' and effect size '{effect_size}':\n{traceback.format_exc()}")

def plot_strain_nucleotide_enrichment_heatmap(df: pd.DataFrame, dfname: str, background_df: pd.DataFrame, compared: str, test: str="kruskal", name_prefix: str="", path: str="", effect_size="eta2", scale: float=1.0, show_strain_brackets: bool=False, correct_for_multiple_comparisons: bool=False)-> None:
    '''
        plot difference of expected vs observed nucleotide enrichment around
        deletion junctions as a single heatmap with nucleotides as rows.
        :param df: The DataFrame containing the data, preprocessed with sequence_df(df)
        :param dfname: The name associated with the DataFrame
        :param background_df: The DataFrame containing the background data, preprocessed with sequence_df(df)
        :param compared: defines in title what data is compared
        :param test: defines the statistical test to use
        :param name_prefix: defines the name prefix of the plot
        :param path: defines where to save the results
        :param effect_size: defines the effect size measure to use
        :param scale: defines the scaling factor of the plot
        :param show_strain_brackets: if True, draw strain grouping brackets left of y labels (not used for single dataset)
        :param correct_for_multiple_comparisons: if True, correct for multiple comparisons
        :return: None
    '''
    width = 14*scale
    height = (3)*scale
    sns.set_theme(style="darkgrid", context="notebook", palette="colorblind")
    fig, ax = plt.subplots(figsize=(width, height))
    #fig = plt.figure(figsize=(width, height))
    #ax = plt.gca()
    #res_display = display("Statistical test results:", display_id=True)
    
    df = df.reset_index()
    probability_matrix = create_nucleotide_ratio_matrix(df, "junction_window")
    n_samples = len(df)
    expected_probability_matrix = create_nucleotide_ratio_matrix(background_df, "junction_window")
    n_samples2 = len(background_df)
    
    x = list()
    y = list()
    vals = list()
    val_labels = list()
    
    if correct_for_multiple_comparisons:
        p_values = []
        statistics = []
    for nuc in NUCLEOTIDES.keys():
        for j in probability_matrix.index:
            x.append(j)
            y.append(nuc)#.append(NUCLEOTIDES[nuc])
            
            p1 = probability_matrix.loc[j, nuc]
            p2 = expected_probability_matrix.loc[j, nuc]
            vals.append(p1 - p2)
            
            test_array = np.concatenate((np.ones(int(n_samples * p1)), np.zeros(int(n_samples - n_samples * p1))))
            test_array2 = np.concatenate((np.ones(int(n_samples2 * p2)), np.zeros(int(n_samples2 - n_samples2 * p2))))
            '''
            if test == "kruskal":
                res = stats.kruskal(test_array, test_array2)
            elif test == "mannwhitney":
                res = stats.mannwhitneyu(test_array, test_array2)
            pval = res.pvalue
            
            # calculate effect size based on version
            if pval < 0.05:
                if test == "kruskal":
                    if version == "jens":
                        eta = get_eta_squared(res.statistic, 2, n_samples)
                    if version == "me":
                        eta = get_eta_squared(res.statistic, 2, n_samples + n_samples2)
                    if eta > 0.06:
                        text = f"{eta:.2f}"
                        text = text[1:]
                        #res_display.update(f"Statistical test results: {dfname} - nucleotide {nuc} position {j} - res: {res}, eta: {eta:.3f}")
                    else:
                        text = ""
                elif test == "mannwhitney":
                    if version == "r":
                        r = get_pearson_r(res.statistic, n_samples, n_samples2)
                        if abs(r) > 0.15:
                            if scale >= 1.0:
                                text = f"{r:.2f}"
                                text = text[1:] if r > 0 else "-" + text[2:]
                            else:
                                text = f"{abs(r):.2f}"
                                text = text[1:]
                            #res_display.update(f"Statistical test results: {dfname} - nucleotide {nuc} position {j} - res: {res}, r: {r:.3f}")
                        else:
                            text = ""
                    if version == "r2":
                        r2 = get_r2(res.statistic, n_samples, n_samples2)
                        if abs(r2) > 0.06:
                            text = f"{r2:.2f}"
                            text = text[1:]
                            #res_display.update(f"Statistical test results: {dfname} - nucleotide {nuc} position {j} - res: {res}, r2: {r2:.3f}")
                        else:
                            text = ""
                    if version == "delta":
                        delta = get_cliffs_delta(res.statistic, n_samples, n_samples2)
                        if abs(delta) > 0.14:
                            if scale >= 1.0:
                                text = f"{delta:.2f}"
                                text = text[1:] if delta > 0 else "-" + text[2:]
                            else:
                                text = f"{abs(delta):.2f}"
                                text = text[1:]# if delta > 0 else text[2:]
                            #res_display.update(f"Statistical test results: {dfname} - nucleotide {nuc} position {j} - res: {res}, delta: {delta:.3f}")
                        else:
                            text = ""
            else:
                text = ""'''
            
            stat_res = get_stat_result(test_array, test_array2, test=test)
            if correct_for_multiple_comparisons:
                p_values.append(stat_res.pvalue)
                statistics.append(stat_res.statistic)
            else:
                effect_size_res = get_effect_size(statistic=stat_res.statistic, n_samples=n_samples, n_samples2=n_samples2, effect_size=effect_size)
                text = get_effect_size_text(val=effect_size_res, effect_size=effect_size, pval=stat_res.pvalue, overwrite_threshold=0)
                val_labels.append(text)
            #text = nuc_enrich_stats(test_array, test_array2, test=test, version=version, scale=scale, n_samples=n_samples, n_samples2=n_samples2)
            #val_labels.append(text)
    if correct_for_multiple_comparisons:
        corrected_p_values = stats.multitest.multipletests(p_values, method='fdr_bh')[1]
        for p_adj, statistic in zip(corrected_p_values, statistics):
            effect_size_res = get_effect_size(statistic=statistic, n_samples=n_samples, n_samples2=n_samples2, effect_size=effect_size)
            text = get_effect_size_text(val=effect_size_res, effect_size=effect_size, pval=p_adj)
            val_labels.append(text)

    if len(vals) != 0:        
        m = abs(min(vals)) if abs(min(vals)) > max(vals) else max(vals)
    else:
        m = 0
    
    ax = plot_heatmap(x, y, vals, ax, format=".1e", cbar=True, vmin=-m, vmax=m, cbar_kws={"pad": 0.01})
    
    # Update annotations with effect size labels
    for v_idx, val_label in enumerate(ax.texts):
        val_label.set_text(val_labels[v_idx])
        val_label.set_size(12)
        if abs(vals[v_idx]) > 0.15:
            val_label.set_color("white")
        else:
            val_label.set_color("black")
    
    ax.set_ylabel("Nucleotide")
    ax.set_yticklabels([nuc for nuc in NUCLEOTIDES.keys()], rotation=0)
    ax.set_xlabel("Position")
    
    # Set x-axis ticks
    quarter = len(probability_matrix.index) // 4
    indexes = [pos for pos in range(1, quarter * 2 + 1)]
    ax.set_xticks([xtick - 0.5 for xtick in probability_matrix.index])
    ax.set_xticklabels(indexes + indexes, rotation=0)
    xlabels = ax.get_xticklabels()
    for x_idx, xlabel in enumerate(xlabels):
        if x_idx < quarter or x_idx >= quarter * 3:
            xlabel.set_color("black")
            xlabel.set_fontweight("bold")
        else:
            xlabel.set_color("grey")
    
    ax.tick_params(left=False, top=False, bottom=False)
    #ax.set_title(f"{dfname} - Nucleotide Enrichment ({compared})")
    fig.suptitle(f"{dfname} - Nucleotide Enrichment")
    ax.set_title(f'({compared.replace("_"," ")})')
    fig.tight_layout()
    
    if path != "":
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f"{name_prefix}_strain_nuc_occ_diff.png"), dpi=300)
    else:
        plt.show()
    plt.close()

def prepare_strain_data_for_nucleotide_enrichment_heatmap(strain, cutoff, pub_overwrite=None, own_synthetic=False, force_recreate=False, **kwargs):
    if pub_overwrite is not None:
        pubs = pub_overwrite
    else:
        pubs = strain_to_pubs.get(strain, [])
    if not pubs:
        logging.warning(f"No publications found for strain '{strain}' in strain_to_pubs. Using all publications as fallback.")
        pubs = ALL_PUBS
    df = load_data(pubs, unpooled=False)
    df = df[df["Strain"]==strain].reset_index(drop=True)
    df = identify_candidates(df)
    intersecting_ids = df.groupby('ID')['Publication'].nunique()
    intersecting_ids = intersecting_ids[intersecting_ids >= max(2, df["Publication"].nunique() / 2)].index.tolist()

    if kwargs.get("only_cleaned_intersecting", True):
        df = cutoff_clean(df, threshold=cutoff, minimum_dataset_size=0)
    else:
        df = cutoff_clean(df, threshold=cutoff, minimum_dataset_size=0, left_out_ids=intersecting_ids)
    df = get_sequence_quicker(df)
    df["junction_window"] = df.apply(lambda row: _extract_junction_window(row["Full_Sequence"], row["Start"], row["End"]), axis=1).transform(lambda x: x.replace("|", ""))
    logging.info(f'Processing {strain} with cutoff {cutoff}: {len(df)} total IDs, {len(intersecting_ids)} intersecting IDs, {len([id for id in intersecting_ids if id in df["ID"].values])} intersecting IDs after cutoff cleaning')
    df = cutoff_clean(df, threshold=cutoff, minimum_dataset_size=40).reset_index(drop=True)
    if own_synthetic:
        background_df = generate_expected_data(strain, df, cutoff=cutoff, seg_sample_size=kwargs.get("seg_sample_size", 35000), force_recreate=force_recreate, multi_source=kwargs.get("multi_source", True))
    else:
        background_df = load_synthetic(pubs)
        background_df = background_df[background_df["Strain"]==strain].reset_index(drop=True)
    background_df = identify_candidates(background_df)
    background_df = get_sequence_quicker(background_df)
    background_df["junction_window"] = background_df.apply(lambda row: _extract_junction_window(row["Full_Sequence"], row["Start"], row["End"]), axis=1).transform(lambda x: x.replace("|", ""))
    #if kwargs.get("only_cleaned_intersecting", True): # get only intesecting DVG that survive cutoff
    intersecting_df = df[df["ID"].isin(intersecting_ids)].drop_duplicates("ID").reset_index(drop=True)
    #intersecting_df = df[df["ID"].isin(intersecting_ids)].drop_duplicates("ID").reset_index(drop=True)
    non_intersecting_df = df[~df["ID"].isin(intersecting_ids)].drop_duplicates("ID").reset_index(drop=True)

    return non_intersecting_df, intersecting_df, background_df

def strain_nucleotide_enrichment_pipeline(cutoff=15, scale=1.0, debug=False):
    test_combs = [("kruskal", "jens"), ("kruskal", "me"), ("mannwhitney", "r"), ("mannwhitney", "r2"), ("mannwhitney", "delta")]
    if debug:
        test_combs = [("mannwhitney", "delta")]
    for strain in strain_to_pubs.keys():
        non_intersecting_df, intersecting_df, background_df = prepare_strain_data_for_nucleotide_enrichment_heatmap(strain=strain, cutoff=cutoff, own_synthetic=True)
        if debug:
            strain_path = ""
        else:
            strain_path = os.path.join(RESULT_PATH, strain, f'Cutoff {cutoff}', "nucleotide_enrichment")
            os.makedirs(strain_path, exist_ok=True)
        for test, version in test_combs:
            #display(f"Strain: {strain}, Test: {test}, Version: {version}")

            plot_strain_nucleotide_enrichment_heatmap(pd.concat([non_intersecting_df, intersecting_df], ignore_index=True), dfname=strain.replace("_","/"), background_df=background_df, compared="observed vs synthetic", test=test, version=version, path=strain_path, scale=scale, name_prefix=f"{test}_{version}_observed_vs_synthetic_")

            plot_strain_nucleotide_enrichment_heatmap(non_intersecting_df, dfname=strain.replace("_","/"), background_df=background_df, compared="non-intersecting vs synthetic", test=test, version=version, path=strain_path, scale=scale, name_prefix=f"{test}_{version}_non_intersecting_vs_synthetic_")

            plot_strain_nucleotide_enrichment_heatmap(intersecting_df, dfname=strain.replace("_","/"), background_df=non_intersecting_df, compared="intersecting vs non-intersecting", test=test, version=version, path=strain_path, scale=scale, name_prefix=f"{test}_{version}_intersecting_vs_non_intersecting_")

            plot_strain_nucleotide_enrichment_heatmap(intersecting_df, dfname=strain.replace("_","/"), background_df=background_df, compared="intersecting vs synthetic", test=test, version=version, path=strain_path, scale=scale, name_prefix=f"{test}_{version}_intersecting_vs_synthetic_")
        if debug:
            break

def execute_per_strain_nucleotide_enrichment_pipeline(cutoff_grid=[0,5,10,15], own_synthetic=False, save_plots=True, per_strain_scale: float=1.0, per_dataset_scale: float=1.5, correct_p: bool=False, debug=False, **kwargs):
    logging.info("Starting nucleotide enrichment pipeline...")
    #test_combs = [("kruskal", "meta2"), ("kruskal", "eta2"), ("mannwhitney", "r"), ("mannwhitney", "r2"), ("mannwhitney", "delta"), ("kruskal", "epsilon2"), ("chisquare", "cramers_v")]
    test_combs = [("mannwhitney", "delta"), ("kruskal", "meta2")]
    if debug:
        test_combs = [("mannwhitney", "delta")]
    for strain in strain_to_pubs.keys():
        for cutoff in cutoff_grid:
            if save_plots:
                strain_path = os.path.join(RESULT_PATH, strain, f'Cutoff {cutoff}', "nucleotide_enrichment")
                os.makedirs(strain_path, exist_ok=True)
            else:
                strain_path = ""
            
            # nucleotide enrichment for each dataset of the strain
            non_intersecting_dfs, dfnames, intersecting_dfs, synthetic_dfs = prepare_inter_nucleotide_enrichment_heatmap_data(strain, cutoff=cutoff, own_synthetic=own_synthetic, **kwargs.get("per_dataset_kwargs", {}))
            per_dataset_nucleotide_plots(non_intersecting_dfs, dfnames, intersecting_dfs, synthetic_dfs, strain_path=strain_path, cutoff=cutoff, test_combs=test_combs, scale=per_dataset_scale, debug=debug, show_strain_brackets=False)

            # nucleotide enrichment for entire strain
            non_intersecting_df, intersecting_df, background_df = prepare_strain_data_for_nucleotide_enrichment_heatmap(strain=strain, cutoff=cutoff, own_synthetic=True, **kwargs.get("per_strain_kwargs", {}))
            single_strain_nucleotide_plots(non_intersecting_df, intersecting_df, background_df, strain=strain, strain_path=strain_path, test_combs=test_combs, scale=per_strain_scale, correct_p=correct_p, debug=debug)
            if debug:
                break



########    Functions to plot intersection rates between publications and resulting label-conflicts    ########
def get_intersection_rates(data) -> pd.DataFrame:
    '''
    Loads data for a given strain, applies cutoff, identifies candidates, and calculates intersection rates between publications.

    :param strain: Strain to load data for
    :param cutoff: Cutoff to apply to data

    :return: Dataframe containing intersection rates between publications
    '''    
    # creating publication vs publication matrix with intersection rates as values
    strain = data["Strain"].unique()[0]
    pubs = STRAIN_TO_PUBS[strain]  # data["Publication"].unique()
    matrix = pd.DataFrame(0, index=pubs, columns=pubs)
    for pub1 in pubs:
        ids_pub1 = set(data[data['Publication'] == pub1]['ID'])
        for pub2 in pubs:
            if pub1 == pub2:
                matrix.loc[pub1, pub2] = 1.0
                continue
            ids_pub2 = set(data[data['Publication'] == pub2]['ID'])
            intersection_count = len(ids_pub1.intersection(ids_pub2))
            # Calculate intersection rate (normalized by the second set)
            if len(ids_pub1) == 0 or len(ids_pub2) == 0:
                intersection_rate = 0
            else:
                intersection_rate = intersection_count / len(ids_pub2)
            matrix.loc[pub1, pub2] = intersection_rate

    return matrix

def get_label_conflict_matrix(data, target_column: str="NGS_log_norm") -> pd.DataFrame:
    '''
    Loads data for a given strain, applies cutoff, identifies candidates, and calculates label conflict rates between publications.

    :param strain: Strain to load data for
    :param cutoff: Cutoff to apply to data
    :param target_column: Column name for the target variable

    :return: Dataframe containing label conflict rates between publications
    '''    
    # creating publication vs publication matrix with label conflict rates as values
    if "label" not in data.columns:
        if target_column not in data.columns:
            calculate_target(data, y_col=target_column, drop_read_count=False)
        data["median_label"] = data.groupby('Publication')[target_column].transform('median')
        data["label"] = data.apply(lambda row: 1 if row[target_column] > row['median_label'] else 0, axis=1)
    strain = data["Strain"].unique()[0]
    pubs = STRAIN_TO_PUBS[strain]  # data["Publication"].unique()
    matrix = pd.DataFrame(0, index=pubs, columns=pubs)
    for pub1 in pubs:
        ids_pub1 = set(data[data['Publication'] == pub1]['ID'])
        labels_pub1 = data[data['Publication'] == pub1].set_index('ID')['label'].to_dict()
        for pub2 in pubs:
            if pub1 == pub2:
                matrix.loc[pub1, pub2] = 0.0
                continue
            ids_pub2 = set(data[data['Publication'] == pub2]['ID'])
            labels_pub2 = data[data['Publication'] == pub2].set_index('ID')['label'].to_dict()
            common_ids = ids_pub1.intersection(ids_pub2)
            if len(common_ids) == 0:
                conflict_rate = 0
            else:
                conflicts = sum(1 for id in common_ids if labels_pub1[id] != labels_pub2[id])
                conflict_rate = conflicts / len(common_ids)
            matrix.loc[pub1, pub2] = conflict_rate

    return matrix

def get_odds_ratio_matrix(data_dict, strain):
    '''
    Transforms the dictionary of pairwise odds ratios into a matrix format for easier visualization and analysis.
    '''
    # keys are tuples of (pub1, pub2), values are odds ratios
    pubs = STRAIN_TO_PUBS[strain]  # sorted(set(pub for pair in data_dict.keys() for pub in pair))
    matrix = pd.DataFrame(index=pubs, columns=pubs, dtype=float)

    for pub1 in pubs:
        for pub2 in pubs:
            if pub1 == pub2:
                matrix.loc[pub1, pub2] = 1.0  # Odds ratio of a publication with itself is 1
            else:
                matrix.loc[pub1, pub2] = data_dict.get((pub1, pub2), np.nan)  # Fill with NaN if no data

    return matrix

def aggregate_by_publication(data, aggregation: str="sum") -> pd.DataFrame:
    '''
    Aggregates data by publication, summing or averaging NGS_read_count for intersecting IDs.

    :param data: Dataframe containing data to aggregate
    :param aggregation: Method of aggregation, either "sum" or "mean"

    :return: Dataframe aggregated by publication
    '''
    if aggregation == "sum":
        agg_func = 'sum'
    elif aggregation == "mean":
        agg_func = 'mean'
    else:
        raise ValueError("Invalid aggregation method. Use 'sum' or 'mean'.")

    behavior = {col: "first" for col in data.columns}
    behavior["NGS_read_count"] = agg_func
    aggregated_data = data.groupby(['ID', 'Publication'], as_index=False).agg(behavior)
    return aggregated_data.drop_duplicates(subset=['ID', 'Publication']).reset_index(drop=True)

def plot_label_scatter(data, pub1, pub2, target_column="NGS_log_norm", aggregation="sum", path=""):
    '''
    Plots a scatter plot comparing NGS_read_count between two publications for intersecting IDs.

    :param data: Dataframe containing data to plot
    :param pub1: First publication to compare
    :param pub2: Second publication to compare

    :return: None
    '''
    sns.set_theme(style="darkgrid", context="talk", palette="colorblind")
    if target_column not in data.columns:
        calculate_target(data, y_col=target_column, drop_read_count=False)
    ids_pub1 = set(data[data['Publication'] == pub1]['ID'])
    ids_pub2 = set(data[data['Publication'] == pub2]['ID'])
    common_ids = ids_pub1.intersection(ids_pub2)
    
    if len(common_ids) == 0:
        logging.info(f"No common IDs between {pub1} and {pub2}. Cannot generate scatter plot.")
        return
    if len(common_ids) < 5:
        logging.info(f"Just {len(common_ids)} between {pub1} and {pub2}. Skipping scatter plot.")
        return
    
    df_pub1 = data[(data['Publication'] == pub1) & (data['ID'].isin(common_ids))].set_index('ID')
    df_pub2 = data[(data['Publication'] == pub2) & (data['ID'].isin(common_ids))].set_index('ID')
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', adjustable='box')
    sns.scatterplot(x=df_pub1[target_column], y=df_pub2[target_column], size=1, markers=".", ax=ax, legend=False)
    if df_pub1[target_column].max() <= 1 and df_pub2[target_column].max() <= 1 and df_pub1[target_column].min() >= 0 and df_pub2[target_column].min() >= 0:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        ax.set_xlim(df_pub1[target_column].min(), df_pub1[target_column].max())
        ax.set_ylim(df_pub2[target_column].min(), df_pub2[target_column].max())

    # mark the median to visualize the binary label thresholds
    x_median = df_pub1[target_column].median()
    y_median = df_pub2[target_column].median()
    ax.axvline(x=x_median, color='grey', linestyle='--', label='Label Threshold')
    ax.axhline(y=y_median, color='grey', linestyle='--', label='Label Threshold')

    # mark area above both medians and below both medians green and area above one median and below the other red to visualize agreement and disagreement between datasets
    ax.fill_betweenx([ax.get_ylim()[0], y_median], ax.get_xlim()[0], x_median, color='blue', alpha=0.3)
    ax.fill_betweenx([y_median, ax.get_ylim()[1]], x_median, ax.get_xlim()[1], color='blue', alpha=0.3)
    ax.fill_betweenx([ax.get_ylim()[0], y_median], x_median, ax.get_xlim()[1], color='yellow', alpha=0.3)
    ax.fill_betweenx([y_median, ax.get_ylim()[1]], ax.get_xlim()[0], x_median, color='yellow', alpha=0.3)

    # create custom legend: Patches for areas and lines for thresholds
    legend_elements = [plt.Line2D([0], [0], color='grey', linestyle='--', label='Label Threshold'),
                       Patch(facecolor='blue', edgecolor='blue', alpha=0.3, label='Agreement'),
                       Patch(facecolor='yellow', edgecolor='yellow', alpha=0.3, label='Conflict')]
    ax.legend(handles=legend_elements, loc='upper right')

    if target_column == "NGS_log_min_max_norm":
        title_norm = f'scaled NGS log norm'
    else:
        title_norm = target_column.replace("_", " ")
    if "NGS" not in target_column:
        title_norm = f'NGS {title_norm}'

    ax.set_xlabel(f'{title_norm}\n{pub1}')#{target_column.replace("_", " ")}')
    ax.set_ylabel(f'{pub2}\n{title_norm}')#{target_column.replace("_", " ")}')
    #title = f'Scatter Plot of {title_norm} for {pub1} vs {pub2}'
    #ax.set_title(title)
    if path != "":
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f'{pub1}_vs_{pub2}_{aggregation}_{target_column}_scatter.png'), dpi=300)
    else:
        plt.show()
    plt.close()

def test_conflict_significance(data, pub1, pub2, target_column="NGS_log_norm"):
    '''
    Performs a statistical test to determine if the label conflict rate between two publications is significant.
    '''
    if "label" not in data.columns:
        if target_column not in data.columns:
            calculate_target(data, y_col=target_column, drop_read_count=False)
        data["median_label"] = data.groupby('Publication')[target_column].transform('median')
        data["label"] = data.apply(lambda row: 1 if row[target_column] > row['median_label'] else 0, axis=1)
    
    ids_pub1 = set(data[data['Publication'] == pub1]['ID'])
    ids_pub2 = set(data[data['Publication'] == pub2]['ID'])
    common_ids = ids_pub1.intersection(ids_pub2)
    
    if len(common_ids) == 0:
        logging.info(f"No common IDs between {pub1} and {pub2}. Cannot perform significance test.")
        return 0, 0, None, None, None, None
    
    df_pub1 = data[(data['Publication'] == pub1) & (data['ID'].isin(common_ids))].set_index('ID')
    df_pub2 = data[(data['Publication'] == pub2) & (data['ID'].isin(common_ids))].set_index('ID')
    
    labels_pub1 = df_pub1['label']
    labels_pub2 = df_pub2['label']
    b = np.sum((labels_pub1 == 1) & (labels_pub2 == 0))
    c = np.sum((labels_pub1 == 0) & (labels_pub2 == 1))
    contingency_table = [[np.sum((labels_pub1 == 1) & (labels_pub2 == 1)),
                          np.sum((labels_pub1 == 1) & (labels_pub2 == 0))],
                          [np.sum((labels_pub1 == 0) & (labels_pub2 == 1)),
                           np.sum((labels_pub1 == 0) & (labels_pub2 == 0))]]
    #contingency_table = pd.crosstab(labels_pub1, labels_pub2)
    if contingency_table[0][1] + contingency_table[1][0] < 25:
        exact = True
    else:
        exact = False
    exact_stat, exact_pvalue, corrected_stat, corrected_pvalue = None, None, None, None
    #display(f'b: {contingency_table[0][1]}, c: {contingency_table[1][0]}')
    mcnemar_result = mcnemar(contingency_table, exact=exact, correction=False)
    exact_stat, exact_pvalue = mcnemar_result.statistic, mcnemar_result.pvalue
    #display(f'McNemar{" exact" if exact else " approximate"} test result: statistic={mcnemar_result.statistic}, p-value={mcnemar_result.pvalue}')
    if not exact:
        mcnemar_result = mcnemar(contingency_table, exact=exact, correction=True)
        corrected_stat, corrected_pvalue = mcnemar_result.statistic, mcnemar_result.pvalue
    #display(f'Corrected McNemar test result: statistic={mcnemar_result.statistic}, p-value={mcnemar_result.pvalue}')
    #chi2, chi2_pvalue, dof, expected = stats.chi2_contingency(contingency_table)
    #display(f"Chi-squared: {chi2}, p-value: {p}")
    return b, c, exact_stat, exact_pvalue, corrected_stat, corrected_pvalue#, chi2, chi2_pvalue

def plot_conflict_matrix(conflict_matrix, stat_df=None, name_prefix="", path="", strain="", cutoff=0, target_column="", get_ax = False):
    sns.set_theme(style="darkgrid", context="notebook", font_scale=1.15, palette="colorblind")
    conflict_matrix.rename(columns=lambda x: get_datasetname(strain, x), index=lambda x: get_datasetname(strain, x), inplace=True)
    annot_mat = conflict_matrix.applymap(lambda x: f"{x*100:.1f}" if x > 0 else "0")
    ax = sns.heatmap(conflict_matrix*100, square=True, annot=annot_mat, fmt="", cmap='magma', vmin=0, vmax=100, cbar_kws={'label': 'Conflict of shared DelVGs [%]', "format": "{x:.0f}"})
    '''
    # Draw yellow rectangles for non-significant cells
    if stat_df is not None:
        non_significant_conflicts = stat_df[(stat_df["exact_pvalue"]>=0.05) | (stat_df["corrected_pvalue"]>=0.05)]
        non_significant_matrix = pd.DataFrame(0, index=STRAIN_TO_PUBS[strain], columns=STRAIN_TO_PUBS[strain])
        for (pub1, pub2) in non_significant_conflicts.index:
            non_significant_matrix.loc[pub1, pub2] = 1
            non_significant_matrix.loc[pub2, pub1] = 1

        
        for i in range(non_significant_matrix.shape[0]):
            for j in range(non_significant_matrix.shape[1]):
                if non_significant_matrix.iloc[i, j] == 1:
                    rect = patches.Rectangle(
                        (j, i), 1, 1,
                        fill=False,
                        edgecolor='yellow',
                        linewidth=2
                    )
                    ax.add_patch(rect)
    '''
    if get_ax:
        return ax
    if path !="":
        plt.savefig(os.path.join(path, f'{name_prefix}_conflict_matrix.png'), dpi=300)
    else:
        plt.show()
    plt.close()

def plot_weighted_conflict_matrix(intersection_matrix, conflict_matrix, name_prefix="", path="", strain="", cutoff=0, target_column="", get_ax = False):
    sns.set_theme(style="darkgrid", context="notebook", font_scale=1.15, palette="colorblind")
    weighed_conflict_matrix = conflict_matrix * intersection_matrix
    weighed_conflict_matrix.rename(columns=lambda x: get_datasetname(strain, x), index=lambda x: get_datasetname(strain, x), inplace=True)
    annot_mat = weighed_conflict_matrix.applymap(lambda x: f"{x*100:.1f}" if x > 0 else "0")
    sns.heatmap(weighed_conflict_matrix*100, square=True, annot=annot_mat, fmt="", cmap='cividis', vmin=0, vmax=100, cbar_kws={'label': 'Conflicting Signals [%]'})
    if get_ax:
        return plt.gca()
    if path != "":
        plt.savefig(os.path.join(path, f'{name_prefix}_weighed_conflict_matrix.png'), dpi=300)
    else:
        plt.show()
    plt.close()

    sns.heatmap(weighed_conflict_matrix*100, square=True, annot=annot_mat, fmt="", cmap='cividis', vmin=0, vmax=50, cbar_kws={'label': 'Conflicting Signals [%]'})
    if path !="":
        plt.savefig(os.path.join(path, f'{name_prefix}_weighed_conflict_matrix_zoomed.png'), dpi=300)
    else:
        plt.show()
    plt.close()

def plot_intersection_matrix(intersection_matrix, name_prefix="", path="", strain="", cutoff=0, target_column="", get_ax = False):
    sns.set_theme(style="darkgrid", context="notebook", font_scale=1.15, palette="colorblind")
    intersection_matrix.rename(columns=lambda x: get_datasetname(strain, x), index=lambda x: get_datasetname(strain, x), inplace=True)
    annot_mat = intersection_matrix.applymap(lambda x: f"{x*100:.1f}" if x < 1 else "100")
    sns.heatmap(intersection_matrix*100, square=True, annot=annot_mat, fmt="", cmap='viridis', vmin=0, vmax=100, cbar_kws={'label': 'Shared DelVGs [%]'})
    if get_ax:
        return plt.gca()
    if path != "":
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f'{name_prefix}_intersection_matrix.png'), dpi=300)
    else:
        plt.show()
    plt.close()
    
def plot_mcnemar_matrix(mcnemar_results_df, odds_ratio_matrix, name_prefix="", path="", strain="", cutoff=0, target_column="", get_ax=False):
    sns.set_theme(style="darkgrid", context="notebook", font_scale=1.1, palette="colorblind")
    #mcnemar_results_df.rename(index=lambda x: f"{get_datasetname(strain, x[0])} vs {get_datasetname(strain, x[1])}", inplace=True)
    odds_ratio_matrix.rename(columns=lambda x: get_datasetname(strain, x), index=lambda x: get_datasetname(strain, x), inplace=True)
    odds_ratio_matrix = odds_ratio_matrix.applymap(lambda x: np.log10(x) if x > 0 else np.nan)  # Log-transform odds ratios for better visualization
    
    #fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(odds_ratio_matrix, square=True, annot=True, fmt=".2f", cmap='vlag', center=0, cbar_kws={'label': r'$\log_{10}$ odds ratio of discordant labels'})
    
    # Draw yellow rectangles for non-significant cells
    if mcnemar_results_df is not None:
        non_significant_conflicts = mcnemar_results_df[(mcnemar_results_df["exact_pvalue"]>=0.05) | (mcnemar_results_df["corrected_pvalue"]>=0.05)]
        non_significant_matrix = pd.DataFrame(0, index=STRAIN_TO_PUBS[strain], columns=STRAIN_TO_PUBS[strain])
        for (pub1, pub2) in non_significant_conflicts.index:
            non_significant_matrix.loc[pub1, pub2] = 1
            non_significant_matrix.loc[pub2, pub1] = 1

        
        for i in range(non_significant_matrix.shape[0]):
            for j in range(non_significant_matrix.shape[1]):
                if non_significant_matrix.iloc[i, j] == 1:
                    rect = patches.Rectangle(
                        (j, i), 1, 1,
                        fill=False,
                        edgecolor='yellow',
                        linewidth=2
                    )
                    ax.add_patch(rect)
    if get_ax:
        return plt.gca()
    
    if path != "":
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f'{name_prefix}_mcnemar_odds_ratio_matrix.png'), dpi=300)
    else:
        plt.show()
    plt.close()

def intersection_and_conflict_pipeline(strain: str, cutoff: int=15, target_columns: str|list="NGS_log_norm", aggregation: str="sum", path="", save_plots: bool=False, **kwargs):
    sns.set_theme(style="darkgrid", context="notebook", font_scale=1.15, palette="colorblind")
    df = load_data(STRAIN_TO_PUBS[strain], unpooled=True)
    df = df[df["Strain"]==strain]
    df = identify_candidates(df)
    df = cutoff_clean(df, cutoff, minimum_dataset_size=0).reset_index(drop=True)

    intersection_matrix = get_intersection_rates(df)
    if not kwargs.get("skip_plots", False) and not kwargs.get("skip_intersection_matrix", False):
        plot_intersection_matrix(intersection_matrix, name_prefix=f'{strain}_cutoff_{cutoff}', path=path, strain=strain, cutoff=cutoff, target_column=target_columns, get_ax=False)

    df = aggregate_by_publication(df, aggregation=aggregation)
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    for target_column in target_columns:
        df = calculate_target(df, y_col=target_column, drop_read_count=False)
        df["median_label"] = df.groupby('Publication')[target_column].transform('median')
        df["label"] = df.apply(lambda row: 1 if row[target_column] > row['median_label'] else 0, axis=1)

        mcnemar_results = {}
        odds_ratios = {}
        for pub1 in df["Publication"].unique():
            for pub2 in df["Publication"].unique():
                if pub1 == pub2:
                    continue
                b, c, exact_stat, exact_pvalue, corrected_stat, corrected_pvalue  = test_conflict_significance(df, pub1, pub2, target_column=target_column)
                # only plot scatter if the conflict is significant in at least one of the tests to avoid plotting insignificant conflicts
                is_significant = (exact_pvalue is not None and exact_pvalue < 0.05) or (corrected_pvalue is not None and corrected_pvalue < 0.05)
                if is_significant and not kwargs.get("skip_plots", False) and not kwargs.get("skip_scatter_plots", False):
                    plot_label_scatter(df, pub1, pub2, target_column=target_column, aggregation=aggregation, path=path)
                mcnemar_results[(pub1, pub2)] = (b, c, exact_stat, exact_pvalue, corrected_stat, corrected_pvalue) #, chi2, chi2_pvalue)
                # Calculate odds ratios for each pair of publications
                odds_ratios[(pub2, pub1)] = b/c if c != 0 else np.inf  # Handle division by zero
        mcnemar_results_df = pd.DataFrame(mcnemar_results, index=["b", "c", "exact_stat", "exact_pvalue", "corrected_stat", "corrected_pvalue"]).T
        # report non-significant results
        if path != "":
            mcnemar_results_df.to_csv(os.path.join(path, f'{strain}_{aggregation}_cutoff_{cutoff}_{target_column}_mcnemar_results.csv'))
            if mcnemar_results_df[(mcnemar_results_df["exact_pvalue"]>=0.05)| (mcnemar_results_df["corrected_pvalue"]>=0.05)].shape[0] > 0:
                logging.info(f"Non-significant conflict between datasets for {strain} with cutoff {cutoff} and target column {target_column}:\n{mcnemar_results_df[(mcnemar_results_df['exact_pvalue']>=0.05) | (mcnemar_results_df['corrected_pvalue']>=0.05)]}")
            else:
                logging.info(f"All conflicts between datasets for {strain} with cutoff {cutoff} and target column {target_column} are significant.")
        else:
            if mcnemar_results_df[(mcnemar_results_df["exact_pvalue"]>=0.05)| (mcnemar_results_df["corrected_pvalue"]>=0.05)].shape[0] > 0:
                logging.info(f"Non-significant conflict between datasets for {strain} with cutoff {cutoff} and target column {target_column}:",mcnemar_results_df[(mcnemar_results_df['exact_pvalue']>=0.05) | (mcnemar_results_df['corrected_pvalue']>=0.05)])
            else:
                logging.info(f"All conflicts between datasets for {strain} with cutoff {cutoff} and target column {target_column} are significant.")

        conflict_matrix = get_label_conflict_matrix(df, target_column=target_column)
        if not kwargs.get("skip_plots", False) and not kwargs.get("skip_conflict_matrix", False):
            plot_conflict_matrix(conflict_matrix, stat_df=mcnemar_results_df, name_prefix=f'{strain}_{aggregation}_cutoff_{cutoff}_{target_column}', path=path, strain=strain, cutoff=cutoff, target_column=target_column)
        '''conflict_matrix.rename(columns=lambda x: get_datasetname(strain, x), index=lambda x: get_datasetname(strain, x), inplace=True)
        annot_mat = conflict_matrix.applymap(lambda x: f"{x*100:.1f}" if x > 0 else "0")
        ax = sns.heatmap(conflict_matrix*100, square=True, annot=annot_mat, fmt="", cmap='magma', vmin=0, vmax=100, cbar_kws={'label': 'Conflict of shared DelVGs [%]', "format": "{x:.0f}"})
        # Draw yellow rectangles for non-significant cells
        for i in range(non_significant_matrix.shape[0]):
            for j in range(non_significant_matrix.shape[1]):
                if non_significant_matrix.iloc[i, j] == 1:
                    rect = patches.Rectangle(
                        (j, i), 1, 1,
                        fill=False,
                        edgecolor='yellow',
                        linewidth=2
                    )
                    ax.add_patch(rect)

        if path !="":
            plt.savefig(os.path.join(path, f'{strain}_{aggregation}_cutoff_{cutoff}_{target_column}_conflict_matrix.png'), dpi=300)
        elif not kwargs.get("skip_plots", False):
            plt.show()
        plt.close()'''
        
        if not kwargs.get("skip_plots", False) and not kwargs.get("skip_weighted_conflict_matrix", False):
            plot_weighted_conflict_matrix(intersection_matrix, conflict_matrix, name_prefix=f'{strain}_{aggregation}_cutoff_{cutoff}_{target_column}', path=path, strain=strain, cutoff=cutoff, target_column=target_column)

        if not kwargs.get("skip_plots", False) and not kwargs.get("skip_mcnemar_matrix", False):
            plot_mcnemar_matrix(mcnemar_results_df, get_odds_ratio_matrix(odds_ratios, strain), name_prefix=f'{strain}_{aggregation}_cutoff_{cutoff}_{target_column}', path=path, strain=strain, cutoff=cutoff, target_column=target_column)

def label_conflicts_per_strain(cutoff_grid=[0, 5, 10, 15], target_columns: str|list=["NGS_log_norm", "NGS_log_min_max_norm", "CLR"], save_plots: bool=True, debug: bool=False, **kwargs):
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    strain_path = ""
    for strain in STRAIN_TO_PUBS.keys():
        for aggregation in ["sum", "mean"]:
            for cutoff in cutoff_grid:
                if save_plots:
                    strain_path = os.path.join(RESULT_PATH, strain, f'Cutoff {cutoff}', "intersection_plots")
                    logging.info(f"Plotting intersection and conflict matrices for strain '{strain}' with cutoff {cutoff} on {aggregation} aggregated data and saving to '{strain_path}'")
                intersection_and_conflict_pipeline(strain=strain, cutoff=cutoff, target_columns=target_columns, aggregation=aggregation, path=strain_path, **kwargs)
                if debug:
                    return

def execute_label_conflicts_per_strain(save_plots=True, debug=False, **kwargs):
    label_conflicts_per_strain(save_plots=save_plots, debug=debug, **kwargs)


########    function to just run all pipelines    ########

def run_all_pipelines():
    logging.info("Running all pipelines...")
    execute_rsc_intersecting_pipeline_per_strain()
    execute_numeric_feature_cutoff_pipeline_per_strain()
    execute_boxplotting_per_strain(cutoff_grid=[0, 5, 10, 15], segments=ALL_SEGMENTS, add_all=True, n_cols=4, figsize_per_panel=(3, 4))
    execute_per_strain_nucleotide_enrichment_pipeline(cutoff_grid=[0,5,10,15], own_synthetic=True, save_plots=True, per_strain_scale=1.0, per_dataset_scale=1.5, debug=False)
    execute_label_conflicts_per_strain(save_plots=True, debug=False)
