import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import os
import gc
import sys
from pathlib import Path
import traceback

import shap

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from utils import get_sequence

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

ALL_STRAINS = ["A_Anhui_1_2013", "A_California_07_2009", "A_Connecticut_Flu122_2013", "A_NewCaledonia_20-JY2_1999",
               "A_Perth_16_2009", "A_PuertoRico_8_1934", "A_turkey_Turkey_1_2005", "A_WSN_33", "B_Brisbane_60_2008",
               "B_Lee_1940", "B_Victoria_504_2000", "B_Yamagata_16_1988"]
ALL_SEGMENTS = ["PB2", "PB1", "PA", "HA", "NP", "NA", "M", "NS"]
PLOT_PATH = os.path.join("plots")
CHARS = ["A", "U", "C", "G"]
DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'data'))
#RESULT_PATH = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'results',datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

def get_param_grid(name, regression=False):
    match name:
        case 'logistic_regression':
            return {
                    "penalty": ["l1", "l2", "elasticnet"],
                    "C" : [0.01, 0.1, 1.0],
                    }
        case 'support_vector':
            return {
                    "kernel": ["linear", "rbf", "sigmoid", "poly"],
                    "C": [0.01, 0.1, 1.0],
                    "gamma": ["scale", "auto", 2],
                    }
        case 'random_forest':
            return {
                    "min_samples_split": [3, 5, 10],
                    "n_estimators": [100, 300],
                    "max_depth": [5, 15, 25],
                    "max_features": [5, 10, 20, 'log2', 'sqrt'],
                    #"criterion": ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'] if regression else ['gini', 'entropy', 'log_loss']
                    }
        case 'adaboost':
            return {
                    "n_estimators": [25, 50, 100, 300, 500],
                    "learning_rate": [0.01, 0.1, 0.25, 0.5, 0.75, 1.0],
                    }
        case 'naive_bayes':
            return {
                    "var_smoothing": [0.000000001, 0.0000000001, 0.00000000001]
                    }
        case 'mlp':
            return {
                    "hidden_layer_sizes": [(50,), (100,), (250,)],
                    "alpha": [0.001, 0.0001, 0.00001]
                    }
        case 'gradient_boost':
            return {
                    'n_estimators': [5, 50, 100, 300, 500],
                    'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.5, 1],
                    'max_depth': [3, 5, 10]
                    }
        case 'xgb':
            return {
                    #'min_child_weight': [1, 5, 10],
                    #'gamma': [0.5, 1, 2, 5],
                    #'subsample': [0.6, 0.8, 1.0],
                    #'colsample_bytree': [0.6, 0.8, 1.0],
                    'max_depth': [3, 4, 5],
                    'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.5, 1],
                    'n_estimators': [5, 50, 100, 300, 500]
                    }
        case 'lgb':
            return {
                    'num_leaves': [10,50,100],
                    'max_depth': [3, 4, 5],
                    'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.5, 1],
                    'n_estimators': [5, 50, 100, 300, 500]
                    }
        case 'linear':
            return {}
        case 'ridge':
            return {
                    'alpha': [0.1, 1.0, 10.0]
                    }
        case 'lasso':
            return {
                    'alpha': [0.001, 0.01, 0.1]
                    }
        case 'knn':
            return {
                    'n_neighbors': [3, 5, 10]
                    }

def select_classifier(name: str, perform_grid_search: bool = False):
    if perform_grid_search:
        match name:
            case 'logistic_regression':
                clf = LogisticRegression(max_iter=10000, solver="saga", l1_ratio=0.5)
            case 'support_vector':
                clf = SVC(gamma=2, C=1)
            case 'random_forest':
                clf = RandomForestClassifier()
            case 'adaboost':
                clf = AdaBoostClassifier(n_estimators=25, learning_rate=0.1)
            case 'naive_bayes':
                clf = GaussianNB(var_smoothing=0.0000000001)
            case 'mlp':
                clf = MLPClassifier(max_iter=10000)
            case 'xgb':
                clf = XGBClassifier(objective = 'binary:hinge')
            case 'lgb':
                clf = LGBMClassifier(objective = 'binary')
            case _:
                logger.error(f'Error: Unknown classifier name {name}. Skipping...')
                return "unknown classifier", None
        param_grid = get_param_grid(name)
    else:    
        match name:
            case 'logistic_regression':
                clf = LogisticRegression(penalty="l1", C=1.0, solver="saga", max_iter=10000)
            case 'support_vector':
                clf = SVC()
            case 'random_forest':
                clf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=10, max_features=20)
            case 'adaboost':
                clf = AdaBoostClassifier(n_estimators=100, learning_rate=1.0)
            case 'naive_bayes':
                clf = GaussianNB(var_smoothing=0.0000000001)
            case 'mlp':
                clf = MLPClassifier(alpha=0.0001, hidden_layer_sizes=(100,), max_iter=10000)
            case 'xgb':
                clf = XGBClassifier(learning_rate=0.1, n_estimators=200, max_depth=2, objective = 'binary:hinge')
            case 'lgb':
                clf = LGBMClassifier(learning_rate=0.1, n_estimators=200, max_depth=2, objective = 'binary')
            case _:
                logger.error(f'Error: Unknown classifier name {name}. Skipping...')
                return "unknown classifier", None
        param_grid = dict()
    return clf, param_grid

def select_regressor(name: str, perform_grid_search: bool = False):
    if perform_grid_search:
        match name:
            case 'linear':
                reg = LinearRegression()
            case 'ridge':
                reg = Ridge(max_iter=10000)
            case 'lasso':
                reg = Lasso(max_iter=10000)
            case 'support_vector':
                reg = SVR()
            case 'random_forest':
                reg = RandomForestRegressor()
            case 'adaboost':
                reg = AdaBoostRegressor()
            case 'gradient_boost':
                reg = GradientBoostingRegressor()
            case 'mlp':
                reg = MLPRegressor(max_iter=10000)
            case 'xgb':
                reg = XGBRegressor(objective='reg:squarederror')
            case 'lgb':
                reg = LGBMRegressor()
            case 'knn':
                reg = KNeighborsRegressor()
            case _:
                logger.error(f'Error: Unknown regressor name {name}. Skipping...')
                return "unknown regressor", None
        param_grid = get_param_grid(name, True)
    else:
        match name:
            case 'linear':
                reg = LinearRegression()
            case 'ridge':
                reg = Ridge(max_iter=10000)
            case 'lasso':
                reg = Lasso(max_iter=10000)
            case 'svr':
                reg = SVR()
            case 'random_forest':
                reg = RandomForestRegressor()
            case 'adaboost':
                reg = AdaBoostRegressor()
            case 'gradient_boost':
                reg = GradientBoostingRegressor()
            case 'mlp':
                reg = MLPRegressor(max_iter=10000)
            case 'xgb':
                reg = XGBRegressor(objective='reg:squarederror')
            case 'lgb':
                reg = LGBMRegressor()
            case 'knn':
                reg = KNeighborsRegressor()
            case _:
                logger.error(f'Error: Unknown regressor name {name}. Skipping...')
                return "unknown regressor", None
        param_grid = dict()

    return reg, param_grid

def make_artificial_set(strain, segment, step_distance=5):
    '''
    Creates an artificial dataset of DelVGs for a given strain and segment.

    :param strain: Strain of the virus
    :param segment: Segment of the virus
    :param step_distance: Distance between possible start- and end-values

    :return: Dataframe with possible DelVGs
    '''
    seq = get_sequence(strain, segment)
    n = len(seq)
    rows = []
    for start in range(25, n-15-step_distance, step_distance):
        for end in range(start, n-15, step_distance):
            new_row = pd.DataFrame.from_dict({"Strain": [strain], "Segment": [segment], "Start": [start], "End": [end], "Full_Sequence": [seq]})
            rows.append(new_row)
    df = pd.concat(rows, ignore_index=True)
    return df

def plot_startvsend(df, name, labels, thresholds, y_column, result_path) -> None:
    '''
    Plots the Start and End location of a deletion site in scatter plots. Creates two separate plots, one using a
    heatmap based on the NGS_log_norm and one splitting the data by the given thresholds.

    :param df: Dataframe containing the Start, End and NGS_read_count and NGS_log_norm columns.
    :param name: Name for the files to save each plot
    :param labels: List of labels for each datapoint (y values)
    :param thresholds: List of thresholds between classes
    :param result_path: Path to save the plots
    '''
    os.makedirs(result_path, exist_ok=True)
    plt.rcParams.update({'font.size': 14})
    df_c = df.copy()
    fig, ax = plt.subplots(figsize=(10,10))
    scatter = ax.scatter(df_c["Start"], df_c["End"], vmin=0, vmax=1, alpha=0.5, cmap="jet", c=df_c[y_column])
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.8, aspect=30)
    cbar.set_label('Normalized log of NGS read count')
    ax.text(.99, 0.5, f'n = {len(df_c["Start"])}', transform=ax.transAxes, fontsize=14, verticalalignment='bottom',
             horizontalalignment='right')
    ax.set_xlim(0, 2400)
    ax.set_ylim(0, 2400)
    ax.set_xlabel(f'Start')
    ax.set_ylabel(f'End')
    ax.set_title(f'{name} Dataset: Scatter Plot of Start and End Positions')
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    if result_path != "":
        os.makedirs(result_path, exist_ok=True)
        plt.savefig(os.path.join(result_path, f'{name}_Scatter_StartVsEnd.png'))
    else:
        plt.show()
    plt.close()
    gc.collect()

    # Making plot with label colors
    classes = np.unique(labels)
    num_classes = len(classes)
    colors = cm.viridis(np.linspace(0, 1, num_classes))
    label_colors = {label: color for label, color in zip(labels, colors)}
    for label in range(num_classes):
        label_colors[label] = colors[label]
    cmap = ListedColormap(colors)
    
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    scatter2 = ax2.scatter(df_c["Start"], df_c["End"], alpha=0.5, cmap=cmap, c=labels)

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markersize=10,
                          label=classes[0])]
    if thresholds is not None:
        for i in range(1, num_classes):
            if num_classes == len(thresholds)-1:
                handles.append(plt.Line2D([0], [0], color='k', linestyle='--',
                                        markerfacecolor=(colors[i]-colors[i-1])/2, markersize=10, label=f'{thresholds[i-1]:.3f}'))
                handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10,
                                        label=classes[i]))

    ax2.legend(handles=handles, title="Classes", loc='lower right', fancybox=True, shadow=True, fontsize='large')
    ax2.text(.99, 0.5, f'n = {len(df_c["Start"])}', transform=ax2.transAxes, fontsize=14, verticalalignment='bottom',
             horizontalalignment='right')

    ax2.set_xlim(0, 2400)
    ax2.set_ylim(0, 2400)
    ax2.set_xlabel('Start')
    ax2.set_ylabel('End')
    ax2.set_title(f'{name} Dataset: Labeled Scatter Plot of Start and End Positions')
    ax2.set_aspect('equal', 'box')
    plt.tight_layout()
    if result_path != "":
        plt.savefig(os.path.join(result_path, f'{name}_labeled_Scatter_StartVsEnd.png'))
    else:
        plt.show()
    plt.close()
    gc.collect()

def plot_actual_vs_pred(y_true, y_pred, model_name, output_dir):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_name} - Actual vs. Predicted')
    plt.grid(True)
    if output_dir != "":
        path = os.path.join(output_dir, f'{model_name}_actual_vs_pred.png')
        plt.savefig(path)
    else:
        plt.show()
    plt.close()

def plot_prediction_startvsend(X_test, y_test, predictions, name, result_path, multi_seg=False, high_only=False) -> None:
    '''
    Plots the Start and End location of a deletion site in scatter plots. Colors each datapoint based on its predicted
    label.

    :param X_test: Dataframe containing the Start, End and NGS_read_count and NGS_log_norm columns.
    :param predictions: List of predictions for each datapoint
    :param name: Name for the files to save each plot
    :param result_path: Path to save the plots

    :return: None
    '''
    os.makedirs(result_path, exist_ok=True)
    logger.debug(f'Creating scatter plot based on predictions of {name} on Start and End positions')

    fig, ax = plt.subplots(figsize=(10,10))
    if multi_seg:
        classes = ALL_SEGMENTS
        num_classes = len(classes)
        colors = cm.tab10(np.linspace(0, 1, num_classes))
    else:
        classes = list(np.unique(predictions))
        num_classes = len(classes)
        colors = cm.viridis(np.linspace(0, 1, num_classes))
    label_colors = {label: color for label, color in zip(classes, colors)}
    if not multi_seg:
        for label in range(num_classes):
            label_colors[label] = colors[label]
    cmap = ListedColormap(colors)
    if multi_seg:
        X_test['pred'] = predictions
        X_test = X_test[X_test['pred']==1]
        X_test['Segment'] = X_test.apply(lambda row: [col for col in ALL_SEGMENTS if row[col] == 1], axis=1)
        labels = X_test['Segment']
        labels = [x[0] for x in labels]
        labels = [labels.index(x) for x in labels]
        scatter = ax.scatter(X_test["Start"], X_test["End"], alpha=0.5, cmap=cmap, c=labels)
    else:
        scatter = ax.scatter(X_test["Start"], X_test["End"], alpha=0.5, cmap=cmap, c=predictions)

    ax.text(.99, 0.5, f'n = {len(X_test["Start"])}', transform=ax.transAxes, fontsize=14, verticalalignment='bottom',
             horizontalalignment='right')
    ax.set_xlim(0, 2400)
    ax.set_ylim(0, 2400)
    ax.set_xlabel(f'Start')
    ax.set_ylabel(f'End')
    if multi_seg:
        ax.set_title(f'{name} Dataset: segment-wise high predictions')
    else:
        ax.set_title(f'{name} Dataset: Predicted Labels')
    ax.set_aspect('equal', 'box')
    ax.legend(handles = [plt.Line2D([i], [i], marker='o', color='w', markerfacecolor=colors[i], markersize=10,
                          label=classes[i]) for i in range(num_classes)], title="Predicted Label", loc='lower right', fancybox=True, shadow=True, fontsize='large')
    plt.tight_layout()
    if result_path != "":
        path = os.path.join(result_path, f'{name}_Prediction_Scatter_StartVsEnd.png')
        plt.savefig(path)
    else:
        plt.show()
    plt.close(fig)
    gc.collect()

    if y_test is not None:
        # plot true labels
        fig2, ax2 = plt.subplots(figsize=(10, 10))
        correct = []
        i = 0
        for idx, true_label in y_test.items():
            if predictions[i] == true_label:
                correct.append(1)
            else:
                correct.append(0)
            i+=1
        num_classes = 2
        colors = ['tomato', 'cornflowerblue']
        classes = ['Incorrect','Correct']
        label_colors = {label: color for label, color in zip(correct, colors)}
        for label in range(num_classes):
            label_colors[label] = colors[label]
        cmap = ListedColormap(colors)
        scatter2 = ax2.scatter(X_test["Start"], X_test["End"], alpha=0.5, cmap=cmap, c=correct)

        ax2.text(.99, 0.5, f'n = {len(X_test["Start"])}', transform=ax2.transAxes, fontsize=14, verticalalignment='bottom',
                 horizontalalignment='right')
        ax2.set_xlim(0, 2400)
        ax2.set_ylim(0, 2400)
        ax2.set_xlabel(f'Start')
        ax2.set_ylabel(f'End')
        if multi_seg:
            ax2.set_title(f'{name} Dataset: Correctness of high Predictions')
        else:
            ax2.set_title(f'{name} Dataset: Correctness of Prediction')
        ax2.set_aspect('equal', 'box')
        ax2.legend(handles = [plt.Line2D([i], [i], marker='o', color='w', markerfacecolor=colors[i], markersize=10,
                          label=classes[i]) for i in range(num_classes)], title="Label Correctness", loc='lower right', fancybox=True, shadow=True, fontsize='large')
        plt.tight_layout()
        if result_path != "":
            path2 = os.path.join(result_path, f'{name}_Correctness_Scatter_StartVsEnd.png')
            plt.savefig(path2)
        else:
            plt.show()
        plt.close(fig2)
        gc.collect()

def decode_ohes(df):
    '''
    Decodes the one-hot-encoded columns in the dataframe. Returns the dataframe with the decoded columns.

    :param df: Dataframe containing the one-hot-encoded columns

    :return: Dataframe with decoded columns
    '''
    logger.debug(f'Decoding one-hot-encoded columns')
    features = []
    sequence_ohe, junction_ohe, strain_ohe, segment_ohe = False,False,False,False
    for col in df.columns:
        if any([f'1_{ch}' == col for ch in CHARS]):
            sequence_ohe = True
        elif "Start_" in col:
            junction_ohe = True
        elif any([strain == col for strain in ALL_STRAINS]):
            strain_ohe = True
        elif any([seg == col for seg in ALL_SEGMENTS]):
            segment_ohe = True
    if sequence_ohe:
        features.append("Sequence")
    if junction_ohe:
        features.append("Junction")
    if strain_ohe:
        features.append("Strain")
        df["Strain"] = df.apply(lambda x: [strain for strain in ALL_STRAINS if x[strain] == 1][0], axis=1)
    if segment_ohe:
        features.append("Segment")
        df["Segment"] = df.apply(lambda x: [seg for seg in ALL_SEGMENTS if x[seg] == 1][0], axis=1)

    return df, features


def make_barplot(df, name, result_path, feature, y_column, heatmap=False) -> None:
    ''' Creates a bar plot for the given feature. If heatmap=True, the function additionally creates a plot with each bar
    colored as a heatmap, according to the normed logarithm of the data's NGS count.

    :param df: Dataframe that includes the NGS_log_norm column and all columns that belong to the feature
    :param name: Name for the files to save each plot
    :param result_path: Path to save the plots
    :param feature: Feature to plot
    :param heatmap: Boolean to decide if a heatmap-barplot should be created

    :return: None
    '''
    os.makedirs(result_path, exist_ok=True)
    plt.rcParams.update({'font.size': 14})
    df_c = df
    sums = {}
    match feature:
        case 'Strain':
            all_groups = ALL_STRAINS
        case 'Segment':
            all_groups = ALL_SEGMENTS
        case _:
            logger.error(f'Feature {feature} not recognized for barplots.')
    for group in all_groups:
        sums[group] = 0
        try:
            sums[group] = sum(df_c[group])
        except:
            pass
    keys = sums.keys()
    values = sums.values()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.barh(keys, values)
    ax.set_title(f'{name}: {feature} distribution')
    ax.set_xlabel('Number of candidates')
    plt.tight_layout()
    if result_path != "":
        os.makedirs(result_path, exist_ok=True)
        plt.savefig(os.path.join(result_path, f'{name}_{feature}_bars.png'))
    else:
        plt.show()
    plt.close()
    gc.collect()

    if heatmap:
        cmap = plt.cm.viridis
        fig, ax = plt.subplots(figsize=(14, 6))
        try:
            for i, group in enumerate(all_groups):
                group_data = df[df[group]>0][y_column].value_counts().sort_index()
                df_t = group_data.reset_index()
                if len(group_data) == 0:
                    ax.barh(group, 0, edgecolor='none')
                    continue

                starting_value = 0
                for index, row in df_t.iterrows():
                    count = row['count']
                    color = cmap(row[y_column])
                    ax.barh(group, count, left=starting_value, color=color, edgecolor='none')
                    starting_value += count
        except:
            for i, group in enumerate(all_groups):
                group_data = df[df[feature]==group][y_column].value_counts().sort_index()
                df_t = group_data.reset_index()
                if len(group_data) == 0:
                    ax.barh(group, 0, edgecolor='none')
                    continue

                starting_value = 0
                for index, row in df_t.iterrows():
                    count = row['count']
                    color = cmap(row[y_column])
                    ax.barh(group, count, left=starting_value, color=color, edgecolor='none')
                    starting_value += count
        
        ax.set_xlabel('Proportion')
        ax.set_title(f'Distribution of NGS-Counts Across {feature}s')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', label='Normed NGS Read Count Log', aspect=50)

        ax.set_title(f'{name}: {feature} distribution')
        ax.set_xlabel('Number of candidates')
        plt.tight_layout()
        if result_path != "":
            os.makedirs(result_path, exist_ok=True)
            plt.savefig(os.path.join(result_path, f'{name}_{feature}_bars_heatmap.png'))
        else:
            plt.show()
        plt.close()
        gc.collect()


def make_histogram(df, name, feature, thresholds, result_path) -> None:
    '''
    Creates a histogram for the given feature. Marks the median and, for normed log of NGS count, any other given
    thresholds in the plot.

    :param df: Dataframe containing the feature to plot and the NGS_read_count column
    :param name: Name for the files to save each plot
    :param feature: Feature to plot
    :param thresholds: List of thresholds to mark in the plot
    :param result_path: Path to save the plots
    '''
    os.makedirs(result_path, exist_ok=True)
    logger.debug(f'Creating histogram for {name}: {feature}')
    plt.rcParams.update({'font.size': 14})
    df_c = df.copy()
    uniques = df_c[feature].nunique()
    if uniques < 20:
        bins = 2*uniques
    elif uniques < 150:
        bins = 40
    else:
        bins = 100
    logger.debug(f'bins are set to {bins}, based on number of uniques {uniques}')
    if feature not in ["Intersections", "Inter_norm"]:
        bins = int(uniques/3)
    else:
        bins = uniques
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df_c[feature], bins=bins, color='b', alpha=0.4)
    if "length_proportion" in df.columns:
        df_long_del = df_c[df_c["length_proportion"] >= 0.85].copy()
        ax.hist(df_long_del[feature], bins=bins, color='r', alpha=0.3, label='Long DelVGs')

    median = np.median(df[feature])
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

    if feature == 'NGS_log_norm' and thresholds is not None:
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
    ax.legend()
    ax.set_title(f'{name}: {feature.replace("_", " ")} Histogram')
    plt.tight_layout()
    if result_path != "":
        os.makedirs(result_path,exist_ok=True)
        plt.savefig(os.path.join(result_path, f'{name}_{feature}_hist.png'))
    else:
        plt.show()
    plt.close(fig)
    gc.collect()

def visualize_segments(df, name, y, thresholds, y_column, result_path) -> None:
    '''
    Creates plots for each segment present in the dataframe. Executes make_histogram and start vs end scatter plot for each
    segment, if applicable. Plots are saved in a subdirectory named after the segment.

    :param df: Dataframe containing the dataset
    :param name: Name of the dataset for save files
    :param y: List of labels for each datapoint
    :param thresholds: List of thresholds between classes
    :param y_column: Name of the target column, which was used for labeling
    :param result_path: Directory to save the plots

    :return: None
    '''
    os.makedirs(result_path, exist_ok=True)
    logger.debug(f'Beginning to visualize Segments.')
    for seg in ALL_SEGMENTS:
        df_c = df.copy()
        df_c['y'] = y
        if seg in df_c.columns:
            df_c = df_c[df_c[seg]>0]
        else:
            df_c = df_c[df_c['Segment']==seg]

        if df_c.shape[0] == 0:
            continue

        seg_path = os.path.join(result_path, seg)
        os.makedirs(seg_path, exist_ok=True)
        y_c = df_c['y']
        df_c.drop(columns=['y'], inplace=True)

        if 'Start' in df.columns and 'End' in df.columns:
            plot_startvsend(df, name, y, thresholds, y_column, seg_path)

        make_histogram(df, name, y_column, thresholds, seg_path)


def visualize_dataset(df, name, y, thresholds, y_column, features, result_path) -> None:
    '''
    Creates plots for the given dataset. Calls the functions make_startvsend to plot Start and End positions,
    make_barplot for Strains and Segments and make_histogram to plot the y_column.

    :param df: Dataframe containing the dataset
    :param name: Name of the dataset for save files
    :param y: List of labels for each datapoint
    :param thresholds: List of thresholds between classes
    :param y_column: Name of the target column, which was used for labeling
    :param features: List of features to plot
    :param result_path: Directory to save the plots

    :return: None
    '''
    os.makedirs(result_path, exist_ok=True)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    logger.debug(f'Visualizing dataset {name}')
    try:
        make_histogram(df, name, y_column, thresholds, result_path)
    except Exception:
        logger.error(f'Issue with histogram plot\n{traceback.format_exc()}')
    if ('Start' in features and 'End' in features) or ('Start' in df.columns and 'End' in df.columns):
        try:
            plot_startvsend(df, name, y, thresholds, y_column, result_path)
        except Exception:
            logger.error(f'Issue with histogram plot\n{traceback.format_exc()}')
    if ('Strain' in features or 'Strain' in df.columns) and df["Strain"].nunique()>1:
        try:
            make_barplot(df, name, result_path, 'Strain', y_column, heatmap=True)
        except Exception:
            logger.error(f'Issue with histogram plot\n{traceback.format_exc()}')
    if 'Segment' in features or 'Segment' in df.columns:
        try:
            make_barplot(df, name, result_path, 'Segment', y_column, heatmap=True)
        except Exception:
            logger.error(f'Issue with histogram plot\n{traceback.format_exc()}')
        try:
            visualize_segments(df, name, y, thresholds, y_column, result_path)
        except Exception as e:
            logger.error(f'Error in visualizing segments: {e}')


def rename_col_labels(label):
    '''
    Renames column labels for better readability in plots. Replaces certain substrings with more descriptive terms and formats the label.
    '''
    replacements = {
        "Celltype": "",
        "Host": "",
        ".0": "",
        "kmeans": "k-means",
        "hdbscan": "HDBSCAN",
        "comb0": "hybrid",
        "scaff0": "scaffold",
        "comb5": "hybrid",
        "scaff5": "scaffold",
        "comb10": "hybrid",
        "scaff10": "scaffold",
        "comb15": "hybrid",
        "scaff15": "scaffold",
        "Distance_to": "Dist",
        "centroid": "Centroid",
        "_": " ",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label.title()

def plot_multiclass_shap_summary(shap_values, X_test, class_names=None, max_display=20, result_path="", name_prefix="model"):
    '''
    Plots SHAP summary plots for each class in a multiclass classification problem.

    :param shap_values: SHAP values array of shape (n_samples, n_features, n_classes).
    :param X_test: Test dataset used for computing SHAP values.
    :param class_names: List of class names.
    :param max_display: Maximum number of features to display in the summary plot.
    :param result_path: Path to save the plots. If empty, plots will be shown instead.
    :param name_prefix: Prefix for the plot filenames.
    '''
    n_classes = shap_values.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    for i in range(n_classes):
        plt.figure(figsize=(8, 6))
        plt.title(class_names[i])
        shap.summary_plot(
            shap_values[:, :, i],
            X_test,
            max_display=max_display,
            show=False,
        )
        plt.xlabel("SHAP Value")
        new_labels = [rename_col_labels(label.get_text()) for label in plt.gca().get_yticklabels()]
        plt.gca().set_yticklabels(new_labels)
        plt.tight_layout()
        if result_path != "" and result_path is not None:
            os.makedirs(result_path, exist_ok=True)
            plt.savefig(os.path.join(result_path, f'{name_prefix}_shap_summary_{class_names[i]}.png'))
        else:
            plt.show()