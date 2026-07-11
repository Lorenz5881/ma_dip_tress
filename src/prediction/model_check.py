import argparse
import traceback
import warnings

from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, confusion_matrix, f1_score, make_scorer, matthews_corrcoef, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split

import argparse
import os.path
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

# Ensure imports work regardless of current working directory.
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR.parent
REPO_ROOT = BASE_DIR.parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, os.path.join(".."))
sys.path.insert(0, os.path.join(os.getcwd(), ".."))

from clf_utils import *
try:
    from utils import calculate_features, load_data, calculate_target, make_multiclass, apply_cutoff, cutoff_clean, drop_non_numeric, split_data, stratified_undersample, identify_candidates, transform_meta_features, UNPOOLED_DATA_DIR
    from utils import calculate_target, cutoff_clean, drop_non_numeric, get_sequence_quicker, make_multiclass
except ImportError:
    import glob
    logging.error("Failed to import from utils.py. Please ensure that utils.py is in the same directory as model_check.py or in the Python path.")
    paths_to_check = [str(BASE_DIR), str(SRC_DIR), os.path.join(os.getcwd(), "..")]
    logging.error(f"Checked the following paths for utils.py:\n{paths_to_check}")
    py_files = glob.glob(os.path.join(BASE_DIR, "*.py")) + glob.glob(os.path.join(SRC_DIR, "*.py")) + glob.glob(os.path.join(os.getcwd(), "..", "*.py"))
    logging.error(f"Python files found in those directories:\n{py_files}")
    logging.error("Current working directory: " + os.getcwd())
    logging.error(f'Python paths: {sys.path}')
    exit(1)
RESULT_PATH = str(REPO_ROOT / 'results' / "mcf")  # "model_check")
#RESULT_PATH = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'dev_results', "fix_unpooled"))
from main import test_classifiers, test_on_artificial

ALL_PUBS = ["Lui2019", "Kupke2020", "Penn2022", "Sheng2018", "Zhuravlev2020", "vdHoecke2015", "Boussier2020", "Southgate2019", "Valesano2020", "Mendes2021", "Alnaji2019", "Berry2021", "Alnaji2021", "Wang2020", "Wang2023", "Pelz2021"]
ALL_STRAINS = ["A_Anhui_1_2013", "A_California_07_2009", "A_Connecticut_Flu122_2013", "A_NewCaledonia_20-JY2_1999", "A_Perth_16_2009", "A_PuertoRico_8_1934", "A_turkey_Turkey_1_2005", "A_WSN_33", "B_Brisbane_60_2008", "B_Lee_1940", "B_Victoria_504_2000", "B_Yamagata_16_1988"]

strain_to_pubs = {'A_PuertoRico_8_1934': ['Kupke2020', 'Zhuravlev2020', 'VdHoecke2015', 'Alnaji2021', 'Wang2020', 'Wang2023', 'Pelz2021'],
                  'A_WSN_33': ['Boussier2020', 'Mendes2021'],
                  'B_Victoria_504_2000': ['Valesano2020', 'Berry2021'],
                  'B_Yamagata_16_1988': ['Southgate2019', 'Valesano2020', 'Berry2021']}
STRAIN_TO_PUBS = strain_to_pubs
STRAIN = 'A_PuertoRico_8_1934'
PUBS_TO_USE = ['Alnaji2021', 'Kupke2020', 'Pelz2021', 'vdHoecke2015', 'Wang2020', 'Wang2023', 'Zhuravlev2020']
STANDARD_FEATURES_DEFAULT = ['Segment', 'Start', 'End', 'Direct_repeat', 'Junction', 'remaining_length', 'deletion_length', '3_5_diff', '3_len', "5_len", 'length_proportion', 'Peptide_Length']


def load_preprocessed_data(strain, logger=logging):
    data_path = os.path.join(UNPOOLED_DATA_DIR, strain, "preprocessed_data.csv")
    col_dict_path = os.path.join(UNPOOLED_DATA_DIR, strain, "processed_column_dict.json")
    
    if not os.path.exists(data_path) or not os.path.exists(col_dict_path):
        logger.warning(f"Preprocessed data or column dictionary not found for strain {strain}. Expected at: {data_path} and {col_dict_path}")
        return None, None
    
    df = pd.read_csv(data_path)
    with open(col_dict_path, "r") as f:
        col_dict = json.load(f)
    
    return df, col_dict


def compute_shap_values(model, X_train, X_test, model_name, result_path, is_regression=False, bin_name="", logger=logging):
    '''
    Computes SHAP values for the given model (regressor or classifier) and saves the explainer, raw SHAP values, and summary plots to the specified result path. Automatically selects the appropriate SHAP explainer based on the model type and name, and handles both regression and classification scenarios.
    '''
    try:
        # Choose the right estimator (GridSearchCV vs direct model)
        estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model

        # Pick explainer
        if model_name.lower() in ["gradient_boost","adaboost","random_forest","xgb","lgb"]: #"xgb" in model_name.lower() or "lgb" in model_name.lower() or "forest" in model_name.lower():
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_test)
        elif model_name.lower() in ["linear","ridge","lasso","logistic_regression"]: #"linear" in model_name.lower() or "lasso" in model_name.lower() or "ridge" in model_name.lower() or "logistic_regression" in model_name.lower():
            explainer = shap.LinearExplainer(estimator, X_train)
            shap_values = explainer.shap_values(X_test)
        else:
            logger.info(f'SHAP values for model {model_name} may take more time.')
            # KernelExplainer (slower, but general)
            background = shap.sample(X_train, 100, random_state=0)
            predict_fn = estimator.predict if is_regression else estimator.predict_proba
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_test, nsamples=100)

        os.makedirs(result_path, exist_ok=True)
        try:
            # save shap explainer to file for later use
            #explainer.save(os.path.join(result_path, f"{model_name}{bin_name}_shap_explainer.pkl"))
            joblib.dump(explainer, os.path.join(result_path, f"{model_name}{bin_name}_shap_explainer.pkl"))
        except Exception as e:
            logger.error(f"Error saving SHAP explainer for {model_name}: {e}")


        # Save raw SHAP values for later analysis / custom plots.
        shap_arr = np.array(shap_values)
        try:
            np.save(os.path.join(result_path, f"{model_name}{bin_name}_shap_values.npy"), shap_arr)
            X_test.to_csv(os.path.join(result_path, f"{model_name}{bin_name}_shap_X_test.csv"), index=True)
        except Exception as e:
            logger.error(f"Error saving SHAP values for {model_name}: {e}")

        try:
            if shap_arr.ndim == 2:
                # Regression or binary classification — one value per sample per feature.
                pd.DataFrame(shap_arr, columns=X_test.columns, index=X_test.index).to_csv(
                    os.path.join(result_path, f"{model_name}{bin_name}_shap_values.csv"), index=True
                )
            elif shap_arr.ndim == 3:
                # Multiclass — shape (n_classes, n_samples, n_features).
                for class_idx in range(shap_arr.shape[0]):
                    pd.DataFrame(shap_arr[class_idx], columns=X_test.columns, index=X_test.index).to_csv(
                        os.path.join(result_path, f"{model_name}{bin_name}_shap_values_class{class_idx}.csv"), index=True
                    )
        except Exception as e:
            logger.error(f"Error saving SHAP values as CSV for {model_name}: {e}")

        # Summary plot (beeswarm)
        try:
            shap.summary_plot(shap_values, X_test, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(result_path, f"{model_name}{bin_name}_shap_summary.png"))
            plt.close()
        except Exception as e:
            logger.error(f"Error saving SHAP summary plot for {model_name}: {e}")

        # Bar plot (global importance)
        try:
            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(result_path, f"{model_name}{bin_name}_shap_bar.png"))
            plt.close()
        except Exception as e:
            logger.error(f"Error saving SHAP bar plot for {model_name}: {e}")

        logger.info(f"Saved SHAP values and plots for {model_name}.")
    except Exception as e:
        logger.error(f"Error during SHAP computation for {model_name}: {e}")


def save_model(model, name, path):
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path,f'{name}.pkl'))


def apply_pooling(dataframe, method, inplace=False, logger=logging):
    '''
    Pools NGS read counts of each ID per publication using the specified method (sum or mean).

    method options: "sum", "mean"
    '''
    df = dataframe if inplace else dataframe.copy()

    if method not in ("sum", "mean"):
        logger.error(f'Unknown method to pool NGS read counts: {method}')
        if not inplace:
            return df
        return

    behavior = {col: "first" for col in df.columns}
    behavior["NGS_read_count"] = method
    logger.debug(f'Pooling NGS read counts using {method}')
    df = df.groupby(["Publication", "ID"], as_index=False).agg(behavior)
    df = df.fillna(np.nan)
    df = df.reset_index(drop=True)

    return None if inplace else df


def normalize_by_length(dataframe, feature_columns, inplace=False, logger=logging):
    df = dataframe if inplace else dataframe.copy()
    if not "Full_Sequence" in df.columns:
        df = get_sequence_quicker(df)
    seq_len_dependent_features = ['Start', 'End', 'remaining_length', 'deletion_length', '3_5_diff', "3_len", "5_len"] # these features are likely to be dependent on the length of the peptide, so we can normalize them by the peptide length
    df = get_sequence_quicker(df)
    df["seq_len"] = df["Full_Sequence"].transform(len)
    for col in [c for c in feature_columns if c in seq_len_dependent_features]:
        if col in df.columns:
            df[col] = df[col] / df["seq_len"]
    df.drop(columns=["Full_Sequence", "seq_len"], inplace=True, errors="ignore")
    return None if inplace else df


def scale_features(dataframe, feature_columns, inplace=False, logger=logging):
    df = dataframe if inplace else dataframe.copy()
    features_to_scale = []
    for col in feature_columns:
        if col not in df.columns:
            logger.warning(f'Feature column {col} not found in dataframe. Skipping scaling for this column.')
            continue

        if not df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            continue
        if df[col].isnull().all():
            continue
        if df[col].nunique() <= 2:
            continue
        features_to_scale.append(col)
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    return None if inplace else df


def train_regressor(X_train, y_train, X_test, y_test, perform_grid_search: bool=True,
                    seed: int = 42, result_path=None, reg_name="random_forest", logger=logging):
    os.makedirs(result_path, exist_ok=True)
    results = dict()
    results["metric"] = ["MSE", "MAE", "R2"]
    models_path = os.path.join(result_path, "Models")
    grid_search_results = {}
    logging.debug(f'Going to train on columns: {list(X_train.columns)}')
    regressor, param_grid = select_regressor(reg_name, perform_grid_search)

    logging.info(f'Training regressor: {reg_name}')
    results[reg_name] = [None,None,None]

    regressor, param_grid = select_regressor(reg_name, perform_grid_search)
    if regressor == "unknown regressor":
        raise ValueError(f"Unknown regressor specified: {reg_name}. Cannot train.")
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    grid = GridSearchCV(regressor, param_grid, scoring='neg_mean_squared_error',
                        refit=True, cv=kf, return_train_score=True)
    grid.fit(X_train, y_train)

    # Save grid search results for this regressor
    cv_results_dict = {}
    for k, v in grid.cv_results_.items():
        try:
            cv_results_dict[k] = v.tolist() if hasattr(v, 'tolist') else v
        except Exception:
            cv_results_dict[k] = str(v)
    try:
        cv_results_dict["best_params"] = grid.best_params_
    except Exception:
        cv_results_dict["best_params"] = str(grid.best_params_)
    grid_search_results[reg_name] = cv_results_dict

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    try:
        save_model(best_model, f'reg_{reg_name}', models_path)
    except Exception:
        logger.error(traceback.format_exc())
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    results[reg_name]= [mse, mae, r2]

    # Plot actual vs predicted
    try:
        plot_actual_vs_pred(y_test, y_pred, reg_name, os.path.join(result_path))
    except Exception as e:
        logger.error(f'Error plotting regression predictions:\n{e}')
    # Save all grid search results to a JSON file
    try:
        json_path = os.path.join(result_path, 'grid_search_results_regressors.json')
        with open(json_path, 'w') as f:
            json.dump(grid_search_results, f, indent=2)
        logger.info(f"Saved grid search results for all regressors to {json_path}")
    except Exception as e:
        logger.error(f"Failed to save grid search results for regressors: {e}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(result_path, 'regression_results.csv'), index=False)
    try:
        with open(os.path.join(result_path, f'regression_{reg_name}_training_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        logger.error(f'Error saving regression training results as json:\n{e}')
    return best_model, results


def train_classifier(X_train, y_train, X_test, y_test, perform_grid_search: bool, n_bins: int = 2,
                     seed: int = 42, result_path=None, clf_name="random_forest", logger=logging):
    os.makedirs(result_path, exist_ok=True)
    data_dict = dict()
    data_dict["param"] = ["accuracy"]
    results = dict()
    results["metric"] = ["accuracy", "F1-Score", "MCC"]
    f1_dict = dict()
    f1_dict["param"] = ["F1-Score"]
    model_dict = dict()
    models_path = os.path.join(result_path, "Models")
    bin_name = "" if n_bins == 2 else "_" + str(n_bins) + "bins"
    os.makedirs(models_path, exist_ok=True)

    grid_search_results = {}

    results[clf_name] = [None,None,None]
    # Setup for fitting (k-fold and grid search)
    logger.info(f'Training classifier: {clf_name}')
    data_dict[clf_name] = list()
    clf, param_grid = select_classifier(clf_name, perform_grid_search)
    if clf == "unknown classifier":
        raise ValueError(f"Unknown classifier specified: {clf_name}. Cannot train.")
    skf = StratifiedKFold(n_splits=5)
    scorers = {"accuracy_score": make_scorer(accuracy_score)}
    grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit='accuracy_score', cv=skf,
                            return_train_score=True)
    # removing 0 in k-vary for clfs that can handle empty features
    if clf_name in ("xgb", "lgb"):
        X_train_fit = X_train.copy()
        X_test_fit = X_test.copy()
        for col in X_train_fit.columns:
            if "k_vary_" in col:
                X_train_fit[col] = X_train_fit[col].replace(0, np.nan)
                X_test_fit[col] = X_test_fit[col].replace(0, np.nan)
    else:
        X_train_fit = X_train
        X_test_fit = X_test

    # Fitting
    grid_search.fit(X_train, y_train)

    # Save grid search results for this classifier
    cv_results_dict = {}
    for k, v in grid_search.cv_results_.items():
        try:
            cv_results_dict[k] = v.tolist() if hasattr(v, 'tolist') else v
        except Exception:
            cv_results_dict[k] = str(v)
    try:
        cv_results_dict["best_params"] = grid_search.best_params_
    except Exception:
        cv_results_dict["best_params"] = str(grid_search.best_params_)
    grid_search_results[clf_name] = cv_results_dict

    # Extracting cross-validation results
    cv_results = grid_search.cv_results_['mean_test_accuracy_score']
    cv_std = grid_search.cv_results_['std_test_accuracy_score']
    logger.info(f"Cross-validation results:  scores - mean: {np.mean(cv_results)}\tstandard deviation - mean: {np.mean(cv_std)}")
    if np.mean(cv_std) > 0.05:
        logger.warning(f'High standard deviation in cross-validation! Mean: {np.mean(cv_std)}')
    best_index = grid_search.best_index_
    logger.info(f'best run - score: {cv_results[best_index]}\tstd: {cv_std[best_index]}')

    logger.info(f"training accuracy: {grid_search.best_score_}")

    # Test set evaluation
    if perform_grid_search:
        logger.info(f"best params: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_
    predicted_val = best_model.predict(X_test)
    try:
        save_model(best_model, f'clf_{clf_name}{bin_name}', models_path)
    except Exception:
        logger.error(traceback.format_exc())
    os.makedirs(result_path, exist_ok=True)
    if {"Start", "End"}.issubset(set(X_test.columns)):
        try:
            plot_prediction_startvsend(X_test, y_test, predicted_val, clf_name, result_path)
        except Exception as e:
            logger.error(f'Error during prediction plotting: {e}\n{traceback.format_exc()}')
    else:
        logger.debug("Skipping Start/End prediction plot because required columns are missing.")
    acc_score = accuracy_score(y_test, predicted_val)
    average = "macro" if len(set(y_test + predicted_val)) != 2 or n_bins != 2 else "binary"
    f1 = f1_score(y_test, predicted_val, average=average)
    confus_matrix = confusion_matrix(y_test, predicted_val)
    mcc = matthews_corrcoef(y_true=y_test, y_pred=predicted_val)
    logger.info(f"Test accuracy:\t{acc_score}\nF1-Score:\t{f1}\nConfusion Matrix:\n{confus_matrix}\nMatthews correlation coefficient: {mcc}")
    results[clf_name] = [acc_score,f1,mcc]
    try:
        disp = ConfusionMatrixDisplay.from_predictions(y_test, predicted_val)
        disp.plot()
        plt.savefig(os.path.join(result_path, f'{clf_name}{bin_name}_confusion.png'))
        plt.close()
    except Exception as e:
        logger.error(f'Issue with confusion matrix:\n{e}\n{traceback.format_exc()}')
    data_dict[clf_name].append(acc_score)
    f1_dict[clf_name] = f1
    model_dict[clf_name] = best_model

    # ROC curve for binary classification
    if n_bins == 2:
        plt.rc('font', size=14)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
        shuffle = y_test.sample(frac=1, random_state=seed, ignore_index=True).to_numpy()
        RocCurveDisplay.from_estimator(best_model, X_test, y_test, name=clf_name, ax=ax)
        RocCurveDisplay.from_estimator(best_model, X_test, shuffle, name="shuffled", ax=ax)
        plt.plot([0, 1], [0, 1])
        path = os.path.join(result_path, f'{clf_name}_ROC.png')
        plt.savefig(path)
        logger.info(f'Saving ROC curve for {clf_name} and shuffled data at {path}.')
        plt.close()
    else:
        shuffle = y_test.sample(frac=1, random_state=seed, ignore_index=True).to_numpy()
        shuffle_acc = accuracy_score(shuffle, predicted_val)
        logger.info(f'Test accuracy after shuffling: {shuffle_acc}')

    # Save all grid search results to a JSON file
    try:
        json_path = os.path.join(result_path, 'grid_search_results.json')
        with open(json_path, 'w') as f:
            json.dump(grid_search_results, f, indent=2)
        logger.info(f"Saved grid search results for all classifiers to {json_path}")
    except Exception as e:
        logger.error(f"Failed to save grid search results: {e}")
    '''try:
        final_df = pd.DataFrame(data_dict)
        final_df = final_df.drop(['param'], axis=1)
        final_df["mean"] = final_df.mean(axis=1)
        f1_df = pd.DataFrame(f1_dict)
        f1_df = f1_df.drop(['param'], axis=1)
        f1_df["mean"] = f1_df.mean(axis=1)
        logging.info(f'Accuracies across classifiers:\n{final_df}\n{f1_df}')
        final_df.to_csv(os.path.join(result_path, f'results_{n_bins}.csv'), index=False)
        f1_df.to_csv(os.path.join(result_path,f'f1_scores_{n_bins}.csv'), index=False)
    except Exception as e:
        logging.error(f'Issue compiling results old way:\n{e}\n{traceback.format_exc()}')'''

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(result_path, f'classifier_{n_bins}_results.csv'), index=False)
    try:
        with open(os.path.join(result_path, f'classifier_{n_bins}_training_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        logger.error(f'Error saving training results as json:\n{e}')
    return best_model, results


def test_regressor(model, X_test, y_test, name, result_path, logger=logging):
    y_pred = model.predict(X_test)
    try:
        plot_actual_vs_pred(y_true=y_test, y_pred=y_pred, model_name=name, output_dir=result_path)
    except Exception as e:
        logger.error(f'Error plotting regression predictions:\n{e}')

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    return mse, mae, r2


def test_classifier(model, X_test, y_test, name, result_path, n_bins=2, labels=None, logger=logging):
    predicted = model.predict(X_test)
    if {"Start", "End"}.issubset(set(X_test.columns)):
        try:
            os.makedirs(result_path, exist_ok=True)
            plot_prediction_startvsend(X_test=X_test, y_test=y_test, predictions=predicted, name=name, result_path=result_path)
        except Exception as e:
            logger.error(f'Error during prediction plotting: {e}\n{traceback.format_exc()}')
    else:
        logger.debug("Skipping Start/End prediction plot because required columns are missing.")
    acc_score = accuracy_score(y_test, predicted)
    average = "macro" if len(set(y_test + predicted)) != 2 or n_bins != 2 else "binary"
    f1 = f1_score(y_test, predicted, average=average)
    confus_matrix = confusion_matrix(y_test, predicted, labels=labels)
    mcc = matthews_corrcoef(y_true=y_test, y_pred=predicted)
    logger.info(f"Test accuracy:\t{acc_score}\nF1-Score:\t{f1}\nConfusion Matrix:\n{confus_matrix}\nMatthews correlation coefficient: {mcc}")
    try:
        disp = ConfusionMatrixDisplay.from_predictions(y_test, predicted)
        disp.plot()
        plt.savefig(os.path.join(os.path.join(result_path), f'{name}_confusion.png'))
        plt.close()
    except Exception as e:
        logger.error(f'Issue with confusion matrix: {e}\n{traceback.format_exc()}')
    return acc_score, f1, mcc

def drop_non_finite(dataframe, inplace=False, logger=logging):
    '''
    Drops all numeric columns that hold non-finite values (NaN, inf, -inf) from the dataframe.
    '''

    df = dataframe if inplace else dataframe.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_finite_columns = df[numeric_cols].apply(lambda x: ~np.isfinite(x)).any()
    if non_finite_columns.any():
        logger.error(f"Of {len(df.columns)} columns, the following {non_finite_columns.sum()} had non-finite values:\n{non_finite_columns[non_finite_columns].index.tolist()}")
    df = df.drop(columns=non_finite_columns[non_finite_columns].index.tolist(), errors="ignore")
    if df.empty:
        raise ValueError("Dataframe is empty after dropping non-finite values. Cannot proceed with training.")

    return None if inplace else df

def drop_nan_columns(dataframe, inplace=False, logger=logging):
    '''
    Drops all columns that contain any NaN values from the dataframe, and logs which columns were dropped. If all columns are dropped, raises an error to prevent training on an empty dataset.
    '''
    df = dataframe if inplace else dataframe.copy()
    nan_columns = df.columns[df.isna().any()].tolist()
    if nan_columns:
        logger.error(f"The following columns contain NaN values and will be dropped:\n{nan_columns}")
    df = df.drop(columns=nan_columns, errors="ignore")
    if df.empty:
        raise ValueError("Dataframe is empty after dropping NaN columns. Cannot proceed with training.")
    return None if inplace else df
    
def undersample_by_quantile(dataframe, target_column, n_bins=3, quantiles=None, random_state=42, logger=logging):
    '''
    Performs stratified undersampling of the dataframe based on quantiles of the target column. The target column is binned into n_bins using the specified quantiles, and then undersampling is performed to balance the number of samples in each bin. If quantiles are not provided, they are calculated automatically. Returns a new dataframe that has been undersampled according to the specified quantile bins.
    '''
    df = dataframe.copy()
    if quantiles is None:
        quantiles = np.linspace(0, 1, n_bins + 1)
    try:
        df["quantile_bin"] = pd.qcut(df[target_column], q=quantiles, labels=False, duplicates='drop')
    except Exception as e:
        logger.error(f"Error during quantile binning: {e}")
        raise
    undersampled_indices = []
    for bin in df["quantile_bin"].unique():
        bin_indices = df[df["quantile_bin"] == bin].index
        if len(bin_indices) == 0:
            continue
        undersampled_bin_indices = np.random.choice(bin_indices, size=min(len(bin_indices), len(df) // n_bins), replace=False)
        undersampled_indices.extend(undersampled_bin_indices)
    undersampled_df = df.loc[undersampled_indices].copy()
    #undersampled_df = stratified_undersample(df, "quantile_bin", random_state=random_state)
    undersampled_df = undersampled_df.drop(columns=["quantile_bin"], errors="ignore")
    logger.info(f"Performed undersampling based on quantiles of {target_column}. Original size: {len(dataframe)}, undersampled size: {len(undersampled_df)}.")
    return undersampled_df

def prepare_data(df,
                 col_dict,
                 test_type="single_dataset",
                 model_type="reg",
                 model_name="random_forest",
                 strain="A_PuertoRico_8_1934",
                 cutoff=15,
                 pub_id=1,
                 package="benchmark",
                 y_column="NGS_log_norm",
                 pooling=None,
                 drop_intersecting=False,
                 logger=logging,
                 **kwargs):
    '''# Load preprocessed data
    df, col_dict = load_preprocessed_data(strain)
    if df is None or col_dict is None:
        logger.error(f"Preprocessed data not found for strain {strain}. Cannot train model.")
        return
    if pub_id not in df["pub_id"].unique():
        logger.error(f"Publication ID {pub_id} not found in data for strain {strain}. Available pub_ids: {df['pub_id'].unique()}. Cannot train model.")
        return

    # Define the results directory
    results_dir = os.path.join(RESULT_PATH, test_type, strain, package, f"{pooling if pooling is not None else 'unpooled'}_{'drop' if drop_intersecting else 'keep'}_{cutoff}_{df[df['pub_id']==pub_id]['Publication'].iloc[0]}")
    os.makedirs(results_dir, exist_ok=True)
    logger.addHandler(logging.FileHandler(os.path.join(results_dir, "results.log")))
    logger.info(f"Starting for strain {strain}, publication ID {pub_id}={df[df['pub_id']==pub_id]['Publication'].iloc[0]}, cutoff {cutoff}, package {package}, pooling {pooling}, drop_intersecting {drop_intersecting}.")'''

    # apply per-publication pooling if specified
    if pooling is not None:
        logger.debug(f'before pooling: {df.shape}\n{df}')
        df = apply_pooling(dataframe=df, method=pooling, logger=logger)
        available_meta = [col for col in col_dict["meta"] if col in df.columns]
        removed_constant_meta = [col for col in available_meta if df[col].nunique(dropna=False) <= 1]
        if removed_constant_meta:
            df.drop(removed_constant_meta, axis=1, errors="ignore", inplace=True)
            logger.info(f'Removed constant meta columns after pooling: {removed_constant_meta}')
        logger.debug(f'after pooling: {df.shape}\n{df}')

    if test_type == "intersection": # intersection test is independent of meta features
        effective_meta_columns = []
    if test_type == "single_dataset" and pooling is not None: # if pooling is applied, meta features are meaningless for single_dataset test
        effective_meta_columns = []
    effective_meta_columns = [col for col in col_dict["meta"] if col in df.columns]
    effective_meta_columns = [col for col in effective_meta_columns if df[col].nunique(dropna=False) > 1] # remove constant meta columns
    if len(effective_meta_columns) != len(col_dict["meta"]):
        logger.info(f'Effective meta columns after pooling/pruning: {effective_meta_columns}')

    if drop_intersecting:
        df = df[df["Intersecting"] == False].copy().reset_index(drop=True)

    # Select feature columns based on package
    if package == "benchmark":
        feature_columns = col_dict["standard"] + effective_meta_columns
    elif package == "relational":
        cutoff_clust_cols = [col for col in col_dict["clustering"] if f'comb{cutoff}_' in col or f'scaff{cutoff}_' in col]
        feature_columns = col_dict["standard"] + effective_meta_columns + cutoff_clust_cols #col_dict["clustering"]
    elif package == "context":
        cutoff_clust_cols = [col for col in col_dict["clustering"] if f'comb{cutoff}_' in col or f'scaff{cutoff}_' in col]
        feature_columns = col_dict["standard"] + effective_meta_columns + cutoff_clust_cols + col_dict["context"]
    else:
        logger.error(f"Invalid package specified: {package}. Must be one of 'benchmark', 'relational', or 'context'.")
        return
    feature_columns = feature_columns + [col for col in STANDARD_FEATURES_DEFAULT if col in df.columns and col not in feature_columns] # ensure all standard features are included
    feature_columns = [col for col in feature_columns if col in df.columns] # ensure that all selected features are actually present in the dataframe
    
    df = cutoff_clean(df, cutoff)
    
    # normalize and scale feature values
    df = normalize_by_length(df, feature_columns, logger=logger)
    df = scale_features(df, feature_columns, logger=logger)

    if not nan_compatible(model_name):
        logger.info(f'Model {model_name} is not compatible with NaN values, dropping NaN columns...')
        feature_columns = drop_nan_columns(df[feature_columns], logger=logger).columns.tolist()

    # split into training and testing datasets: for single_dataset, train on pub_id and test on the rest. For leave_one_out, train on everything except pub_id and test on pub_id.
    if test_type == "single_dataset":
        base_df = df[df["pub_id"] == pub_id].copy().reset_index(drop=True)
        cross_df = df[df["pub_id"] != pub_id].copy().reset_index(drop=True)
    elif test_type == "leave_one_out":
        base_df = df[df["pub_id"] != pub_id].copy().reset_index(drop=True)
        cross_df = df[df["pub_id"] == pub_id].copy().reset_index(drop=True)
    elif test_type == "intersection":
        base_df = df.drop_duplicates("ID").copy().reset_index(drop=True)
        cross_df = base_df.copy()
    else:
        logger.error(f"Invalid test type specified: {test_type}. Must be one of 'single_dataset' or 'leave_one_out'.")
        return
    
    # remove columns with constant values in base_df from both base_df and cross_df, since they won't contribute to model training and may cause issues with some models
    constant_cols = [col for col in base_df.columns if base_df[col].nunique(dropna=False) <= 1]
    logger.info(f'Removing constant columns from base_df and cross_df: {constant_cols}')
    feature_columns = [col for col in feature_columns if col not in constant_cols]


    if base_df.empty or len(base_df) < 10: # if there are less than 10 samples left after cutoff, it's unlikely we can train a meaningful model
        raise ValueError(f"Not enough training data left after applying cutoff for strain {strain}, pub_id {pub_id}, cutoff {cutoff}. Cannot train model.")
    
    cross_df = cutoff_clean(cross_df, cutoff)
    if test_type != "intersection":
        base_df = calculate_target(base_df, y_col=y_column)

    X_base = base_df[feature_columns]

    if X_base.empty:
        logger.error(f"Training or validation data is empty after applying cutoff for strain {strain}, pub_id {pub_id}, cutoff {cutoff}. Cannot train model.")
        return

    if model_type == "reg":
        if test_type == "intersection":
            y_column = "Num_Publications"
            y_base = base_df[y_column]/base_df[y_column].max() # min-max normalization to keep target values between 0 and 1, which can help with training stability for regression models
        else:
            # apply undersampling by quantiles to regression target to ensure that training and testing sets have similar distributions of the target variable, which can help with model generalization, especially when the target variable has a skewed distribution
            undersampled_base_df = undersample_by_quantile(base_df, target_column=y_column, n_bins=10, logger=logger)
            y_base = undersampled_base_df[y_column]
            X_base = undersampled_base_df[feature_columns]
    elif model_type == "clf":
        if test_type == "intersection":
            y_column = "Num_Publications"
            # threshold is half the number of unique publications (rounded up), but at least 2
            threshold = max(2, int(np.ceil(len(STRAIN_TO_PUBS[strain])/2)))
            # classification with one class per publication count, with any count above the threshold considered as single category
            y_base = base_df[y_column].apply(lambda x: x if x < threshold else threshold).astype(int)
            logger.info(f'For classification in intersection test, using publication count threshold of {threshold}. Classes will be: {y_base.unique()} with {y_base.max()} as the maximum class representing intersecting DelVGs.')

            # if classes are imbalanced, undersample the majority class to have at most 4 times more samples than the minority class
            if y_base.value_counts().max() > 4*y_base.value_counts().min():
                majority_count = y_base.value_counts().max()
                minority_count = y_base.value_counts().min()
                undersample_count = min(majority_count, 4*minority_count)
                # ensure that each class is at most 4 times more prevalent than the minority class
                undersampled_indices = []
                for cls in y_base.unique(): # ensuring that each class is represented in the undersampled dataset, but if a class has more than 4 times the samples of the minority class, we only take a random subset of it
                    cls_count = y_base.value_counts()[cls]
                    if cls_count > 4*minority_count:
                        undersample_count = min(undersample_count, 4*minority_count)
                    undersampled_indices.append(np.random.choice(y_base[y_base == cls].index, undersample_count, replace=False) if cls_count > 4*minority_count else y_base[y_base == cls].index)
                undersampled_indices = np.concatenate(undersampled_indices)
                X_base = X_base.loc[undersampled_indices].reset_index(drop=True)
                y_base = y_base.loc[undersampled_indices].reset_index(drop=True)
        else:
            y_base, thresholds = make_multiclass(base_df, y_column=y_column, n_bins=2)
    else:
        logger.error(f"Invalid model type specified: {model_type}. Must be one of 'reg' or 'clf'.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X_base, y_base, test_size=0.2, random_state=42)

    # ensure that X contains only numeric values for model training
    X_train = drop_non_numeric(X_train)
    X_test = drop_non_numeric(X_test)

    '''try:
        # ensure that X contains only finite values for model training
        X_train = drop_non_finite(X_train, logger=logger)
        X_test = drop_non_finite(X_test, logger=logger)
    except Exception as e:
        logger.error(f'Error during dropping non-finite values:\n{e}\n{traceback.format_exc()}')
        return'''

    return X_train, X_test, y_train, y_test, cross_df, feature_columns

def calculate_deltas(training_results, testing_results, logger=logging):
    '''
    Calculate the difference between training and testing accuracies.
    The dictionaries for classifiers are formatted as {"metric": ["accuracy", "F1-Score", "MCC"], "classifier1": [train_acc, train_f1, train_mcc], "classifier2": [train_acc, train_f1, train_mcc], ...}
    The dictionaries for regressors are formatted as {"metric": ["MSE", "MAE", "R2"], "regressor1": [train_mse, train_mae, train_r2], "regressor2": [train_mse, train_mae, train_r2], ...}
    '''
    deltas = {}
    for model in training_results:
        if model == "metric":
            continue
        if model not in testing_results:
            logger.warning(f'Model {model} found in training results but not in testing results. Skipping delta calculation for this model.')
            continue
        deltas[model] = []
        for train_metric, test_metric in zip(training_results[model], testing_results[model]):
            if train_metric is None or test_metric is None:
                deltas[model].append(None)
            else:
                deltas[model].append(test_metric - train_metric)
    return deltas

def nan_compatible(model_name):
    is_nan_compatible = {"random_forest": True, 
                         "xgb": True,
                         "lgb": True,
                         "catboost": True,
                         "logistic_regression": False,
                         "linear": False,
                         "ridge": False,
                         "lasso": False}
    return is_nan_compatible.get(model_name, False)


def train_and_evaluate_model(test_type="single_dataset",
                             model_type="reg",
                             model_name="random_forest",
                             strain="A_PuertoRico_8_1934",
                             cutoff=15,
                             pub_id=1,
                             package="benchmark",
                             y_column="NGS_log_norm",
                             pooling=None,
                             drop_intersecting=False,
                             logger=logging,
                             **kwargs):
    # Load preprocessed data
    df, col_dict = load_preprocessed_data(strain)
    if df is None or col_dict is None:
        logger.error(f"Preprocessed data not found for strain {strain}. Cannot train model.")
        return
    if pub_id not in df["pub_id"].unique():
        logger.error(f"Publication ID {pub_id} not found in data for strain {strain}. Available pub_ids: {df['pub_id'].unique()}. Cannot train model.")
        return

    # Define the results directory
    results_dir = os.path.join(RESULT_PATH, test_type, strain, package, f"{pooling if pooling is not None else 'unpooled'}_{'drop' if drop_intersecting else 'keep'}_{cutoff}_{df[df['pub_id']==pub_id]['Publication'].iloc[0]}")
    os.makedirs(results_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(results_dir, f"{model_type}_results.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    start_time = datetime.datetime.now()
    logger.info(f"Starting for strain {strain}, publication ID {pub_id}={df[df['pub_id']==pub_id]['Publication'].iloc[0]}, cutoff {cutoff}, package {package}, pooling {pooling}, drop_intersecting {drop_intersecting}.")
    
    '''
    # apply per-publication pooling if specified
    if pooling is not None:
        logging.debug(f'before pooling: {df.shape}\n{df}')
        df = apply_pooling(dataframe=df, method=pooling)
        available_meta = [col for col in col_dict["meta"] if col in df.columns]
        removed_constant_meta = [col for col in available_meta if df[col].nunique(dropna=False) <= 1]
        if removed_constant_meta:
            df.drop(removed_constant_meta, axis=1, errors="ignore", inplace=True)
            logging.info(f'Removed constant meta columns after pooling: {removed_constant_meta}')
        logging.debug(f'after pooling: {df.shape}\n{df}')

    effective_meta_columns = [col for col in col_dict["meta"] if col in df.columns]
    if removed_constant_meta:
        logging.info(f'Effective meta columns after pooling/pruning: {effective_meta_columns}')

    if drop_intersecting:
        df = df[df["Intersecting"] == False].copy().reset_index(drop=True)

    # Select feature columns based on package
    if package == "benchmark":
        feature_columns = col_dict["standard"] + effective_meta_columns
    elif package == "relational":
        feature_columns = col_dict["standard"] + effective_meta_columns + col_dict["vip"] + col_dict["clustering"]
    elif package == "context":
        feature_columns = col_dict["standard"] + effective_meta_columns + col_dict["vip"] + col_dict["clustering"] + col_dict["context"]
    else:
        logger.error(f"Invalid package specified: {package}. Must be one of 'benchmark', 'relational', or 'context'.")
        return
    feature_columns = feature_columns + [col for col in STANDARD_FEATURES_DEFAULT if col in df.columns and col not in feature_columns] # ensure all standard features are included

    # normalize and scale feature values
    df = normalize_by_length(df, feature_columns)
    df = scale_features(df, feature_columns)

    # split into training and testing datasets: for single_dataset, train on pub_id and test on the rest. For leave_one_out, train on everything except pub_id and test on pub_id.
    if test_type == "single_dataset":
        base_df = df[df["pub_id"] == pub_id].copy().reset_index(drop=True)
        cross_df = df[df["pub_id"] != pub_id].copy().reset_index(drop=True)
    elif test_type == "leave_one_out":
        base_df = df[df["pub_id"] != pub_id].copy().reset_index(drop=True)
        cross_df = df[df["pub_id"] == pub_id].copy().reset_index(drop=True)
    else:
        logger.error(f"Invalid test type specified: {test_type}. Must be one of 'single_dataset' or 'leave_one_out'.")
        return

    base_df = cutoff_clean(base_df, cutoff)
    if base_df.empty or len(base_df) < 10: # if there are less than 10 samples left after cutoff, it's unlikely we can train a meaningful model
        raise ValueError(f"Not enough training data left after applying cutoff for strain {strain}, pub_id {pub_id}, cutoff {cutoff}. Cannot train model.")
    cross_df = cutoff_clean(cross_df, cutoff)

    base_df = calculate_target(base_df, y_col=y_column)

    X_base = base_df[feature_columns]

    if X_base.empty:
        logger.error(f"Training or validation data is empty after applying cutoff for strain {strain}, pub_id {pub_id}, cutoff {cutoff}. Cannot train model.")
        return

    if model_type == "reg":
        y_base = base_df[y_column]
    elif model_type == "clf":
        y_base, thresholds = make_multiclass(base_df, y_column=y_column)
    else:
        logger.error(f"Invalid model type specified: {model_type}. Must be one of 'reg' or 'clf'.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X_base, y_base, test_size=0.2, random_state=42)

    X_train = drop_non_numeric(X_train)
    X_test = drop_non_numeric(X_test)'''
    # datasplit and processing for later tests
    X_train, X_test, y_train, y_test, cross_df, feature_columns = prepare_data(df=df,
                                                                               col_dict=col_dict,
                                                                               test_type=test_type,
                                                                               model_type=model_type,
                                                                               model_name=model_name,
                                                                               strain=strain,
                                                                               cutoff=cutoff,
                                                                               pub_id=pub_id,
                                                                               package=package,
                                                                               y_column=y_column,
                                                                               pooling=pooling,
                                                                               drop_intersecting=drop_intersecting,
                                                                               logger=logger)

    logger.info(f"Data preparation completed. Columns: {len(feature_columns)}, Training samples: {len(X_train)}, Testing samples: {len(X_test)}, Cross-publication samples: {len(cross_df)}")
    if model_type == "reg":
        model, training_results = train_regressor(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, perform_grid_search=True, seed=42, result_path=results_dir, reg_name=model_name, logger=logger)
    else:
        model, training_results = train_classifier(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, perform_grid_search=True, seed=42, n_bins=len(set(y_train)), result_path=results_dir, clf_name=model_name, logger=logger)
    train_end_time = datetime.datetime.now()
    logger.info(f"Training completed. Time taken: {train_end_time - start_time}")
    # test on the cross dataset
    cross_df = calculate_target(cross_df, y_col=y_column)
    results = {}
    if model_type == "reg":
        results["metric"] = ["MSE", "MAE", "R2"]
    elif model_type == "clf":
        results["metric"] = ["accuracy", "F1-Score", "MCC"]
    try:
        for pub in cross_df["Publication"].unique():
            pub_cross_df = cross_df[cross_df["Publication"] == pub]
            if pub_cross_df.empty:
                continue
            pub_res = dict()
            X_pub_cross = pub_cross_df[feature_columns]
            if model_type == "reg":
                if test_type == "intersection":
                    y_column = "Num_Publications"
                    threshold = max(2, int(np.ceil(len(STRAIN_TO_PUBS[strain])/2)))
                    y_pub_cross = pub_cross_df[y_column].apply(lambda x: x if x < threshold else threshold).astype(int)/threshold 
                    #y_pub_cross = pub_cross_df[y_column]/pub_cross_df[y_column].max() 
                else:
                    y_pub_cross = pub_cross_df[y_column]
            elif model_type == "clf":
                if test_type == "intersection":
                    y_column = "Num_Publications"
                    threshold = max(2, int(np.ceil(len(STRAIN_TO_PUBS[strain])/2)))
                    y_pub_cross = pub_cross_df[y_column].apply(lambda x: x if x < threshold else threshold).astype(int)
                    #threshold = max(2, df["Publication"].nunique()/2)
                    #y_pub_cross = (pub_cross_df[y_column] >= threshold).astype(int) 
                else:
                    y_pub_cross, pub_cross_thresholds = make_multiclass(pub_cross_df, y_column=y_column, n_bins=2)
            try:
                X_pub_cross = drop_non_numeric(X_pub_cross)
                #X_pub_cross = drop_non_finite(X_pub_cross, logger=logger)
                #y_pub_cross = drop_non_numeric(y_pub_cross)
                if X_pub_cross.empty or y_pub_cross.empty:
                    continue
                if model_type == "reg":
                    results[pub] = test_regressor(model, X_pub_cross, y_pub_cross, name=f'{model_name}_{pub}', result_path=results_dir, logger=logger)
                else:
                    results[pub] = test_classifier(model, X_pub_cross, y_pub_cross, name=f'{model_name}_{pub}', result_path=results_dir, n_bins=len(set(y_train)), labels=sorted(list(set(y_train))), logger=logger)
            except Exception as e:
                logger.error(f'Error during testing on publication {pub}:\n{e}\n{traceback.format_exc()}')
    except Exception as e:
        logger.error(f'Error during cross-publication testing:\n{e}\n{traceback.format_exc()}')
    
    
    try:
        json_path = os.path.join(results_dir, f'{model_type}_cross_publication_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved cross-publication results to {json_path}")
    except Exception as e:
        logger.error(f"Failed to save cross-publication results: {e}")
    try:    
        deltas = calculate_deltas(training_results=training_results, testing_results=results, logger=logger)
        with open(os.path.join(results_dir, f'{model_type}_cross_publication_deltas.json'), 'w') as f:
            json.dump(deltas, f, indent=2)
        logger.info(f"Saved cross-publication deltas to {os.path.join(results_dir, f'{model_type}_cross_publication_deltas.json')}")
    except Exception as e:
        logger.error(f"Failed to calculate/save deltas: {e}")
    test_end_time = datetime.datetime.now()
    logger.info(f"Testing completed. Time taken: {test_end_time - train_end_time}")

    shap_path = os.path.join(results_dir, "regressor shaps" if model_type=="reg" else "classifier shaps")
    try:
        compute_shap_values(model, X_train, X_test, model_name, shap_path, is_regression=model_type=="reg", logger=logger)
    except Exception as e:
        logger.error(f'Issue with shap value calculation:\n{e}')
    shap_end_time = datetime.datetime.now()
    logger.info(f"SHAP value computation completed. Time taken: {shap_end_time - test_end_time}")
    logger.info(f"Total time taken for training, testing, and SHAP computation: {shap_end_time - start_time}")

def setup_logging(verbose=False):
    fmt_debug = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s'
    fmt_info  = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(handlers=[logging.StreamHandler()],
                        format=fmt_debug if verbose else fmt_info,
                        force=True)
    logging.getLogger('shap').setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib\..*")
    logger = logging.getLogger("ModelTraining")
    #logger.setFormatter(logging.Formatter(fmt_debug if verbose else fmt_info))
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='Short model testing.')
    parser.add_argument('-t', '--test_type', type=str, help='Type of test to perform: single_dataset or leave_one_out', default='leave_one_out')
    parser.add_argument('-m', '--model_type', type=str, help='Type of model to train: regression (reg) or classification (clf)', default='reg')
    parser.add_argument('-d', '--strain', type=str, help='Strain to test on.', default='A_PuertoRico_8_1934')
    parser.add_argument('-i', '--pub_id', type=int, help='ID of the publication to filter for', default='1')
    parser.add_argument('--package', type=str, help='Named feature preset (e.g. benchmark, relational, context).', default="benchmark")
    parser.add_argument('-g', '--grid_search', action='store_true', help='Perform grid search')
    parser.add_argument('--debug', action='store_true', help='Use debug settings')
    parser.add_argument('-b', '--n_bins', nargs='+', help='Number of bins for multiclass classification', default=2)
    parser.add_argument('-y', '--y_column', help='Column to predict, also used to choose norm (e.g. NGS_log_norm or NGS_log_min_max_norm)', default='NGS_log_norm')
    parser.add_argument('-o', '--cutoff', type=int, help='Cutoff for NGS count', default=15)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('-s', '--seed', type=int, help='Random seed', default=42)
    parser.add_argument('-p', '--pooling', type=str, help='Choose the way to pool data', default=None)
    parser.add_argument('-x', '--drop_intersecting', action='store_true', help='Choose whether to drop intersections')
    args = parser.parse_args()

    logger = setup_logging(verbose=args.verbose or args.debug)
    params = {key: value for key, value in vars(args).items()}
    train_and_evaluate_model(test_type=params["test_type"], model_type=params["model_type"], strain=params["strain"], cutoff=params["cutoff"], pub_id=params["pub_id"], package=params["package"], pooling=params["pooling"], drop_intersecting=params["drop_intersecting"], debug=params["debug"], y_column=params["y_column"], seed=params["seed"], logger=logger)