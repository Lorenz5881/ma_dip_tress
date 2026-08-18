# permutation tests with energy distance, wasserstein distance and MMD to compare distributions of clusters in umap space across different categories (e.g. publications, ACC groups, etc.) and see if they are significantly different from each other. This can help us understand if certain clusters are enriched in certain categories and if the overall distribution of clusters differs between categories. We can also do this for the segments themselves to see if the distribution of segments in umap space differs between categories. This can provide insights into whether certain types of data (e.g. from certain publications or ACC groups) tend to cluster together in umap space, which could indicate underlying similarities in the data that are captured by the umap embedding.

from model_check import *
import seaborn as sns
SEGMENTS = ["PB2", "PB1", "PA", "HA", "NP", "NA", "M", "NS"]
SEGMENT_COLORS = {segment: color for segment, color in zip(SEGMENTS, sns.color_palette("Set2", len(SEGMENTS)))}
strain = "A_PuertoRico_8_1934"
df, column_dict = load_preprocessed_data(strain)

import seaborn as sns
from scipy.spatial import ConvexHull
from scipy.stats import bootstrap

clustering_plot_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "dev_results", "compilation_dev", "clustering_plots_smol"))#os.path.abspath(os.path.join(os.getcwd(), "..", "..", "dev_results", "compilation_dev", "clustering_plots"))
SEGMENTS = ["PB2", "PB1", "PA", "HA", "NP", "NA", "M", "NS"]
SEGMENT_COLORS = {segment: color for segment, color in zip(SEGMENTS, sns.color_palette("Set2", len(SEGMENTS)))}
import distinctipy
import datetime
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
df = df[df["Publication"].isin(["Alnaji2021","Pelz2021","Wang2020","Wang2023"])]
df = df.sort_values(by="Publication")

from preprocess_data import (
    load_scaffold,
    load_comb_umap,
    cluster_intersecting_on_embedding,
    get_centroid_distance,
    get_cluster_motif_identities,
)


def regain_meta_col(dataframe, meta_col):
    data = dataframe.copy()
    if meta_col in dataframe.columns:
        return data
    else:
        match meta_col:
            case "Cells":
                cell_cols = [column for column in dataframe.columns if "Celltype_" in column]
                # assign each row the cell type set to 1 in the cell type columns (only one should be 1)
                data["Cells"] = dataframe[cell_cols].idxmax(axis=1).str.replace("Celltype_", "")
            case "Host":
                host_cols = [column for column in dataframe.columns if "Host_" in column]
                data["Host"] = dataframe[host_cols].idxmax(axis=1).str.replace("Host_", "")
            case "Compartment":
                comp_cols = ["extracellular", "intracellular"]
                data["Compartment"] = dataframe[comp_cols].idxmax(axis=1)
            case "Resolution":
                res_cols = ["singlecell", "bulk"]
                data["Resolution"] = dataframe[res_cols].idxmax(axis=1)
            case "Context":
                con_cols = ["in vitro", "in vivo"]
                data["Context"] = dataframe[con_cols].idxmax(axis=1)
            case _:
                raise ValueError(f"Invalid meta column: {meta_col}. Expected one of 'Cells', 'Host', 'Compartment', 'Resolution', or 'Context'.")
        return data

def energy_distance(group1, group2):
    from scipy.spatial.distance import cdist
    d_xx = cdist(group1, group1).mean()
    d_yy = cdist(group2, group2).mean()
    d_xy = cdist(group1, group2).mean()
    return 2 * d_xy - d_xx - d_yy

def wasserstein_distance(group1, group2):
    from scipy.stats import wasserstein_distance
    # Calculate Wasserstein distance for each dimension and average them
    distances = []
    for i in range(group1.shape[1]):
        dist = wasserstein_distance(group1[:, i], group2[:, i])
        distances.append(dist)
    return np.mean(distances)

def mmd(group1, group2, kernel="rbf", bandwidth=1.0):
    from sklearn.metrics.pairwise import pairwise_kernels
    K_xx = pairwise_kernels(group1, group1, metric=kernel, gamma=1/(2*bandwidth**2))
    K_yy = pairwise_kernels(group2, group2, metric=kernel, gamma=1/(2*bandwidth**2))
    K_xy = pairwise_kernels(group1, group2, metric=kernel, gamma=1/(2*bandwidth**2))
    mmd_value = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd_value

def permutation_test(group1, group2, distance_func, n_permutations=1000, random_state=None):
    np.random.seed(random_state)
    observed_distance = distance_func(group1, group2)
    combined = np.vstack([group1, group2])
    count = 0
    for _ in tqdm(range(n_permutations), desc="Permutations", leave=False, mininterval=10):
        np.random.shuffle(combined)
        perm_group1 = combined[:len(group1)]
        perm_group2 = combined[len(group1):]
        perm_distance = distance_func(perm_group1, perm_group2)
        if perm_distance >= observed_distance:
            count += 1
    p_value = (count + 1) / (n_permutations + 1)  # Add 1 to numerator and denominator to avoid p-value of 0
    return observed_distance, p_value

def iterate_category_pairs(dataframe, coord_columns, category_column):
    categories = dataframe[category_column].unique()
    category_pairs = [(cat1, cat2) for i, cat1 in enumerate(categories) for cat2 in categories[i+1:]]
    for cat1, cat2 in category_pairs:
        group1 = dataframe[dataframe[category_column] == cat1][coord_columns].values
        group2 = dataframe[dataframe[category_column] == cat2][coord_columns].values
        yield (cat1, cat2), group1, group2

def perform_pairwise_permutation_tests(dataframe, coord_columns, category_column, distance_funcs, distance_names, n_permutations=1000, random_state=None):
    results = {}
    for (cat1, cat2), group1, group2 in tqdm(iterate_category_pairs(dataframe, coord_columns, category_column), desc="Processing category pairs for permutation tests", leave=False, mininterval=10, total=dataframe[category_column].nunique() * (dataframe[category_column].nunique() - 1) // 2):
        cur_dict = {}
        for distance_func, distance_name in tqdm(zip(distance_funcs, distance_names), desc=f"Processing distance functions for {cat1}_{cat2}", leave=False, mininterval=10):
            try:
                observed_distance, p_value = permutation_test(group1, group2, distance_func, n_permutations, random_state)
                cur_dict[distance_name] = observed_distance
                cur_dict[f"{distance_name} p_value"] = p_value
            except Exception as e:
                print(f"Error occurred while processing category pair ({cat1}, {cat2}) with distance function {distance_name}: {e}")
        results[(cat1, cat2)] = cur_dict
    return results

def energy_distance_permutation_test(dataframe, coord_columns, category_column, n_permutations=1000, random_state=None, disp_id=None, disp_prefix="", start_time=None):
    #from scipy.stats import energy_distance
    from scipy.spatial.distance import cdist # using this to overcome depth issues with scipy's energy_distance function when working with large datasets in high-dimensional space. We can calculate the energy distance manually using pairwise distances.
    def energy_distance(group1, group2):
        d_xx = cdist(group1, group1).mean()
        d_yy = cdist(group2, group2).mean()
        d_xy = cdist(group1, group2).mean()
        return 2 * d_xy - d_xx - d_yy
    
    if disp_id is not None:
        logging.info(f"{disp_prefix}Processing energy distance permutation test for {category_column}...")
    np.random.seed(random_state)
    categories = dataframe[category_column].unique()
    category_pairs = [(cat1, cat2) for i, cat1 in enumerate(categories) for cat2 in categories[i+1:]]
    results = {}
    for cat1, cat2 in tqdm(category_pairs, desc="Processing energy distance category pairs", leave=False, mininterval=10):
        group1 = dataframe[dataframe[category_column] == cat1].drop_duplicates(subset="ID")[coord_columns].values
        group2 = dataframe[dataframe[category_column] == cat2].drop_duplicates(subset="ID")[coord_columns].values
        observed_distance = energy_distance(group1, group2)
        combined = np.vstack([group1, group2])
        count = 0
        for _ in tqdm(range(n_permutations), desc="Processing energy distance permutations", leave=False, mininterval=10):
            np.random.shuffle(combined)
            perm_group1 = combined[:len(group1)]
            perm_group2 = combined[len(group1):]
            perm_distance = energy_distance(perm_group1, perm_group2)
            if perm_distance >= observed_distance:
                count += 1
        p_value = (count + 1) / (n_permutations + 1)  # Add 1 to numerator and denominator to avoid p-value of 0
        results[(cat1, cat2)] = {"observed_distance": observed_distance, "p_value": p_value}
    return results

def wasserstein_distance_permutation_test(dataframe, coord_columns, category_column, n_permutations=1000, random_state=None, disp_id=None, disp_prefix="", start_time=None):
    from scipy.stats import wasserstein_distance
    if disp_id is not None:
        logging.info(f"{disp_prefix}Processing Wasserstein distance permutation test for {category_column}...")
    np.random.seed(random_state)
    categories = dataframe[category_column].unique()
    category_pairs = [(cat1, cat2) for i, cat1 in enumerate(categories) for cat2 in categories[i+1:]]
    results = {}
    for cat1, cat2 in tqdm(category_pairs, desc="Processing Wasserstein distance category pairs", leave=False, mininterval=10):
        group1 = dataframe[dataframe[category_column] == cat1].drop_duplicates(subset="ID")[coord_columns].values
        group2 = dataframe[dataframe[category_column] == cat2].drop_duplicates(subset="ID")[coord_columns].values
        observed_distance = wasserstein_distance(group1.flatten(), group2.flatten())
        combined = np.vstack([group1, group2])
        count = 0
        for _ in tqdm(range(n_permutations), desc="Processing Wasserstein permutations", leave=False, mininterval=10):
            np.random.shuffle(combined)
            perm_group1 = combined[:len(group1)]
            perm_group2 = combined[len(group1):]
            perm_distance = wasserstein_distance(perm_group1.flatten(), perm_group2.flatten())
            if perm_distance >= observed_distance:
                count += 1
        p_value = (count + 1) / (n_permutations + 1)  # Add 1 to numerator and denominator to avoid p-value of 0
        results[(cat1, cat2)] = {"observed_distance": observed_distance, "p_value": p_value}
    return results

def mmd_permutation_test(dataframe, coord_columns, category_column, n_permutations=1000, random_state=None, disp_id=None, disp_prefix="", start_time=None):
    from sklearn.metrics.pairwise import pairwise_kernels
    if disp_id is not None:
        logging.info(f"{disp_prefix}Processing MMD permutation test for {category_column}...")
    np.random.seed(random_state)
    categories = dataframe[category_column].unique()
    category_pairs = [(cat1, cat2) for i, cat1 in enumerate(categories) for cat2 in categories[i+1:]]
    results = {}
    for cat1, cat2 in tqdm(category_pairs, desc="Processing MMD category pairs", leave=False, mininterval=10):
        group1 = dataframe[dataframe[category_column] == cat1].drop_duplicates(subset="ID")[coord_columns].values
        group2 = dataframe[dataframe[category_column] == cat2].drop_duplicates(subset="ID")[coord_columns].values
        K_xx = pairwise_kernels(group1, group1, metric='rbf')
        K_yy = pairwise_kernels(group2, group2, metric='rbf')
        K_xy = pairwise_kernels(group1, group2, metric='rbf')
        observed_mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        combined = np.vstack([group1, group2])
        count = 0
        for _ in tqdm(range(n_permutations), desc="Processing MMD permutations", leave=False, mininterval=10):
            np.random.shuffle(combined)
            perm_group1 = combined[:len(group1)]
            perm_group2 = combined[len(group1):]
            K_xx_perm = pairwise_kernels(perm_group1, perm_group1, metric='rbf')
            K_yy_perm = pairwise_kernels(perm_group2, perm_group2, metric='rbf')
            K_xy_perm = pairwise_kernels(perm_group1, perm_group2, metric='rbf')
            perm_mmd = K_xx_perm.mean() + K_yy_perm.mean() - 2 * K_xy_perm.mean()
            if perm_mmd >= observed_mmd:
                count += 1
        p_value = (count + 1) / (n_permutations + 1)  # Add 1 to numerator and denominator to avoid p-value of 0
        results[(cat1, cat2)] = {"observed_mmd": observed_mmd, "p_value": p_value}
    return results

def plot_dist_heatmaps(results_df, category_col, distance_cols, title="", path=""):
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Create a pivot table for each distance metric
    for dist_col in distance_cols:
        pivot_table = results_df.pivot(index=category_col, columns=category_col, values=dist_col)
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, square=True, fmt=".2f", cmap="viridis")
        plt.title(f"{title} - {dist_col}")
        plt.tight_layout()
        if path != "":
            os.makedirs(path, exist_ok=True)
            base, ext = os.path.splitext(path)
            dist_path = f"{base}_{dist_col.replace(' ', '_')}{ext}"
            plt.savefig(dist_path)
        else:
            plt.show()

def iterate_clusterings_permutation_tests(dataframe, strain, algorithm, logger=logging, output_dir="", titles=True, **kwargs):
    if not kwargs.get("mute_updates", False):
        try:
            start_time = datetime.datetime.now()
        except Exception as e:
            logger.warning(f"Could not initialize progress timing for permutation tests: {e}")
            kwargs["mute_updates"] = True
    for cutoff in tqdm([15,10,5,0], desc="Cutoffs"):#[0]:#[15,10,5,0]:#[0, 5, 10, 15]:
        for umap_name, load_func in tqdm([("scaff", load_scaffold), ("comb", load_comb_umap)], desc="UMAPs"):#, ("feature", load_feature_umap)]:
            if "scaff" in umap_name:
                continue
            logger.debug(f"Processing {algorithm} clustering with cutoff {cutoff} on {umap_name} embedding for permutation tests...")
            if not kwargs.get("mute_updates", False):
                logger.info(f"Elapsed time: {datetime.datetime.now() - start_time} | Processing {algorithm} clustering with cutoff {cutoff} on {umap_name} embedding for permutation tests. Loading embedding and clustering data...")
            clustering_df = load_func(strain=strain, cutoff=cutoff, clustering=algorithm)
            if kwargs.get("recluster",False) and clustering_df is None:
                clustering_df = load_func(strain=strain, cutoff=cutoff, clustering="kmeans") # kmeans is usually available for all cutoffs, so we can use it as a fallback if we will recluster anyway
            if clustering_df is not None:
                if kwargs.get("recluster",False):
                    logger.info(f"Reclustering {algorithm} on {umap_name} embedding with cutoff {cutoff} for intersecting candidates before permutation tests...")
                    intersecting_ids = dataframe[dataframe["Intersecting"] == True]["ID"].unique()
                    clustering_df = cluster_intersecting_on_embedding(clustering_df, intersecting_ids, algorithm, logger=logger, kwargs=kwargs.get("clustering_kwargs", {}))
                    cluster_id_col = "Intersecting Cluster"
                    if cluster_id_col not in clustering_df.columns:
                        logger.warning(f"Expected cluster ID column '{cluster_id_col}' not found in reclustered dataframe. Found columns: {clustering_df.columns.tolist()}. This may cause issues with distance calculations.")
                    logging.info(f"Finished reclustering {algorithm} on {umap_name} embedding with cutoff {cutoff} for permutation tests. Clustering dataframe has {clustering_df.shape[0]} rows and {clustering_df.shape[1]} columns.")
                else:
                    cluster_id_col = "Cluster"
                if not kwargs.get("mute_updates", False):
                    logger.info(f"Elapsed time: {datetime.datetime.now() - start_time} | Finished loading data for {algorithm} clustering with cutoff {cutoff} on {umap_name} embedding. Running permutation tests...")
                intersecting_ids = dataframe[dataframe["Intersecting"] == True]["ID"].unique()
                # map umap coordinates to ID
                sub_df = dataframe[dataframe["ID"].isin(set(clustering_df["ID"].unique()))].copy()
                sub_df["UMAP1"] = clustering_df.set_index("ID").loc[sub_df["ID"], "UMAP1"].values
                sub_df["UMAP2"] = clustering_df.set_index("ID").loc[sub_df["ID"], "UMAP2"].values
                sub_df[cluster_id_col] = clustering_df.set_index("ID").loc[sub_df["ID"], cluster_id_col].values
                for cat_column in tqdm(["Context"], desc="Permutation Test Categories"): #["ACC_num","Publication","Cells","Host","Compartment","Resolution","Context"]
                    if not cat_column in sub_df.columns:
                        sub_df = regain_meta_col(sub_df, cat_column)
                    if cat_column in sub_df.columns:
                        if not kwargs.get("mute_updates", False):
                            logger.info(f"Elapsed time: {datetime.datetime.now() - start_time} | Running permutation tests for {algorithm} clustering with cutoff {cutoff} on {umap_name} embedding for category '{cat_column}'...")
                        #energy_results = energy_distance_permutation_test(dataframe=sub_df, coord_columns=["UMAP1", "UMAP2"], category_column=cat_column, n_permutations=1000, random_state=42, disp_id=disp_id, disp_prefix="")
                        #wasserstein_results = wasserstein_distance_permutation_test(dataframe=sub_df, coord_columns=["UMAP1", "UMAP2"], category_column=cat_column, n_permutations=1000, random_state=42, disp_id=disp_id, disp_prefix="")
                        #mmd_results = mmd_permutation_test(dataframe=sub_df, coord_columns=["UMAP1", "UMAP2"], category_column=cat_column, n_permutations=1000, random_state=42, disp_id=disp_id, disp_prefix="")
                        results = perform_pairwise_permutation_tests(dataframe=sub_df, coord_columns=["UMAP1", "UMAP2"], category_column=cat_column, distance_funcs=[energy_distance, wasserstein_distance, mmd], distance_names=["Energy Distance", "Wasserstein Distance", "MMD"], n_permutations=1000, random_state=42)
                        # Save results to a file
                        results_df = pd.DataFrame({
                            "Category Pair": list(results.keys()),
                            "Energy Distance": [res["Energy Distance"] for res in results.values()],
                            "Energy p-value": [res["Energy Distance p_value"] for res in results.values()],
                            "Wasserstein Distance": [res["Wasserstein Distance"] for res in results.values()],
                            "Wasserstein p-value": [res["Wasserstein Distance p_value"] for res in results.values()],
                            "MMD": [res["MMD"] for res in results.values()],
                            "MMD p-value": [res["MMD p_value"] for res in results.values()]
                        })
                        '''results_df = pd.DataFrame({
                            "Category Pair": list(energy_results.keys()),
                            "Energy Distance": [res["observed_distance"] for res in energy_results.values()],
                            "Energy p-value": [res["p_value"] for res in energy_results.values()],
                            "Wasserstein Distance": [res["observed_distance"] for res in wasserstein_results.values()],
                            "Wasserstein p-value": [res["p_value"] for res in wasserstein_results.values()],
                            "MMD": [res["observed_mmd"] for res in mmd_results.values()],
                            "MMD p-value": [res["p_value"] for res in mmd_results.values()]
                        })'''
                        if output_dir != "":
                            os.makedirs(os.path.join(output_dir, cat_column), exist_ok=True)
                            results_path = os.path.join(output_dir, cat_column, f"{umap_name}_{cutoff}_{algorithm}_permut_results.csv")
                            results_df.to_csv(results_path, index=False)
                        else:
                            print(f"Permutation test results for {algorithm} clustering with cutoff {cutoff} on {umap_name} embedding for category '{cat_column}':")
                            print(results_df)
                            
                        plot_dist_heatmaps(results_df, category_col="Category Pair", distance_cols=["Energy Distance", "Wasserstein Distance", "MMD"], title=f"Permutation Test Distances for {algorithm} Clustering on {umap_name} UMAP Embedding | Cutoff {cutoff} | Category: {cat_column}", path=os.path.join(output_dir, cat_column, f"{umap_name}_{cutoff}_{algorithm}_permut.png"))
                    else:
                        logging.warning(f"Category column '{cat_column}' not found in clustering dataframe for {algorithm} on {umap_name} embedding with cutoff {cutoff}. Skipping permutation tests for this category.")
            else:
                logging.error(f"Clustering dataframe for {algorithm} on {umap_name} embedding with cutoff {cutoff} is None. Skipping permutation tests for this clustering.")
    print(f"Finished permutation tests for all UMAP embeddings after {datetime.datetime.now()-start_time}.{' Saved results to ' + output_dir if output_dir != '' else ''}")

                
iterate_clusterings_permutation_tests(df, strain=strain, algorithm="hdbscan", logger=logging, output_dir=clustering_plot_dir, titles=True)#, output_dir=clustering_plot_dir)
