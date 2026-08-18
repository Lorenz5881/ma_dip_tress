from analysis_utils import *
from cluster_analysis import plot_feature_umap, plot_segment_wise_umaps
import os
import logging
import datetime
import joblib
import pyarrow.parquet as pq
from funcy import log_durations

RESULT_PATH = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'results',"final_scaffolds"))
MULTI_PUB_STRAINS = ["A_PuertoRico_8_1934","A_WSN_33","B_Victoria_504_2000","B_Yamagata_16_1988"]
SCAFFOLD_PATH = os.path.abspath(os.path.join(os.getcwd(), '..', '..', "scaffolds"))
os.makedirs(RESULT_PATH, exist_ok=True)

def setup_logging(verbose = False, path="", name=""):
    if path == "":
        log_path = os.path.join(RESULT_PATH, name+'results.log')
    else:
        log_path = os.path.join(path, name+'results.log')
    if verbose:
        logging.basicConfig(handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG, force=True)
    else:
        logging.basicConfig(handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO, force=True)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.info(f"Finished log setup and saving in {RESULT_PATH}/results.log")

def get_everything(strain="A_PuertoRico_8_1934", feature_list=['Segment', 'Start', 'End', 'Direct_repeat', 'Junction', '3_5_diff', 'length_proportion']):
    try:
        dataframe = pd.read_csv(f"artificials_{strain}.csv")
        dataframe = get_sequence_quicker(dataframe)
        dataframe["s_len"] = dataframe.apply(lambda row: len(row["Full_Sequence"]), axis=1)
        dataframe = identify_candidates(dataframe)
        pivot_data = calculate_standard_features(dataframe,feature_list)
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
    except Exception as e:
        logging.info(f'Failed to load artificials file:\n{e}\nCreating new dataframe instead.')
        dataframe, pivot_data, embedding = get_feature_umap_embedding([strain], feature_list, step_size=1)
    return dataframe, pivot_data, embedding

def feature_umaps(strain="A_PuertoRico_8_1934"):
    logging.info("Creating feature UMAPS")
    feat_path = os.path.join(RESULT_PATH,"feature_embedding")
    os.makedirs(feat_path,exist_ok=True)
    features=['Segment', 'Start', 'End', 'Direct_repeat', 'Junction', '3_5_diff', 'length_proportion']
    # Feature strain-wise
    logging.info(f'Starting on {strain}')
    feat_str_path = os.path.join(feat_path,strain)
    os.makedirs(feat_str_path,exist_ok=True)
    artificial_strain_data, strain_pivot, strain_embedding = get_everything(strain, features)
    try:
        save_path = os.path.join(feat_str_path,f"embedding_{strain}.sav")
        joblib.dump(strain_embedding, save_path)
        pivot_path = os.path.join(feat_str_path,f"index_{strain}.csv")
        strain_pivot.to_csv(pivot_path,columns=["ID"])
        pivot_path = os.path.join(feat_str_path,f"index_{strain}.parquet")
        strain_pivot["ID"].to_parquet(pivot_path)
    except Exception as e:
        logging.error(f'Unable to save Embedding:\n{e}')
    logging.debug("Plotting based on created feature embedding")
    title = f'UMAP scaffold of artificial data, based on calculated features for {strain.replace("_"," ")}'
    name = f"{strain}"
    make_umap_DelVG_plot(strain_pivot,get_coloring(artificial_strain_data,"strain")[0],0.25,strain_embedding,title=title,path=feat_str_path,name=name+"_art_str")
    make_umap_DelVG_plot(strain_pivot,get_coloring(artificial_strain_data,"segment")[0],0.25,strain_embedding,title=title,path=feat_str_path,name=name+"_art_seg")
    plot_segment_wise_umaps(strain_embedding,strain_pivot,cutoff=0,test_dataframe=artificial_strain_data,title=title,path=feat_str_path,name=name+"art")
    for i in [20,15,10,5,2,0]:
        logging.info(f"Cutoff: {i}")
        title = f'Projection of experimental data on scaffold, based on calculated features for {strain.replace("_"," ")}  (cutoff: {i})'
        plot_feature_umap(strain_embedding, strain_pivot, cutoff=i,path=feat_str_path,name=name+f"_c{i}_exp")

def plot_scaffold(scaff, data_df, coloring=[], alpha=0.5, title="", path="", name=""):
    '''
    Draws scatter plot for a given scaffold, using the full scaffold as coloring for the background.

    :param scaff: Two-dimensional numpy array of the scaffold embedding.
    :param data_df: Pandas dataframe of the experimental data meant to be drawn on top of scaffold.
        Must include the columns ID (identifier of candidate) and index (spot in embedding).
    :param coloring: List of 3-tuples of the form (*label*, *color*, *list of ids*). Used for setting the colors of
        specified ids. The list is plottet from front to back, so the first tuple will be on the very back of the final
        plot and may be overdrawn. If none is provided, will instead draw the background alone and a second plot with all
        foreground points in blue.
    :param alpha: The alpha argument to use for pyplot scatter function.
    :param title: Title to put at the top of the figure.
    :param path: Path to the directory in which to save the plot. If none is provided, plt.show is called instead.
    :param name: Name of the file to save the resulting plot in. ".png" will be appended.
    '''
    if len(coloring)==0:
        # empty background
        plt.figure(figsize=(16,16))
        ax = plt.gca()
        set_plot_background(scaff, ax)
        plt.title(title.split("  ")[0])
        if path!="":
            os.path.join(path,name+"_background.png")
            plt.savefig(path)
        else:
            plt.show()
        plt.close()
        
        plt.figure(figsize=(16,16))
        ax = plt.gca()
        set_plot_background(scaff, ax)
        filtered = scaff[data_df["index"].unique()]
        plt.scatter(filtered[:,0], filtered[:,1], s=10, c="blue",label=f'Found Candidates ({len(filtered)})')
        plt.legend()
    else:
        plt.figure(figsize=(16,16))
        ax = plt.gca()
        set_plot_background(scaff, ax)
        for group in coloring:
            filtered = scaff[data_df[data_df["ID"].isin(group[2])]["index"].unique()]
            plt.scatter(filtered[:,0], filtered[:,1], c=group[1], s=10, label=f'{group[0]} ({len(group[2])})', alpha=alpha)
    
        ax = plt.gca()
        if coloring[0][0]=="":
            cbar=plt.colorbar(boundaries=range(1,len(coloring)+2),values=[group[1] for group in coloring],aspect=50)
            cbar.set_ticks(np.linspace(1,len(coloring),len(coloring))+0.5)
            cbar.ax.set_yticklabels(range(1,len(coloring)+1))
        else:
            plt.legend()
    
    plt.title(title)
    if path!="":
        os.path.join(path,name+".png")
        plt.savefig(path)
    else:
        plt.show()
    plt.close()

@log_durations(logging.info)
def make_scaffold_plots(scaff, id_df, data_df, title="", path="", name=""):
    '''
    Makes segment, intersection and num-pub plots using the provided scaffold and experimental data.
    Base scaffold is colored in as light grey background on each plot.

    :param scaff: Two-dimensional numpy array of the scaffold embedding.
    :param id_df: Dataframe holding all the indices and respective IDs for the provided scaffold.
    :param data_df: Pandas dataframe of the experimental data meant to be drawn on top of scaffold.
        Must include the columns ID (identifier of candidate) and index (spot in embedding).
    :param title: Title to put at the top of the figure.
    :param path: Path to the directory in which to save the plots. If none is provided, plt.show is usually called instead.
    :param name: Base name of the files to save the resulting plots in. Base name will be extended by a separate suffix for each plot and
        ".png" will be appended. Any cutoff applied to data_df should be signified by a "c*your cutoff*" at the very beginning of the name,
        to adjust titles accordingly.
    '''
    if len(title.split("  ")) == 1: # if no cutoff info already included
        cut_info = name.split("_")[0]
        try:
            if int(cut_info[1:]) > 0:
                cut_info = f'  (cutoff: {int(cut_info[1:])})'
            else:
                cut_info = ""
        except ValueError as e:
            cut_info = name.split("_")[-1] # in case cutoff info is on the back
            try:
                if int(cut_info[1:]) > 0:
                    cut_info = f'  (cutoff: {int(cut_info[1:])})'
                else:
                    cut_info = ""
            except ValueError as e:
                logging.warning(f'Failed to get cutoff from name:\n{e}')
                cut_info = ""
    else:
        cut_info = ""

    plot_scaffold(scaff=scaff, data_df=data_df, coloring=[], title=f'{title}{cut_info}', path=path, name=name+"_base")
    plot_seg_mosaic(scaffold=scaff, scaff_ids=id_df, data_df=None, path=path, name=name+"_base_seg_mosaic", title=f'{title} Overview{cut_info}')
    plot_seg_mosaic(scaffold=scaff, scaff_ids=id_df, data_df=data_df, path=path, name=name+"_seg_mosaic", title=f'{title} filtered Overview{cut_info}')

    for (col_type, sub_name) in [("segment","seg"), ("intersections","inter"), ("intersections_extra","interEx"), ("num_publications","nPub")]:
        highlights, alpha = get_coloring(data=data_df, group_by=col_type)
        plot_scaffold(scaff=scaff, data_df=data_df, coloring=highlights, alpha=alpha, title=f'{title}{cut_info}', path=path, name=f'{name}_{sub_name}')

def cluster_on_scaffold(scaff, data_df, name, path, title, grid_search=True): # TODO: Make this much better
    logging.info(f'Beginning with clustering on scaffold')
    clustering_index = np.array([[i,x] for i,x in enumerate(data_df["index"].unique())])
    pd.DataFrame(clustering_index, columns=["clustering_index","scaff_index"]).to_csv(path+f"{name}_cluster_index.csv")
    sub_scaff = scaff[clustering_index[:,1]]
    clustering, results = get_clustering(source_data=sub_scaff, set_name="full", path=path, name=name, grid_search=grid_search, results=[])
    
    num_clusters = set(clustering.labels_)
    distinct_cmap = distinctipy.get_colormap(distinctipy.get_colors(len(num_clusters)-1,n_attempts=5000,rng=42,exclude_colors=[(0,0,0),(1,1,1),(1,0,0)]))

    plot_scaffold_cluster(whole_scaff=scaff, part_scaff=sub_scaff, plot_clustering=clustering, data_df=data_df, cmap=distinct_cmap, name=name, path=path, title=title, results=results)    
    plot_cluster(plot_base_data=sub_scaff, plot_embedding=sub_scaff, plot_clustering=clustering, cmap=distinct_cmap, plot_path=path, plot_name=name+f"_DBSCAN", plot_title=title)
    plot_silhouette(sub_scaff, clustering,distinct_cmap,path=path,name=name+f"_clustering_silhouette",title="Silhouette Plot of DBSCAN Clustering")

    if isinstance(results[-1],tuple) or isinstance(results[-1],list):
        n_samples = results[-1][-1]["best_min_samples"] if results else 5
    else:
        n_samples = results[-1]["best_min_samples"] if results else 5
    misc_plots(sub_scaff,n_samples,path,name+"_clustering")

    intersection_grouping, alpha = get_coloring(data_df, "intersections")
    inter_df = data_df[data_df["ID"].isin(intersection_grouping[1][2] + intersection_grouping[2][2])]
    inter_scaff = scaff[inter_df["index"].unique()]
    inter_clustering, inter_results = get_clustering(source_data=inter_scaff, set_name="onlyInter", path=path, name=name, grid_search=grid_search, results=[],grid = [np.linspace(1,5,50), np.arange(10,50,step=1)])

    num_clusters = set(inter_clustering.labels_)
    distinct_cmap = distinctipy.get_colormap(distinctipy.get_colors(len(num_clusters)-1,n_attempts=5000,rng=42,exclude_colors=[(0,0,0),(1,1,1),(1,0,0)]))

    plot_scaffold_cluster(whole_scaff=scaff, part_scaff=inter_scaff, plot_clustering=inter_clustering, data_df=data_df, cmap=distinct_cmap, name=name+'_onlyInter', path=path, title=title, results=results)    
    plot_cluster(plot_base_data=inter_scaff, plot_embedding=inter_scaff, plot_clustering=inter_clustering, cmap=distinct_cmap, plot_path=path, plot_name=name+f"_inter_DBSCAN", plot_title=title)
    plot_silhouette(inter_scaff, inter_clustering,distinct_cmap,path=path,name=name+f"_inter_clustering_silhouette",title="Silhouette Plot of DBSCAN Clustering")

    if isinstance(inter_results[-1],tuple) or isinstance(inter_results[-1],list):
        n_samples = inter_results[-1][-1]["best_min_samples"] if inter_results else 5
    else:
        n_samples = inter_results[-1]["best_min_samples"] if inter_results else 5
    misc_plots(inter_scaff,n_samples,path,name+"_clustering_inter")
    logging.info(f'Finished clustering on scaffold:\nbase results: {results}\nonly intersections: {inter_results}')

def plot_scaffold_cluster(whole_scaff, part_scaff, plot_clustering, data_df, cmap, name, path="", title="", results=None):
    def make_single_clust_plot(label):
        plt.figure(figsize=(16,16))
        ax = plt.gca()
        ax = set_plot_background(whole_scaff, ax)
        mask = plot_clustering.labels_ == label
        col = cmap(label) if label>=0 else "red"
        plt.scatter(part_scaff[mask, 0], part_scaff[mask, 1], color=col, s=10, label=label, edgecolors='k', alpha=0.5)
        plt.title(title+f'{f" Cluster {label}" if label>=0 else " Noise"}')
        if path!="":
            plt.savefig(os.path.join(path,name+f'_cluster_{label}.png'))
        else:
            plt.show()
        plt.close()
    
    current_clusters = set(plot_clustering.labels_)
    plt.figure(figsize=(16,16))
    ax = plt.gca()
    ax = set_plot_background(whole_scaff, ax)
    legend_handles = []
    for label in current_clusters:
        if label == -1:
            continue  # Skip noise since it's already added
        mask = plot_clustering.labels_ == label
        color = cmap(label)
        plt.scatter(part_scaff[mask, 0], part_scaff[mask, 1], color=color, s=10, alpha=0.6)
        legend_handles.append(mpatches.Patch(color=color, label=f"Cluster {label}"))
    noise_mask = plot_clustering.labels_ == -1
    plt.scatter(part_scaff[noise_mask, 0], part_scaff[noise_mask, 1], color='red', s=10, label="Noise", edgecolors='k', alpha=0.5)
    legend_handles.append(mpatches.Patch(color='red', label="Noise"))
    plt.legend(handles=legend_handles, title="Clusters")
    plt.title(title)
    plt.annotate(f'Silhouette Score: {shs(part_scaff,plot_clustering.labels_):.4f}',(min(whole_scaff[:,0]),min(whole_scaff[:,1])))
    if path!="":
        plt.savefig(os.path.join(path,name+"_clustering.png"))
    else:
        plt.show()
    plt.close()
    
    for label in current_clusters:
        make_single_clust_plot(label)

@log_durations(logging.info)
def build_scaffold(strain, feature_list=['Segment', 'Start', 'End', 'Direct_repeat', 'Junction', '3_5_diff', 'length_proportion'], name=""):
    try:
        dataframe = pd.read_csv(f"artificials_{strain}.csv")
        # Calculating features
        dataframe = get_sequence_quicker(dataframe)
        dataframe["s_len"] = dataframe.apply(lambda row: len(row["Full_Sequence"]), axis=1)
        dataframe = identify_candidates(dataframe)
        pivot_data = calculate_standard_features(dataframe,feature_list)
        pivot_data.drop_duplicates(["ID"],inplace=True)
        pivot_data.set_index(pivot_data["ID"],inplace=True)
        
        pivot_data = drop_non_numeric(pivot_data)
        try:
            pivot_data.drop(["Start","End","s_len","Full_Sequence"],inplace=True,axis=1)
        except Exception as e:
            logging.warning(f'Exception when trying to remove leftover columns in pivot data:\n{e}\n')
        logging.info("Creating Embedding for calculated features.")
        embedding = umap.UMAP(random_state=42).fit_transform(pivot_data)
        pivot_data.reset_index(inplace=True)
        logging.debug(f'pivot data:\n{pivot_data.head()}\n{pivot_data.describe()}\ndataframe:\n{dataframe.head()}\n{dataframe.describe()}\nembedding:\n{embedding}')
    except Exception as e:
        logging.info(f'Failed to load artificials file:\n{e}\nCreating new dataframe instead.')
        dataframe, pivot_data, embedding = get_feature_umap_embedding([strain], feature_list, step_size=1)

    local_path = os.path.join(RESULT_PATH,strain)
    os.makedirs(local_path,exist_ok=True)

    save_path = os.path.join(local_path,f"{name}_scaffold")
    try:
        np.savez_compressed(save_path+'.npz', embedding)
    except Exception as e:
        logging.error(f'Unable to save scaffold embedding to npz:\n{e}')
    try:
        emb_df = pd.DataFrame(embedding)
        emb_df.to_parquet(save_path+'.parquet')
    except Exception as e:
        logging.error(f'Unable to save scaffold embedding to parquet:\n{e}')
    
    pivot_path = os.path.join(local_path,f"{name}_index")
    try:
        pivot_data.to_csv(pivot_path+".csv",columns=["ID"])
    except Exception as e:
        logging.error(f'Unable to save index to csv:\n{e}')
    try:
        pivot_data["ID"].to_parquet(pivot_path+".parquet")
    except Exception as e:
        logging.error(f'Unable to save index to parquet:\n{e}')
    try:
        np.savez_compressed(pivot_path+".npz", pivot_data["ID"].to_numpy())
    except Exception as e:
        logging.error(f'Unable to save index to npz:\n{e}')
    return dataframe, pivot_data, embedding

@log_durations(logging.info)
def get_dataframe(pivot_id):
    dataframe = pivot_id.copy()
    dataframe[["Strain","Segment","Start","End"]] = dataframe["ID"].str.rsplit("_", n=3, expand=True)
    return dataframe

@log_durations(logging.info)
def load_scaffold_deprecated(strain: str, name=""):
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
    
    return get_dataframe(pivot_id), pivot_id, scaffold

@log_durations(logging.info)
def load_scaffold(strain: str, sample_size: int=None, ids=[]):
    '''
    Loads precomputed umap scaffold and its respective index mapping.

    :param strain: Name of the strain for which to load a scaffold.
    :param sample_size: integer to return only sample of scaffold index (for debugging).
        If provided, returns a sample of the chosen size. Index dataframe then includes
        "index" for sample embedding and "old_index" for full scaffold.
    :param ids: Iterable of ids to load index for. If given, function will return the
        index for specified ids only, ignoring sample_size parameter and returning the
        full scaffold.

    :return: umap embedding, pivot index
    '''
    logging.info(f'Attempting to load scaffold for {strain}')
    try:
        scaff_path = os.path.abspath(os.path.join(SCAFFOLD_PATH,f'{strain}_scaffold.npz'))
        id_path = os.path.abspath(os.path.join(SCAFFOLD_PATH,f'{strain}_index.csv'))
        try:
            embedding = np.load(scaff_path)["arr_0"].astype(float)
        except FileNotFoundError as e:
            embedding = joblib.load(scaff_path.split(".")[0]+".sav")
        if isinstance(embedding, list):
            embedding = np.vstack(embedding)
            logging.info("vstack done")
        if len(ids) > 0:
            pivot_id = pd.DataFrame({"ID": ids, "index": [None] * len(ids)})
            pivot_id.set_index("ID", inplace=True)  # makes lookup easier

            # Read the large file in chunks
            for chunk in pd.read_csv(id_path, index_col=0, chunksize=10000):
                local_ids = chunk[chunk["ID"].isin(pivot_id.index)]
                if not local_ids.empty:
                    # Set the 'index' column in pivot_id for matching IDs
                    for id_, idx in zip(local_ids["ID"], local_ids.index):
                        if pd.isna(pivot_id.at[id_, "index"]):
                            pivot_id.at[id_, "index"] = idx

            # Reset index if you need "ID" back as a column
            pivot_id.reset_index(inplace=True)
        elif sample_size:
            pivot_id = pd.read_csv(id_path, index_col=0).sample(sample_size)
            pivot_id["old_index"] = pivot_id.index
            embedding = embedding[pivot_id.index]
            pivot_id.reset_index(inplace=True,drop=True)
        else:
            pivot_id = pd.read_csv(id_path, index_col=0)
    except FileNotFoundError as e:
        logging.error(f'Did not find File for {strain} in scaffolds directory: {e}\ntrying old version instead')
        depr_df, pivot_id, embedding = load_scaffold_deprecated(strain)
    pivot_id["index"] = pivot_id.index
    return embedding, pivot_id

def load_na_scaffold():
    scaff_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'results',"na_scaffold_pr8","na_scaffold_pr8.csv"))
    pivot_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'results',"na_scaffold_pr8","na_scaffold_pr8_pivot.csv"))
    logging.info(f'Loading na-Scaffold')
    scaff = pd.read_csv(scaff_path)
    logging.info(f'Switching to numpy format')
    embedding = scaff[["UMAP1","UMAP2"]].to_numpy()
    logging.info(f'Loading chunked pivot')
    pivot_chunked = pd.read_csv(pivot_path,chunksize=5000)
    chunks = []
    for chunk in pivot_chunked:
        chunk["index"] = chunk.index
        chunk = chunk[["ID","index"]]
        chunks.append(chunk)
    logging.info(f'Merging pivot chunks')
    pivot_id = pd.concat(chunks)
    return embedding, pivot_id


def get_matched_samples(strain:str, cutoff:int=0, sample_size:int=None):
    '''
    Loads scaffold, index and dataframe for the given cutoff.
    '''
    scaff, id_df = load_scaffold(strain, sample_size)
    logging.debug(f'scaffold: {len(scaff)}\n{scaff}\n\nindex: {len(id_df)}\n{id_df.head()}')
    data_df = get_data(strain="A_PuertoRico_8_1934",cutoff=cutoff)
    if sample_size:
        data_df = data_df[data_df["ID"].isin(id_df["ID"])] # filtering for current sample
    data_df = data_df.merge(id_df[["ID","index"]], on="ID", how='left') # add index to experimental data
    return scaff, id_df, data_df

def fix_index_df(index_df):
    if ("Strain" not in index_df.columns) or ("Segment" not in index_df.columns):
        index_df["Strain"] = index_df["ID"].str.split("_")[0:-4]
        index_df["Segment"] = index_df["ID"].str.split("_")[-3]
    return index_df

@log_durations(logging.info)
def work_on_scaffold(strain, dataframe=None, index=None, scaffold=None, name="", overwrite_path="", overwrite_title=""):
    logging.debug("Plotting based on scaffold")
    if overwrite_title == "":
        title = f'UMAP scaffold, based on calculated features for {strain.replace("_"," ")}'
    else:
        title = overwrite_title
    if overwrite_path == "":
        local_path = os.path.join(RESULT_PATH,strain)
    else:
        local_path = overwrite_path
    os.makedirs(local_path,exist_ok=True)
    os.makedirs(os.path.join(local_path,"basic"),exist_ok=True)
    try:
        fake_df = get_dataframe(index)
        make_umap_DelVG_plot(index,get_coloring(fake_df,"strain")[0],0.25,scaffold,title=title,path=os.path.join(local_path,"basic"),name=name+"_art_str")
        make_umap_DelVG_plot(index,get_coloring(fake_df,"segment")[0],0.25,scaffold,title=title,path=os.path.join(local_path,"basic"),name=name+"_art_seg")
    except Exception as e:
        logging.error(f'Problem when using old make_umap_DelVG_plot function:\n{e}')
    try:
        plot_segment_wise_umaps(scaffold,index,cutoff=0,test_dataframe=fake_df,title=title,path=os.path.join(local_path,"basic"),name=name+"art")
    except Exception as e:
        logging.error(f'Problem when trying to do segment-wise umaps:\n{e}')

    # using test-data
    for i in [20,15,10,5,0]:
        logging.debug(f"Cutoff: {i}")
        test_data = get_data(strain, cutoff=i)
        test_data = test_data.merge(index[["ID","index"]], on="ID", how='left')
        title = f'Projection of experimental data on scaffold for {strain.replace("_"," ")}  (cutoff: {i})'
        cutoff_path = os.path.join(local_path,f"cutoff {i}")
        os.makedirs(cutoff_path,exist_ok=True)
        try:
            make_scaffold_plots(scaff=scaffold, id_df=index, data_df=test_data, title=f'{strain}', path=cutoff_path, name=f'{name}_c{i}')
        except Exception as e:
            logging.error(f'Problem when trying to run make_scaffold_plots function:\n{e}')
        try:
            cluster_on_scaffold(scaff=scaffold, data_df=test_data, name=f'{name}_c{i}_cluster', path=cutoff_path, title=f'{strain}', grid_search=True)
        except Exception as e:
            logging.error(f'Problem when trying to run cluster_on_scaffold function:\n{e}')
        try:
            plot_feature_umap(scaffold, index, cutoff=i,path=cutoff_path,name=name+f"_c{i}_exp",title=f'{strain} scaffold',test_dataframe=test_data)
        except Exception as e:
            logging.error(f'Problem when trying to run plot_feature_umap function:\n{e}')

def run_task(strain:str, use_existing:bool, name:str, feature_list:list):
    os.makedirs(os.path.join(RESULT_PATH,strain),exist_ok=True)
    if use_existing:
        scaffold, id_df = load_scaffold(strain)
        work_on_scaffold(strain=strain, index=id_df, scaffold=scaffold, name=name)
    else:
        work_on_scaffold(strain=strain, *build_scaffold(strain,feature_list,name), name=name)

def na_task(strain, use_existing, name, feature_list):
    na_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'results',"na_scaffold_analysis"))
    os.makedirs(na_path, exist_ok=True)
    scaffold, id_df = load_na_scaffold()
    work_on_scaffold(strain=strain, index=id_df, scaffold=scaffold, name=name, overwrite_path=na_path, overwrite_title=f"Null-Scaffold UMAP for Strain {strain.replace("_"," ")}")

def test_k_vary(strain, use_existing, name, feature_list):
    logging.error("This is a stub.")
    return

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='Create feature scaffolds by strain.')
    parser.add_argument('-s', '--strain', type=str, help='Name of the strain to be used.', default="A_PuertoRico_8_1934")
    parser.add_argument('-n', '--name', type=str, help='Name of files to save to.', default="")
    parser.add_argument('-f', '--features', nargs='+', help='Features to use for the scaffold', default=['Segment', 'Start', 'End', 'Direct_repeat', 'Junction', '3_5_diff', 'length_proportion'])
    parser.add_argument('-x', '--remove_segment', action='store_true', help='Remove Segment ohe from features?')
    parser.add_argument('-e', '--existing', action='store_true', help='Set if an existing scaffold should be used.')
    parser.add_argument('-t', '--test', type=int, help='ID of the test to run on the chosen scaffold. 0: standard, 1: na scaffold', default=0)
    args = parser.parse_args()

    setup_logging(verbose=False)
    start_time = datetime.datetime.now()
    
    file_name = args.name
    strain = args.strain
    features = args.features
    test = args.test

    if file_name == "":
        file_name = str(strain)
    if args.remove_segment:
        features.remove("Segment")
    
    logging.info(f"Starting script for {strain} at {start_time}")
    match test:
        case 0:
            run_task(strain,args.existing,file_name,features)
        case 1:
            na_task(strain, args.existing, file_name, features)
    logging.info(f'Finished {args.strain} feature scaffold. Elapsed time: {datetime.datetime.now()-start_time}.')
    
