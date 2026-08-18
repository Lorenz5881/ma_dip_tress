from analysis_utils import *
import os
import logging
import datetime
import joblib

RESULT_PATH=os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'results',"umaps"))
MULTI_PUB_STRAINS = ["A_PuertoRico_8_1934","A_WSN_33","B_Victoria_504_2000","B_Yamagata_16_1988"]

def setup_logging(verbose = False, path="", name=""):
    #log_folder = os.path.join(RESULT_PATH, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(RESULT_PATH, exist_ok=True)
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

def umap_stuff(dataframe, pivot_data, embedding, title, path, name):
        logging.debug(f'Drawing NGS UMAPs with all datasets "unpooled" and saving in paths with {path}')
        logging.debug(f'Using following input:\ndataframe: {dataframe.shape}\n{dataframe}\n\npivot: {pivot_data.shape}\n{pivot_data}\n\nembedding: {embedding.shape}\n{embedding}')
        # Drawing basic umaps with everything
        make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"strain"),embedding,title,path,name+"_basic_str") # base ref by strain
        make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"type"),embedding,title,path,name+"_basic_type") # base ref by type
        make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"intersections"),embedding,title,path,name+"_basic_inter") # base ref by intersections
        make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"num_publications"),embedding,title,path,name+"_basic_nPub") # base ref by number of publications

        # Looking at strains with multiple publications
        multi_pub_strains = filter_ids(dataframe, "Strain", keep=["A_PuertoRico_8_1934","A_WSN_33","B_Victoria_504_2000","B_Yamagata_16_1988"],drop=None)
        logging.debug(f'multi-pub strains {len(multi_pub_strains)}')
        make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"num_publications",multi_pub_strains),embedding,title+": Strains with multiple References",path,name+"_multiPub_nPub")

        # Filtering out non-intersections for comparison
        all_non_intersections = filter_ids(dataframe, "intersections", keep="non-intersecting", drop=None)
        all_intersections = filter_ids(dataframe, "intersections", keep=None, drop="non-intersecting")
        all_test = filter_ids(dataframe, "intersections", keep=None, drop="any")
        logging.debug(f'all non-inter: {len(all_non_intersections)}\nall inter: {len(all_intersections)}\nall test: {len(all_test)}={len(all_non_intersections)}')
        make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"intersections",all_non_intersections),embedding,title,path,name+"_non") # base of non-intersecting
        make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"intersections",all_intersections),embedding,title,path,name+"_inter") # base of any intersecting

# UMAPs based on NGS read counts from all datasets
def draw_umaps_with_all(cutoff=10, exp_col="ACC_num", title="UMAP projection of DelVGs, based on normalized NGS read counts from all datasets", path="", name=""):
    '''
    Draws UMAPS based on NGS read counts from all datasets.
    Some filters applied for exploration.
    '''
    logging.debug(f'Preparing to draw NGS UMAPs with all datasets {"unpooled" if exp_col=="ACC_num" else "pooled"} and saving in paths with {path}')
    # Preparing data
    dataframe = get_data(pubs=ALL_PUBS, unpooled=exp_col=="ACC_num", exp_col=exp_col, cutoff=cutoff)
    pivot_data = dataframe.pivot(index="ID",columns=exp_col,values="NGS_log_min_max_norm").fillna(0)
    embedding = umap.UMAP(random_state=42).fit_transform(pivot_data)
    pivot_data.reset_index(inplace=True)
    best_results = None
    try:
        best_results = get_cluster_plots(embedding, pivot_data, title=f'Clustering all DelVGs, based on normalized NGS read counts from all datasets  (cutoff: {cutoff})', path=path, name=f"{name}_clusters")
    except Exception as e:
        logging.error("Couldn't do clusters:\n{e}")
    
    umap_stuff(dataframe, pivot_data, embedding, title, path, name)
    '''
    logging.info(f'Drawing NGS UMAPs with all datasets {"unpooled" if exp_col=="ACC_num" else "pooled"} and saving in paths with {path}')
    # Drawing basic umaps with everything
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"strain"),embedding,title,path,name+"_basic_str") # base ref by strain
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"type"),embedding,title,path,name+"_basic_type") # base ref by type
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"intersections"),embedding,title,path,name+"_basic_inter") # base ref by intersections
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"num_publications"),embedding,title,path,name+"_basic_nPub") # base ref by number of publications

    # Looking at strains with multiple publications
    multi_pub_strains = filter_ids(dataframe, "Strain", keep=["A_PuertoRico_8_1934","A_WSN_33","B_Victoria_504_2000","B_Yamagata_16_1988"],drop=None)
    logging.info(f'multi-pub strains {len(multi_pub_strains)}')
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"num_publications",multi_pub_strains),embedding,title+": Strains with multiple References",path,name+"_multiPub_nPub")

    # Filtering out non-intersections for comparison
    all_non_intersections = filter_ids(dataframe, "intersections", keep="non-intersecting", drop=None)
    all_intersections = filter_ids(dataframe, "intersections", keep=None, drop="non-intersecting")
    all_test = filter_ids(dataframe, "intersections", keep=None, drop="any")
    logging.info(f'all non-inter: {len(all_non_intersections)}\nall inter: {len(all_intersections)}\nall test: {len(all_test)}={len(all_non_intersections)}')
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"intersections",all_non_intersections),embedding,title,path,name+"_non") # base of non-intersecting
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"intersections",all_intersections),embedding,title,path,name+"_inter") # base of any intersecting
    '''
    
    # Concentrating on pr8 ids due to focus of data
    pr8_all = filter_ids(dataframe, "Strain", keep="A_PuertoRico_8_1934", drop=None)
    pr8_non_intersections =  filter_ids(dataframe, ["intersections","Strain"], keep=["A_PuertoRico_8_1934","non-intersecting"], drop=None)
    pr8_intersections =  filter_ids(dataframe, ["intersections","Strain"], keep="A_PuertoRico_8_1934", drop="non-intersecting")
    pr8_test_intersections =  filter_ids(dataframe, ["intersections","Strain"], keep="A_PuertoRico_8_1934", drop="any")
    logging.debug(f'pr8 non-inter: {len(pr8_non_intersections)}\npr8 inter: {len(pr8_intersections)}\npr8 test: {len(pr8_test_intersections)}={len(pr8_non_intersections)}')
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"strain",pr8_all),embedding,title+": A PuertoRico 8 1934",path,name+"_pr8_str")
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"intersections",pr8_all),embedding,title+": A PuertoRico 8 1934",path,name+"_pr8_inter")
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"intersections",pr8_non_intersections),embedding,title+": A PuertoRico 8 1934",path,name+"_pr8_spec_non")
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"intersections",pr8_intersections),embedding,title+": A PuertoRico 8 1934",path,name+"_pr8_spec_inter")

    # Looking at Pelz2021 specifically for time-series data
    pelz_all = filter_ids(dataframe, "Publication", keep="Pelz2021", drop=None)
    pelz_non_intersections =  filter_ids(dataframe, ["intersections","Publication"], keep="Pelz2021", drop="any")
    pelz_intersections =  filter_ids(dataframe, ["intersections","Publication"], keep="Pelz2021", drop="non-intersecting")
    pelz_test_intersections =  filter_ids(dataframe, ["intersections","Publication"], keep=["Pelz2021","any"], drop=None)
    logging.debug(f'pelz non-inter: {len(pelz_non_intersections)}\npelz inter: {len(pelz_intersections)}\npelz test: {len(pelz_test_intersections)}={len(pelz_intersections)}')
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"intersections",pelz_all),embedding,title+": Pelz2021",path,name+"_pelz")
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"intersections",pelz_non_intersections),embedding,title+": Pelz2021",path,name+"_pelz_spec_non")
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"intersections",pelz_intersections),embedding,title+": Pelz2021",path,name+"_pelz_spec_inter")
    #return dataframe, pivot_data, embedding
    return best_results

# UMAPs based on NGS read counts of one strain
def draw_umaps_of_strain(cutoff=10, exp_col="ACC_num", strain="A_PuertoRico_8_1934", title="", path="", name=""):
    '''
    Draws UMAPs based on NGS read counts of specified strain.
    '''
    logging.debug(f'Preparing to draw NGS UMAPS for strain {strain.replace("_"," ")}')
    if title == "":
        title=f"UMAP projection of {strain.replace('_',' ')} DelVGs, based on normalized NGS read counts from all datasets"
    # Preparing data
    dataframe = get_data(strain=strain,pubs=ALL_PUBS, unpooled=exp_col=="ACC_num", exp_col=exp_col, cutoff=cutoff)
    pivot_data = dataframe.pivot(index="ID",columns=exp_col,values="NGS_log_min_max_norm").fillna(0)
    embedding = umap.UMAP(random_state=42).fit_transform(pivot_data)
    pivot_data.reset_index(inplace=True)
    best_results = "None"
    try:
        best_results = get_cluster_plots(embedding, pivot_data, title=f'Clustering {strain.replace("_","/")} DelVGs, based on normalized NGS read counts from all datasets  (cutoff: {cutoff})', path=path, name=f"{name}_clusters")
    except Exception as e:
        logging.error("Couldn't do clusters:\n{e}")

    logging.info(f'Drawing NGS UMAPS for strain {strain.replace("_"," ")}')
    # Drawing basic umap with everything
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"strain"),embedding,title,path,name+"_basic_str")
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"intersections"),embedding,title,path,name+"_basic_inter")
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"num_publications"),embedding,title,path,name+"_basic_nPub")

    # Filtering out non-intersections for comparison
    all_intersections = filter_ids(dataframe, "intersections", keep=None, drop="non-intersecting")
    all_non_intersections = filter_ids(dataframe, "intersections", keep="non-intersecting", drop=None)
    logging.debug(f'all non-inter: {len(all_non_intersections)}\nall inter: {len(all_intersections)}')
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"intersections",all_non_intersections),embedding,title,path,name+"_spec_non")
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"intersections",all_intersections),embedding,title,path,name+"_spec_inter")

    # Looking at Pelz2021 specifically for time-series data.
    pelz_all = filter_ids(dataframe, "Publication", keep="Pelz2021", drop=None)
    pelz_intersections =  filter_ids(dataframe, ["intersections","Publication"], keep="Pelz2021", drop="non-intersecting")
    pelz_non_intersections =  filter_ids(dataframe, ["intersections","Publication"], keep="Pelz2021", drop="any")
    logging.debug(f'pelz non-inter: {len(pelz_non_intersections)}\npelz inter: {len(pelz_intersections)}')
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"intersections",pelz_all),embedding,title+": Pelz 2021",path,name+"_pelz")
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"intersections",pelz_non_intersections),embedding,title+": Pelz 2021",path,name+"_pelz_spec_non")
    make_umap_DelVG_plot(pivot_data,*get_coloring(dataframe,"intersections",pelz_intersections),embedding,title+": Pelz 2021",path,name+"_pelz_spec_inter")
    #return dataframe, pivot_data, embedding
    return best_results

def plot_segment_wise_umaps(embedding, pivot_data, exp_col="ACC_num", cutoff=10, title="Feature embedding", test_dataframe=[], path="", name=""):
    '''
    Draws separate UMAP of each segment
    '''
    logging.info(f"Plotting UMAPs by segment")
    if len(test_dataframe)==0:
        test_dataframe = get_data(pubs=ALL_PUBS, unpooled=exp_col=="ACC_num", exp_col=exp_col, cutoff=cutoff)
    test_dataframe = test_dataframe[test_dataframe["ID"].isin(pivot_data["ID"].unique())]
    try:
        plot_seg_mosaic(scaffold=embedding,scaff_ids=pivot_data,path=path,name=name,title=title)
    except Exception as e:
        logging.error(f'Problem when making segment mosaic:\n{e}')
    for seg in SEGMENTS:
        logging.info(f"Segment {seg}")
        seg_title = title
        if "  " in seg_title:
            seg_title.replace("  ", f" Segment {seg}  ")
        else:
            seg_title = seg_title+f" Segment {seg}"
        seg_df = test_dataframe[test_dataframe["Segment"]==seg]
        if len(seg_df)>0:
            make_umap_DelVG_plot(pivot_data,*get_coloring(seg_df,"segment"),standard_embedding=embedding,title=seg_title,path=path,name=name+f"_{seg}_seg") # base ref by strain
            make_umap_DelVG_plot(pivot_data,*get_coloring(seg_df,"intersections"),standard_embedding=embedding,title=seg_title,path=path,name=name+f"_inter_{seg}_inter") # base ref by strain
            make_umap_DelVG_plot(pivot_data,*get_coloring(seg_df,"intersections_extra"),standard_embedding=embedding,title=seg_title,path=path,name=name+f"_{seg}_interEx") # base ref by strain
            make_umap_DelVG_plot(pivot_data,*get_coloring(seg_df,"num_publications"),standard_embedding=embedding,title=seg_title,path=path,name=name+f"_{seg}_nPubs") # base ref by strain

def plot_feature_umap(embedding, pivot_data, exp_col="ACC_num", cutoff=10, title="Feature embedding", test_dataframe=[],path="",name=""):
    '''
    Draws UMAP of all ids found in test data, based on given embedding and pivot data.
    If no test data is provided, uses all experimental data.
    '''
    if len(test_dataframe)==0:
        test_dataframe = get_data(pubs=ALL_PUBS, unpooled=exp_col=="ACC_num", exp_col=exp_col, cutoff=cutoff)
    test_dataframe = test_dataframe[test_dataframe["ID"].isin(pivot_data["ID"].unique())]
    logging.info(f"Beginning to draw feature UMAPs for experimental data.\nOverlap between test data and artificial data: {len(test_dataframe)}")
    make_umap_DelVG_plot(pivot_data,*get_coloring(test_dataframe,"strain"),embedding,title,path,name+"_str") # base ref by strain
    make_umap_DelVG_plot(pivot_data,*get_coloring(test_dataframe,"segment"),embedding,title,path,name+"_seg") # base ref by strain
    make_umap_DelVG_plot(pivot_data,*get_coloring(test_dataframe,"type"),embedding,title,path,name+"_type") # base ref by type
    make_umap_DelVG_plot(pivot_data,*get_coloring(test_dataframe,"intersections"),embedding,title,path,name+"_inter") # base ref by intersections
    make_umap_DelVG_plot(pivot_data,*get_coloring(test_dataframe,"intersections_extra"),embedding,title,path,name+"_interEx") # base ref by intersections
    make_umap_DelVG_plot(pivot_data,*get_coloring(test_dataframe,"num_publications"),embedding,title,path,name+"_nPubs") # base ref by number of publications
    plot_segment_wise_umaps(embedding, pivot_data, exp_col, cutoff, title, test_dataframe, path, name)
    logging.info(f"Done with current set of feature umaps\n\n")

def feature_umaps_strain(strain="A_PuertoRico_8_1934",features=['Strain', 'Segment', 'Start', 'End', 'Direct_repeat', 'Junction', '3_5_diff', 'length_proportion'],cutoff=5,step_size=2,title="",path="",name="",artificial_data=[],pivot=[],embedding=[],chosen_ids=[]):
    '''
    Gets feature embedding for specified strain and draws respective UMAPs.
    Artificial data, pivot data and embedding can also be handed over, but if any are missing, all are newly
    '''
    if title == "":
        title = f'UMAP projection of artificial data, based on calculated features for {strain.replace("_"," ")}'
    if any([len(artificial_data)==0, len(pivot)==0, len(embedding)==0]):
        logging.info(f"Creating new feature embedding for {strain}, using {features}")
        artificial_data, pivot, embedding = get_feature_umap_embedding([strain],features, step_size=step_size, chosen_ids=chosen_ids)
    logging.info("Plotting based on created feature embedding")
    make_umap_DelVG_plot(pivot,get_coloring(artificial_data,"strain")[0],0.5,embedding,title=title,path=path,name=name+"_basic_str")
    make_umap_DelVG_plot(pivot,get_coloring(artificial_data,"segment")[0],0.5,embedding,title=title,path=path,name=name+"_basic_seg")
    plot_feature_umap(embedding, pivot, cutoff=cutoff,path=path,name=name+"_exp")
    return artificial_data, pivot, embedding

def ngs_umaps(strain_wise=False):
    logging.info("Creating NGS UMAPS, looping over cutoff")
    ngs_path = os.path.join(RESULT_PATH, "ngs_embedding")
    os.makedirs(ngs_path,exist_ok=True)
    setup_logging(verbose=False,path=ngs_path)
    best_results = {}
    if strain_wise:
        for i in [20,15,10,5,2,0]:
            logging.info(f"Cutoff: {i}")
            # NGS strain-wise
            for strain in MULTI_PUB_STRAINS:
                ngs_str_path = os.path.join(ngs_path, strain)
                os.makedirs(ngs_str_path,exist_ok=True)
                best_results[f'{strain}_{i}'] = draw_umaps_of_strain(cutoff=i,strain=strain,title=f'UMAP projection of {strain.replace("_","/")} DelVGs, based on NGS read counts  (cutoff {i})',path=ngs_str_path,name=f"{strain}_c{i}")
    else:
        for i in [20,15,10,5,2,0]:
            logging.info(f"Cutoff: {i}")
            # NGS all
            best_results[f'{i}'] = draw_umaps_with_all(cutoff=i,title=f'UMAP projection of DelVGs, based on NGS read counts  (cutoff {i})',path=ngs_path,name=f"FullNGS_c{i}")
    logging.info(f'All best results from grid_search:\n{best_results}')

def feature_umaps(strain_wise=False):
    logging.info("Creating feature UMAPS")
    feat_path = os.path.join(RESULT_PATH,"feature_embedding")
    os.makedirs(feat_path,exist_ok=True)
    if strain_wise:
        setup_logging(verbose=False, path=feat_path, name="strains_")
        features=['Segment', 'Start', 'End', 'Direct_repeat', 'Junction', '3_5_diff', 'length_proportion']
        # Feature strain-wise
        for single_strain in MULTI_PUB_STRAINS:
            logging.info(f'Starting on {single_strain}')
            feat_str_path = os.path.join(feat_path,single_strain)
            os.makedirs(feat_str_path,exist_ok=True)
            artificial_strain_data, strain_pivot, strain_embedding = get_feature_umap_embedding([single_strain], features, step_size=1)
            logging.debug("Plotting based on created feature embedding")
            title = f'UMAP scaffold of artificial data, based on calculated features for {single_strain.replace("_"," ")}'
            name = f"{single_strain}"
            make_umap_DelVG_plot(strain_pivot,get_coloring(artificial_strain_data,"strain")[0],0.25,strain_embedding,title=title,path=feat_str_path,name=name+"_art_str")
            make_umap_DelVG_plot(strain_pivot,get_coloring(artificial_strain_data,"segment")[0],0.25,strain_embedding,title=title,path=feat_str_path,name=name+"_art_seg")
            plot_segment_wise_umaps(strain_embedding,strain_pivot,cutoff=0,test_dataframe=artificial_strain_data,title=title,path=feat_str_path,name=name+"art")
            for i in [20,15,10,5,2,0]:
                logging.info(f"Cutoff: {i}")
                title = f'Projection of experimental data on scaffold, based on calculated features for {single_strain.replace("_"," ")}  (cutoff: {i})'
                plot_feature_umap(strain_embedding, strain_pivot, cutoff=i,path=feat_str_path,name=name+f"_c{i}_exp")
                #artificial_strain_data, strain_pivot, strain_embedding = feature_umaps_strain(strain=single_strain,step_size=1,cutoff=i,path=feat_str_path,name=f"{single_strain}_c{i}",artificial_data=artificial_strain_data,pivot=strain_pivot,embedding=strain_embedding)
    else:
        setup_logging(verbose=False, path=feat_path, name="full_")
        features=['Strain', 'Segment', 'Start', 'End', 'Direct_repeat', 'Junction', '3_5_diff', 'length_proportion']
        # Feature all
        artificial_data, artificial_pivot, feature_embedding = get_feature_umap_embedding(features=features, step_size=1)
        logging.debug("Drawing basic feature UMAPS")
        title = f'UMAP scaffold of artificial data, based on calculated features of all Strains'
        name = f"all"
        make_umap_DelVG_plot(artificial_pivot,get_coloring(artificial_data,"strain")[0],0.25,feature_embedding,title=title,path=feat_path,name="art_str")
        make_umap_DelVG_plot(artificial_pivot,get_coloring(artificial_data,"segment")[0],0.25,feature_embedding,title=title,path=feat_path,name="art_seg")
        plot_segment_wise_umaps(feature_embedding,artificial_pivot,cutoff=0,test_dataframe=artificial_data,title=title,path=feat_str_path,name=name+"art")
        for i in [20,15,10,5,2,0]:
            logging.info(f"Cutoff: {i}")
            title = f'Projection of experimental data on scaffold, based on calculated features of all Strains  (cutoff: {i})'
            plot_feature_umap(feature_embedding, artificial_pivot, cutoff=i, title=title, path=feat_path, name=name+f"_c{i}_exp")

def reduced_feature_umaps(strain_wise=False):
    logging.info("Creating feature UMAPS with reduced features")
    # Feature all
    reduced_features = ['Direct_repeat', 'Junction', '3_5_diff', 'length_proportion']
    reduced_path = os.path.join(RESULT_PATH,"reduced_features")
    os.makedirs(reduced_path,exist_ok=True)
    setup_logging(verbose=False, path=reduced_path)

    if strain_wise:
        # Reduced feature strain-wise
        for single_strain in MULTI_PUB_STRAINS:
            reduced_str_path = os.path.join(reduced_path,single_strain)
            os.makedirs(reduced_str_path,exist_ok=True)
            logging.info(f'Starting on {single_strain}')
            artificial_strain_data, strain_pivot, strain_embedding = [], [], []
            for i in [20,15,10,5,2,0]:
                logging.info(f'Starting with Cutoff {i}')
                artificial_strain_data, strain_pivot, strain_embedding = feature_umaps_strain(strain=single_strain,step_size=1,features=reduced_features,cutoff=i,path=reduced_str_path,name=f"{single_strain}_c{i}",artificial_data=artificial_strain_data,pivot=strain_pivot,embedding=strain_embedding)
    else:
        # Reduced feature all
        logging.debug("Drawing basic reduced feature UMAPS")
        artificial_data, artificial_pivot, feature_embedding = get_feature_umap_embedding(step_size=1,features=reduced_features)
        make_umap_DelVG_plot(artificial_pivot,get_coloring(artificial_data,"strain")[0],0.5,feature_embedding,"UMAP projection of artificial data, based on calculated features",reduced_path,name="basic_str")
        make_umap_DelVG_plot(artificial_pivot,get_coloring(artificial_data,"segment")[0],0.5,feature_embedding,"UMAP projection of artificial data, based on calculated features",reduced_path,name="basic_seg")
        for i in [20,15,10,5,2,0]:
            logging.info(f"Cutoff: {i}")
            plot_feature_umap(feature_embedding, artificial_pivot, cutoff=i,path=reduced_path,name=f"all_c{i}")

def test_seq_umaps(strain_wise=False):
    logging.info("Creating feature UMAPS with just sequence ohe")
    seq_features = ['Sequence']
    seq_feat_path = os.path.join(RESULT_PATH,"test_seq")
    os.makedirs(seq_feat_path,exist_ok=True)
    setup_logging(verbose=False, path=seq_feat_path)

    if strain_wise:
        # Sequence ohe of all test candidates strain-wise
        for single_strain in MULTI_PUB_STRAINS:
            seq_feat_str_path = os.path.join(seq_feat_path,single_strain)
            os.makedirs(seq_feat_str_path,exist_ok=True)
            logging.info(f'Starting on {single_strain}')
            for i in [20,15,10,5,2,0]:
                logging.info(f"Cutoff: {i}")
                logging.debug(f'Getting relevant ids.')
                real_ids = get_data(single_strain,cutoff=i)["ID"].tolist()
                artificial_strain_data, strain_pivot, strain_embedding = [], [], []
                artificial_strain_data, strain_pivot, strain_embedding = feature_umaps_strain(strain=single_strain,step_size=1,features=seq_features,cutoff=i,title=f"UMAP projection of {single_strain.replace('_','/')} DelVGs, based on sequence one-hot-encoding  (cutoff {i})",path=seq_feat_str_path,name=f"{single_strain}_c{i}",artificial_data=artificial_strain_data,pivot=strain_pivot,embedding=strain_embedding,chosen_ids=real_ids)
                try:
                    save_path = os.path.join(seq_feat_str_path,f"test_seq_emb_{single_strain}_{i}.sav")
                    joblib.dump(strain_embedding, save_path)
                except Exception as e:
                    logging.error(f'Could not save Embedding:\n{e}')
    else:
        for i in [20,15,10,5,2,0]:
            logging.info(f"Cutoff: {i}")
            logging.debug(f'Getting relevant ids.')
            real_ids = get_data(cutoff=i)["ID"].tolist()
            # Sequence ohe of all test candidates total
            logging.debug("Drawing basic reduced feature UMAPS")
            artificial_data, artificial_pivot, feature_embedding = get_feature_umap_embedding(step_size=1,features=seq_features,chosen_ids=real_ids)
            make_umap_DelVG_plot(artificial_pivot,get_coloring(artificial_data,"strain")[0],0.5,feature_embedding,title=f"UMAP projection of DelVGs, based on sequence one-hot-encoding",path=seq_feat_path,name="basic_str")
            make_umap_DelVG_plot(artificial_pivot,get_coloring(artificial_data,"segment")[0],0.5,feature_embedding,title=f"UMAP projection of DelVGs, based on sequence one-hot-encoding",path=seq_feat_path,name="basic_seg")
            plot_feature_umap(feature_embedding, artificial_pivot, cutoff=i, title=f"UMAP projection of DelVGs, based on sequence one-hot-encoding  (cutoff {i})",path=seq_feat_path,name=f"all_c{i}")
            try:
                save_path = os.path.join(seq_feat_path,f"test_seq_emb_full_{i}.sav")
                joblib.dump(feature_embedding, save_path)
            except Exception as e:
                logging.error(f'Could not save Embedding:\n{e}')

def test_feat_umaps(strain_wise=False):
    logging.info("Creating feature UMAPS for candidates in test data")
    feat_path = os.path.join(RESULT_PATH,"test_feat")
    os.makedirs(feat_path,exist_ok=True)
    setup_logging(verbose=False, path=feat_path)
    if strain_wise:
        # Feature candidates in test data strain-wise
        for single_strain in MULTI_PUB_STRAINS:
            logging.info(f'Starting on {single_strain}')
            feat_str_path = os.path.join(feat_path,single_strain)
            os.makedirs(feat_str_path,exist_ok=True)
            for i in [20,15,10,5,2,0]:
                logging.info(f"Cutoff: {i}")
                logging.debug(f'Getting relevant ids.')
                real_ids = get_data(single_strain,cutoff=i)["ID"].tolist()
                artificial_strain_data, strain_pivot, strain_embedding = [], [], []
                artificial_strain_data, strain_pivot, strain_embedding = feature_umaps_strain(strain=single_strain,step_size=1,cutoff=i,title=f"UMAP projection of {single_strain.replace('_','/')} DelVGs, based on calculated features  (cutoff: {i})",path=feat_str_path,name=f"{single_strain}_c{i}",artificial_data=artificial_strain_data,pivot=strain_pivot,embedding=strain_embedding,chosen_ids=real_ids)
                try:
                    save_path = os.path.join(feat_str_path,f"test_feat_emb_{single_strain}_{i}.sav")
                    joblib.dump(strain_embedding, save_path)
                except Exception as e:
                    logging.error(f'Could not save Embedding:\n{e}')
    else:
        for i in [20,15,10,5,2,0]:
            logging.info(f"Cutoff: {i}")
            logging.debug(f'Getting relevant ids.')
            real_ids = get_data(cutoff=i)["ID"].tolist()
            # Feature all candidates in test data
            artificial_data, artificial_pivot, feature_embedding = get_feature_umap_embedding(step_size=1,chosen_ids=real_ids)
            logging.debug("Drawing basic feature UMAPS")
            make_umap_DelVG_plot(artificial_pivot,get_coloring(artificial_data,"strain")[0],0.5,feature_embedding,title=f"UMAP projection of DelVGs, based on calculated features and NGS read counts",path=feat_path,name="basic_str")
            make_umap_DelVG_plot(artificial_pivot,get_coloring(artificial_data,"segment")[0],0.5,feature_embedding,title=f"UMAP projection of DelVGs, based on calculated features and NGS read counts",path=feat_path,name="basic_seg")
            plot_feature_umap(feature_embedding, artificial_pivot, cutoff=i,title=f"UMAP projection of DelVGs, based on calculated features and NGS read counts  (cutoff: {i})",path=feat_path,name=f"all_c{i}")
            try:
                save_path = os.path.join(feat_path,f"test_feat_emb_full_{i}.sav")
                joblib.dump(feature_embedding, save_path)
            except Exception as e:
                logging.error(f'Could not save Embedding:\n{e}')

def test_red_umaps(strain_wise=False):
    logging.info("Creating feature UMAPS with reduced features")
    # Feature all
    reduced_features = ['Direct_repeat', 'Junction', '3_5_diff', 'length_proportion']
    reduced_path = os.path.join(RESULT_PATH,"test_reduced")
    os.makedirs(reduced_path,exist_ok=True)
    setup_logging(verbose=False, path=reduced_path)

    if strain_wise:
        # Reduced feature strain-wise
        for single_strain in MULTI_PUB_STRAINS:
            reduced_str_path = os.path.join(reduced_path,single_strain)
            os.makedirs(reduced_str_path,exist_ok=True)
            logging.info(f'Starting on {single_strain}')
            for i in [20,15,10,5,2,0]:
                logging.info(f'Starting with Cutoff {i}')
                logging.debug(f'Getting relevant ids.')
                real_ids = get_data(single_strain,cutoff=i)["ID"].tolist()
                artificial_strain_data, strain_pivot, strain_embedding = [], [], []
                artificial_strain_data, strain_pivot, strain_embedding = feature_umaps_strain(strain=single_strain,step_size=1,features=reduced_features,cutoff=i,title=f"UMAP projection of {single_strain.replace('_','/')} DelVGs, based on calculated features  (cutoff: {i})",path=reduced_str_path,name=f"{single_strain}_c{i}",artificial_data=artificial_strain_data,pivot=strain_pivot,embedding=strain_embedding,chosen_ids=real_ids)
                try:
                    save_path = os.path.join(reduced_str_path,f"test_red_emb_{single_strain}_{i}.sav")
                    joblib.dump(strain_embedding, save_path)
                except Exception as e:
                    logging.error(f'Could not save Embedding:\n{e}')
    else:
        logging.info(f'Getting relevant ids.')
        # Reduced feature all
        for i in [20,15,10,5,2,0]:
            logging.info(f"Cutoff: {i}")
            logging.debug(f'Getting relevant ids.')
            real_ids = get_data(cutoff=i)["ID"].tolist()
            logging.debug("Drawing basic reduced feature UMAPS")
            artificial_data, artificial_pivot, feature_embedding = get_feature_umap_embedding(step_size=1,features=reduced_features,chosen_ids=real_ids)
            make_umap_DelVG_plot(artificial_pivot,get_coloring(artificial_data,"strain")[0],0.5,feature_embedding,title="UMAP projection of DelVGs, based on calculated features",path=reduced_path,name="basic_str")
            make_umap_DelVG_plot(artificial_pivot,get_coloring(artificial_data,"segment")[0],0.5,feature_embedding,title="UMAP projection of DelVGs, based on calculated features",path=reduced_path,name="basic_seg")
            plot_feature_umap(feature_embedding, artificial_pivot, cutoff=i,title=f"UMAP projection of DelVGs, based on calculated features  (cutoff: {i})",path=reduced_path,name=f"all_c{i}")
            try:
                save_path = os.path.join(reduced_path,f"test_red_emb_full_{i}.sav")
                joblib.dump(feature_embedding, save_path)
            except Exception as e:
                logging.error(f'Could not save Embedding:\n{e}')

def save_umap(embedding, index, name, result_path=RESULT_PATH):
    index[["UMAP1","UMAP2"]] = embedding
    index.to_csv(os.path.join(result_path,f'{name}.csv'))

def test_comb_umaps(strain_wise=False):
    logging.info("Creating UMAPS for candidates in test data, using standard features and ngs counts")
    comb_path = os.path.join(RESULT_PATH,"test_comb")
    os.makedirs(comb_path,exist_ok=True)
    setup_logging(verbose=False, path=comb_path)
    best_results = {}
    if strain_wise:
        # Feature candidates in test data strain-wise
        for single_strain in MULTI_PUB_STRAINS:
            comb_str_path = os.path.join(comb_path,single_strain)
            os.makedirs(comb_str_path,exist_ok=True)
            logging.info(f'Starting on {single_strain}')
            for i in [20,15,10,5,2,0]:
                logging.info(f"Cutoff: {i}")
                logging.debug(f'Getting relevant ids.')
                real_ids = get_data(single_strain,cutoff=i)["ID"].tolist()
                strain_data, strain_pivot, strain_embedding = get_combined_umap_embedding(strains=single_strain,cutoff=i)
                try:
                    save_umap(strain_embedding, strain_pivot, f'comb_umap_{single_strain}_c{i}_changed', comb_str_path)
                except Exception as e:
                    logging.error(f'Problem with save_umap function:\n{e}')
                strain_data, strain_pivot, strain_embedding = feature_umaps_strain(strain=single_strain,step_size=1,cutoff=i,title=f"UMAP projection of {single_strain.replace('_','/')} DelVGs, based on calculated features and NGS read counts  (cutoff: {i})",path=comb_str_path,name=f"{single_strain}_c{i}",artificial_data=strain_data,pivot=strain_pivot,embedding=strain_embedding,chosen_ids=real_ids)
                try:
                    best_results[f'{single_strain}_{i}'] = get_cluster_plots(strain_embedding, strain_pivot, title=f"Clustering of {single_strain.replace('_','/')} DelVGs, based on calculated features and NGS read counts  (cutoff: {i})", path=comb_str_path, name=f"c{i}_{single_strain}_comb_clusters", grid_search=False)
                except Exception as e:
                    logging.error(f'Could not do clustering:\n{e}')
                try:
                    save_path = os.path.join(comb_str_path,f"test_comb_emb_{single_strain}_{i}.sav")
                    joblib.dump(strain_embedding, save_path)
                    pivot_path = os.path.join(comb_str_path,f"test_comb_emb_{single_strain}_{i}_index.csv")
                    strain_pivot.to_csv(pivot_path,columns=["ID"])
                except Exception as e:
                    logging.error(f'Could not save Embedding:\n{e}')
    else:
        # Feature all candidates in test data
        for i in [20,15,10,5,2,0]:
            logging.info(f"Cutoff: {i}")
            logging.debug(f'Getting relevant ids.')
            real_ids = get_data(cutoff=i)["ID"].tolist() # useless by now TODO: remove those parts
            dataframe, pivot_data, feature_embedding = get_combined_umap_embedding(cutoff=i)
            try:
                save_umap(feature_embedding, pivot_data, f'comb_umap_full_c{i}_changed', comb_path)
            except Exception as e:
                logging.error(f'Problem with save_umap function:\n{e}')
            logging.debug("Drawing basic ngs+feature UMAPS")
            make_umap_DelVG_plot(pivot_data,get_coloring(dataframe,"strain")[0],0.5,feature_embedding,title=f"UMAP projection of DelVGs, based on calculated features and NGS read counts",path=comb_path,name="basic_str")
            make_umap_DelVG_plot(pivot_data,get_coloring(dataframe,"segment")[0],0.5,feature_embedding,title=f"UMAP projection of DelVGs, based on calculated features and NGS read counts",path=comb_path,name="basic_seg")
            plot_feature_umap(feature_embedding, pivot_data, cutoff=i,title=f"UMAP projection of DelVGs, based on calculated features and NGS read counts  (cutoff: {i})",path=comb_path,name=f"all_c{i}")
            try:
                best_results[f'{i}'] = get_cluster_plots(feature_embedding, pivot_data, title=f"Clustering of DelVGs, based on calculated features and NGS read counts  (cutoff: {i})", path=comb_path, name=f"all_c{i}_clusters", grid_search=False)
            except Exception as e:
                logging.error(f'Could not do clustering:\n{e}')
            try:
                save_path = os.path.join(comb_path,f"test_comb_emb_full_{i}.sav")
                joblib.dump(feature_embedding, save_path)
                pivot_path = os.path.join(comb_path,f"test_comb_emb_full_{i}_index.csv")
                pivot_data.to_csv(pivot_path,columns=["ID"])
            except Exception as e:
                logging.error(f'Could not save Embedding:\n{e}')
    logging.info(f'All best results from grid_search:\n{best_results}')

#TODO: rethink and redo this entirely
def test_meta_umaps():
    '''
    Creates UMAPs with meta features for the A/PuertoRico/8/1934 strain.
    Ignores seed virus data and drops additional datapoints missing the used meta information.
    The separate embeddings include the following sets of meta features:
    1. Resolution, Context, Cells
    2. Resolution, Context, Cells, Compartment
    3. Resolution, Context, Cells, Compartment, Time
    '''
    # getting all meta-info and progressivley filtering out rows with missing info, before using more columns
    # preparing data
    meta_path = os.path.join(RESULT_PATH,"meta")
    os.makedirs(meta_path,exist_ok=True)
    setup_logging(verbose=False, path=meta_path)
    strain = "A_PuertoRico_8_1934"
    features = ['Segment', 'Direct_repeat', 'Junction', '3_5_diff', 'length_proportion']
    meta_features = ['Host','Context','Resolution']

    for i in [20,15,10,5,2,0]:
        logging.info(f'Beginning with cutoff {i}')
        # prepare base dataframe and filter out seed virus rows
        dataframe = get_data(strain=strain,cutoff=i)
        dataframe = dataframe[~dataframe["ACC_num"].isin(["SRR15084925","SRR14352113"])]
        ngs_pivot = dataframe.pivot(index="ID",columns=exp_col,values="NGS_log_min_max_norm").fillna(0)
        start_columns = set(dataframe.columns)
        start_columns.discard("ID")
        feature_pivot = calculate_standard_features(dataframe,features)
        feature_pivot = feature_pivot[[col for col in feature_pivot.columns if col not in start_columns]].set_index("ID")
        feature_columns = set(feature_pivot.columns)
        feature_columns.discard("ID")
        
        logging.info(f'Minimal meta features')
        meta_features = ['Host','Context','Resolution']
        meta_pivot = transform_meta_features(dataframe,meta_features).set_index("ACC_num")
        try: # removing non meta columns for cleaner concatenation later
            meta_pivot.drop([col for col in meta_pivot.columns if col in start_columns or col in feature_columns], inplace=True, axis=1)
        except Exception as e:
            logging.error(f'Problem when trying to remove superfluous columns: {e}')

        # UMAPs with Resolution, Context and Cells as meta info
        logging.info(f'Dataframe:\n{dataframe.head()}\n{dataframe.columns}')
        logging.info(f'dataframe id duplicated: {dataframe["ID"].duplicated().sum()}')
        logging.info(f'feature pivot duplicated: {feature_pivot.index.duplicated().sum()}')  # Check for duplicates in feature_pivot
        logging.info(f'meta pivot duplicated: {meta_pivot.index.duplicated().sum()}')
        base_text = f'based on NGS read counts and meta features  (cutoff: {i})'
        try:
            meta_x_ngs = pd.concat([ngs_pivot, meta_pivot],axis=1)
            embedding = umap.UMAP(random_state=42).fit_transform(meta_x_ngs)
            umap_stuff(dataframe, meta_x_ngs, embedding, f'UMAP projection {base_text}', meta_path, f'c{i}_min_ngs')
        except Exception as e:
            logging.error(f'Exception when trying to merge pivots:\n{e}\n{meta_pivot.head()}\n{ngs_pivot.head()}\n\nColumns across pivots:\nmeta features: {set(meta_pivot.columns)}\nngs: {set(ngs_pivot.columns)}')
        try:
            get_cluster_plots(embedding, meta_x_ngs, title=f"Clustering of DelVGs {base_text}", path=meta_path, name=f"c{i}_min_ngs_clusters")
        except Exception as e:
            logging.error(f'Could not do clustering:\n{e}')
        base_text = f'based on NGS read counts, standard and meta features  (cutoff: {i})'
        try:
            meta_x_ngs_x_features = pd.concat([ngs_pivot, feature_pivot, meta_pivot],axis=1)
            embedding = umap.UMAP(random_state=42).fit_transform(meta_x_ngs_x_features)
            umap_stuff(dataframe, meta_x_ngs_x_features, embedding, f'UMAP projection {base_text}', meta_path, f'c{i}_min_all')
        except Exception as e:
            logging.error(f'Exception when trying to merge pivots:\n{e}\n{meta_pivot.head()}\n{ngs_pivot.head()}\n{feature_pivot.head()}\n\nColumns across pivots:\nmeta features: {set(meta_pivot.columns)}\nfeatures: {set(feature_pivot.columns)}\nngs: {set(ngs_pivot.columns)}')
        try:
            get_cluster_plots(embedding, meta_x_ngs_x_features, title=f"Clustering of DelVGs {base_text}", path=meta_path, name=f"c{i}_min_all_clusters")
        except Exception as e:
            logging.error(f'Could not do clustering:\n{e}')
            
        # filter out data missing Compartment
        dataframe = dataframe[dataframe["Compartment"]!="unknown"]
        ngs_pivot = dataframe.pivot(index="ID",columns=exp_col,values="NGS_log_min_max_norm").fillna(0)
        start_columns = set(dataframe.columns)
        start_columns.discard("ID")
        feature_pivot = calculate_standard_features(dataframe,features)
        feature_pivot = feature_pivot[[col for col in feature_pivot.columns if col not in start_columns]].set_index("ID")
        feature_columns = set(feature_pivot.columns)
        feature_columns.discard("ID")
        
        logging.info(f'Extended meta features')
        meta_features = ['Cells','Context','Resolution','Compartment']
        meta_pivot = transform_meta_features(dataframe,meta_features).set_index("ID")
        try: # removing non meta columns for cleaner concatenation later
            meta_pivot.drop([col for col in meta_pivot.columns if col in start_columns or col in feature_columns], inplace=True, axis=1)
        except Exception as e:
            logging.error(f'Problem when trying to remove superfluous columns: {e}')

        # UMAPs with Resolution, Context, Cells and Compartment as meta info
        base_text = f'based on NGS read counts and meta features  (cutoff: {i})'
        try:
            meta_x_ngs = pd.concat([ngs_pivot, meta_pivot],axis=1)
            embedding = umap.UMAP(random_state=42).fit_transform(meta_x_ngs)
            umap_stuff(dataframe, meta_x_ngs, embedding, f'UMAP projection {base_text}', meta_path, f'c{i}_mid_ngs')
        except Exception as e:
            logging.error(f'Exception when trying to merge pivots:\n{e}\n{meta_pivot.head()}\n{ngs_pivot.head()}\n')
        try:
            get_cluster_plots(embedding, meta_x_ngs, title=f"Clustering of DelVGs {base_text}", path=meta_path, name=f"c{i}_mid_ngs_clusters")
        except Exception as e:
            logging.error(f'Could not do clustering:\n{e}')
        base_text = f'based on NGS read counts, standard and meta features  (cutoff: {i})'
        try:
            meta_x_ngs_x_features = pd.concat([ngs_pivot, feature_pivot, meta_pivot],axis=1)
            embedding = umap.UMAP(random_state=42).fit_transform(meta_x_ngs_x_features)
            umap_stuff(dataframe, meta_x_ngs_x_features, embedding, f'UMAP projection {base_text}', meta_path, f'c{i}_mid_all')
        except Exception as e:
            logging.error(f'Exception when trying to merge pivots:\n{e}\n{meta_pivot.head()}\n{ngs_pivot.head()}\n{feature_pivot.head()}\n')
        try:
            get_cluster_plots(embedding, meta_x_ngs_x_features, title=f"Clustering of DelVGs {base_text}", path=meta_path, name=f"c{i}_mid_all_clusters")
        except Exception as e:
            logging.error(f'Could not do clustering:\n{e}')


        # filter out data missing Time
        dataframe = dataframe[dataframe["Time"].notnull()]
        ngs_pivot = dataframe.pivot(index="ID",columns=exp_col,values="NGS_log_min_max_norm").fillna(0)
        start_columns = set(dataframe.columns)
        start_columns.discard("ID")
        feature_pivot = calculate_standard_features(dataframe,features)
        feature_pivot = feature_pivot[[col for col in feature_pivot.columns if col not in start_columns]].set_index("ID")
        feature_columns = set(feature_pivot.columns)
        feature_columns.discard("ID")
        
        logging.info(f'Maximal meta features')
        meta_features = ['Cells','Context','Resolution','Compartment','Time']
        meta_pivot = transform_meta_features(dataframe,meta_features).set_index("ID")
        try: # removing non meta columns for cleaner concatenation later
            meta_pivot.drop([col for col in meta_pivot.columns if col in start_columns or col in feature_columns], inplace=True, axis=1)
        except Exception as e:
            logging.error(f'Problem when trying to remove superfluous columns: {e}')

        # UMAPs with Resolution, Context, Cells, Compartment and Time as meta info
        base_text = f'based on NGS read counts and meta features  (cutoff: {i})'
        try:
            meta_x_ngs = pd.concat([ngs_pivot, meta_pivot],axis=1)
            embedding = umap.UMAP(random_state=42).fit_transform(meta_x_ngs)
            umap_stuff(dataframe, meta_x_ngs, embedding, f'UMAP projection {base_text}', meta_path, f'c{i}_max_ngs')
        except Exception as e:
            logging.error(f'Exception when trying to merge pivots:\n{e}\n{meta_pivot.head()}\n{ngs_pivot.head()}\n')
        try:
            get_cluster_plots(embedding, meta_x_ngs, title=f"Clustering of DelVGs {base_text}", path=meta_path, name=f"c{i}_max_ngs_clusters")
        except Exception as e:
            logging.error(f'Could not do clustering:\n{e}')
        base_text = f'based on NGS read counts, standard and meta features  (cutoff: {i})'
        try:
            meta_x_ngs_x_features = pd.concat([ngs_pivot, feature_pivot, meta_pivot],axis=1)
            embedding = umap.UMAP(random_state=42).fit_transform(meta_x_ngs_x_features)
            umap_stuff(dataframe, meta_x_ngs_x_features, embedding, f'UMAP projection {base_text}', meta_path, f'c{i}_max_all')
        except Exception as e:
            logging.error(f'Exception when trying to merge pivots:\n{e}\n{meta_pivot.head()}\n{ngs_pivot.head()}\n{feature_pivot.head()}\n')
        try:
            get_cluster_plots(embedding, meta_x_ngs_x_features, title=f"Clustering of DelVGs {base_text}", path=meta_path, name=f"c{i}_max_all_clusters")
        except Exception as e:
            logging.error(f'Could not do clustering:\n{e}')

def ngs_corr_umap():
    corr_path = os.path.join(RESULT_PATH,"ngs_corr")
    os.makedirs(corr_path,exist_ok=True)
    setup_logging(verbose=False, path=corr_path)
    best_results = {}

    # Dealing with strains separately
    for strain in MULTI_PUB_STRAINS:
        strain_path = os.path.join(corr_path,strain)
        os.makedirs(strain_path,exist_ok=True)
        for i in [20,15,10,5,2,0]:
            data = get_data(strain=strain,cutoff=i)
            pivot = data.pivot(index="ID",columns=exp_col,values="NGS_log_min_max_norm").fillna(0)
            corr_matrix = pivot.corr()
            abs_corr_matrix = pivot.corr().abs()
            corr_embedding = umap.UMAP(random_state=42).fit_transform(corr_matrix)
            corr_matrix.reset_index(inplace=True)
            abs_corr_embedding = umap.UMAP(random_state=42).fit_transform(abs_corr_matrix)
            abs_corr_matrix.reset_index(inplace=True)
            base_text = f'Pearson Correlation of NGS read counts  (cutoff: {i})'
            umap_stuff(data, pivot, corr_embedding, f'UMAP projection of {strain.replace("_","/")} DelVGs, based on {base_text}', strain_path, f"c{i}_{strain}_full_corr")
            umap_stuff(data, pivot, abs_corr_embedding, f'UMAP projection of {strain.replace("_","/")} DelVGs, based on absolute {base_text}', strain_path, f"c{i}_{strain}_abs_full_corr")
            try:
                best_results[f'{strain}_{i}'] = get_cluster_plots(corr_embedding, corr_matrix, title=f'Clustering {strain.replace("_","/")} DelVGs, based on {base_text}', path=strain_path, name=f"c{i}_{strain}_corr_clusters")
                best_results[f'abs_{strain}_{i}'] = get_cluster_plots(abs_corr_embedding, abs_corr_matrix, title=f'Clustering {strain.replace("_","/")} DelVGs, based on absolute {base_text}', path=strain_path, name=f"c{i}_{strain}_abs_corr_clusters")
            except Exception as e:
                logging.error(f'Could not do clustering:\n{e}')
    
    # Starting with full dataset
    for i in [20,15,10,5,2,0]:
        data = get_data(cutoff=i)
        pivot = data.pivot(index="ID",columns=exp_col,values="NGS_log_min_max_norm").fillna(0)
        corr_matrix = pivot.corr()
        abs_corr_matrix = pivot.corr().abs()
        corr_embedding = umap.UMAP(random_state=42).fit_transform(corr_matrix)
        corr_matrix.reset_index(inplace=True)
        abs_corr_embedding = umap.UMAP(random_state=42).fit_transform(abs_corr_matrix)
        abs_corr_matrix.reset_index(inplace=True)
        base_text = f'Pearson Correlation of NGS read counts  (cutoff: {i})'
        umap_stuff(data, pivot, corr_embedding, f"UMAP projection of DelVGs, based on {base_text}", corr_path, f"c{i}_full_corr")
        umap_stuff(data, pivot, abs_corr_embedding, f"UMAP projection of DelVGs, based on absolute {base_text}", corr_path, f"c{i}_abs_full_corr")
        try:
            best_results[f'{i}'] = get_cluster_plots(corr_embedding, corr_matrix, title=f"Clustering of DelVGs, based on {base_text}", path=corr_path, name=f"c{i}_full_corr_clusters")
            best_results[f'abs_{i}'] = get_cluster_plots(abs_corr_embedding, abs_corr_matrix, title=f"Clustering of DelVGs, based on absolute {base_text}", path=corr_path, name=f"c{i}_abs_full_corr_clusters")
        except Exception as e:
            logging.error(f'Could not do clustering:\n{e}')
    logging.info(f'All best results from grid_search:\n{best_results}')


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='Run classifiers on DI-RNA data')
    parser.add_argument('-t', '--test', type=int, help='0=ngs full, 1=ngs strain-wise, 2=feature full, 3=feature strain-wise, 4=reduced feature full, 5=reduced feature strain-wise', default='0')
    parser.add_argument('-n', '--name', type=str, help='Name to to define result paths', default="")
    args = parser.parse_args()

    test_name = args.name
    test_type = args.test

    if len(test_name)>0:
        RESULT_PATH = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'results', f'umaps_{test_name}'))
    
    setup_logging(verbose=False)
    start_time = datetime.datetime.now()
    logging.info(f"Starting test {test_type} at {start_time}")
    match test_type:
        case 0: # Embedding of NGS read counts for all test data
            ngs_umaps(False)
        case 1: # Embedding of NGS read counts for test data by strain
            ngs_umaps(True)
        case 2: # Embedding of standard features for artificial set of all possible DelVGs
            feature_umaps(False)
        case 3: # Embedding of standard features for artificial set of all possible DelVGs by strain
            feature_umaps(True)
        case 4: # Embedding of standard features without Strain and Segment ohes for all test data
            test_red_umaps(False)
        case 5: # Embedding of standard features without Strain and Segment ohes for test data by strain
            test_red_umaps(True)
        case 6: # Embedding of Sequence ohe for all test data
            test_seq_umaps(False)
        case 7: # Embedding of Sequence ohe for test data by strain
            test_seq_umaps(True)
        case 8: # Embedding of standard features for all test data
            test_feat_umaps(False)
        case 9: # Embedding of standard features for test data by strain
            test_feat_umaps(True)
        case 10: # Embedding of standard features and NGS read counts for all test data
            test_comb_umaps(False)
        case 11: # Embedding of standard features and NGS read counts for test data by strain
            test_comb_umaps(True)
        case 12: # Embedding with meta some features and NGS read counts by strain
            test_meta_umaps()
        case 13: # Embedding from correlation between NGS read counts
            ngs_corr_umap()
        case _:
            logging.error(f"Unknown Test number: {test_type}")

    logging.info(f"Finished with umap creation task {test_type} und name {test_name}\nTime elapsed: {datetime.datetime.now()-start_time}")
    
