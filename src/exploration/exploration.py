from exploration_utils import *

if __name__ == "__main__":
    setup_logging(RESULT_PATH, verbose=False, ignore_warnings=True)

    logging.info("Running all pipelines...")
    execute_boxplotting_per_strain(cutoff_grid=[0, 5, 10, 15], segments=ALL_SEGMENTS, add_all=True, n_cols=4, figsize_per_panel=(3, 4))
    execute_rsc_intersecting_pipeline_per_strain()
    execute_numeric_feature_cutoff_pipeline_per_strain()
    execute_per_strain_nucleotide_enrichment_pipeline(own_synthetic=True, save_plots=True, debug=False, force_recreate=True, seg_sample_size=35000, multi_source=True)
    execute_label_conflicts_per_strain(save_plots=True, debug=False)
    