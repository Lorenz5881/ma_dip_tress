### Leveraging cross-dataset intersections for improved machine learning-based assessment of defective viral genome antiviral potential

This repository contains computational approaches to employ deletion-containing defective viral genomes (DelVGs) that were observed in multiple independent influenza virus (IV) studies, for the improvement of cross-dataset generalization and the priotization of candidates for future research and development of therapeutic applications.
##### General Overview
The contents can be grouped into three main categories:
1. Analysis of intersecting DelVGs.
	- Nucleotide enrichment around the deletion site
	- Length of direct repeats
	- Analysis of signal conflicts between datasets that share DelVGs
	- Visualization of numeric sequence features
2. Clustering and encoding of DelVGs for the generation of features that describe relative sequence characteristics and observed abundances
	- Generation of feature scaffolds to reduce data bias with a theoretical feature space
	- Generation of hybrid embeddings to enrich generated features with abundance data
	- Abundance clustering to use intersecting DelVGs as descriptors of biological conditions within a given experimental sample
3. Evaluation of prediction models
	- Performance and cross-dataset generalization of models predicting individual DelVG abundances, based on the features calculated generated from clusterings
	- Prediction of intersecting DelVGs that are more consistently expressed under varying biological conditions
##### Extension with additional datasets
This project specifies the employed datasets and influenza A and B virus strains. In order to extend that list, new data must be added to the data folder in a subdirectory named after the viral strain. Additional strains must be included in influenza_info.json and novel datasets named in utils.py line 37. Additionally, the dictionary in utils.py line 45-48 must be updated to include the novel dataset names listed in their respective strains.

#### Analysis of intersecting DelVGs
This part is largely independent of the rest and can be performed without prerequisites. The sequence_analysis.sh script executes the full pipeline. If no synthetic data is available, new data will be generated and saved automatically.

#### Clustering and encoding of DelVGs
The feature_scaffold.sh generates a feature scaffold and subsequently generates plots for visualization. This script takes quite a while to complete, so it is best to skip this part if a previously made scaffold is available. The generate_feature_umaps.py includes additional options for umap embeddings, including the hybrid embedding (aka "comb") and various ngs read count embeddings. Again, a large amount of data makes it preferable to use premade results. Once these embeddings are available, the assign_clustering_labels.sh script demonstrates how to perform HDBSCAN and K-means clustering with them. The ngs_communities.sh script generates correlation networks to based on NGS read counts in order to perform community detection on them and thereby cluster shared DelVGs.

#### Evaluation of prediction models
Using preprocess.sh generates a single dataframe holding all data for a given strain, including the calculated features. Additionally, a dictionary lists the columns by feature type (init, meta, standard, cluster and context) and is saved as a column_dict.json file. The script naturally requires that the clusterings are available in the clustering directory. After preprocessing, the run_prediction_tests.sh shows how and which tests were performed to evaluate prediction models. It is best to run tests separately, rather than execute the entire script, since it trains a lot of models and therefore takes a lot of time. There are three test types:
1. Single-dataset - trains on a single dataset and evaluate abundance prediction on all others.
2. Leave-one-out - trains on all but one dataset and evaluate abundance prediction on the held out set.
3. Intersection - trains on all deduplicated DelVGs to predict the number of datasets it is observed within.
Each test automatically calculates and plots SHAP values to estimate feature importance. Single-dataset and leave-one-out are both done with classifiers and regression models, while intersection prediction is done via multi-label classification, with the number of classes set to half the number of datasets rounded up (minimum 2). All tests apply stratified undersampling to balance the data.

#### Scientific Context
This work was conducted as part of a Master's thesis in Bioinformatics at the University of Hamburg.

#### Notes
##### Loading additional data from SRR files
The src/srr_to_unpooled directory contains a script to convert the contained SRR files to the unpooled data format used in this project. This script was adapted from the meta-analysis conducted by Lohmann et al. (available at: https://github.com/viraidip/DIP_meta-study), to parse additional metadata provided within the original publications. The resulting data of the given SRR files is already provided within the data directory, but the script can be extended to convert novel SRR files. Put the new SRR files in the srr_data directory and update the dictionaries in parsing_utils.py and parse_to_csv accordingly. The execute_parsing.sh script will convert the data. You can ignore resulting discrepancies in the generated test log.

#### License
This project is intended for academic and research purposes.
Please contact the authors for usage beyond this scope.

This project is licensed under the MIT License - see the LICENSE file for details.

The data employed in this project was gathered and provided by Lohmann, whose meta-analysis  is the basis for this work. Sections of the code were adapted from the meta-analysis (available at: https://github.com/viraidip/DIP_meta-study).