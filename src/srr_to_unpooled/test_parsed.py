import logging
import pandas as pd
import numpy as np
import os
import glob
import sys


sys.path.insert(0, "..")
from utils import load_data, calculate_standard_features, make_multiclass, get_length_proportion, apply_cutoff, read_json_lists, log_and_norm

PUBLICATIONS = ["Alnaji2019", "Alnaji2021", "Berry2021", "Boussier2020", "Kupke2020", "Lui2019", "Mendes2021",
                "Pelz2021", "Penn2022", "Sheng2018", "Southgate2019", "Valesano2020", "vdHoecke2015", "Wang2020",
                "Wang2023", "Zhuravlev2020"]

STRAINS = read_json_lists('influenza_info.json')['strains']
UNPOOLED_DATA_DIR = os.path.join(os.path.dirname(__file__), "data_unpooled")

def setup_logging():
    '''
    Set up logging for the script
    :return:
    '''
    logging.basicConfig(handlers=[logging.StreamHandler()],
                        format='%(asctime)s - %(module)s - %(levelname)s - %(message)s', level=logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def identify_candidates(df):
    '''
    Create a unique ID for each DelVG candidate based on Strain, Segment, Start and End
    :param df: pandas dataframe with DelVG candidates, including columns Strain, Segment, Start and End
    :return: pandas dataframe with additional column ID
    '''
    df['ID'] = df.apply(lambda row: str(row['Strain']) + '_' + str(row['Segment']) + '_' + str(row['Start']) + '_' + str(row['End']), axis=1)
    return df


def load_unpooled_data(names: list) -> pd.DataFrame:
    '''
    Loads data from any csv files, corresponding to the given list of publication names.

    :param names: List of publication names to load data from

    :return: Dataframe containing all data from the given publications
    '''
    data_dir = UNPOOLED_DATA_DIR
    if isinstance(names, str):
        names = [names]
    csv_paths = []
    pubs = []
    for publication in names:
        publication_paths = glob.glob(os.path.join(data_dir, '**', f'{publication}*.csv'), recursive=True)
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
        df = pd.read_csv(file_path, keep_default_na=False)
        df['Strain'] = os.path.basename(os.path.dirname(file_path))
        pub = os.path.basename(file_path).split('/')[-1].split('.')[0]
        if "_" in pub:
            pub = pub.split('_')[0]
        #index = csv_paths.index(file_path)
        df['Publication'] = pub
        return df

    dfs = [load_and_label(file_path) for file_path in csv_paths]
    final_df = pd.concat(dfs, ignore_index=True)
    logging.debug(f'Loaded data for {names} with shape {final_df.shape}\nColumns: {final_df.columns}')

    #get_duplicates(final_df)

    return final_df


def compare_data(local = False):
    '''
    Compare the number of unique candidates between pooled and unpooled data
    :param pooled: dataframe with pooled data
    :param unpooled: dataframe with unpooled data
    :return:
    '''
    logging.info(f'Comparing given pooled and self-parsed unpooled data by publication')
    for pub in PUBLICATIONS:
        given = load_data([pub], include_publication=True, unpooled=False)
        if given.empty:
            logging.error(f'No pre-parsed data found for {pub}')
            continue
        given = identify_candidates(given)
        if local:
            logging.info(f'Using local parsed data')
            unpooled = load_unpooled_data([pub])
        else:
            unpooled = load_data([pub], include_publication=True, unpooled=True)
        if unpooled.empty:
            logging.error(f'No SRR-based data found for {pub}')
            continue
        unpooled = identify_candidates(unpooled)
        logging.info(f'Comparing data from {pub}:')
        given['PubxID'] = given.apply(lambda row: str(row['Publication']) + '_' + str(row['ID']), axis=1)
        unpooled['PubxID'] = unpooled.apply(lambda row: str(row['Publication']) + '_' + str(row['ID']), axis=1)

        # ID overlap and difference
        logging.info(
            f'Comparing IDs:\ngiven data IDs\tSRR-parsed IDs\n{given["ID"].nunique()}\t{unpooled["ID"].nunique()}\tdifference: {abs(given["ID"].nunique() - unpooled["ID"].nunique())}')
        given_ids = set(given['ID'].unique())
        unpooled_ids = set(unpooled['ID'].unique())
        given_pubxid = set(given['PubxID'].unique())
        unpooled_pubxid = set(unpooled['PubxID'].unique())
        only_given = given_ids.difference(unpooled_ids)
        only_unpooled = unpooled_ids.difference(given_ids)
        only_given_pubxid = given_pubxid.difference(unpooled_pubxid)
        only_unpooled_pubxid = unpooled_pubxid.difference(given_pubxid)
        if len(only_given_pubxid) == 0 and len(only_unpooled_pubxid) == 0:
            logging.info(f'All IDs are the same between sources.')
        else:
            if len(only_given) > 0:
                logging.info(f'IDs exclusive to meta analysis:\t{len(only_given)}\n{only_given}')
            if len(only_unpooled) > 0:
                logging.info(f'IDs exclusive to SRR-parsed:\t{len(only_unpooled)}\n{only_unpooled}')
            if len(only_given_pubxid) > 0:
                logging.info(f'IDs with pub exclusive to meta analysis:\t{len(only_given_pubxid)}\n{only_given_pubxid}')
            if len(only_unpooled_pubxid) > 0:
                logging.info(f'IDs with pub exclusive to SRR-parsed:\t{len(only_unpooled_pubxid)}\n{only_unpooled_pubxid}')
            # checking IDs with ngs read counts > 1
            if len(only_unpooled) > 0:
                logging.info(
                    f'Description of IDs exclusive to SRR-parsed data:\n{unpooled[unpooled["ID"].isin(list(only_unpooled))].describe()}')
                new_rows = unpooled[unpooled["ID"].isin(list(only_unpooled))]
                logging.info(f'grouped sizes:\n{new_rows.groupby("ID").size().describe()}')
                new_rows = new_rows.groupby("ID")[["NGS_read_count"]].sum()
                ngs_freq = []
                for score in list(new_rows["NGS_read_count"].unique()):
                    freq = len(new_rows[new_rows["NGS_read_count"] == score])
                    ngs_freq.append((score, freq))
                ngs_freq.sort(key=lambda x: x[0], reverse=False)
                logging.info(f'pooled NGS read counts:\n{new_rows.describe()}\nNGS Counts:\n{ngs_freq}\nTotal:\n{new_rows}')

        # NGS read count
        pooled = unpooled.groupby(['Strain', 'Segment', 'Start', 'End', 'ID'], as_index=False)['NGS_read_count'].sum()
        pooled_given = given.groupby(['Strain', 'Segment', 'Start', 'End', 'ID'], as_index=False)['NGS_read_count'].sum()
        combined = pd.merge(pooled_given[['NGS_read_count', 'ID']], pooled[['NGS_read_count', 'ID']], on='ID', how='outer', suffixes=('_given', '_srr'))
        combined['same'] = combined.apply(lambda row: row['NGS_read_count_given'] == row['NGS_read_count_srr'], axis=1)
        # pooled NGS read count difference
        if False in combined['same'].unique():
            logging.warning(f'Found difference in pooled NGS read counts!\nsame: {combined["same"].value_counts()[True]}'
                            f'\tdifferent: {combined["same"].value_counts()[False]}\n'
                            f'{combined[combined["same"] == False][["ID", "NGS_read_count_given", "NGS_read_count_srr"]]}\n'
                            f'IDs with different NGS read counts:\n{list(combined[combined["same"] == False]["ID"].unique())}')
        else:
            logging.info(f'All pooled NGS read counts are the same between sources.')
        # NGS read count difference without extra pooling for pubs with unpooled data
        if len(pooled_given) != len(given):
            comb = pd.merge(given[['NGS_read_count', 'ID']], unpooled[['NGS_read_count', 'ID']], on='ID', how='outer', suffixes=('_given', '_srr'))
            '''comb['same'] = comb.groupby('ID').apply(
                lambda x: x['NGS_read_count_given'].value_counts().eq(x['NGS_read_count_srr'].value_counts()).all())'''
            #print('after apply\n', comb)
            comb['same'] = comb.groupby('ID').apply(
                lambda x: pd.Series([x['NGS_read_count_given'].value_counts().eq(x['NGS_read_count_srr'].value_counts()).all()] * len(x), index=x.index)
                ).reset_index(level=0, drop=True)

            if False in comb['same'].unique():
                num_diff = comb.groupby('ID').apply(lambda x: 1 if x['same'].eq(False).all() else 0).sum()
                logging.warning(f'Found difference in NGS read counts without extra pooling on given data!\n'
                                f'Total number: {comb["ID"].nunique()}\nNumber of differing values: {num_diff}')
                #logging.warning(f'IDs with different NGS read counts:\n{list(comb[comb["same"] == False]["ID"].unique())}')
                grouped = comb.groupby('ID')

                id_tuples = []
                for name, group in grouped:
                    if not group['same'].all():
                        metas = [(x,y) for x,y in (zip(list(group["NGS_read_count_given"].value_counts().index), list(group["NGS_read_count_given"].value_counts().values)))]
                        srrs = [(x,y) for x,y in (zip(list(group["NGS_read_count_srr"].value_counts().index), list(group["NGS_read_count_srr"].value_counts().values)))]
                        id_tuples.append((name, f'meta: {metas}', f'srr: {srrs}'))
                logging.warning(f'Differing ngs counts:\n{id_tuples}')

                return
        logging.info(f'Finished with {pub}.\n\n')

if __name__ == "__main__":
    setup_logging()
    compare_data()