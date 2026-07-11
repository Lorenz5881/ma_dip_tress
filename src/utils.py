import glob
import logging
import os
import re
from typing import Tuple
import json
import joblib
try:
    import RNA
except:
    pass
import numpy as np
import pandas as pd
from Bio import SeqIO, Align
from Bio.Seq import Seq
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import types
from skbio.stats.composition import clr, ilr
from sklearn.preprocessing import StandardScaler
import distinctipy
import seaborn as sns
from functools import lru_cache
# from nupack import complex_analysis, Model

def read_json_lists(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), 'r') as file:
        data = json.load(file)
    return data

influenza_info = read_json_lists('influenza_info.json')#os.path.abspath(os.path.join(os.getcwd(), 'influenza_info.json')))

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
CLUSTERING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Clustering'))
UNPOOLED_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_unpooled'))
PUBLICATIONS = ["Lui2019", "Kupke2020", "Penn2022", "Sheng2018", "Zhuravlev2020", "vdHoecke2015", "Boussier2020", "Southgate2019", "Valesano2020", "Mendes2021", "Alnaji2019", "Berry2021", "Alnaji2021", "Wang2020", "Wang2023", "Pelz2021"]
CHARS = influenza_info['rna_bases'] #"ACGU"
CHARS_COUNT = len(CHARS)
SEGMENTS = influenza_info['segments'] #["PB2", "PB1", "PA", "HA", "NP", "NA", "M", "NS"]
SEGMENTS_COUNT = len(SEGMENTS)
STRAINS = influenza_info['strains'] #["A_Anhui_1_2013", "A_California_07_2009", "A_Connecticut_Flu122_2013", "A_NewCaledonia_20-JY2_1999", "A_Perth_16_2009", "A_PuertoRico_8_1934", "A_turkey_Turkey_1_2005", "A_WSN_33", "B_Brisbane_60_2008", "B_Lee_1940", "B_Victoria_504_2000", "B_Yamagata_16_1988"]
STRAINS_COUNT = len(STRAINS)
MAX_LEN = 2396
STRAIN_WISE_PUBLICATIONS = {'A_PuertoRico_8_1934': ['Alnaji2021', 'Pelz2021', 'Wang2023', 'Wang2020', 'Zhuravlev2020', 'Kupke2020', 'VdHoecke2015'],
                            'A_WSN_33': ['Boussier2020', 'Mendes2021'],
                            'B_Victoria_504_2000': ['Valesano2020', 'Berry2021'],
                            'B_Yamagata_16_1988': ['Southgate2019', 'Valesano2020', 'Berry2021']}
MULTI_PUB_STRAINS = list(STRAIN_WISE_PUBLICATIONS.keys())

# colors for standardized visualizations
colorblind_type = None
STRAIN_COLORS = {strain: color for strain, color in zip(STRAINS, distinctipy.get_colors(STRAINS_COUNT, n_attempts=5000, pastel_factor=0.2, colorblind_type=colorblind_type, rng=42))}
SEGMENT_COLORS = {segment: color for segment, color in zip(SEGMENTS, sns.color_palette("Set2", len(SEGMENTS)))}#{segment: color for segment, color in zip(SEGMENTS, distinctipy.get_colors(SEGMENTS_COUNT, n_attempts=5000, pastel_factor=0.2, colorblind_type=colorblind_type, rng=42))}
STRAIN_WISE_PUB_COLORS = {strain: {pub: color for pub, color in zip(pubs, sns.color_palette("Accent", len(pubs)))} for strain, pubs in STRAIN_WISE_PUBLICATIONS.items()}
#{strain: {pub: color for pub, color in zip(pubs, distinctipy.get_colors(len(pubs), n_attempts=5000, pastel_factor=0.2, colorblind_type=colorblind_type, rng=42))} for strain, pubs in STRAIN_WISE_PUBLICATIONS.items()}

# take looger from main
logger = logging.getLogger(__name__)
logging.debug(f'Loaded influenza_info.json:\n{influenza_info}\nCHARS: {CHARS}\nSEGMENTS: {SEGMENTS}\nSTRAINS: {STRAINS}')
# TODO: Comment
# TODO: Update other files for reorganized feature calculation

def load_data(names: list, include_publication: bool = True, unpooled = False) -> pd.DataFrame:
    '''
    Loads data from any csv files, corresponding to the given list of publication names.

    :param names: List of publication names to load data from
    :param include_publication: Whether or not to include Publciation column in dataframe
    :param unpooled: Whether or not to use unpooled data

    :return: Dataframe containing all data from the given publications
    '''
    if unpooled:
        data_dir = UNPOOLED_DATA_DIR
    else:
        data_dir = DATA_DIR
    if isinstance(names, str):
        names = [names]
    csv_paths = []
    pubs = []
    for publication in names:
        publication_paths = glob.glob(os.path.join(data_dir, '**', f'{publication}*.csv'), recursive=True)
        if len(publication_paths) < 1 and publication == "VdHoecke2015": # temporary fix for capitalization issue
            publication_paths = glob.glob(os.path.join(data_dir, '**', f'vdHoecke2015*.csv'), recursive=True)
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
        df = pd.read_csv(file_path, low_memory=False, keep_default_na=False)
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
    if unpooled: # handle wang2020 duplicates by summing read counts, keeping other features the same
        final_df = final_df.groupby(["Strain","Segment","Start","End","ACC_num"]).agg({col: "first" if col != "NGS_read_count" else "sum" for col in final_df.columns}).reset_index(drop=True)

    return final_df


def get_short_pub_name(pub):
    start = pub[:2]
    end = pub[-2:]
    return start+end

def get_sequence(strain: str, segment: str) -> str:
    '''
    Reads the sequence corresponding to given strain and segment from a fasta file and returns it as a string.

    :param strain: Strain of the virus
    :param segment: Segment of the virus

    :return: Sequence of the strains segment as a string
    '''
    #logging.debug(f'DATA_DIR: {DATA_DIR}\nstrain: {strain}\nsegment: {segment}')
    fasta_file = os.path.join(DATA_DIR, strain, 'fastas', f'{segment}.fasta')
    seq_obj = SeqIO.read(fasta_file, 'fasta')
    return str(seq_obj.seq.transcribe())

def get_sequence_quicker(df):
    if isinstance(df,types.GeneratorType):
        logging.info("Getting sequences for Generator")
        return pd.concat([get_sequence_quicker(chunk) for chunk in df],ignore_index=True)
    df["Full_Sequence"] = ''
    if "Strain" not in df.columns or "Segment" not in df.columns or df["Strain"].isna().any() or df["Segment"].isna().any():
        if "ID" in df.columns:
            logging.warning(f'Missing Strain and/or Segment info for some rows. Attempting to use ID column as fallback...')
            df[["Strain", "Segment", "tmp_Start", "tmp_End"]] = df["ID"].str.rsplit('_', n=3, expand=True)
            df.drop(columns=["tmp_Start","tmp_End"], inplace=True, errors="ignore")

    for id, group in df.groupby(["Strain","Segment"]):
        strain = group["Strain"].values[0]
        segment = group["Segment"].values[0]
        
        #logging.info(f'id: {id}\nseg: {segment}\tstrain: {strain}')
        fasta_file = os.path.join(DATA_DIR, strain, 'fastas', f'{segment}.fasta')
        seq_obj = SeqIO.read(fasta_file, 'fasta')
        seq = str(seq_obj.seq.transcribe()) 
        df.loc[group.index, "Full_Sequence"] = seq
        if len(seq)==0:
            logging.warning(f'Empty sequence for {id} at {fasta_file}')
    return df


def strain_ohe(df: pd.DataFrame):
    '''
    Encodes the strain column of the dataframe into one hot encoding. Uses a predefined List of strains.

    :param df: Dataframe containing the column 'Strain'

    :return: Original dataframe, now including the columns of the one hot encoding for the strain column
    '''
    n = df.shape[0]
    res = np.zeros((n, STRAINS_COUNT), dtype=np.uint8)
    for i, r in df.iterrows():
        try:
            pos = STRAINS.index(r['Strain'])
            res[i][pos] = 1
        except ValueError:
            logging.error(f'Strain {r["Strain"]} not found in list of Strains')
        except Exception as e:
            logging.error(f'Error during Strain ohe calc: {e}')

    encoded_df = pd.DataFrame(res)
    encoded_df.columns = STRAINS
    df = pd.concat([df, encoded_df], axis=1)

    return df


def segment_ohe(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Encodes the segment column of the dataframe into one hot encoding. Uses a predefined List of segments.

    :param df: Dataframe containing the column 'Segment'

    :return: Original dataframe, now including the columns of the one hot encoding for the segment column
    '''
    n = df.shape[0]
    res = np.zeros((n, SEGMENTS_COUNT), dtype=np.uint8)
    #logging.debug(f'Calculating one hot encoding for {n} rows in res: {res.shape}')
    for i, r in df.iterrows():
        try:
            pos = SEGMENTS.index(r['Segment'])
            res[i][pos] = 1
        except ValueError:
            logging.error(f'Segment {r["Segment"]} not found in list of segments')
        except Exception as e:
            logging.error(f'Error during Segment ohe calc: {e}')

    encoded_df = pd.DataFrame(res)
    encoded_df.columns = SEGMENTS
    df = pd.concat([df, encoded_df], axis=1)

    return df

def segment_ohe_quicker(df: pd.DataFrame) -> pd.DataFrame: 
    for seg in SEGMENTS:
        df[seg] = 0
    for id, group in df.groupby(["Segment"]):
        try:
            seg = group["Segment"].values[0]
            df.loc[group.index, seg] = 1
        except ValueError:
            logging.error(f'Segment {seg} not found in list of segments')
        except Exception as e:
            logging.error(f'Error during Segment ohe calc: {e}')
    return df


def sequence_ohe(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculates a one hot encoding for the sequence column of the dataframe into one hot encoding.
    Uses a predefined List of base characters.

    :param df: Dataframe containing the column 'Full_Sequence'

    :return: original Dataframe, now including the new columns of the one hot encoding for the sequence column
    '''
    n = df.shape[0]
    res = np.zeros((n, CHARS_COUNT * MAX_LEN), dtype=np.uint8)
    for i, r in df.iterrows():
        seq = r['Full_Sequence']
        seq = seq + "*" * (MAX_LEN - len(seq))
        for j, char in enumerate(seq):
            try:
                if char == '*':
                    break
                if char == "Y":
                    char = "C"
                if char == "R":
                    char = "A"
                pos = CHARS.index(char)
                res[i][j * CHARS_COUNT + pos] = 1
            except ValueError:
                logging.error(f'(Sequence ohe) Character {char} not found in list of characters!\nChars: {CHARS}')

    encoded_df = pd.DataFrame(res)
    col_names = [f"{i}_{ch}" for i in range(1, MAX_LEN + 1) for ch in CHARS]
    encoded_df.columns = col_names
    df = pd.concat([df, encoded_df], axis=1)  # df.join(encoded_df)

    return df


def get_junction_ohe(df):
    '''
    Calculates a one hot encoding for the sequence at the start and end of each deletion site.

    :param df: Dataframe including the columns 'Start', 'End' and 'Full_Sequence'
    :return: Dataframe including the new columns of the one hot encoding for both start and end of each deletion site
    '''

    def get_balanced_window(seq, anchor, window_size=10):
        half = int(window_size / 2)
        left = int(anchor) - half
        right = int(anchor) + half
        if left < 0:
            right = min(len(seq), right + abs(left))
            left = 0
        if right > len(seq):
            left = max(0, left - (right - len(seq)))
            right = len(seq)
        return seq[left:right]

    def get_subseq_ohe(position):
        '''
        Gets a one hot encoding for the sub-sequence of length 10 at the given site ('Start' or 'End'). Shifts the
        window if the site is too close to the start or end of the whole sequence.

        :param position: 'Start' or 'End'

        :return: Dataframe with one hot encoding for the sequence at the given position
        '''
        win_size = 10
        if win_size % 2 != 0:
            raise ValueError(f"Window size must be even, got {win_size}")
        res = np.zeros((df.shape[0], CHARS_COUNT * win_size), dtype=np.uint8)
        for i, r in df.iterrows():
            seq = r["Full_Sequence"]
            if position == "Start":
                # Start is first removed base (1-indexed).
                anchor = int(r["Start"])
            else:
                # End is last removed base (1-indexed), boundary anchor is End.
                anchor = int(r["End"])-1

            assert 0 <= anchor <= len(seq), f"Site {position} outside of sequence: {anchor}"
            site = get_balanced_window(seq, anchor, window_size=win_size)
            for j, char in enumerate(site):
                try:
                    if char == "Y":
                        char = "C"
                    if char == "R":
                        char = "A"
                    pos = CHARS.index(char)
                    res[i][j * CHARS_COUNT + pos] = 1
                except ValueError:
                    logging.error(f'Character {char} not found in list of characters!\nChars: {CHARS}\tSite: {site}')
        encoded_df = pd.DataFrame(res)
        col_names = [f"{position}_{i}_{ch}" for i in range(1, win_size + 1) for ch in CHARS]
        encoded_df.columns = col_names
        return encoded_df

    df_start = get_subseq_ohe("Start")
    df_end = get_subseq_ohe("End")
    df = pd.concat([df, df_start, df_end], axis=1)

    return df

def get_junction_ohe_quicker(df, win_size=10):
    def get_balanced_window(anchor, seq):
        # anchor is a 0-based boundary index. We center a fixed-size window on it.
        half = int(win_size / 2)
        left = int(anchor) - half
        right = int(anchor) + half
        if left < 0:
            right = min(len(seq), right + abs(left))
            left = 0
        if right > len(seq):
            left = max(0, left - (right - len(seq)))
            right = len(seq)
        return seq[left:right]

    def one_hot_window(anchor, seq):
        seq = seq.upper()
        res = np.zeros(CHARS_COUNT * win_size, dtype=np.uint8)
        assert 0 <= anchor <= len(seq), f"Site outside of sequence: {anchor}"
        site = get_balanced_window(anchor, seq)
        for j, char in enumerate(site):
            try:
                if char == "Y":
                    char = "C"
                if char == "R":
                    char = "A"
                idx = CHARS.index(char)
                res[j * CHARS_COUNT + idx] = 1
            except ValueError:
                logging.error(f'Character {char} not found in list of characters!\nChars: {CHARS}\tSite: {site}')
        #print(f'{site}\n{[CHARS[i%4] for i in range(len(res)) if res[i]==1]}')
        return res
    
    if win_size % 2 != 0:
        raise ValueError(f"win_size must be even to split deleted/remaining halves evenly, got {win_size}")
    if "Full_Sequence" not in df.columns:
        df = get_sequence_quicker(df)
    start_names = [f"Start_{i}_{ch}" for i in range(1, win_size+1) for ch in CHARS]
    end_names = [f"End_{i}_{ch}" for i in range(1, win_size+1) for ch in CHARS]
    for col in start_names:
        df[col] = 0
    for col in end_names:
        df[col] = 0
    
    for name, group in df.groupby(["Strain","Segment"]):
        sequence = group["Full_Sequence"].iloc[0]
        start_anchors = group["Start"].astype(int).unique()
        end_anchors = (group["End"].astype(int) - 1).unique()
        unique_anchors = np.array(list(set(start_anchors).union(end_anchors)))
        ohe_series = pd.Series((one_hot_window(pos, sequence) for pos in unique_anchors), index=unique_anchors)
        
        matching_rows = group.index
        df.loc[matching_rows, start_names] = df.loc[matching_rows, "Start"].astype(int).map(ohe_series).values.tolist()
        df.loc[matching_rows, end_names] = (df.loc[matching_rows, "End"].astype(int) - 1).map(ohe_series).values.tolist()
    return df

def get_ohes(df, features):
    '''
    Creates one hot encodings for a specified subset of the given features. Specified subset includes Strain, Segment,
    Sequence and Junction. Organizes their respective functions in a match-case block.

    :param df: Dataframe including the columns needed for each respective one hot encoding
    :param features: List of all features

    :return: Dataframe with all one hot encodings concatenated to the original Data
    '''
    ohes = ["strain", "segment", "sequence", "junction"]
    try:
        non_tuple = [feature for feature in features if not isinstance(feature, tuple)]
        ohe_features = [feature for feature in non_tuple if feature.lower() in ohes]
        logging.debug(f'Features to calculate one hot encodings for: {ohe_features}')
    except Exception as e:
        logging.error(f'Error: {e}')
    for feature in ohe_features:
        try:
            match feature.lower():
                case "strain":
                    logging.debug(f'Calculating Strain ohe')
                    df = strain_ohe(df)
                case "segment":
                    logging.debug(f'Calculating Segment ohe')
                    #df = segment_ohe(df)
                    df = segment_ohe_quicker(df)
                case "sequence":
                    logging.debug(f'Calculating Sequence ohe')
                    df = sequence_ohe(df)
                case "junction":
                    logging.debug(f'Calculating Junction ohe')
                    df = get_junction_ohe_quicker(df)
        except Exception as e:
            logging.error(f'Error: {e}')
    return df


def get_direct_repeat_length(row):
    '''
    Calculates the length of the longest subsequence, which equals both the suffix of the start portion and the prefix of
    the end portion of the DelVG.

    :param row: Series containing the indices Full Sequence, Start and End

    :return: Length of the longest directly repeating subsequence.
    '''
    #if max < 1:
    #    max = min(start, len(sequence) - end)
    #counter = 0
    #for i in range(max):
    #    if sequence[start - i:start] == sequence[end - 1:end - 1 + i]:
    #        counter = i
    #return counter
    counter = 0

    seq = row['Full_Sequence']
    s = row['Start']
    e = row['End']
    w_len = 15
    m = 1

    if m == 1:
        if s < w_len or e-1 > len(seq) - w_len:
            w_len = min(s, len(seq) - e)
        start_window = seq[s - w_len: s]
        end_window = seq[e - 1 - w_len: e - 1]

        # if they are the same return directly to avoid off-by-one error
        if start_window == end_window:
            return len(start_window)#, start_window

        for i in range(w_len - 1, -1, -1):
            if start_window[i] == end_window[i]:
                counter += 1
            else:
                break
        overlap_seq = str(start_window[i + 1:w_len])

    elif m == 2:
        for i in range(w_len):
            if seq[s - i:s] == seq[e - 1:e - 1 + i]:
                counter = i
                overlap_seq = str(seq[s - i:s])

    assert counter == len(overlap_seq), f"{counter=}, {len(overlap_seq)}"

    return counter


def get_peptide_len(row):
    '''
    Calculates the length of the peptide, resulting from the given Del VG.

    :param row: Series containing the indeces Full Sequence, Start and End

    :return: Length of the peptide for the remaining sequence
    '''
    # total_seq = get_sequence(row['Strain'], row['Segment'])
    total_seq = row['Full_Sequence']
    seq = Seq(total_seq[:row["Start"] - 1] + total_seq[row["End"]:])
    pep_seq = seq.translate(to_stop=True)
    return len(pep_seq)


def get_delta_G(row):
    '''
    Calculates the Gibbs free Energy (Delta G) for a sequence, after removing the deletion site.

    :param row: Series containing the indeces Full Sequence, Start and End

    :return: Gibbs free energy of the remaining sequence
    '''
    rest_seq = row['Full_Sequence'][:row['Start']]+row['Full_Sequence'][row['End']+1:]
    mfe = RNA.fold_compound(rest_seq).mfe()[1]
    return mfe/len(rest_seq)
    #return RNA.duplexfold(row['Full_Sequence'][row['Start']:row['End']], row['Full_Sequence'][row['Start']:row['End']])[1]

'''
# old junction calc functions
def get_junction_start_ohe(df, reach=5):
    n = df.shape[0]
    res = np.zeros((n, CHARS_COUNT * 2 * reach), dtype=np.uint8)
    for i, r in df.iterrows():
        seq = r["Full_Sequence"]
        if reach > r['Start']:
            reach = r['Start']
        start = seq[r['Start'] - reach:r['Start']+reach]
        start = start + "*" * (reach - len(start))
        for j, char in enumerate(start):
            pos = CHARS.rfind(char)
            res[i][j * CHARS_COUNT + pos] = 1

    encoded_df = pd.DataFrame(res)
    col_names = [f"JuncStart_{i}_{ch}" for i in range(1, 2 * reach + 1) for ch in CHARS]
    encoded_df.columns = col_names
    df = pd.concat([df, encoded_df], axis=1)
    return df
    
def get_junction_end_ohe(df, reach=5):
    n = df.shape[0]
    res = np.zeros((n, CHARS_COUNT * 2 * reach), dtype=np.uint8)
    for i, r in df.iterrows():
        seq = r["Full_Sequence"]
        if reach > len(seq) - r['End']:
            reach = len(seq) - r['End']
        end = seq[r['End']-reach:r['End'] + reach]
        end = end + "*" * (reach - len(end))
        for j, char in enumerate(end):
            pos = CHARS.rfind(char)
            res[i][j * CHARS_COUNT + pos] = 1

    encoded_df = pd.DataFrame(res)
    col_names = [f"JuncEnd_{i}_{ch}" for i in range(1, 2 * reach + 1) for ch in CHARS]
    encoded_df.columns = col_names
    df = pd.concat([df, encoded_df], axis=1)
    return df'''

def get_3_5_diff(row):
    '''
    Calculates the difference between the length of the 3' end and the length of the 5' end of a DelVG.

    :param row: Series containing the indices Full_Sequence, Start and End

    :return: Difference between the section lengths.
    '''
    return row['Start'] - (len(row['Full_Sequence']) - row['End'] - 1)


def get_3_5_ratio(row):
    '''
    Calculates the ratio of the length of the 3' end to the length of the 5' end of a DelVG.

    :param row: Series containing the indices Full_Sequence, Start and End

    :return: Ratio of the section lengths.
    '''
    return row['Start'] / (len(row['Full_Sequence']) - row['End'] - 1)


def get_3_len(row):
    '''
    Calculates the length of the 3' end of a DelVG, meaning the length from the end position to the end of the sequence.

    :param row: Series containing the indices Full_Sequence, Start and End

    :return: Length of the 3' end.
    '''
    return len(row['Full_Sequence']) - row['End'] - 1

def get_5_len(row):
    '''
    Calculates the length of the 5' end of a DelVG, meaning the length from the start position to the beginning of the deletion.

    :param row: Series containing the indices Full_Sequence, Start and End

    :return: Length of the 5' end.
    '''
    return row['Start']


def get_DI_Length(row):
    '''
    Calculates the length of the remaining sequence after removal of the deletion site.

    :param row: Series containing the indices Full Sequence, Start and End
    :return:
    '''
    if 'Full_Sequence' not in row:
        row['Full_Sequence'] = get_sequence(row['Strain'],row['Segment'])
    di_len = (len(row['Full_Sequence']) - row['End'] + 1) + row['Start']
    return di_len


def get_deletion_length(row):
    '''
    Calculates the length of the remaining sequence after removal of the deletion site.

    :param row: Series containing the indices Full Sequence, Start and End
    :return:
    '''
    deletion_len = row['End'] - (row['Start']-1)
    return deletion_len


def get_length_proportion(row):
    '''
    Calculates the length proportion of the leftover sequence after removing the deletion site, in comparison to the
    full sequence length.

    :param row: Series containing the DI_Length or Full_Sequence index

    :return: Proportion of DI_Length and Full_Sequence
    '''
    if 'DI_Length' in row.index:
        di_len = row['DI_Length'] = get_DI_Length(row)
    else:
        di_len = get_DI_Length(row)
    return di_len / (len(row['Full_Sequence']) if len(row['Full_Sequence'])>0 else len(get_sequence(row["Strain"], row["Segment"])))

def log_and_norm(df, norm = 'NGS_log_norm', experiment_col = 'Publication', drop_read_count=True, drop_0=True):
    '''
    Calculates the normed logarithm of the NGS read count for a given dataframe.

    :param norm: Normalization method to use. Options: NGS_log_norm, NGS_log_min_max_norm
    :param df: Dataframe containing the NGS_read_count column

    :return: Dataframe with the NGS_log_norm instead of NGS_read_count
    '''
    '''for i, row in df.iterrows():
        df.at[i, 'NGS_log'] = np.log(row['NGS_read_count'].astype(float))
    df['NGS_log_norm'] = df['NGS_log'] / max(df['NGS_log'])
    df = df.drop(['NGS_read_count', 'NGS_log'], axis=1)
    '''
    logging.info('Applying log and normalization to NGS read count.')
    assert experiment_col in df.columns, f'Column {experiment_col} not found in dataframe'
    if 'NGS' in norm:
        df["NGS_read_count"] = df["NGS_read_count"].astype(float)
        if len(df[df["NGS_read_count"]==0])>0:
            if drop_0:
                df = df[df["NGS_read_count"] > 0].copy()
            else:
                df["NGS_read_count"] = df["NGS_read_count"]+1
        df["NGS_log"] = np.log(df["NGS_read_count"]).astype(float)
        match norm:
            case 'NGS_log_norm':
                #df["NGS_log_norm"] = df["NGS_log"] / max(df["NGS_log"])
                df["NGS_log_norm"] = df.groupby(experiment_col)["NGS_log"].transform(lambda x: x / x.max() if x.max()>0 else 0)
            case 'NGS_log_min_max_norm':
                #df["NGS_log_min_max_norm"] = (df["NGS_log"]-min(df["NGS_log"])) / (max(df["NGS_log"])-min(df["NGS_log"]))
                df["NGS_log_min_max_norm"] = df.groupby(experiment_col)["NGS_log"].transform(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max()>x.min() else 0)
            case 'NGS_log_sig_norm':
                df["NGS_log_sig_norm"] = df.groupby(experiment_col)["NGS_log"].transform(lambda x: sigmoid_normalize(x, gain=1))
            case 'NGS_log_robust_sig_norm':
                df["NGS_log_robust"] = df.groupby(experiment_col)["NGS_log"].transform(robust_scale)
                df["NGS_log_robust_sig_norm"] = sigmoid_normalize(df["NGS_log_robust"])
                df.drop("NGS_log_robust", axis=1,inplace=True)
            case _:
                logging.error(f'Unknown norm method {norm} for log_and_norm! Applying standard log_norm instead.\nOptions: NGS_log_norm, NGS_log_min_max_norm')
                #df["NGS_log_norm"] = df["NGS_log"] / max(df["NGS_log"])
                df["NGS_log_norm"] = df.groupby(experiment_col)["NGS_log"].transform(lambda x: x / x.max())
        if drop_read_count:
            df = df.drop(["NGS_read_count", "NGS_log"], axis=1)
        else:
            df = df.drop(["NGS_log"], axis=1)
            if not drop_0:
                df["NGS_read_count"] = df["NGS_read_count"]-1
    elif 'inter' in norm.lower():
        match norm:
            case 'Inter_norm':
                df[norm] = df["Intersections"]/df[experiment_col].nunique()
                df = df.drop(["Intersections"], axis=1)
            case _:
                logging.error(f'Unknown norm method {norm} for log_and_norm! Applying standard norm instead.\nOptions: NGS_log_norm, NGS_log_min_max_norm')
                df[norm] = df["Intersections"]/df[experiment_col].nunique()
                df = df.drop(["Intersections"], axis=1)
    #    if drop_read_count:
    #        df = df.drop(["NGS_read_count"], axis=1)
    return df

def normalize_feature(df: pd.DataFrame, column):
    maximum = df[column].max()
    minimum = df[column].min()
    df[column] = (df[column]-minimum)/(maximum-minimum)
    return df


def sigmoid_normalize(series: pd.Series, gain: float = 1.0) -> pd.Series:
    '''
    Applies a sigmoid transformation to the input series.
    Larger gain -> steeper curve -> values pushed closer to 0/1.
    '''
    mu = series.mean()
    sigma = series.std() if series.std() > 0 else 1.0
    normalized = 1 / (1 + np.exp(-gain * (series - mu) / sigma))
    return normalized

def robust_scale(series: pd.Series) -> pd.Series:
    '''
    Applies robust scaling to the input series.
    '''
    median = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1 if q3 > q1 else 1.0
    
    scaled = (series - median) / iqr
    return scaled

def get_remaining_sequence(df: pd.DataFrame):
    if "Remaining" in df.columns:
        return df
    else:
        if not "Full_Sequence" in df.columns:
            df = get_sequence_quicker(df)
        remove_id = False
        if not "ID" in df.columns:
            df = identify_candidates(df)
            remove_id = True
        for id, group in df.groupby("ID"):
            seq = group["Full_Sequence"].values[0]
            start = group["End"].values[0]
            end = group["Start"].values[0]
            df.loc[group.index,"Remaining"] = seq[:int(start)-1]+seq[int(end):] #df.apply(lambda row: row["Full_Sequence"][:int(row["Start"])-1]+row["Full_Sequence"][int(row["End"]):], axis=1)
        if remove_id:
            df.drop(columns=["ID"], inplace=True)
        return df

def clustering_to_vip_candidates(clustering_data: dict, strategy: str = "centroid", include_noise: bool = False):
    '''
    Converts clustering assignments to representative IDs (one per cluster).
    '''
    cluster_df = clustering_data["clustering_df"]
    noise_label = clustering_data["metadata"].get("noise_label", -1)
    coord_columns = clustering_data.get("metadata", {}).get("coord_columns", [])
    centroid_columns = clustering_data.get("metadata", {}).get("centroid_coord_columns", [])
    representatives = []

    for cid, group in cluster_df.groupby("cluster_id"):
        if (cid == noise_label) and (not include_noise):
            continue
        member_ids = group["ID"].tolist()
        if len(member_ids) == 0:
            continue
        if strategy == "first" or len(coord_columns) == 0:
            representatives.append(member_ids[0])
            continue

        # pick the member closest to centroid in embedding space
        coord_frame = group[["ID"] + coord_columns].dropna(axis=0)
        if coord_frame.empty:
            representatives.append(member_ids[0])
            continue
        coords = coord_frame[coord_columns].to_numpy(dtype=float)
        centroid = None
        if len(centroid_columns) == len(coord_columns):
            centroid_rows = group[centroid_columns].dropna(axis=0)
            if not centroid_rows.empty:
                centroid = centroid_rows.iloc[0].to_numpy(dtype=float)
        if centroid is None:
            centroid = coords.mean(axis=0)
        dists = np.linalg.norm(coords - centroid, axis=1)
        representatives.append(coord_frame.iloc[int(np.argmin(dists))]["ID"])
    return representatives

def get_vip_distance_features(df: pd.DataFrame, clustering_data: dict, vip_candidates: list, summary: bool = False, summary_only: bool = False):
    '''
    Adds UMAP-distance features relative to VIP reference points (one column per VIP). Summary features (min/mean/max) are added if summary is True and multiple VIPs are provided.
    '''
    df = ensure_id_column(df)
    if not vip_candidates:
        return df
    coords, valid_mask = _get_embedding_for_ids(df, clustering_data)
    cluster_df = clustering_data["clustering_df"]
    coord_columns = clustering_data.get("metadata", {}).get("coord_columns", [])
    coord_lookup = cluster_df.set_index("ID")[coord_columns]

    dist_matrix = {}
    for vip in vip_candidates:
        if vip not in coord_lookup.index:
            continue
        row = coord_lookup.loc[vip]
        if row.isna().any():
            continue
        col_name = f"umap_dist_vip_{vip}"
        values = np.full(len(df), np.nan, dtype=float)
        if valid_mask.any():
            values[valid_mask] = np.linalg.norm(coords[valid_mask] - row.to_numpy(dtype=float), axis=1)
        if not summary_only:
            df[col_name] = values
        dist_matrix[col_name] = values

    if len(dist_matrix) == 0:
        logging.warning("No valid VIP coordinate references found for distance features")
        return df

    if summary and len(dist_matrix) > 1:
        stacked = np.vstack(list(dist_matrix.values())).T
        df["umap_dist_min"] = np.nanmin(stacked, axis=1)
        df["umap_dist_mean"] = np.nanmean(stacked, axis=1)
        df["umap_dist_max"] = np.nanmax(stacked, axis=1)
    return df


def get_centroid_distance_features(df: pd.DataFrame, clustering_data: dict, summary: bool = False, summary_only: bool = False):
    '''
    Adds UMAP-distance features relative to cluster centroids (one column per cluster).
    Centroid coordinates are taken from stored centroid columns when available,
    otherwise computed as the mean of member point coordinates.
    '''
    df = ensure_id_column(df)
    coords, valid_mask = _get_embedding_for_ids(df, clustering_data)
    cluster_df = clustering_data["clustering_df"]
    coord_columns = clustering_data.get("metadata", {}).get("coord_columns", [])
    centroid_columns = clustering_data.get("metadata", {}).get("centroid_coord_columns", [])
    noise_label = clustering_data["metadata"].get("noise_label", -1)

    dist_matrix = {}
    for cid, group in cluster_df.groupby("cluster_id"):
        if cid == noise_label:
            continue
        centroid = None
        if len(centroid_columns) == len(coord_columns):
            centroid_rows = group[centroid_columns].dropna(axis=0)
            if not centroid_rows.empty:
                centroid = centroid_rows.iloc[0].to_numpy(dtype=float)
        if centroid is None:
            valid_group = group[coord_columns].dropna(axis=0)
            if valid_group.empty:
                continue
            centroid = valid_group.to_numpy(dtype=float).mean(axis=0)
        col_name = f"umap_dist_cluster_{cid}"
        values = np.full(len(df), np.nan, dtype=float)
        if valid_mask.any():
            values[valid_mask] = np.linalg.norm(coords[valid_mask] - centroid, axis=1)
        if not summary_only:
            df[col_name] = values
        dist_matrix[col_name] = values

    if len(dist_matrix) == 0:
        logging.warning("No valid cluster centroid references found for distance features")
        return df

    if summary and len(dist_matrix) > 1:
        stacked = np.vstack(list(dist_matrix.values())).T
        df["umap_dist_min"] = np.nanmin(stacked, axis=1)
        df["umap_dist_mean"] = np.nanmean(stacked, axis=1)
        df["umap_dist_max"] = np.nanmax(stacked, axis=1)
    return df


def _extract_balanced_window(seq: str, anchor: int, flank: int) -> str:
    window_size = 2 * int(flank)
    left = int(anchor) - int(flank)
    right = int(anchor) + int(flank)
    if left < 0:
        right = min(len(seq), right + abs(left))
        left = 0
    if right > len(seq):
        left = max(0, left - (right - len(seq)))
        right = len(seq)
    window = seq[left:right]
    # For very short sequences we may not reach full window_size; keep the available span.
    return window[:window_size]


def _extract_junction_window(seq: str, start: int, end: int, flank: int = 5) -> str:
    # Start/End are 1-indexed first/last removed positions.
    # start_anchor gives flank remaining (left) + flank deleted (right).
    # end_anchor gives flank deleted (left) + flank remaining (right).
    start_anchor = int(start)
    end_anchor = int(end) - 1
    left = _extract_balanced_window(seq, start_anchor, flank)
    right = _extract_balanced_window(seq, end_anchor, flank)
    return left + "|" + right


def _identity_similarity(s1: str, s2: str) -> float:
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return np.nan
    s1 = s1.ljust(max_len, "-")
    s2 = s2.ljust(max_len, "-")
    matches = sum(a == b for a, b in zip(s1, s2))
    return matches / max_len

def _hamming_distance(s1: str, s2: str) -> float:
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return np.nan
    s1 = s1.ljust(max_len, "-")
    s2 = s2.ljust(max_len, "-")
    differences = sum(a != b for a, b in zip(s1, s2))
    return differences / max_len

def _consensus_motif(motifs: list[str]) -> str:
    if len(motifs) == 0:
        return ""
    max_len = max(len(m) for m in motifs)
    padded = [m.ljust(max_len, "-") for m in motifs]
    consensus = []
    for i in range(max_len):
        col = [m[i] for m in padded]
        values, counts = np.unique(col, return_counts=True)
        consensus.append(values[int(np.argmax(counts))])
    return "".join(consensus)


def _get_motif_for_id(candidate_id: str, source_df: pd.DataFrame, flank: int = 5) -> str:
    if "ID" in source_df.columns:
        rows = source_df[source_df["ID"] == candidate_id]
        if len(rows) > 0:
            row = rows.iloc[0]
            seq = row["Full_Sequence"] if "Full_Sequence" in row.index else get_sequence(row["Strain"], row["Segment"])
            return _extract_junction_window(seq, int(row["Start"]), int(row["End"]), flank=flank)
    strain, segment, start, end = parse_candidate_id(candidate_id)
    seq = get_sequence(strain, segment)
    return _extract_junction_window(seq, start, end, flank=flank)


def get_vip_motif_identity_features(df: pd.DataFrame, vip_candidates: list, flank: int = 5):
    '''
    Adds motif-identity features comparing each row's junction window to each VIP's junction window.
    '''
    df = ensure_id_column(df)
    if not vip_candidates:
        return df
    if "Full_Sequence" not in df.columns:
        df = get_sequence_quicker(df)
    df["_junction_window"] = df.apply(
        lambda row: _extract_junction_window(row["Full_Sequence"], int(row["Start"]), int(row["End"]), flank=flank),
        axis=1,
    )
    for vip in vip_candidates:
        ref_motif = _get_motif_for_id(vip, df, flank=flank)
        df[f"junction_identity_vip_{vip}"] = df["_junction_window"].map(
            lambda window: _identity_similarity(window, ref_motif)
        )
    df.drop(columns=["_junction_window"], inplace=True)
    return df


def get_cluster_motif_identity_features(df: pd.DataFrame, clustering_data: dict, flank: int = 5):
    '''
    Adds motif-identity features comparing each row's junction window to the consensus
    motif of each cluster.
    '''
    df = ensure_id_column(df)
    if "Full_Sequence" not in df.columns:
        df = get_sequence_quicker(df)
    df["_junction_window"] = df.apply(
        lambda row: _extract_junction_window(row["Full_Sequence"], int(row["Start"]), int(row["End"]), flank=flank),
        axis=1,
    )
    cluster_df = clustering_data["clustering_df"]
    noise_label = clustering_data["metadata"].get("noise_label", -1)
    for cid, group in cluster_df.groupby("cluster_id"):
        if cid == noise_label:
            continue
        motifs = []
        for member_id in group["ID"].tolist():
            try:
                motifs.append(_get_motif_for_id(member_id, df, flank=flank))
            except Exception:
                continue
        if motifs:
            ref_motif = _consensus_motif(motifs)
            df[f"junction_identity_motif_{cid}"] = df["_junction_window"].map(
                lambda window: _identity_similarity(window, ref_motif)
            )
    df.drop(columns=["_junction_window"], inplace=True)
    return df


def get_cluster_context_features(df: pd.DataFrame, clustering_data: dict, metric: str, target_column: str = "NGS_log_norm", experiment_col: str = "auto"):
    '''
    Adds context features derived from cluster-level signal per experiment.
    '''
    df = ensure_id_column(df)
    resolved_exp_col = resolve_experiment_column(df, experiment_col)

    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column} not found in dataframe")

    work = df[["ID", target_column, resolved_exp_col]].copy()
    cluster_df = clustering_data.get("clustering_df", None)
    if cluster_df is None:
        raise ValueError("clustering_data is missing clustering_df")
    id_to_cluster = cluster_df.set_index("ID")["cluster_id"]
    noise_label = clustering_data.get("metadata", {}).get("noise_label", -1)
    work["cluster_id"] = work["ID"].map(id_to_cluster)

    missing_cluster = work["cluster_id"].isna()
    if missing_cluster.any():
        missing_count = int(missing_cluster.sum())
        logging.warning(
            f"{missing_count} rows are missing cluster assignments for context feature '{metric}'. "
            f"Assigning noise_label={noise_label} as fallback."
        )
        work.loc[missing_cluster, "cluster_id"] = noise_label

    work["cluster_id"] = work["cluster_id"].astype(int)
    signal = (
        work.groupby([resolved_exp_col, "cluster_id"], as_index=False)
        .agg(
            cluster_mean_signal=(target_column, "mean"),
            cluster_median_signal=(target_column, "median"),
            cluster_count_signal=(target_column, "count"),
        )
    )

    cluster_sizes = (
        cluster_df
        .groupby("cluster_id", as_index=False)
        .size()
        .rename(columns={"size": "cluster_global_size"})
    )
    signal = signal.merge(cluster_sizes, on="cluster_id", how="left")

    signal["cluster_global_size"] = signal["cluster_global_size"].fillna(signal["cluster_count_signal"])
    signal["cluster_coverage"] = signal["cluster_count_signal"] / signal["cluster_global_size"].replace(0, np.nan)

    signal["cluster_rank_percentile"] = signal.groupby(resolved_exp_col)["cluster_mean_signal"].rank(
        pct=True,
        ascending=True,
        method="average",
    )

    work = work.merge(signal, on=[resolved_exp_col, "cluster_id"], how="left")
    feature_col = {
        "cluster_mean_signal": "cluster_mean_signal",
        "cluster_median_signal": "cluster_median_signal",
        "cluster_rank_percentile": "cluster_rank_percentile",
        "cluster_coverage": "cluster_coverage",
    }.get(metric, None)
    if feature_col is None:
        raise ValueError(f"Unknown cluster context metric: {metric}")

    if feature_col == "cluster_coverage":
        work[feature_col] = work[feature_col].fillna(0.0)
    elif feature_col == "cluster_rank_percentile":
        work[feature_col] = work[feature_col].fillna(0.5)
    else:
        work[feature_col] = work[feature_col].fillna(work.groupby(resolved_exp_col)[feature_col].transform("median"))
        work[feature_col] = work[feature_col].fillna(work[feature_col].median())

    df[feature_col] = work[feature_col].values
    return df


def get_cluster_membership_onehot_features(df: pd.DataFrame, clustering_data: dict, column_prefix: str = "cluster_membership"):
    '''
    Adds one-hot encoded cluster membership columns based on clustering assignments.
    '''
    df = ensure_id_column(df)
    cluster_df = clustering_data.get("clustering_df", None)
    if cluster_df is None:
        raise ValueError("clustering_data is missing clustering_df")
    id_to_cluster = cluster_df.set_index("ID")["cluster_id"]
    noise_label = clustering_data.get("metadata", {}).get("noise_label", -1)

    cluster_series = df["ID"].map(id_to_cluster)
    missing_mask = cluster_series.isna()
    if missing_mask.any():
        missing_count = int(missing_mask.sum())
        logging.warning(
            f"{missing_count} rows are missing cluster assignments for one-hot context feature. "
            f"Assigning noise_label={noise_label} as fallback."
        )
        cluster_series = cluster_series.fillna(noise_label)

    cluster_series = cluster_series.astype(int)

    dummies = pd.get_dummies(cluster_series, prefix=column_prefix, prefix_sep="_")
    
    dummies = dummies.astype(int)

    noise_col = f"{column_prefix}_{noise_label}"
    if noise_col in dummies.columns:
        dummies.rename(columns={noise_col: f"{column_prefix}_noise"}, inplace=True)

    for col in dummies.columns:
        df[col] = dummies[col].values
    return df


def get_sample_cluster_context_features(df: pd.DataFrame, clustering_data: dict, metric: str, target_column: str = "NGS_log_norm", experiment_col: str = "auto"):
    '''
    Adds sample-level context features for all clusters in the same experiment.

    For each experiment/sample, this creates one column per cluster with the
    chosen metric value for that cluster in that experiment. Values depend only
    on sample context, not on the candidate's own cluster ID.
    '''
    df = ensure_id_column(df)
    resolved_exp_col = resolve_experiment_column(df, experiment_col)
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column} not found in dataframe")

    noise_label = clustering_data.get("metadata", {}).get("noise_label", -1)
    cluster_df = clustering_data.get("clustering_df", None)
    if cluster_df is None:
        raise ValueError("clustering_data is missing clustering_df")
    id_to_cluster = cluster_df.set_index("ID")["cluster_id"]
    work = df[["ID", target_column, resolved_exp_col]].copy()
    work["cluster_id"] = work["ID"].map(id_to_cluster)

    missing_cluster = work["cluster_id"].isna()
    if missing_cluster.any():
        missing_count = int(missing_cluster.sum())
        logging.warning(
            f"{missing_count} rows are missing cluster assignments for sample-level context feature '{metric}'. "
            f"Assigning noise_label={noise_label} as fallback."
        )
        work.loc[missing_cluster, "cluster_id"] = noise_label

    work["cluster_id"] = work["cluster_id"].astype(int)
    signal = (
        work.groupby([resolved_exp_col, "cluster_id"], as_index=False)
        .agg(
            cluster_mean_signal=(target_column, "mean"),
            cluster_median_signal=(target_column, "median"),
            cluster_count_signal=(target_column, "count"),
        )
    )

    cluster_sizes = (
        cluster_df
        .groupby("cluster_id", as_index=False)
        .size()
        .rename(columns={"size": "cluster_global_size"})
    )
    signal = signal.merge(cluster_sizes, on="cluster_id", how="left")
    signal["cluster_global_size"] = signal["cluster_global_size"].fillna(signal["cluster_count_signal"])
    signal["cluster_coverage"] = signal["cluster_count_signal"] / signal["cluster_global_size"].replace(0, np.nan)
    signal["cluster_rank_percentile"] = signal.groupby(resolved_exp_col)["cluster_mean_signal"].rank(
        pct=True,
        ascending=True,
        method="average",
    )

    metric_col = {
        "sample_cluster_mean_signal": "cluster_mean_signal",
        "sample_cluster_rank_percentile": "cluster_rank_percentile",
        "sample_cluster_coverage": "cluster_coverage",
    }.get(metric, None)
    if metric_col is None:
        raise ValueError(f"Unknown sample-level cluster context metric: {metric}")

    prefix_map = {
        "sample_cluster_mean_signal": "sample_cluster_mean",
        "sample_cluster_rank_percentile": "sample_cluster_rank",
        "sample_cluster_coverage": "sample_cluster_coverage",
    }
    prefix = prefix_map[metric]

    pivot = signal.pivot(index=resolved_exp_col, columns="cluster_id", values=metric_col)
    
    pivot = pivot.fillna(0.0)
    pivot.columns = [f"{prefix}_{int(cid)}" for cid in pivot.columns]

    noise_col = f"{prefix}_{noise_label}"
    if noise_col in pivot.columns:
        pivot.rename(columns={noise_col: f"{prefix}_noise"}, inplace=True)

    merged = df.merge(pivot.reset_index(), on=resolved_exp_col, how="left")
    new_cols = [c for c in merged.columns if c not in df.columns]
    if new_cols:
        merged[new_cols] = merged[new_cols].fillna(0.0)
    return merged

def get_k_identities(df: pd.DataFrame, candidates: list):

    remove_id = False
    if not "ID" in df.columns:
        df = identify_candidates(df)
        remove_id = True
    remove_remaining = False
    if not "Remaining" in df.columns:
        df = get_remaining_sequence(df)
        remove_remaining = True

    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    #aligner.match_score = 1
    #aligner.mismatch_score = -1
    #aligner.open_gap_score = 0
    #aligner.extend_gap_score = 0
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -1


    prefilled_cols = set([prefil for prefil in [f'{cand}_identity' for cand in candidates] if prefil in df.columns])
    if len(prefilled_cols) > 0:
        logging.warning(f'The following {len(prefilled_cols)} identity columns already exist and will be skipped: {list(prefilled_cols)}')
    present_vips = [vip for vip in candidates if vip in set(df["ID"].unique().tolist()) and f'{vip}_identity' not in prefilled_cols]
    not_present_vips = [vip for vip in candidates if vip not in present_vips and f'{vip}_identity' not in prefilled_cols]
    if len(not_present_vips) > 0:
        logging.warning(f'The following candidates were not found in the dataframe and will have their remaining sequences calculated from their ID: {list(not_present_vips)}')
    logging.debug(f'Getting remaining sequences for {len(present_vips)} from precomputed remaining sequence')
    vip_seqs = {}
    for vip in present_vips:
        rem = df.loc[df["ID"]==vip, 'Remaining'].values[0]
        vip_seqs[vip] = rem
    logging.debug(f'Getting remaining sequences for {len(not_present_vips)} from ID parsing and sequence retrieval')
    for vip in not_present_vips:
        strain, seg, start, end = parse_candidate_id(vip)
        full = get_sequence(strain, seg)
        rem = full[:start-1]+full[end:]
        vip_seqs[vip] = rem

    try:
        import edlib
        def identity(ref, seq):
            result = edlib.align(ref, seq, mode="NW")  # global
            edit_dist = result["editDistance"]
            max_len = max(len(ref), len(seq))
            return 1 - edit_dist / max_len
    except ImportError:
        logging.warning('edlib not found, falling back to slower pairwise aligner for identity calculations. Install edlib for faster performance.')
        def identity(ref, seq):
            alignment = aligner.align(ref, seq)[0]
            matches = sum(1 for a, b in zip(alignment.aligned[0], alignment.aligned[1]))
            return matches / max(len(ref), len(seq))
        
    logging.debug(f'Calculating identities for {len(candidates)-len(prefilled_cols)} candidates against {df["ID"].nunique()} dataframe sequences')
    if df["ID"].nunique() < 100 or len(candidates) - len(prefilled_cols) < 10:
        logging.debug('Small scale identity calculation, using simple for loop.')
        for id, group in df.groupby("ID"):
            for cand in vip_seqs.keys():
                if f'{cand}_identity' in prefilled_cols:
                    continue
                #strain, segment, start, end = cand.rsplit('_',3)
                #ref_seq = get_sequence(strain, segment)
                #ref_seq = ref_seq[:int(start)-1]+ref_seq[int(end):]
                ref_seq = vip_seqs[cand]
                #df[cand+"_identity"] = df["Remaining"].transform(lambda seq: aligner.score(ref_seq,seq)/max(len(ref_seq),len(seq)))
                seq = group["Remaining"].values[0]
                #alignment = aligner.align(ref_seq, seq)[0]
                #matches = sum(1 for a, b in zip(alignment.aligned[0], alignment.aligned[1]))
                df.loc[group.index, f'{cand}_identity'] = identity(ref_seq, seq)#matches/max(len(ref_seq),len(seq))
                #df.loc[group.index, f'{cand}_identity'] = aligner.score(ref_seq,seq)/max(len(ref_seq),len(seq))
    else:
        logging.debug('Large scale identity calculation, using alternative approach.')
        
        unique_seqs = df["Remaining"].unique()
        for cand, ref_seq in vip_seqs.items():
            col = f"{cand}_identity"
            if col in prefilled_cols:
                continue
            score_map = {
                seq: identity(ref_seq, seq)
                for seq in unique_seqs
            }

            df[col] = df["Remaining"].map(score_map)
    if remove_id:
        df.drop(columns=["ID"], inplace=True, errors="ignore")
    if remove_remaining:
        df.drop(columns=["Remaining"], inplace=True, errors="ignore")
    return df

def get_k_diffs(df: pd.DataFrame, candidates: list):
    logging.info("Calculating vip ngs count differences")
    def min_max_scale(group):
        for cand in candidates:
            col = f'{cand}_diff'
            col_values = group[col]
            if col_values.notna().any():
                min_val = col_values.min()
                max_val = col_values.max()
                if min_val == max_val:
                    group[col] = 0.0
                else:
                    group[col] = (col_values - min_val) / (max_val - min_val)
        return group

    # Any missing vip would have ngs read count 0
    for cand in candidates:
        df[f'{cand}_diff'] = -df["NGS_read_count"]
    
    exp_col = "ACC_num" if "ACC_num" in df.columns else "Publication"
    # Compute differences per experiment
    for exp, group in df.groupby(exp_col):
        for cand in candidates:
            vip_rows = group.loc[group['ID'] == cand, 'NGS_read_count']
            if vip_rows.empty:
                continue
            vip_count = vip_rows.values[0]
            df.loc[group.index, f'{cand}_diff'] = vip_count - group['NGS_read_count']
        
    df = df.groupby(exp_col, group_keys=False).apply(min_max_scale)
    return df


def get_kmer_jaccard_similiarities(df:pd.DataFrame, candidates:list, k_list:list = [4,5,6]):
    def get_multi_kmers(seq):
        #kmers = set()
        #for k in k_list:
        #    kmers.update(seq[i:i+k] for i in range(len(seq) - k + 1))
        #return kmers
        return {seq[i:i+k] for k in k_list for i in range(len(seq) - k + 1)}

    logging.info("Calculating kmer jaccard-similarities")
    remove_id = False
    if not "ID" in df.columns:
        df = identify_candidates(df)
        remove_id = True
    remove_remaining = False
    if not "Remaining" in df.columns:
        df = get_remaining_sequence(df)
        remove_remaining = True
        
    '''try:
        vip_kmers = {vip: df.loc[df['ID'] == vip, 'kmers'].values[0] for vip in candidates}
    except Exception as e:
        logging.error(f'Issue with kmer jaccard-similarities:\n{e}\nTrying extra safe method.')
        vip_kmers = {}
        for vip in candidates:
            if "ID" not in df["ID"].unique():
                logging.debug(f'VIP not found in dataframe: {vip}\nCalculating kmers from ID..')
                strain, seg, start, end = parse_candidate_id(vip)
                full = get_sequence(strain, seg)
                rem = full[:start-1]+full[end:]
                vip_kmers[vip] = get_multi_kmers(rem)
            elif len(df.loc[df["ID"]==vip, 'kmers'].values) < 1:
                logging.error(f'VIP without kmers: {vip}')
            else:
                vip_kmers[vip] = df.loc[df['ID'] == vip, 'kmers'].values[0]'''
    vip_kmers = {}
    seq_to_kmers = {}
    present_vips = [vip for vip in candidates if vip in set(df["ID"].unique().tolist())]
    not_present_vips = [vip for vip in candidates if vip not in present_vips]
    if len(not_present_vips) > 0:
        logging.warning(f'The following candidates were not found in the dataframe and will have their kmers calculated from their ID: {list(not_present_vips)}')
    logging.debug(f'Getting kmers for {len(present_vips)} from precomputed remaining sequence')
    for vip in present_vips:
        rem = df.loc[df["ID"]==vip, 'Remaining'].values[0]
        vip_kmers[vip] = get_multi_kmers(rem)
        seq_to_kmers[rem] = vip_kmers[vip]
    logging.debug(f'Getting kmers for {len(not_present_vips)} from ID parsing and sequence retrieval')
    for vip in not_present_vips:
        strain, seg, start, end = parse_candidate_id(vip)
        full = get_sequence(strain, seg)
        rem = full[:start-1]+full[end:]
        vip_kmers[vip] = get_multi_kmers(rem)
        seq_to_kmers[rem] = vip_kmers[vip]
    prefilled_cols = set([prefil for prefil in [f'{cand}_kmer_sim' for cand in candidates] if prefil in df.columns])
    if len(prefilled_cols) > 0:
        logging.warning(f'The following {len(prefilled_cols)} kmer similarity columns already exist and will be skipped: {list(prefilled_cols)}')

    logging.debug(f'Calculating kmer similarities for {df["ID"].nunique()} IDs to {len(candidates)} candidates')
    seq_to_kmers.update({seq: get_multi_kmers(seq) for seq in df["Remaining"].unique() if seq not in seq_to_kmers})
    results = {cand: {} for cand in vip_kmers}

    for seq, kmers in seq_to_kmers.items():
        for cand, vip_km in vip_kmers.items():
            col = f"{cand}_kmer_sim"
            if col in prefilled_cols:
                continue

            if not kmers and not vip_km:
                sim = np.nan
            else:
                inter = len(kmers & vip_km)
                sim = inter / (len(kmers) + len(vip_km) - inter)
                #sim = len(kmers & vip_km) / len(kmers | vip_km)

            results[cand][seq] = sim

    for cand in vip_kmers:
        col = f"{cand}_kmer_sim"
        if col in prefilled_cols:
            continue

        df[col] = df["Remaining"].map(results[cand])

    '''# handle self-match
    df.loc[df["ID"] == cand, col] = 1.0
    for id, group in df.groupby("ID"):
        seq = group["Remaining"].values[0]
        kmers = get_multi_kmers(seq)
        for cand in vip_kmers.keys():
            if f'{cand}_kmer_sim' in prefilled_cols:
                continue
            if id == cand:
                df.loc[group.index, f'{cand}_kmer_sim'] = 1.0
                continue
            vip_km = vip_kmers[cand]
            df.loc[group.index, f'{cand}_kmer_sim'] = len(kmers & vip_km) / len(kmers | vip_km) if kmers or vip_km else np.nan'''
    
    #df.drop(columns=["kmers"],inplace=True)
    if remove_id:
        df.drop(columns=["ID"], inplace=True, errors="ignore")
    if remove_remaining:
        df.drop(columns=["Remaining"], inplace=True, errors="ignore")
    return df



def calculate_standard_features(dataframe: pd.DataFrame, features: list, seq_req=True, only_numeric=True, scaling="standard", inplace=True):
    '''
        Calculates the features for the given dataframe and returns the dataframe with all features.

        :param df: DataFrame with the data
        :param features: List of features to calculate
        :param seq_req: Boolean if sequence is needed for calculations

        :return: DataFrame with all features calculated and non-numerical columns removed
    '''
    logging.debug(f'Loaded influenza_info.json:\n{influenza_info}\nCHARS: {CHARS}\nSEGMENTS: {SEGMENTS}\nSTRAINS: {STRAINS}')

    if inplace:
        df = dataframe
    else:
        df = dataframe.copy()

    if isinstance(df, types.GeneratorType):
        return pd.concat([calculate_standard_features(chunk,features,seq_req,only_numeric) for chunk in df],ignore_index=True)

    # Helper to speed up calculations that use the full sequence
    if seq_req:
        logging.debug(f'Calculating full sequence for each input row.')
        #df["Full_Sequence"] = df.apply(lambda row: get_sequence(row['Strain'], row['Segment']), axis=1)
        df = get_sequence_quicker(df)

    for feat in features:
        if isinstance(feat, tuple):
            match feat[0]:
                case 'K-VIPs':
                    vip_lists = read_json_lists("vips.json")
                    #"scaffold_hdbscan_A_PuertoRico_8_1934"
                    logging.debug('Calculating K-Varying')
                    chosen_list = list(vip_lists)[feat[1]]
                    k_vary_candidates = vip_lists[chosen_list]
                    #print(k_vary_candidates)
                    logging.info(f'Using K-Varying VIPs of list {chosen_list}: {list(k_vary_candidates)}\n\n')
                    df = get_k_varying(df, k_vary_candidates, feat[2], feat[3])
                    df = get_k_identities(df, k_vary_candidates)
                    if len(feat) > 4:
                        df = get_kmer_jaccard_similiarities(df, k_vary_candidates, feat[4])
                case 'K-Varying':
                    #try:
                    logging.debug('Calculating K-Varying')
                    k_vary_candidates = find_k_varying(df, feat[1], feat[2], feat[4] if len(feat)==5 else 'std')
                    #print(k_vary_candidates)
                    logging.info(f'K-Varying Candidates: {list(k_vary_candidates)}\n\n')
                    df = get_k_varying(df, k_vary_candidates, feat[2], feat[3])
                    df = get_k_identities(df, k_vary_candidates)
                    if len(feat) > 5:
                        df = get_kmer_jaccard_similiarities(df, k_vary_candidates, feat[5])
                    #except Exception as e:
                    #    logging.error(f'Error: {e}')
                case 'Radius':
                    logging.debug('Calculating fuzzy locations')
                    df = get_fuzzy_locations(df)
                case _:
                    logging.info(f'\n {feat[0]} not found and therefore ignored')
        else:
            match feat.lower():
                case '3_5_ratio':
                    assert 'Full_Sequence' in df.columns, 'Full_Sequence column needed for 3_5_ratio calculation'
                    logging.debug('Calculating 3_5_ratio')
                    df["3_5_ratio"] = df.apply(lambda row: get_3_5_ratio(row), axis=1)
                case 'di_length':
                    assert 'Full_Sequence' in df.columns, 'Full_Sequence column needed for DI_Length calculation'
                    logging.debug('Calculating DI_Length')
                    df["DI_Length"] = df.apply(lambda row: get_DI_Length(row), axis=1)
                    if scaling == "standard":
                        df["DI_Length"] = StandardScaler().fit_transform(pd.DataFrame(df["DI_Length"]))
                    else:
                        minimum = df["DI_Length"].min()
                        maximum = df["DI_Length"].max()
                        df["DI_Length"] = (df["DI_Length"]-minimum)/(maximum-minimum) # feature scaling to [0,1]
                case 'remaining_length':
                    assert 'Full_Sequence' in df.columns, 'Full_Sequence column needed for DI_Length calculation'
                    logging.debug('Calculating remaining_length (DI_Length)')
                    df["remaining_length"] = df.apply(lambda row: get_DI_Length(row), axis=1)
                    if scaling == "standard":
                        df["remaining_length"] = StandardScaler().fit_transform(pd.DataFrame(df["remaining_length"]))
                    else:
                        minimum = df["remaining_length"].min()
                        maximum = df["remaining_length"].max()
                        df["remaining_length"] = (df["remaining_length"]-minimum)/(maximum-minimum) # feature scaling to [0,1]
                case 'deletion_length':
                    logging.debug('Calculating deletion_length')
                    df["deletion_length"] = df.apply(lambda row: get_deletion_length(row), axis=1)
                    if scaling == "standard":
                        df["deletion_length"] = StandardScaler().fit_transform(pd.DataFrame(df["deletion_length"]))
                    else:
                        minimum = df["deletion_length"].min()
                        maximum = df["deletion_length"].max()
                        df["deletion_length"] = (df["deletion_length"]-minimum)/(maximum-minimum) # feature scaling to [0,1]
                case 'direct_repeat':
                    assert 'Full_Sequence' in df.columns, 'Full_Sequence column needed for Direct_repeat calculation'
                    logging.debug('Calculating Direct_repeat')
                    df["Direct_repeat"] = df.apply(lambda row: get_direct_repeat_length(row), axis=1)
                    #logging.info(f'Direct_repeat columns:\n{df["Direct_repeat"].head()}\n{df["Direct_repeat"].describe()}\n\nFull df:\n{df.head()}\n{df.describe()}')
                    # normalizing to [0,1]
                    #minimum = df["Direct_repeat"].min()
                    #maximum = df["Direct_repeat"].max()
                    if scaling == "standard":
                        df["Direct_repeat"] = StandardScaler().fit_transform(pd.DataFrame(df["Direct_repeat"]))
                    else:
                        minimum = 0
                        maximum = 15
                        df["Direct_repeat"] = (df["Direct_repeat"]-minimum)/(maximum-minimum)
                case '3_5_diff':
                    assert 'Full_Sequence' in df.columns, 'Full_Sequence column needed for 3_5_diff calculation'
                    logging.debug('Calculating 3_5_diff')
                    df["3_5_diff"] = df.apply(lambda row: get_3_5_diff(row), axis=1)
                    if scaling == "standard":
                        df["3_5_diff"] = StandardScaler().fit_transform(pd.DataFrame(df["3_5_diff"]))
                    else:
                        # normalizing to [0,1]
                        #minimum = df["3_5_diff"].min()
                        minimum = 20-max([len(r) for r in df["Full_Sequence"]])
                        df["3_5_diff"] = df["3_5_diff"]+abs(minimum)
                        #maximum = df["3_5_diff"].max()
                        maximum = abs(minimum)+20
                        df["3_5_diff"] = df["3_5_diff"]/(2*maximum)
                case 'length_proportion':
                    assert 'Full_Sequence' in df.columns, 'Full_Sequence column needed for length_proportion calculation'
                    logging.debug('Calculating length_proportion')
                    df["length_proportion"] = df.apply(lambda row: get_length_proportion(row), axis=1)
                    if scaling == "standard":
                        df["length_proportion"] = StandardScaler().fit_transform(pd.DataFrame(df["length_proportion"]))
                    # normalizing to [0,1]... proportion is already in that interval
                    #minimum = df["length_proportion"].min()
                    #maximum = df["length_proportion"].max()
                    #df["length_proportion"] = (df["length_proportion"]-minimum)/(maximum-minimum)
                    logging.debug(f'Found {df[df["length_proportion"]>=0.85].shape[0]} long DelVGs out of {df.shape[0]} total.')
                case 'peptide_length':
                    assert 'Full_Sequence' in df.columns, 'Full_Sequence column needed for Peptide_Length calculation'
                    logging.debug('Calculating Peptide_Length')
                    df['Peptide_Length'] = df.apply(lambda row: get_peptide_len(row), axis=1)
                    if scaling == "standard":
                        df["Peptide_Length"] = StandardScaler().fit_transform(pd.DataFrame(df["Peptide_Length"]))
                    else:
                        # normalizing to [0,1]
                        minimum = df["Peptide_Length"].min()
                        maximum = df["Peptide_Length"].max()
                        df["Peptide_Length"] = (df["Peptide_Length"]-minimum)/(maximum-minimum)
                case 'delta_g':
                    assert 'Full_Sequence' in df.columns, 'Full_Sequence column needed for Delta_G calculation'
                    logging.debug('Calculating Delta_G')
                    df['Delta_G'] = df.apply(lambda row: get_delta_G(row), axis=1)
                    if scaling == "standard":
                        df["Delta_G"] = StandardScaler().fit_transform(pd.DataFrame(df["Delta_G"]))
                    else:
                        # normalizing to [0,1]
                        minimum = df["Delta_G"].min()
                        maximum = df["Delta_G"].max()
                        df["Delta_G"] = (df["Delta_G"]-minimum)/(maximum-minimum)
                case _:
                    if not (feat.lower() in ["start", "end", "junction", "strain", "segment", "sequence"]):
                        logging.info(f'\n {feat:s} not found and therefore ignored')
    df = get_ohes(df, features)
    if 'Start' not in features:
        df = df.drop("Start", axis=1)
    else: # scale
        if scaling == "standard":
            df["Start"] = StandardScaler.fit_transform(pd.DataFrame(df["Start"]))
        else:
            df["Start"] = df["Start"]/df["Full_Sequence"].transform(len) # to [0,1]
    if 'End' not in features:
        df = df.drop("End", axis=1)
    else: # scale
        if scaling == "standard":
            df["End"] = StandardScaler.fit_transform(pd.DataFrame(df["End"]))
        else:
            df["End"] = df["End"]/df["Full_Sequence"].transform(len) # to [0,1]
    #if only_numeric:
    #    df = drop_non_numeric(df)

    logging.debug(f'Resulting dataframe after feature calculation: {list(df.columns)}\n{df.head()}\n')
    # TODO: Go over uses and make sure this is right before you change it.
    #if inplace:
        #return 
    return df.copy()

def get_standard_feature(dataframe:pd.DataFrame,
                         feature:str,
                         scale:str="standard",
                         normalize_by_length:bool=True,
                         inplace:bool=True):
    if inplace:
        df = dataframe
    else:
        df = dataframe.copy()
    if "Full_Sequence" not in df.columns:
        df = get_sequence_quicker(df)

    match feature.lower():
        case '3_5_ratio':
            assert 'Full_Sequence' in df.columns, 'Full_Sequence column needed for 3_5_ratio calculation'
            logging.debug('Calculating 3_5_ratio')
            df["3_5_ratio"] = df.apply(lambda row: get_3_5_ratio(row), axis=1)
            if scale == "standard":
                df["3_5_ratio"] = StandardScaler().fit_transform(pd.DataFrame(df["3_5_ratio"]))
        case 'di_length':
            assert 'Full_Sequence' in df.columns, 'Full_Sequence column needed for DI_Length calculation'
            logging.debug('Calculating DI_Length')
            df["DI_Length"] = df.apply(lambda row: get_DI_Length(row), axis=1)
            if normalize_by_length:
                df["DI_Length"] = df["DI_Length"]/df["Full_Sequence"].transform(len) # to [0,1]
            if scale == "standard":
                df["DI_Length"] = StandardScaler().fit_transform(pd.DataFrame(df["DI_Length"]))
            elif scale == "none":
                pass
            else:
                minimum = df["DI_Length"].min()
                maximum = df["DI_Length"].max()
                df["DI_Length"] = (df["DI_Length"]-minimum)/(maximum-minimum) # feature scaling to [0,1]
        case 'remaining_length':
            assert 'Full_Sequence' in df.columns, 'Full_Sequence column needed for DI_Length calculation'
            logging.debug('Calculating remaining_length (DI_Length)')
            df["remaining_length"] = df.apply(lambda row: get_DI_Length(row), axis=1)
            if normalize_by_length:
                df["remaining_length"] = df["remaining_length"]/df["Full_Sequence"].transform(len) # to [0,1]
            if scale == "standard":
                df["remaining_length"] = StandardScaler().fit_transform(pd.DataFrame(df["remaining_length"]))
            elif scale == "none":
                pass
            else:
                minimum = df["remaining_length"].min()
                maximum = df["remaining_length"].max()
                df["remaining_length"] = (df["remaining_length"]-minimum)/(maximum-minimum) # feature scaling to [0,1]
        case 'deletion_length':
            logging.debug('Calculating deletion_length')
            df["deletion_length"] = df.apply(lambda row: get_deletion_length(row), axis=1)
            if normalize_by_length:
                df["deletion_length"] = df["deletion_length"]/df["Full_Sequence"].transform(len) # to [0,1]
            if scale == "standard":
                df["deletion_length"] = StandardScaler().fit_transform(pd.DataFrame(df["deletion_length"]))
            elif scale == "none":
                pass
            else:
                minimum = df["deletion_length"].min()
                maximum = df["deletion_length"].max()
                df["deletion_length"] = (df["deletion_length"]-minimum)/(maximum-minimum) # feature scaling to [0,1]
        case 'direct_repeat':
            assert 'Full_Sequence' in df.columns, 'Full_Sequence column needed for Direct_repeat calculation'
            logging.debug('Calculating Direct_repeat')
            df["Direct_repeat"] = df.apply(lambda row: get_direct_repeat_length(row), axis=1)
            #logging.info(f'Direct_repeat columns:\n{df["Direct_repeat"].head()}\n{df["Direct_repeat"].describe()}\n\nFull df:\n{df.head()}\n{df.describe()}')
            if scale == "standard":
                df["Direct_repeat"] = StandardScaler().fit_transform(pd.DataFrame(df["Direct_repeat"]))
            elif scale == "none":
                pass
            else:
                # normalizing to [0,1]
                #minimum = df["Direct_repeat"].min()
                #maximum = df["Direct_repeat"].max()
                minimum = 0
                maximum = 15
                df["Direct_repeat"] = (df["Direct_repeat"]-minimum)/(maximum-minimum)
        case '3_5_diff':
            assert 'Full_Sequence' in df.columns, 'Full_Sequence column needed for 3_5_diff calculation'
            logging.debug('Calculating 3_5_diff')
            df["3_5_diff"] = df.apply(lambda row: get_3_5_diff(row), axis=1)
            if normalize_by_length:
                df["3_5_diff"] = df["3_5_diff"]/df["Full_Sequence"].transform(len) # to [0,1]
            if scale == "standard":
                df["3_5_diff"] = StandardScaler().fit_transform(pd.DataFrame(df["3_5_diff"]))
            elif scale == "none":
                pass
            else:
                # normalizing to [0,1]
                #minimum = df["3_5_diff"].min()
                minimum = 20-max([len(r) for r in df["Full_Sequence"]])
                df["3_5_diff"] = df["3_5_diff"]+abs(minimum)
                #maximum = df["3_5_diff"].max()
                maximum = abs(minimum)+20
                df["3_5_diff"] = df["3_5_diff"]/(2*maximum)
        case 'length_proportion':
            assert 'Full_Sequence' in df.columns, 'Full_Sequence column needed for length_proportion calculation'
            logging.debug('Calculating length_proportion')
            df["length_proportion"] = df.apply(lambda row: get_length_proportion(row), axis=1)
            if scale == "standard":
                df["length_proportion"] = StandardScaler().fit_transform(pd.DataFrame(df["length_proportion"]))
            elif scale == "none":
                pass
            else:
                # normalizing to [0,1]... proportion is already in that interval
                minimum = df["length_proportion"].min()
                maximum = df["length_proportion"].max()
                df["length_proportion"] = (df["length_proportion"]-minimum)/(maximum-minimum)
            logging.debug(f'Found {df[df["length_proportion"]>=0.85].shape[0]} long DelVGs out of {df.shape[0]} total.')
        case 'peptide_length':
            assert 'Full_Sequence' in df.columns, 'Full_Sequence column needed for Peptide_Length calculation'
            logging.debug('Calculating Peptide_Length')
            df['Peptide_Length'] = df.apply(lambda row: get_peptide_len(row), axis=1)
            if scale == "standard":
                df["Peptide_Length"] = StandardScaler().fit_transform(pd.DataFrame(df["Peptide_Length"]))
            elif scale == "none":
                pass
            else:
                # normalizing to [0,1]
                minimum = df["Peptide_Length"].min()
                maximum = df["Peptide_Length"].max()
                df["Peptide_Length"] = (df["Peptide_Length"]-minimum)/(maximum-minimum)
        case 'delta_g':
            assert 'Full_Sequence' in df.columns, 'Full_Sequence column needed for Delta_G calculation'
            logging.debug('Calculating Delta_G')
            df['Delta_G'] = df.apply(lambda row: get_delta_G(row), axis=1)
            if scale == "standard":
                df["Delta_G"] = StandardScaler().fit_transform(pd.DataFrame(df["Delta_G"]))
            elif scale == "none":
                pass
            else:
                # normalizing to [0,1]
                minimum = df["Delta_G"].min()
                maximum = df["Delta_G"].max()
                df["Delta_G"] = (df["Delta_G"]-minimum)/(maximum-minimum)
        case '3_len':
            assert 'Full_Sequence' in df.columns, 'Full_Sequence column needed for 3_len calculation'
            logging.debug('Calculating 3_len')
            df["3_len"] = df.apply(lambda row: get_3_len(row), axis=1)
            if normalize_by_length:
                df["3_len"] = df["3_len"]/df["Full_Sequence"].transform(len) # to [0,1]
            if scale == "standard":
                df["3_len"] = StandardScaler().fit_transform(pd.DataFrame(df["3_len"]))
            elif scale == "none":
                pass
            else:
                # normalizing to [0,1]
                #minimum = df["3_len"].min()
                minimum = 20-max([len(r) for r in df["Full_Sequence"]])
                df["3_len"] = df["3_len"]+abs(minimum)
                #maximum = df["3_len"].max()
                maximum = abs(minimum)+20
                df["3_len"] = df["3_len"]/(2*maximum)
        case '5_len':
            assert 'Full_Sequence' in df.columns, 'Full_Sequence column needed for 5_len calculation'
            logging.debug('Calculating 5_len')
            df["5_len"] = df.apply(lambda row: get_5_len(row), axis=1)
            if normalize_by_length:
                df["5_len"] = df["5_len"]/df["Full_Sequence"].transform(len) # to [0,1]
            if scale == "standard":
                df["5_len"] = StandardScaler().fit_transform(pd.DataFrame(df["5_len"]))
            elif scale == "none":
                pass
            else:
                # normalizing to [0,1]
                #minimum = df["5_len"].min()
                minimum = 20-max([len(r) for r in df["Full_Sequence"]])
                df["5_len"] = df["5_len"]+abs(minimum)
                #maximum = df["5_len"].max()
                maximum = abs(minimum)+20
                df["5_len"] = df["5_len"]/(2*maximum)
        case _:
            if not (feature.lower() in ["start", "end", "junction", "strain", "segment", "sequence"]):
                logging.info(f'\n {feature:s} not found and therefore ignored')
    
    return df

def get_intersection_feature(dataframe:pd.DataFrame,
                             feature:str,
                             vips:list=None,
                             clustering_data:dict|None=None,
                             motif_flank:int=5,
                             use_vip_references:bool=True,
                             use_centroid_references:bool=True,
                             distance_summary_only:bool=False,
                             scale:str="standard",
                             inplace:bool=True):
    if inplace:
        df = dataframe
    else:
        df = dataframe.copy()
    
    match feature.lower():
        case 'kmer_jaccard':
            if vips:
                df = get_kmer_jaccard_similiarities(df=df, candidates=vips, k_list=[4,5,6])
            else:
                logging.warning("Intersection feature kmer_jaccard requested without VIP candidates; skipping")
        case 'identities':
            if vips:
                df = get_k_identities(df=df, candidates=vips)
            else:
                logging.warning("Intersection feature identities requested without VIP candidates; skipping")
        case 'umap_distances' | 'embedding_distances':
            if clustering_data is None:
                logging.warning("Intersection feature umap_distances requested without clustering_data; skipping")
            else:
                if use_vip_references and vips:
                    df = get_vip_distance_features(
                        df=df,
                        clustering_data=clustering_data,
                        vip_candidates=vips,
                        summary_only=distance_summary_only,
                    )
                if use_centroid_references:
                    df = get_centroid_distance_features(
                        df=df,
                        clustering_data=clustering_data,
                        summary_only=distance_summary_only,
                    )
        case 'motif_identity' | 'junction_motif_identity':
            if use_vip_references and vips:
                df = get_vip_motif_identity_features(
                    df=df,
                    vip_candidates=vips,
                    flank=motif_flank,
                )
        case 'motif_consensus_identity' | 'cluster_motif_identity':
            if clustering_data is None:
                logging.warning("Motif consensus feature requested without clustering_data; skipping")
            else:
                df = get_cluster_motif_identity_features(
                    df=df,
                    clustering_data=clustering_data,
                    flank=motif_flank,
                )
        case _:
            logging.info(f'Intersection feature {feature} not found and therefore ignored')
    return df

def get_context_feature(dataframe:pd.DataFrame,
                        feature:str, vips:list=None,
                        clustering_data:dict|None=None,
                        target_column:str="NGS_log_norm",
                        experiment_col:str="ACC_num",
                        scale:str="standard",
                        inplace:bool=True):
    if inplace:
        df = dataframe
    else:
        df = dataframe.copy()
    
    match feature.lower():
        case 'current_results':
            if vips:
                df = get_current_values(df=df, candidates=vips, y_column=target_column, experiment_col=experiment_col)
            else:
                logging.warning("Context feature current_results requested without VIP candidates; skipping")
        case 'last_results':
            if vips:
                df = get_last_values(df=df, candidates=vips, y_column=target_column, experiment_col=experiment_col)
            else:
                logging.warning("Context feature last_results requested without VIP candidates; skipping")
        case 'cluster_mean_signal':
            if clustering_data is None:
                logging.warning("Context feature cluster_mean_signal requested without clustering_data; skipping")
            else:
                df = get_cluster_context_features(
                    df=df,
                    clustering_data=clustering_data,
                    metric='cluster_mean_signal',
                    target_column=target_column,
                    experiment_col=experiment_col,
                )
        case 'cluster_median_signal':
            if clustering_data is None:
                logging.warning("Context feature cluster_median_signal requested without clustering_data; skipping")
            else:
                df = get_cluster_context_features(
                    df=df,
                    clustering_data=clustering_data,
                    metric='cluster_median_signal',
                    target_column=target_column,
                    experiment_col=experiment_col,
                )
        case 'cluster_rank_percentile':
            if clustering_data is None:
                logging.warning("Context feature cluster_rank_percentile requested without clustering_data; skipping")
            else:
                df = get_cluster_context_features(
                    df=df,
                    clustering_data=clustering_data,
                    metric='cluster_rank_percentile',
                    target_column=target_column,
                    experiment_col=experiment_col,
                )
        case 'cluster_coverage':
            if clustering_data is None:
                logging.warning("Context feature cluster_coverage requested without clustering_data; skipping")
            else:
                df = get_cluster_context_features(
                    df=df,
                    clustering_data=clustering_data,
                    metric='cluster_coverage',
                    target_column=target_column,
                    experiment_col=experiment_col,
                )
        case 'sample_cluster_mean_signal':
            if clustering_data is None:
                logging.warning("Context feature sample_cluster_mean_signal requested without clustering_data; skipping")
            else:
                df = get_sample_cluster_context_features(
                    df=df,
                    clustering_data=clustering_data,
                    metric='sample_cluster_mean_signal',
                    target_column=target_column,
                    experiment_col=experiment_col,
                )
        case 'sample_cluster_rank_percentile':
            if clustering_data is None:
                logging.warning("Context feature sample_cluster_rank_percentile requested without clustering_data; skipping")
            else:
                df = get_sample_cluster_context_features(
                    df=df,
                    clustering_data=clustering_data,
                    metric='sample_cluster_rank_percentile',
                    target_column=target_column,
                    experiment_col=experiment_col,
                )
        case 'sample_cluster_coverage':
            if clustering_data is None:
                logging.warning("Context feature sample_cluster_coverage requested without clustering_data; skipping")
            else:
                df = get_sample_cluster_context_features(
                    df=df,
                    clustering_data=clustering_data,
                    metric='sample_cluster_coverage',
                    target_column=target_column,
                    experiment_col=experiment_col,
                )
        case 'cluster_membership_onehot' | 'cluster_onehot' | 'cluster_id_onehot':
            if clustering_data is None:
                logging.warning("Context feature cluster_membership_onehot requested without clustering_data; skipping")
            else:
                df = get_cluster_membership_onehot_features(
                    df=df,
                    clustering_data=clustering_data,
                    column_prefix='cluster_membership',
                )
        case _:
            logging.info(f'Context feature {feature} not found and therefore ignored')
    return df
    

def calculate_features(dataframe:pd.DataFrame,
                       standard_features:list|None=None,
                       context_features:list|None=None,
                       intersection_features:list|None=None,
                       vips:str|int|list|None=None,
                       target_column:str="NGS_log_norm",
                       experiment_column:str="ACC_num",
                       clustering_data:dict|None=None,
                       clustering_dir:str|None=None,
                       clustering_strain:str|None=None,
                       clustering_type:str="scaffold",
                       clustering_subtype:str|None=None,
                       clustering_cutoff:int|str|None=None,
                       cluster_labels_path:str|None=None,
                       motif_flank:int=5,
                       use_vip_references:bool=True,
                       use_centroid_references:bool=True,
                       distance_summary_only:bool=False,
                       scale:str="standard",
                       normalize_by_length:bool=True,
                       inplace=True):
    if inplace:
        df = dataframe
    else:
        df = dataframe.copy()
    
    # starting with standard features
    if standard_features is not None:
        for feat in standard_features:
            get_standard_feature(dataframe=df, feature=feat, scale=scale, normalize_by_length=normalize_by_length, inplace=True)
        df = get_ohes(df, standard_features)

    # If neither feature family is requested, return the current dataframe as-is.
    if not context_features and not intersection_features:
        logging.info("No context or intersection features requested, skipping vip and clustering processing.")
        return df
    if isinstance(vips, int) and vips < 0:
        logging.info(f"Ignoring negative VIP selector ({vips}); using no VIP references")
        vips = None

    if isinstance(vips, (str, int)):
        try:
            vip_lists = read_json_lists("vips.json")
            if isinstance(vips, int):
                chosen_list = list(vip_lists)[vips]
                vips = vip_lists[chosen_list]
            elif isinstance(vips, str):
                if vips in vip_lists:
                    vips = vip_lists[vips]
                elif vips.isdigit():
                    chosen_list = list(vip_lists)[int(vips)]
                    vips = vip_lists[chosen_list]
                else:
                    logging.warning(f'VIP list key {vips} not found in vips.json')
                    vips = None
            logging.info(f'Loaded VIP reference set with {0 if vips is None else len(vips)} candidates')
        except Exception as e:
            logging.warning(f'Failed to resolve vip list {vips}: {e}')
            vips = None

    requested_context = {f.lower() for f in (context_features or []) if isinstance(f, str)}
    requested_intersection = {f.lower() for f in (intersection_features or []) if isinstance(f, str)}
    clustering_context_features = {
        "cluster_mean_signal", "cluster_median_signal", "cluster_rank_percentile", "cluster_coverage",
        "sample_cluster_mean_signal", "sample_cluster_rank_percentile", "sample_cluster_coverage",
        "cluster_membership_onehot", "cluster_onehot", "cluster_id_onehot",
    }
    clustering_intersection_features = {
        "umap_distances", "embedding_distances", "motif_consensus_identity", "cluster_motif_identity"
    }
    needs_clustering = bool(
        requested_context.intersection(clustering_context_features)
        or requested_intersection.intersection(clustering_intersection_features)
    )

    if needs_clustering and clustering_data is None and (
        clustering_dir is not None
        or clustering_strain is not None
        or clustering_subtype is not None
        or cluster_labels_path is not None
    ):
        try:
            clustering_data = load_clustering_artifacts(
                strain=clustering_strain,
                clustering_dir=clustering_dir,
                clustering_type=clustering_type,
                clustering_subtype=clustering_subtype,
                clustering_cutoff=clustering_cutoff,
                labels_path=cluster_labels_path,
            )
            logging.info(
                f"Loaded clustering dataframe with {clustering_data['metadata']['n_rows']} assignments and "
                f"{clustering_data['metadata']['n_clusters']} clusters"
            )
        except Exception as e:
            logging.warning(f'Failed to load clustering artifacts: {e}')
            clustering_data = None


    # starting with context features
    if context_features is not None:
        for feat in context_features:
            get_context_feature(
                dataframe=df,
                feature=feat,
                vips=vips,
                clustering_data=clustering_data,
                target_column=target_column,
                experiment_col=experiment_column,
                scale=scale,
                inplace=True,
            )

    # starting with intersection features
    if intersection_features is not None:
        for feat in intersection_features:
            get_intersection_feature(
                dataframe=df,
                feature=feat,
                vips=vips,
                clustering_data=clustering_data,
                motif_flank=motif_flank,
                use_vip_references=use_vip_references,
                use_centroid_references=use_centroid_references,
                distance_summary_only=distance_summary_only,
                scale=scale,
                inplace=True,
            )
    
    return df

def transform_meta_features(dataframe,
                            features=["Host","Cells","Context","Compartment","Resolution","Time","MOI"],
                            any=False,
                            get_columns=False):
    '''
        Transforms the column for each meta-feature of the given dataframe and returns the adjusted dataframe.
        Normalizes numeric values to [0,1] and creates one-hot-encodings for non-numeric meta-features.

        :param df: DataFrame with the data
        :param features: List of meta-features to calculate
        :param any: Boolean value to simply transform any applicable meta-feature columns

        :return: DataFrame with all features calculated and non-numerical columns removed
    '''
    #dataframe = df.copy()
    meta_columns = []
    if any:
        features = ["Host","Cells","Context","Compartment","Resolution","Time","MOI"]
    drop_cells = False
    for feature in features:
        match feature:
            case "Host":
                #assert "Cells" in dataframe.columns, 'Cells column required in dataframe!'
                if "Cells" in dataframe.columns:
                    #type_to_host = {0:'unknown_celltype', '':'unknown_celltype', 'unknown':'unknown_celltype', 'Host_human': 'Host_human', 'Host_WI38': 'Host_human', 'HEK293FT': 'Host_human', 'Host_A549': 'Host_human', 'MDCK-SIAT1': 'Host_canine', 'MDCK': 'Host_canine', 'HBEpC': 'Host_human', 'Mouse': 'Host_mouse', 'MRC5': 'Host_human'}
                    type_to_host = {0:'unknown_celltype', '':'unknown_celltype', 'unknown':'unknown_celltype', 'Host_human': 'Host_human', "Human": "Host_human", "human": "Host_human", "A549": "Host_human", 'Host_WI38': 'Host_human', 'HEK293FT': 'Host_human', 'Host_A549': 'Host_human', 'MDCK-SIAT1': 'Host_canine', 'MDCK': 'Host_canine', 'HBEpC': 'Host_human', "mouse": "Host_mouse", 'Mouse': 'Host_mouse', 'MRC5': 'Host_human'}
                    host_names = set(type_to_host.values())
                    host_names.remove('unknown_celltype')
                    for host in host_names:
                        dataframe[host] = (dataframe["Cells"].map(type_to_host) == host).astype(int)
                        meta_columns.append(host)
                    drop_cells = True
                    dataframe.drop("Host", inplace=True, axis=1, errors='ignore')
                else:
                    logging.error(f'Missing Cells column in dataframe')
            case "Cells":
                if "Cells" in dataframe.columns:
                    cell_to_simpler = {0:'unknown_celltype','':'unknown_celltype','human': 'unknown_celltype', 'WI38': 'Celltype_WI38', 'HEK293FT': 'Celltype_HEK293FT', 'A549': 'Celltype_A549', 'MDCK-SIAT1': 'Celltype_MDCK', 'MDCK': 'Celltype_MDCK', 'HBEpC': 'Celltype_HBEpC', 'Mouse': 'Host_mouse', 'MRC5': 'Celltype_MRC5'}
                    cell_names = set(cell_to_simpler.values())
                    cell_names.remove('unknown_celltype')
                    for cell in cell_names:
                        dataframe[cell] = (dataframe["Cells"].map(cell_to_simpler) == cell).astype(int)
                        meta_columns.append(cell)
                    drop_cells = True
                else:
                    logging.error(f'Missing Cells column in dataframe')
            case "Context":
                #assert "Context" in dataframe.columns, 'Context column required in dataframe!'
                if "Context" in dataframe.columns:
                    for value in ["in vitro", "in vivo"]:
                        dataframe[value] = (dataframe[feature] == value).astype(int)
                        meta_columns.append(value)
                    dataframe.drop(feature, inplace=True, axis=1)
                else:
                    logging.error(f'Missing Context column in dataframe')
            case "Compartment":
                #assert "Compartment" in dataframe.columns, 'Compartment column required in dataframe!'
                if "Compartment" in dataframe.columns:
                    for value in ["intracellular", "extracellular"]:
                        dataframe[value] = (dataframe[feature] == value).astype(int)
                        meta_columns.append(value)
                    dataframe.drop(feature, inplace=True, axis=1)
                else:
                    logging.error(f'Missing Compartment column in dataframe')
            case "Resolution":
                #assert "Resolution" in dataframe.columns, 'Resolution column required in dataframe!'
                if "Resolution" in dataframe.columns:
                    for value in ["singlecell", "bulk"]:
                        dataframe[value] = (dataframe[feature] == value).astype(int)
                        meta_columns.append(value)
                    dataframe.drop(feature, inplace=True, axis=1)
                else:
                    logging.error(f'Missing Resolution column in dataframe')
            case "Time":
                #assert "Time" in dataframe.columns, 'Time column required in dataframe!'
                if "Time" in dataframe.columns:
                    def translate_time(entry:str): # Turns entry value into float that refers to hours past infection
                        if entry is None:
                            return entry
                        elif isinstance(entry, float):
                            return entry
                        elif isinstance(entry, int):
                            return float(entry)
                        else:
                            if entry == "seed":
                                return float(0)
                            if "hpi" in entry:
                                return float(entry.replace("hpi",""))
                            if "dpi" in entry:
                                return float(entry.replace("dpi",""))*24
                            try:
                                return float(entry)
                            except Exception as e:
                                logging.error(f'Time entry could not be translated to hpi!')
                    dataframe["Time"] = dataframe["Time"].map(translate_time)
                    meta_columns.append("Time")
                else:
                    logging.error(f'Missing Time column in dataframe')
            case "MOI":
                #assert "MOI" in dataframe.columns, 'MOI column required in dataframe!'
                if "MOI" in dataframe.columns:
                    dataframe["MOI"] = dataframe["MOI"].transform(lambda x: x if isinstance(x, float) else 0 if x=='' else None if x is None else float(x))
                    meta_columns.append("MOI")
                else:
                    logging.error(f'Missing MOI column in dataframe')
            # case "Series": pass # TODO: if I ever add memory or other context values
            case _:
                logging.error(f"Unknown meta-feature requested: {feature}")
    if drop_cells:
        dataframe.drop("Cells", inplace=True, axis=1)
    if get_columns:
        return dataframe, meta_columns
    return dataframe

def drop_non_numeric(df: pd.DataFrame):
    '''
    Drops all duplicate and non-numerical columns from the dataframe.

    :param df: Dataframe to drop columns from

    :return: Dataframe with only numerical columns
    '''
    df = df.loc[:, ~df.columns.duplicated()] # drop duplicate columns if any
    weird_columns = []
    for col in df.columns:
        try:
            if isinstance(df[col].dtype, pd.StringDtype):
                logging.debug(f'Dropping non-numeric column {col}: {df[col].iloc[0]}')
                #logging.debug(f'Dropping non-numeric column {col}\n{list(df[col])}')
                continue
            if not np.issubdtype(df[col].dtype, np.number):
                logging.debug(f'Dropping non-numeric column {col}: {df[col].iloc[0]}')
                #logging.debug(f'Dropping non-numeric column {col}\n{list(df[col])}')
        except Exception as e:
            logging.error(f'Error checking column {col} with dtype {df[col].dtype}: {e}\n{traceback.format_exc()}')
            weird_columns.append((col, df[col].dtype))
    if len(weird_columns)>0:
        logging.error(f'Weird columns with unexpected behaviour: {weird_columns}')
    df = df.select_dtypes(include=[np.number])
    logging.debug(f'Columns left after dropping non-numeric columns:\n{list(df.columns)}')
    return df

def make_multiclass(df: pd.DataFrame, n_bins: int, y_column: str = 'NGS_log_norm', style: str = 'quantile', preset_thresholds=None):
    '''
    Labels each row of a dataframe as one of n_bins classes, based on the values of y_column. Unless preset_thresholds
    are given, thresholds between classes are created using either quantiles or the pd.cut function (style). If a list
    of preset_thresholds is entered, the dataframe will be labeled based on those.

    :param df: Dataframe including the target column
    :param n_bins: Number of classes to create
    :param y_column: Target column to label
    :param style: Type of labelling to use. Either 'quantile' or 'pd.cut'
    :param preset_thresholds: List of thresholds to use for labelling (optional)

    :return: Tuple with two entries:
        - y: Series with the labels for each row
        - thresholds: List of thresholds between the classes
    '''

    def label(row, class_dividers):
        '''
        Labels a row based on the given thresholds.
        :param row:
        :param class_dividers:
        :return:
        '''
        # only in case we want to label by intersection and ngs count
        if style == "combined":
            assert "Intersections" in df.columns and "NGS_log_min_max_norm" in df.columns, f'Did not find necessary columns to apply combined labeling style.'
            for j in range(n_bins-1):
                if class_dividers[j] >= row[y_column]:
                    # place in higher classes if intersecting candidate
                    if row["Intersections"] > 0:
                        return n_bins+j
                    else:
                        return j
        else:
            for j in range(n_bins-1):
                if class_dividers[j] >= row[y_column]:
                    return j
        return n_bins-1

    logging.debug(f'Labelling {y_column} into {n_bins} classes.')
    labels = list(range(n_bins))

    if preset_thresholds is None:
        match style:
            case 'pd.cut':
                y, thresholds = pd.cut(df[y_column], bins=n_bins, labels=labels, ordered=False, retbins=True)
                logging.debug(f'Calculated pd.cut thresholds: {thresholds}')
            case 'quantile':
                y = list()
                thresholds = list()

                # calculated thresholds between classes
                for i in range(n_bins - 1):
                    q = (i + 1) / n_bins
                    quantile = df[y_column].quantile(q)
                    # df[y_column] = df[y_column].apply(lambda x: labels[0] if x < median else labels[2] if x > median else labels[1])
                    thresholds.append(quantile)

                # label each row based on the calculated thresholds
                y = df.apply(lambda row: label(row, thresholds), axis=1)
                '''
                for row in df.iterrows():
                    r = row[1]
                    for i in range(n_bins-1):
                        if thresholds[i] >= r[y_column]:
                            y.append(labels[i])
                            break
                        if i == n_bins-2:
                            y.append(labels[i+1])
                y = pd.Series(y)'''
                logging.debug(f'Calculated quantile thresholds: {thresholds}')
            case 'combined':
                y = list()
                thresholds = list()

                # calculated thresholds between classes
                for i in range(n_bins - 1):
                    q = (i + 1) / n_bins
                    quantile = df[y_column].quantile(q)
                    thresholds.append(quantile)

                # label each row based on the calculated thresholds
                y = df.apply(lambda row: label(row, thresholds), axis=1)
                logging.debug(f'Calculated quantile thresholds: {thresholds}')
            case _:
                logging.error(f'Unknown style {style} for labelling')
                y = None
                thresholds = None
    else:
        logging.debug(f'Using given thresholds: {preset_thresholds}')
        assert isinstance(preset_thresholds, list), f'Given thresholds are not a list: {type(preset_thresholds)}'
        assert len(preset_thresholds) == n_bins-1, f'Given thresholds do not match the number of bins: {len(preset_thresholds)} != {n_bins-1}'

        # label each row based on the given thresholds
        thresholds = preset_thresholds
        y = df.apply(lambda row: label(row, thresholds), axis=1)
        '''
        for row in df.iterrows():
            r = row[1]
            for i in range(n_bins-1):
                if thresholds[i] >= r[y_column]:
                    y.append(labels[i])
                    break
                if i == n_bins-2:
                    y.append(labels[i+1])
        y = pd.Series(y)'''

    logging.debug(f'Finished multiclass conversion:\n{list(y)}\n{list(thresholds)}')
    return y, thresholds

def stratified_undersample(df: pd.DataFrame, target_col: str = 'NGS_log_norm',
                           n_bins: int = 10, samples_per_bin: int = None,
                           random_state: int = 42) -> pd.DataFrame:
    '''
    Returns balanced dataframe by sampling the same number of rows from each quantile.

    :param df: Dataframe that needs balancing
    :param target_col: Column to base balancing on
    :param n_bin: Number of bins for quantile determination
    :param samples_per_bin: Preset size of sample per bin. If None, uses size of smallest quantile instead.
    :param random_state: Fandom state for pandas sampling function
    :return: Balanced dataframe
    '''
    # Create quantile bins
    df = df.copy()
    df['target_bin'] = pd.qcut(df[target_col], q=n_bins, duplicates='drop')

    # Determine how many samples per bin
    bin_counts = df['target_bin'].value_counts()
    min_bin_size = bin_counts.min()
    if samples_per_bin is None:
        samples_per_bin = min_bin_size

    # Undersample each bin
    balanced_df = (
        df.groupby('target_bin', group_keys=False)
          .apply(lambda x: x.sample(n=min(samples_per_bin, len(x)), random_state=random_state))
    )

    return balanced_df.drop(columns='target_bin')

def get_duplicates(df):
    '''
    Returns a list of duplicates in the dataframe.

    :param df: Dataframe to check for duplicates
    :return: List of duplicates
    '''
    duplicates = df.groupby(["Segment","Start","End"])
    logging.info(f'Found {duplicates.ngroups} groupings of Segment x Start x End in dataframe')
    duplicates = duplicates.filter(lambda x: len(x) > 1)
    duplicates = duplicates.groupby(["Segment","Start","End"])
    logging.info(f'Found {duplicates.ngroups} groupings with multiple occurences in the dataframe')
    return duplicates

def merge_duplicates(df):
    '''
    Merges duplicates in the dataframe by summing up the NGS_read_counts of the duplicates.
    :param df: Dataframe with the columns Segment, Start, End and NGS_read_count
    :return: Datagrame with merged duplicates
    '''
    df = df.groupby(["Segment","Start","End"], as_index=False).sum(["NGS_log_norm"]).reset_index()
    return df

def find_k_varying(df, k, y_column, metric = 'std'):
    '''
    Returns a list of the k DelVGs whose y_column values vary the most between experiments. Assumes that duplicate DIs
    are from different experiment.

    :param df: Dataframe including all experiments
    :param k: Number of DelVGs to find
    :param y_column: Target column
    :param experiment_col: Column to differentiate between experiments
    :param metric: Metric to use for variance calculation. Options: 'range', 'variance', 'std'

    :return: List of k DelVG IDs
    '''
    logging.debug(f'Finding k-varying candidates')
    if "ID" not in df.columns:
        assert "Strain" in df.columns, "Strain column not found in dataframe"
        assert "Segment" in df.columns, "Segment column not found in dataframe"
        assert "Start" in df.columns, "Start column not found in dataframe"
        assert "End" in df.columns, "End column not found in dataframe"
        df["ID"] = df["Strain"] + "_" + df["Segment"] + "_" + df["Start"].astype(str) + "_" + df["End"].astype(str)

    match metric.lower():
        case 'range':
            grouped_filtered = df.groupby("ID").filter(lambda x: len(x) > 1)
            grouped = grouped_filtered.groupby("ID")[y_column].agg(['min', 'max']).reset_index()
            grouped['diff'] = grouped['max'] - grouped['min']
            logging.debug(f'Range grouping:\n{grouped}')
            top_ids = grouped.nlargest(k, 'diff')['ID'].to_list()
            top_k = df[df['ID'].isin(top_ids)]['ID'].unique()
            logging.debug(f'Top K: {list(top_k)}')
            return top_k
        case 'cv':
            grouped_filtered = df.groupby("ID").filter(lambda x: len(x) > 1)
            grouped = grouped_filtered.groupby("ID")[y_column].agg(['std', 'mean']).reset_index()
            grouped = grouped[grouped["mean"]>0] # Should not be necessary, but just in case ensure we don't divide by 0
            grouped["cv"] = grouped["std"]/grouped["mean"]
            logging.debug(f'Coefficient of variation grouping:\n{grouped}')
            top_ids = grouped.nlargest(k, 'cv')['ID'].to_list()
            top_k = df[df['ID'].isin(top_ids)]['ID'].unique()
            logging.debug(f'Top K: {list(top_k)}')
            return top_k
        case 'std':
            grouped_filtered = df.groupby("ID").filter(lambda x: len(x) > 1)
            grouped = grouped_filtered.groupby("ID")[y_column].std().reset_index(name='std')
            logging.debug(f'Standard Deviation grouping:\n{grouped}')
            top_ids = grouped.nlargest(k, 'std')['ID'].to_list()
            top_k = df[df['ID'].isin(top_ids)]['ID'].unique()
            logging.debug(f'Top K: {list(top_k)}')
            return top_k
        case 'count':
            grouped_filtered = df.groupby("ID").filter(lambda x: len(x) > 1)
            grouped = grouped_filtered.groupby("ID")[y_column].count().reset_index(name='count')
            logging.debug(f'Count grouping:\n{grouped}')
            top_ids = grouped.nlargest(k, 'count')['ID'].to_list()
            top_k = df[df['ID'].isin(top_ids)]['ID'].unique()
            logging.debug(f'Top K: {list(top_k)}')
            return top_k
        case 'count_cv':
            grouped_filtered = df.groupby("ID").filter(lambda x: len(x) > 1)
            grouped = grouped_filtered.groupby("ID")[y_column].agg(['std', 'mean', 'count']).reset_index()
            grouped = grouped[grouped["mean"]>0] # Should not be necessary, but just in case ensure we don't divide by 0
            grouped["cv"] = grouped["std"]/grouped["mean"]
            logging.debug(f'Count grouping:\n{grouped}')
            rest_k = k
            top_ids = []
            count_groups = grouped.groupby(["count"])
            for i in -np.sort(-grouped["count"].unique()):
                if rest_k == 0:
                    break
                group = count_groups.get_group((i,))
                max_added = len(group)
                if max_added <= rest_k:
                    top_ids += group['ID'].to_list()
                    rest_k -= max_added
                else:
                    top_ids += group.nlargest(rest_k, 'count')['ID'].to_list()
                    break
            top_k = df[df['ID'].isin(top_ids)]['ID'].unique()
            logging.debug(f'Top K: {list(top_k)}')
            return top_k
        case 'zscore':
            if "ACC_num" in df.columns:
                exp_col = "ACC_num"
            elif "Publication" in df.columns and df["Publication"].nunique()>1:
                exp_col = "Publication"
            else:
                logging.error(f'Did not find useful experiment column to calculate z-scores.')
                return []
            grouped_filtered = df.groupby("ID").filter(lambda x: len(x) > 1)
            stats = df.groupby(exp_col)[y_column].agg(["mean","std"]).reset_index()
            stats = stats.rename(columns={"mean": "bg_mean", "std": "bg_std"})
            tmp_df = df.merge(stats, on=exp_col, how='left')
            tmp_df["z_score"] = (tmp_df[y_column]-tmp_df["bg_mean"])/tmp_df["bg_std"]
            z_agg = tmp_df.groupby("ID")["z_score"].agg(['mean', 'max', 'count']).reset_index()
            logging.debug(f'Aggregated z-score results:\n{z_agg}')
            top_ids = z_agg.nlargest(k, "mean")['ID'].to_list()
            top_k = df[df['ID'].isin(top_ids)]['ID'].unique()
            logging.debug(f'Top K: {list(top_k)}')
            return top_k
        case _:
            logging.error(f'Unknown metric {metric} for variance calculation')
            return None
    return None

def get_k_varying(df, candidates, y_column, experiment_col):
    '''
    Returns a Dataframe with each row now containing the y_column values of the k DelVGs whose y_column values vary the
    most between experiments. The experiment_row should differentiate the same DI between different experiments.

    :param df: Dataframe including all experiments
    :param k: Number of DelVGs to find
    :param y_column: Target column
    :param experiment_row: Column to differentiate between experiments

    :return: List of k DelVG IDs
    '''
    logging.debug(f'Getting k-varying candidate values')
    if "ID" not in df.columns:
        #logging.debug('adding ID to columns')
        assert "Strain" in df.columns, "Strain column not found in dataframe"
        assert "Segment" in df.columns, "Segment column not found in dataframe"
        assert "Start" in df.columns, "Start column not found in dataframe"
        assert "End" in df.columns, "End column not found in dataframe"
        df["ID"] = df["Strain"] + "_" + df["Segment"] + "_" + df["Start"].astype(str) + "_" + df["End"].astype(str)

    def find_local_candidate(exp_df, y_column, candidate):
        if candidate in exp_df['ID'].values:
            return exp_df[exp_df['ID'] == candidate][y_column].values[0]
        else:
            return 0
    
    def find_local_by_col(df, y_column, candidate):
        # initialize new column
        df['k_vary_'+candidate] = 0

        # iterate over experiments
        grouped = df.groupby([experiment_col])
        for exp, group in grouped:
            # get candidate value within experiment
            if candidate in group['ID'].values:
                candidate_value = group.loc[group['ID']==candidate, y_column].values[0]
            else:
                candidate_value = 0 # if candidate not in experiment, assume 0
            # apply found value to all rows of current experiment
            df.loc[group.index, 'k_vary_'+candidate] = candidate_value
        return df

    for i in candidates:
        df = find_local_by_col(df,y_column, i)
        logging.debug(f'Candidate {i}:\n{df["k_vary_"+i].head()}\n{df["k_vary_"+i].describe()}')
    #for i in candidates:
    #    df['k_vary_'+i] = 0
    #    df['k_vary_'+i] = df.apply(lambda row: find_local_candidate(df[df[experiment_col]==row[experiment_col]], y_column, i), axis=1)
    
    logging.debug(f'Finished getting {len(candidates)} k-varying candidates based on {experiment_col}\n\n')
    return df

def get_current_values(df, candidates, y_column, experiment_col):
    '''
    Returns a Dataframe with each row now containing the y_column values of the k DelVGs whose y_column values vary the
    most between experiments. The experiment_row should differentiate the same DI between different experiments.

    :param df: Dataframe including all experiments
    :param k: Number of DelVGs to find
    :param y_column: Target column
    :param experiment_row: Column to differentiate between experiments

    :return: List of k DelVG IDs
    '''
    logging.debug(f'Getting k-varying candidate values')
    if "ID" not in df.columns:
        #logging.debug('adding ID to columns')
        assert "Strain" in df.columns, "Strain column not found in dataframe"
        assert "Segment" in df.columns, "Segment column not found in dataframe"
        assert "Start" in df.columns, "Start column not found in dataframe"
        assert "End" in df.columns, "End column not found in dataframe"
        df["ID"] = df["Strain"] + "_" + df["Segment"] + "_" + df["Start"].astype(str) + "_" + df["End"].astype(str)

    def find_local_candidate(exp_df, y_column, candidate):
        if candidate in exp_df['ID'].values:
            return exp_df[exp_df['ID'] == candidate][y_column].values[0]
        else:
            return 0
    
    def find_local_by_col(df, y_column, candidate):
        # initialize new column
        df['k_vary_'+candidate] = 0

        # iterate over experiments
        grouped = df.groupby([experiment_col])
        for exp, group in grouped:
            # get candidate value within experiment
            if candidate in group['ID'].values:
                candidate_value = group.loc[group['ID']==candidate, y_column].values[0]
            else:
                candidate_value = 0 # if candidate not in experiment, assume 0
            # apply found value to all rows of current experiment
            df.loc[group.index, 'k_vary_'+candidate] = candidate_value
        return df

    for i in candidates:
        df = find_local_by_col(df,y_column, i)
        logging.debug(f'Candidate {i}:\n{df["k_vary_"+i].head()}\n{df["k_vary_"+i].describe()}')
    #for i in candidates:
    #    df['k_vary_'+i] = 0
    #    df['k_vary_'+i] = df.apply(lambda row: find_local_candidate(df[df[experiment_col]==row[experiment_col]], y_column, i), axis=1)
    
    logging.debug(f'Finished getting {len(candidates)} k-varying candidates based on {experiment_col}\n\n')
    return df

def get_last_values(df, candidates, y_column, experiment_col):
    '''
    Returns a Dataframe with each row now containing the y_column values of the k DelVGs whose y_column values vary the
    most between experiments. The experiment_row should differentiate the same DI between different experiments.

    :param df: Dataframe including all experiments
    :param k: Number of DelVGs to find
    :param y_column: Target column
    :param experiment_row: Column to differentiate between experiments

    :return: List of k DelVG IDs
    '''
    # TODO: Make this get the last known value of candidates by time, rather than value in current experiment
    logging.debug(f'Getting k-varying candidate values')
    if "ID" not in df.columns:
        #logging.debug('adding ID to columns')
        assert "Strain" in df.columns, "Strain column not found in dataframe"
        assert "Segment" in df.columns, "Segment column not found in dataframe"
        assert "Start" in df.columns, "Start column not found in dataframe"
        assert "End" in df.columns, "End column not found in dataframe"
        df["ID"] = df["Strain"] + "_" + df["Segment"] + "_" + df["Start"].astype(str) + "_" + df["End"].astype(str)

    def find_local_candidate(exp_df, y_column, candidate):
        if candidate in exp_df['ID'].values:
            return exp_df[exp_df['ID'] == candidate][y_column].values[0]
        else:
            return 0
    
    def find_local_by_col(df, y_column, candidate):
        # initialize new column
        df['k_vary_'+candidate] = 0

        # iterate over experiments
        grouped = df.groupby([experiment_col])
        for exp, group in grouped:
            # get candidate value within experiment
            if candidate in group['ID'].values:
                candidate_value = group.loc[group['ID']==candidate, y_column].values[0]
            else:
                candidate_value = 0 # if candidate not in experiment, assume 0
            # apply found value to all rows of current experiment
            df.loc[group.index, 'k_vary_'+candidate] = candidate_value
        return df

    for i in candidates:
        df = find_local_by_col(df,y_column, i)
        logging.debug(f'Candidate {i}:\n{df["k_vary_"+i].head()}\n{df["k_vary_"+i].describe()}')
    #for i in candidates:
    #    df['k_vary_'+i] = 0
    #    df['k_vary_'+i] = df.apply(lambda row: find_local_candidate(df[df[experiment_col]==row[experiment_col]], y_column, i), axis=1)
    
    logging.debug(f'Finished getting {len(candidates)} k-varying candidates based on {experiment_col}\n\n')
    return df

def get_fuzzy_locations(df): # not looking into this anymore
    return df

def apply_cutoff(df, cutoff, method = 'quick_alternative', exp_col='', forced_depr=False):
    '''
    Applies a cutoff to the dataframe. Has different methods of application, depending on the method parameter. Pools
    leftover rows of the same DelVG.
    basic: Pools all rows of each DelVG and removes any with a summed y_column value below the cutoff.
    individual: Removes any row with a y_column value below the cutoff.
    combined: Pools all rows of each DelVG that reaches the cutoff with at least one of its occurences. Removes the rest.

    :param df: Dataframe to apply the cutoff to
    :param cutoff: Minimum value to keep a row
    :param method: Name of cutoff application (Options: basic, individual, combined)

    :return: Dataframe with the cutoff applied and no doubled DelVGs
    '''
    logging.warning(f'This function is deprecated.{"" if forced_depr else " Switching to cutoff_clean()."}')
    if not forced_depr:
        return cutoff_clean(data=df, threshold=cutoff, method=method if method=="basic" else "accumulated",
                            dataset_id_column="dataset_id" if exp_col=="dataset_id" else "Publication",
                            inplace=False)
    logging.info(f'Applying cutoff of {cutoff} to the dataframe, using {method.replace("_"," ")} method.')
    logging.debug(f'pd.options.mode.chained_assignment = None')
    logging.debug(f'Columns before cutoff {list(df.columns)}')
    identifier = ["Strain", "Segment", "Start", "End"]
    assert all([x in df.columns for x in identifier]), "No identifier found in dataframe"
    if exp_col != '':
        assert exp_col in df.columns, "Experiment Column is not in dataframe"
        identifier.append(exp_col)
    if "ID" in df.columns:
        identifier.append("ID")
    if "Publication" in df.columns and exp_col != "Publication":
        identifier.append("Publication")

    # TODO: Removes most cols not part of [identifier, NGS_read_count]. Should find better way than adding all to identifier.
    match method:
        case 'basic':
            df = df.groupby(identifier, as_index=False).sum(['NGS_read_count'])
            logging.debug(f'During Cutoff:\n{list(df.columns)}')
            df = df[df['NGS_read_count'] >= cutoff]
            logging.info(f'Pooled NGS_read_count values and applied {method} cutoff of {cutoff} to the dataframe. {df.shape[0]} rows left {df.shape[1]} columns left.')
        case 'individual':
            df = df[df['NGS_read_count'] >= cutoff]
            logging.info(f'Applied {method} cutoff of {cutoff} to the dataframe. {df.shape[0]} rows left.')
            logging.info(f'Pooling NGS_read_count values')
            df = df.groupby(identifier, as_index=False).sum(['NGS_read_count'])
        case 'combined':
            grouped = df.groupby(identifier, as_index=False).sum(['NGS_read_count']).reset_index()
            to_drop = []
            for name, group in grouped:
                if not (group['NGS_read_count'] >= cutoff).any():
                    #df = df.drop(group.index)
                    to_drop.extend(group.index)
            df = df.drop(to_drop)
            logging.info(f'Applied {method} cutoff of {cutoff} to the dataframe. {df.shape[0]} rows left.')
            logging.info(f'Pooling NGS_read_count values')
            df = df.groupby(identifier, as_index=False).sum(['NGS_read_count'])
        case 'quick_alternative':
            if "Wang2020" in df["Publication"].unique() and "ACC_num" in df.columns:
                # Wang2020 has some annoying bits in unpooled version.
                # TODO: Need to ask Jens what he did about that.
                logging.info(f'Found unpooled Wang2020, so dropping doubled ACC_num+ID cols, only keeping max NGS')
                max_ids = df.groupby(["ACC_num","Strain","Segment","Start","End"])["NGS_read_count"].idxmax()
                df = df.loc[max_ids].reset_index(drop=True)
            # TODO: Standardise this and add pooling option
            logging.debug(f'Applying quick alternative cutoff without pooling')
            df = identify_candidates(df)
            survivors = df[df["NGS_read_count"]>=cutoff]["ID"]
            df = df[df["ID"].isin(survivors)]
            if "Publication" in df.columns:
                pub_dfs = []
                for pub in df["Publication"].unique():
                    pub_df = df[df["Publication"]==pub]
                    pub_survivors = pub_df[pub_df["NGS_read_count"]>=cutoff]["ID"]
                    pub_dfs.append(pub_df[pub_df["ID"].isin(pub_survivors)])
                try:
                    df = pd.concat(pub_dfs)
                except ValueError as e:
                    logging.debug(f'ValueError during concatenation in cutoff: {e}')
                except Exception as e:
                    logging.error(f'Problem during cutoff application:\n{e}\n{df.head()}\n{df.describe()}\n{pub_dfs}\n{df["Publication"].nunique()}')
        case _:
            logging.error(f'Unknown type {method} for cutoff application')
    df.reset_index(drop=True, inplace=True)
    logging.debug(f'Columns after cutoff {list(df.columns)}')
    return df

def cutoff_clean(data, threshold:int=None, method="accumulated", y_column="NGS_read_count", dataset_id_column="Publication", minimum_dataset_size=40, inplace=False, fix_wang=True, cutoff:int=None, left_out_ids: list = None):
    '''
    Applies an RSC threshold (cutoff) to the dataframe. Application depends on method input. 
    basic: Removes all rows y_column value below the cutoff.
    accumulated: Considers datasets separately, based on dataset_id_column. Per dataset, accumulates y_column values
    for of each DVG and removes respective rows if cutoff is not reached.

    :param data: Pandas dataframe to apply the cutoff to
    :param threshold: Minimum value to keep entry
    :param method: Name of cutoff application (Options: basic, accumulated)
    :param y_column: Target value to apply cutoff to
    :param dataset_id_column: Column holding the unique identifiers for each dataset
    :param minimum_dataset_size: Minimum number of unique DVGs required for a dataset to be included
    :param inplace: Change original data object (True) or generate a copy (False)
    :param fix_wang: Whether to fix the unpooled Wang2020 data
    :param cutoff: Deprecated parameter, use threshold instead
    :param left_out_ids: List of DVG IDs to exclude from cutoff application, regardless of their y_column value.

    :return: Dataframe with the cutoff applied and no doubled DelVGs
    '''
    logging.info(f'Applying cutoff {threshold} to {y_column} using {method} method.')
    if inplace:
        dataframe = data
    else:
        dataframe = data.copy()
    if threshold is None:
        if cutoff is None:
            raise ValueError(f'Not threshold provided for cutoff application.')
        threshold = cutoff
        logging.warning(f'Using deprecated cutoff parameter for threshold value. Please switch to threshold parameter in future calls.')
    dataframe = identify_candidates(dataframe)
    if fix_wang:
        dataframe = fix_unpooled_wang2020(dataframe)
    match method:
        case "basic":
            dataframe = dataframe[dataframe[y_column]>=threshold]
        case "accumulated":
            to_keep = []
            for name, dataset in dataframe.groupby(dataset_id_column):
                for dvg, dvg_entries in dataset.groupby("ID"):
                    if left_out_ids is not None and dvg in left_out_ids:
                        to_keep.extend(dvg_entries.index)
                    if dvg_entries[y_column].sum()>=threshold:
                        to_keep.extend(dvg_entries.index)
            dataframe = dataframe.loc[to_keep]
        case _:
            logging.error(f'Unknown method for application of cutoff: {method}')
    for dataset, dataset_entries in dataframe.groupby(dataset_id_column):
        if minimum_dataset_size > 0 and dataset_entries["ID"].nunique() < minimum_dataset_size:
            logging.warning(f'Dataset {dataset} has only {dataset_entries["ID"].nunique()} unique DVGs after cutoff application, which is below the minimum of {minimum_dataset_size}. Removing all entries of this dataset.')
            dataframe = dataframe[dataframe[dataset_id_column]!=dataset]
    if not inplace:
        return dataframe
    
def fix_unpooled_wang2020(data, inplace=False):
    if inplace:
        dataframe = data
    else:
        dataframe = data.copy()
    dataframe = identify_candidates(dataframe)
    if "ACC_num" in dataframe.columns and dataframe.duplicated(["ACC_num","ID"]).any():
        logging.debug(f'Found duplicate ACC_num/ID entries. Aggregating NGS read counts to remove duplicates.')
        behavior = {col: "first" for col in dataframe.columns}
        behavior["NGS_read_count"] = "sum"
        dataframe = dataframe.groupby(["Publication","ACC_num","ID"], as_index=False).agg(behavior).reset_index(drop=True)
        dataframe = dataframe.fillna(np.nan) # resetting na values
    if not inplace:
        return dataframe

'''
def apply_cutoff(df, cutoff, method = 'basic', exp_col=''):
    ''''''
    Applies a cutoff to the dataframe. Has different methods of application, depending on the method parameter. Pools
    leftover rows of the same DelVG.
    basic: Pools all rows of each DelVG and removes any with a summed y_column value below the cutoff.
    individual: Removes any row with a y_column value below the cutoff.
    combined: Pools all rows of each DelVG that reaches the cutoff with at least one of its occurences. Removes the rest.

    :param df: Dataframe to apply the cutoff to
    :param cutoff: Minimum value to keep a row
    :param method: Name of cutoff application (Options: basic, individual, combined)

    :return: Dataframe with the cutoff applied and no doubled DelVGs'''
'''
    logging.info(f'Applying cutoff of {cutoff} to the dataframe.')
    logging.debug(f'pd.options.mode.chained_assignment = None')
    logging.debug(f'Columns before cutoff {list(df.columns)}')
    identifier = ["Strain", "Segment", "Start", "End"]
    assert all([x in df.columns for x in identifier]), "No identifier found in dataframe"
    if exp_col != '':
        assert exp_col in df.columns, "Experiment Column is not in dataframe"
        identifier.append(exp_col)
    if "ID" in df.columns:
        identifier.append("ID")
    if "Publication" in df.columns and exp_col != "Publication":
        identifier.append("Publication")

    # TODO: Removes most cols not part of [identifier, NGS_read_count]. Should find better way than adding all to identifier.
    match method:
        case 'basic':
            df = df.groupby(identifier, as_index=False).sum(['NGS_read_count'])
            logging.debug(f'During Cutoff:\n{list(df.columns)}')
            df = df[df['NGS_read_count'] >= cutoff]
            logging.info(f'Pooled NGS_read_count values and applied {method} cutoff of {cutoff} to the dataframe. {df.shape[0]} rows left {df.shape[1]} columns left.')
        case 'individual':
            df = df[df['NGS_read_count'] >= cutoff]
            logging.info(f'Applied {method} cutoff of {cutoff} to the dataframe. {df.shape[0]} rows left.')
            logging.info(f'Pooling NGS_read_count values')
            df = df.groupby(identifier, as_index=False).sum(['NGS_read_count'])
        case 'combined':
            grouped = df.groupby(identifier, as_index=False).sum(['NGS_read_count']).reset_index()
            to_drop = []
            for name, group in grouped:
                if not (group['NGS_read_count'] >= cutoff).any():
                    #df = df.drop(group.index)
                    to_drop.extend(group.index)
            df = df.drop(to_drop)
            logging.info(f'Applied {method} cutoff of {cutoff} to the dataframe. {df.shape[0]} rows left.')
            logging.info(f'Pooling NGS_read_count values')
            df = df.groupby(identifier, as_index=False).sum(['NGS_read_count'])
        case 'quick_alternative':
            # TODO: Standardise this and add pooling option
            logging.info(f'Applying quick alternative cutoff without pooling')
            df = identify_candidates(df)
            survivors = df[df["NGS_read_count"]>=cutoff]["ID"]
            df = df[df["ID"].isin(survivors)]
            if "Publication" in df.columns:
                pub_dfs = []
                for pub in df["Publication"].unique():
                    pub_df = df[df["Publication"]==pub]
                    pub_survivors = pub_df[pub_df["NGS_read_count"]>=cutoff]["ID"]
                    pub_dfs.append(pub_df[pub_df["ID"].isin(pub_survivors)])
                df = pd.concat(pub_dfs)
        case _:
            logging.error(f'Unknown type {method} for cutoff application')
    df.reset_index(drop=True, inplace=True)
    logging.debug(f'Columns after cutoff {list(df.columns)}')
    return df'''

def identify_candidates(df):
    '''
    Create a unique ID for each DelVG candidate based on Strain, Segment, Start and End
    :param df: pandas dataframe with DelVG candidates, including columns Strain, Segment, Start and End
    :return: pandas dataframe with additional column ID
    '''
    if 'ID' not in df.columns:
        logging.info("Identifying candidates of dataframe")
        df["ID"] = df["Strain"] + "_" + df["Segment"] + "_" + df["Start"].astype(str) + "_" + df["End"].astype(str)
        #df['ID'] = df.apply(lambda row: str(row['Strain']) + '_' + str(row['Segment']) + '_' + str(row['Start']) + '_' + str(row['End']), axis=1)
    return df

def count_intersections(df, exp_col):
    logging.info('Counting intersections')
    df = identify_candidates(df)
    grouped = df.groupby('ID')
    df['Intersections'] = 0
    for idx, group in grouped:
        num = group[exp_col].nunique()
        df.loc[group.index, 'Intersections'] = num-1
    return df

def calculate_target(df, y_col, exp_col='Publication', drop_read_count=True):
    logging.info('Calculating target Column')
    if y_col != 'NGS_read_count':
        if 'inter' in y_col.lower():
            # Only on publications
            df = count_intersections(df, exp_col)
        if 'norm' in y_col:
            df = log_and_norm(df, y_col, exp_col, drop_read_count=drop_read_count)
        if 'clr' in y_col.lower():
            df = central_log_ratio_transform(df, exp_col, drop_read_count=drop_read_count)
        if 'ilr' in y_col.lower():
            df = isometric_log_ratio_transform(df, exp_col, drop_read_count=drop_read_count)
    logging.debug(f'After calculating target column:\n{list(df.columns)}\n{df.dtypes}\n{df.head()}')
    return df

def central_log_ratio_transform(data, exp_col, ratio=False, drop_read_count=True, inplace=False):
    '''
    Performs central log-ratio transformation on the specified column. Use exp_col to differentiate between compositions.
    :param data: pandas dataframe
    :param y_col: column to perform transformation on
    :param exp_col: column that holds the composition identifiers
    :param ratio: if True, will divide the values of each composition by their respective totals within the dataframe, before executing the transformation
    
    :return: dataframe with new clr column
    '''
    assert "NGS_read_count" in data.columns, f'NGS_read_count is missing in dataframe!'
    assert exp_col in data.columns, f'{exp_col} is missing in dataframe!'
    if inplace:
        dataframe = data
    else:
        dataframe = data.copy()

    dataframe[f'NGS temp'] = dataframe["NGS_read_count"].astype(float)
    dataframe.loc[dataframe[f'NGS temp']==0.0,f'NGS temp'] = 1e-6
    if ratio:
        for exp, group in dataframe.groupby(exp_col):
            dataframe.loc[group.index, f'NGS temp'] = group[f'NGS temp']/group[f'NGS temp'].sum()
    for exp, group in dataframe.groupby(exp_col):
        dataframe.loc[group.index, "CLR"] = clr(dataframe.loc[group.index,f'NGS temp'])
    dataframe.drop(f'NGS temp', axis=1, inplace=True)
    if drop_read_count:
        dataframe.drop("NGS_read_count", axis=1, errors="ignore")
    if not inplace:
        return dataframe
    
def isometric_log_ratio_transform(data, exp_col, ratio=False, drop_read_count=True, inplace=False):
    '''
    Performs isometric log-ratio transformation on the specified column. Use exp_col to differentiate between compositions.
    :param data: pandas dataframe
    :param y_col: column to perform transformation on
    :param exp_col: column that holds the composition identifiers
    :param ratio: if True, will divide the values of each composition by their respective totals within the dataframe, before executing the transformation

    :return: dataframe with new clr column
    '''
    assert "NGS_read_count" in data.columns, f'NGS_read_count is missing in dataframe!'
    assert exp_col in data.columns, f'{exp_col} is missing in dataframe!'
    if inplace:
        dataframe = data
    else:
        dataframe = data.copy()

    dataframe[f'NGS temp'] = dataframe["NGS_read_count"].astype(float)
    dataframe.loc[dataframe[f'NGS temp']==0.0,f'NGS temp'] = 1e-6
    if ratio:
        for exp, group in dataframe.groupby(exp_col):
            dataframe.loc[group.index, f'NGS temp'] = group[f'NGS temp']/group[f'NGS temp'].sum()
    for exp, group in dataframe.groupby(exp_col):
        dataframe.loc[group.index, "ILR"] = ilr(dataframe.loc[group.index,f'NGS temp'])
    dataframe.drop(f'NGS temp', axis=1, inplace=True)
    if drop_read_count:
        dataframe.drop("NGS_read_count", axis=1, errors="ignore")
    if not inplace:
        return dataframe

def get_sizes_publication():
    '''
    (Deprecated) Calculates the amount of DelVGs per strain for each publication in the data directory and saves them to
    a csv file.

    :return: Dictionary with the amount of DelVGs of all publications split up by strain
    '''
    sizes = dict()
    pub_files = glob.glob(os.path.join(DATA_DIR, '**', f'*.csv'), recursive=True)
    for _ in pub_files:
        publication = os.path.basename(_).split('.')[0].split('_')[0]
        sizes[publication] = dict()
    for _ in pub_files:
        publication = os.path.basename(_).split('.')[0].split('_')[0]
        strain = os.path.basename(os.path.dirname(_))
        df = pd.read_csv(_)
        sizes[publication][strain] = len(df)
    df = pd.DataFrame.from_dict(sizes, orient='index')
    df.fillna(0, inplace=True)
    df = df.astype(int)
    print(df)
    df.to_csv('sizes.csv')
    return sizes

def split_data(df, y, test_size=0.2, seed=42, thresh=0.1, stratified=False, exp_col="Publication", strat_id=True):
    '''
    Splits the dataframe and label column up into train and test sets. If stratified, the split will create the
    same ratio of each Publication in both train and test set.
    '''
    logging.info(f'df:\n{df.head()}\ny:\n{y.head()}\n{y.value_counts()}')
    if not stratified:
        if strat_id:
            # Ensure that same ID is only in either train or test set, but never in both
            if "ID" not in df.columns:
                df = identify_candidates(df)
        
            df["strat"] = df["ID"].map(df["ID"].value_counts())
            reduced = df.drop_duplicates(["ID"])

            # Splitting single value classes separately and using stratify for the others
            unique_comb = reduced[reduced["strat"]==1]
            multi_comb = reduced[reduced["strat"]>1]
            df.drop(["strat"],axis=1,inplace=True)
        
            if len(unique_comb):
                unique_comb_train_ids, unique_comb_test_ids = train_test_split(unique_comb["ID"], test_size=test_size)
            else:
                unique_comb_train_ids, unique_comb_test_ids = pd.Series(),pd.Series()
            if len(multi_comb):
                multi_comb_train_ids, multi_comb_test_ids = train_test_split(multi_comb["ID"], test_size=test_size, stratify=multi_comb["strat"])
            else:
                multi_comb_train_ids, multi_comb_test_ids = pd.Series(),pd.Series()
                
            train_idx_bool = df["ID"].isin(multi_comb_train_ids.to_list()+unique_comb_train_ids.to_list())
            test_idx_bool = df["ID"].isin(multi_comb_test_ids.to_list()+unique_comb_test_ids.to_list())
            assert len(train_idx_bool)==len(test_idx_bool), f'Lengths of Boollists do not match'
            overlap = [x for x in range(len(train_idx_bool)) if train_idx_bool[x]==test_idx_bool[x]]
            if len(overlap):
                print(f'ID crossover between train and test set!\n{overlap}')
                logging.error(f'ID crossover between train and test set!\n{overlap}')
            
            X_train = df[train_idx_bool].reset_index(drop=True)
            X_test = df[test_idx_bool].reset_index(drop=True)
            y_train = y[train_idx_bool].reset_index(drop=True)
            y_test = y[test_idx_bool].reset_index(drop=True)
        else:
            X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=seed)
    else:
        if "ID" not in df.columns:
            df = identify_candidates(df)
        
        grouped = df.groupby(["ID"])[exp_col].apply(lambda x: x.value_counts()).unstack(fill_value=0)
        grouped["strat"] = grouped.apply(lambda x: tuple(x), axis=1)
        grouped["ID"] = grouped.index

        # Splitting single value classes separately and using stratify for the others
        val_counts = grouped["strat"].value_counts()
        unique = val_counts[val_counts==1].index
        multi = val_counts[val_counts>1].index
        unique_comb = grouped[grouped["strat"].isin(unique)]
        multi_comb = grouped[grouped["strat"].isin(multi)]
        
        if len(unique_comb):
            unique_comb_train_ids, unique_comb_test_ids = train_test_split(unique_comb["ID"], test_size=test_size)
        else:
            unique_comb_train_ids, unique_comb_test_ids = pd.Series(),pd.Series()
        if len(multi_comb):
            multi_comb_train_ids, multi_comb_test_ids = train_test_split(multi_comb["ID"], test_size=test_size, stratify=multi_comb["strat"])
        else:
            multi_comb_train_ids, multi_comb_test_ids = pd.Series(),pd.Series()
            
        train_idx_bool = df["ID"].isin(multi_comb_train_ids.to_list()+unique_comb_train_ids.to_list())
        test_idx_bool = df["ID"].isin(multi_comb_test_ids.to_list()+unique_comb_test_ids.to_list())
        assert len(train_idx_bool)==len(test_idx_bool), f'Lengths of Boollists do not match'
        overlap = [x for x in range(len(train_idx_bool)) if train_idx_bool[x]==test_idx_bool[x]]
        if len(overlap):
            print(f'ID crossover between train and test set!\n{overlap}')
            logging.error(f'ID crossover between train and test set!\n{overlap}')
        
        X_train = df[train_idx_bool].reset_index(drop=True)
        X_test = df[test_idx_bool].reset_index(drop=True)
        y_train = y[train_idx_bool].reset_index(drop=True)
        y_test = y[test_idx_bool].reset_index(drop=True)

    return X_train, X_test, y_train, y_test

def identify_duplicates(df):
    '''
    Identifies duplicates in the dataframe and includes a column to identify them with boolean values.
    :param df:
    :return:
    '''
    df['ID'] = df["Segment"] + '_' + df["Start"].astype(str) + df["End"].astype(str)
    tmp = pd.Dataframe(df.groupby('ID').size())
    tmp = tmp.rename(columns={0: "Occurrences"})
    dup_list = tmp[tmp['Occurrences'] > 1].index.values.tolist()

    df['Duplicate'] = 0
    df['Duplicate'] = df['ID'].apply(lambda x: 1 if x in dup_list else 0)
    df.drop('ID', axis=1, inplace=True)

    return df

def balance_dataset(df, y_col, target_ratio=0, seed=42):
    '''
    Balances binary classes of the dataframe. If target_ratio is given, reduces one subset to achieve larger_set/smaller_set = target_ratio.
    '''
    logging.debug(f'Balancing Dataframe\n{df.head()}')
    if y_col == "Intersections":
        start_dist = (len(df[df[y_col]==0]),len(df[df[y_col]>0]))
        # If target_ratio given, rebalance according to it
        if target_ratio != 0:
            if min(start_dist)/max(start_dist) == target_ratio or max(start_dist)/min(start_dist) == target_ratio:
                logging.info(f'Dataframe already balanced according to target ratio')
                return df
            target_max = min(start_dist)*target_ratio
            if target_max > max(start_dist):
                # If ratio is only reached by reducing smaller subset
                removal_amount = int(min(start_dist)-max(start_dist)/target_ratio)
                # Take from the smaller subset
                if start_dist[0]<start_dist[1]:
                    sample = df[df[y_col]==0].sample(n=removal_amount,random_state=seed)
                else:
                    sample = df[df[y_col]>0].sample(n=removal_amount,random_state=seed)
            else:
                # If ratio is reached by reducing greater or either subset
                removal_amount = int(max(start_dist)-target_max)
                # Take from the larger subset
                if start_dist[0]>start_dist[1]:
                    sample = df[df[y_col]==0].sample(n=removal_amount,random_state=seed)
                else:
                    sample = df[df[y_col]>0].sample(n=removal_amount,random_state=seed)
            df = df.drop(sample.index)
            end_dist = (len(df[df[y_col]==0]),len(df[df[y_col]>0]))
            logging.info(f'Balanced dataframe on Intersection counts by removing {removal_amount} rows:\nStarting ratio: {start_dist[0]} vs. {start_dist[1]}\nFinal ratio: {end_dist[0]} vs. {end_dist[1]}')
            
        # Otherwise check if dataframe is unbalanced
        elif min(start_dist)/max(start_dist)>0.2 and min(start_dist)/max(start_dist)<0.8:
            # Remove 80% of the difference in rows from the larger class
            removal_amount = int(abs(start_dist[0]-start_dist[1])*0.8)
            if len(df[df[y_col]==0])>len(df[df[y_col]>0]):
                sample = df[df[y_col]==0].sample(n=removal_amount,random_state=seed)
            else:
                sample = df[df[y_col]>0].sample(n=removal_amount,random_state=seed)
            df = df.drop(sample.index)
            end_dist = (len(df[df[y_col]==0]),len(df[df[y_col]>0]))
            logging.info(f'Balanced dataframe on Intersection counts by removing {removal_amount} rows:\nStarting ratio: {start_dist[0]} vs. {start_dist[1]}\nFinal ratio: {end_dist[0]} vs. {end_dist[1]}')
        else:
            logging.info(f'No target ratio given and Dataframe is already balanced well enough: {start_dist}')
    df = df.reset_index(drop=True)
    return df

def find_cutoff(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Adds a column to the dataframe, that contains the maximum cutoff the given row would still survive.
    Assumes that quick_alternative method.

    :param df: Dataframe with the NGS_read_count and Publication columns

    :result: Dataframe with the additional max_cutoff column
    '''
    df = identify_candidates(df)
    df["max_cutoff"] = df.groupby(["Publication", "ID"])["NGS_read_count"].transform("max")
    return df




'''
case 'quick_alternative':
    # TODO: Standardise this and add pooling option
    logging.debug(f'Applying quick alternative cutoff without pooling')
    df = identify_candidates(df)
    survivors = df[df["NGS_read_count"]>=cutoff]["ID"]
    df = df[df["ID"].isin(survivors)]
    if "Publication" in df.columns:
        pub_dfs = []
        for pub in df["Publication"].unique():
            pub_df = df[df["Publication"]==pub]
            pub_survivors = pub_df[pub_df["NGS_read_count"]>=cutoff]["ID"]
            pub_dfs.append(pub_df[pub_df["ID"].isin(pub_survivors)])
        try:
            df = pd.concat(pub_dfs)
        except Exception as e:
            logging.error(f'Problem during cutoff application:\n{e}\n{df.head()}\n{df.describe()}\n{pub_dfs}\n{df["Publication"].nunique()}')
            '''
