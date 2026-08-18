import pandas as pd
import logging
import os

from parsing_utils import load_dataset, ACCNUMDICT

RESULT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data_unpooled")
#RESULT_PATH = os.path.join(os.path.dirname(__file__), "data_unpooled")

STRAIN_DICT = dict({
    # H1N1
    "Alnaji2021": "A_PuertoRico_8_1934",
    "Pelz2021": "A_PuertoRico_8_1934",
    "Wang2023": "A_PuertoRico_8_1934",
    "Wang2020": "A_PuertoRico_8_1934",
    "Zhuravlev2020": "A_PuertoRico_8_1934",
    "Kupke2020": "A_PuertoRico_8_1934",
    "VdHoecke2015": "A_PuertoRico_8_1934",
    "Alnaji2019_Cal07": "A_California_07_2009",
    "Alnaji2019_NC": "A_NewCaledonia_20-JY2_1999",
    "Mendes2021": "A_WSN_33",
    "Boussier2020": "A_WSN_33",
    # H3N2
    "Alnaji2019_Perth": "A_Perth_16_2009",
    "Berry2021_A": "A_Connecticut_Flu122_2013",
    # H5N1
    "Penn2022": "A_turkey_Turkey_1_2005",
    # H7N9
    "Lui2019": "A_Anhui_1_2013",
    # B
    "Alnaji2019_BLEE": "B_Lee_1940",
    "Berry2021_B": "B_Victoria_504_2000",
    "Valesano2020_Vic": "B_Victoria_504_2000",
    "Sheng2018": "B_Brisbane_60_2008",
    "Berry2021_B_Yam": "B_Yamagata_16_1988",
    "Southgate2019": "B_Yamagata_16_1988",
    "Valesano2020_Yam": "B_Yamagata_16_1988"
})

def setup_logging():
    '''
    Set up logging for the script
    :return:
    '''
    logging.basicConfig(handlers=[logging.StreamHandler()],
                        format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def parse_to_csv(data, filename):
    """
    Parse the raw data to a csv file.

    :param data: the raw data
    :param filename: the name of the csv file
    :return: None
    """
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data has been saved to {filename}")

def parse_to_csv_unpooled():
    '''
    Parse the raw data to a csv file for each experiment.

    :return:
    '''
    for exp in ACCNUMDICT.keys():
        logging.info(f'Parsing data from {exp}')
        data = load_dataset(exp).reset_index()
        data.drop(columns=["index"], inplace=True)
        strain = STRAIN_DICT[exp]
        path = os.path.join(RESULT_PATH, strain)
        logging.info(f'saving data to {path}')
        os.makedirs(path, exist_ok=True)
        if exp == "VdHoecke2015":
            exp = "vdHoecke2015"
        data.to_csv(path + f'/{exp}.csv', index=False)

if __name__ == "__main__":
    setup_logging()
    parse_to_csv_unpooled()