from pathlib import Path


def get_silso_dataset_info() -> (str, str):
    dataset_url = 'https://www.sidc.be/silso/DATA/SN_m_tot_V2.0.txt'
    dataset_filename = Path(dataset_url).stem + '.csv'
    dataset_filepath = f'../data/{dataset_filename}'
    return dataset_url, dataset_filepath
