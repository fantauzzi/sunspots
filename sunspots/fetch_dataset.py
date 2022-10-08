import urllib.request
from pathlib import Path
import pandas as pd
import re

dataset_url = 'https://www.sidc.be/silso/DATA/SN_d_tot_V2.0.txt'
dataset_filename = Path(dataset_url).stem+'.csv'
dataset_filepath = f'../data/{dataset_filename}'


def fetch_silso_dataset() -> str:
    with urllib.request.urlopen(dataset_url) as response:
        text = response.read().decode('utf-8')
    text_lines = text.split('\n')
    if len(text_lines[-1]) == 0:
        text_lines = text_lines[:-1]
    text_lines = [re.split(r'\s+', line) for line in text_lines]
    text_lines = [line[:-1]+['n'] if line[-1] =='' else line[:-1]+['y'] for line in text_lines]
    text_lines = [','.join(line) for line in text_lines]
    with open(dataset_filepath, 'wt') as text_file:
        for line in text_lines:
            text_file.write(f'{line}\n')

    return dataset_filepath


def main():
    fetch_silso_dataset()
    df = pd.read_csv(dataset_filepath)
    ...


if __name__ == '__main__':
    main()
