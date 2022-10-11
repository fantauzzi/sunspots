import urllib.request
import re
from pathlib import Path
import os
import click

# mlflow run . -P url="https://www.sidc.be/silso/DATA/SN_m_tot_V2.0.txt" -P path="../../data"

print(f'Working dir is {os.getcwd()}')


def fetch_silso_dataset(dataset_url: str, dataset_filepath: str) -> str:
    with urllib.request.urlopen(dataset_url) as response:
        text = response.read().decode('utf-8')
    text_lines = text.split('\n')
    if len(text_lines[-1]) == 0:
        text_lines = text_lines[:-1]
    text_lines = [re.split(r'\s+', line) for line in text_lines]
    text_lines = [line[:-1] + ['n'] if line[-1] == '' else line[:-1] + ['y'] for line in text_lines]
    text_lines = [','.join(line) for line in text_lines]
    with open(dataset_filepath, 'wt') as text_file:
        text_file.write('year,month,time,mean,std,n_observations,provisional\n')
        for line in text_lines:
            text_file.write(f'{line}\n')

    return dataset_filepath


@click.command()
@click.option('--url',
              required=True,
              help='URL where to fetch the SILSO dataset from.')
@click.option('--path',
              required=True,
              help='Path where to download the dataset, can be relative or absolute.')
def main(url: str, path: str) -> None:
    click.echo(f'URL is {url} and path is {path}')
    dataset_filename = Path(url).stem + '.csv'
    dataset_filepath = f'{path}/{dataset_filename}'
    fetch_silso_dataset(url, dataset_filepath)


if __name__ == '__main__':
    main()
