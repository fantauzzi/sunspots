import urllib.request
import re
from utils import get_silso_dataset_info


def fetch_silso_dataset() -> str:
    dataset_url, dataset_filepath = get_silso_dataset_info()
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


def main():
    fetch_silso_dataset()
    # _, dataset_filepath = get_silso_dataset_info()
    # df = pd.read_csv(dataset_filepath)
    ...


if __name__ == '__main__':
    main()
