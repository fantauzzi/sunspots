from pathlib import Path
import pandas as pd
import click


#  mlflow run . -P input_file="data/SN_m_tot_V2.0.csv" -P train_size=0.8 -P train_file="data/train.csv" -P test_file="data/test.csv"

@click.command()
@click.option('--input_file',
              required=True,
              help='File with the dataset for training/validation/test.',
              type=str)
@click.option('--train_size',
              required=True,
              help='''Size of the train/validation set. If less than 1, it is meant to be the fraction of the overall 
              dataset to be used for training; if at least one, it is meant to be the size in samples.''',
              type=float)
@click.option('--train_file',
              required=True,
              help='Output file with the pre-processed training/validation set.',
              type=str)
@click.option('--test_file',
              required=True,
              help='Output file with the pre-processed test set.',
              type=str)
def main(input_file: str,
         train_size: float,
         train_file: str,
         test_file: str):
    path_to_prj_root = '../..'
    input_file_corrected = str(path_to_prj_root / Path(input_file))
    train_file_corrected = str(path_to_prj_root / Path(train_file))
    test_file_corrected = str(path_to_prj_root / Path(test_file))

    df = pd.read_csv(input_file_corrected)
    df = df[(df.n_observations > 0) & (df.provisional == 'n')]
    dataset_size = len(df)
    train_val_set_size = int(train_size) if train_size >= 1 else int(train_size * dataset_size)
    train_val_df = df.iloc[:train_val_set_size]
    test_set_size = dataset_size - train_val_set_size
    test_df = df.iloc[train_val_set_size:]
    assert len(test_df) == test_set_size
    train_val_df.to_csv(train_file_corrected, index=False)
    test_df.to_csv(test_file_corrected, index=False)


if __name__ == '__main__':
    main()
