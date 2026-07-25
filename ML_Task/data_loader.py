import pandas as pd


def load_dataset(file_path):
    """
    Load classification dataset.
    """

    df = pd.read_csv(file_path)

    print("Dataset Loaded Successfully\n")

    return df