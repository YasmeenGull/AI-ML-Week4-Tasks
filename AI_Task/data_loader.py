# data_loader.py

import pandas as pd


def load_dataset(file_path):
    """
    Load dataset from CSV file.
    """

    try:
        df = pd.read_csv(file_path)

        print("Dataset loaded successfully.\n")

        return df

    except FileNotFoundError:

        print("Dataset not found.")

        return None