import kagglehub
import pandas as pd
import sys
from sklearn import datasets

def download_dataset():
    """
    download dataset and return path to dataset csv file
    """

    path = kagglehub.dataset_download("shraddha4ever20/california-housing-dataset")
    print("path to datatset: ", path)
    return path

def load_dataset():
    """
    return dataset dataframe
    """
    dataset = datasets.fetch_california_housing(as_frame=True)
    return dataset


def verify_load():
    dataset = load_dataset()
    print(dataset.data.shape, dataset.target.shape)
    print(dataset.feature_names)
    print(dataset.frame.head())