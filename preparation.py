from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import numpy as np


def generate_data():

    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_clusters_per_class=1
    )

    return X, y


def normalize_data(X_train, X_val, X_test):

    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)

    X_train = (X_train - mu) / sigma
    X_val = (X_val - mu) / sigma
    X_test = (X_test - mu) / sigma

    return X_train, X_val, X_test


def prepare_data():

    X, y = generate_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=42,
        stratify=y_train
    )

    X_train, X_val, X_test = normalize_data(
        X_train,
        X_val,
        X_test
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )