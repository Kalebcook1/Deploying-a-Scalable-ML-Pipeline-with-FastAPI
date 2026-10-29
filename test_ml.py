import os

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "census.csv")


@pytest.fixture(scope="module")
def data():
    """A reproducible 2000 row sample of data"""
    return pd.read_csv(DATA_PATH).sample(2000, random_state=0)


@pytest.fixture(scope="module")
def processed(data):
    """Train and test arrays produced by data"""
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    X_train, y_train, encoder, lb, = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    return X_train, y_train, X_test, y_test

# TODO: implement the first test. Change the function name and input as needed


def test_train_model_algorithm(processed):
    """
    # train_model should return a fitted RandomForestClassifier
    """
    X_train, y_train, _, _ = processed
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_inference_output(processed):
    """
    # Inference should return a numpy array of 0's or 1's, one per row
    """
    X_train, y_train, X_test, _ = processed
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    # compute_model_metrics should return known precision, recall, and F1
    """
    y = np.array([1, 0, 1, 1])
    preds = np.array([1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(0.8)


def test_train_test_split_sizes(data):
    """
    # An 80/20 split should give the expected size and lose no rows
    """
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    assert len(train) == 1600
    assert len(test) == 400
    assert len(train) + len(test) == len(data)
