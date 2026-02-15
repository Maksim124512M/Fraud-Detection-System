import numpy as np
import pandas as pd

from src.data_preprocessing import build_preprocessor_pipeline

def test_data_preprocessing():
    test_df = pd.read_csv('data/credit_risk_dataset.csv')
    X_train, X_test, y_train, y_test = build_preprocessor_pipeline(test_df)

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)