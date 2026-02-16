import pandas as pd

from src.train import train


def test_train_function():
    df = pd.read_csv('data/credit_risk_dataset.csv').sample(n=50, random_state=42)

    result = train(df)

    assert result is True