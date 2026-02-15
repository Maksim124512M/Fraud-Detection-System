import joblib
import pandas as pd

def predict_new_data(df: pd.DataFrame) -> None:
    """
    Load the best model and make predictions on the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to make predictions on.
    Returns:
        None
    """

    model = joblib.load('best_model.pkl')
    predictions = model.predict(df)

    return predictions