import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config.logger import setup_logger

logger = setup_logger()


def preprocess_data(df: pd.DataFrame) -> tuple:
    '''
    Build a preprocessing pipeline for the credit risk dataset.

    Args:
        df (pd.DataFrame): The input DataFrame to preprocess.
    Returns:
        X_train_processed (np.ndarray): The preprocessed training features.
        X_test_processed (np.ndarray): The preprocessed testing features.
        y_train (pd.Series): The training target variable.
        y_test (pd.Series): The testing target variable.
    '''

    y = df['cb_person_default_on_file'].map({'N': 0, 'Y': 1})
    X = df.drop(columns=['cb_person_default_on_file'])

    logger.info(f'Initial dataset shape: X={X.shape}, y={y.shape}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8, test_size=0.2)
    
    logger.info(f'Split data: X_train={X_train.shape}, X_test={X_test.shape}')
    logger.info(f'Split targets: y_train={y_train.shape}, y_test={y_test.shape}')

    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['string', 'object']).columns.tolist()
    

    # Define transformers for numeric features
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ])

    # Define transformer for categorical features
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    logger.debug(f'Numeric columns: {numeric_cols}')
    logger.debug(f'Categorical columns: {categorical_cols}')

    # Combine transformers into a preprocessor
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    logger.info(f'Preprocessed X_train shape: {X_train_processed.shape}')
    logger.info(f'Preprocessed X_test shape: {X_test_processed.shape}')
    logger.info(f'Is any NaNs in X_train? {np.isnan(X_train_processed).any()}')

    return X_train_processed, X_test_processed, y_train, y_test