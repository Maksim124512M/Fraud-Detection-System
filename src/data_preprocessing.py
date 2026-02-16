import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Build a preprocessing pipeline for the credit risk dataset.

    Args:
        df (pd.DataFrame): The input DataFrame to preprocess.
    Returns:
        X_train_processed (np.ndarray): The preprocessed training features.
        X_test_processed (np.ndarray): The preprocessed testing features.
        y_train (pd.Series): The training target variable.
        y_test (pd.Series): The testing target variable.
    """

    y = df['cb_person_default_on_file']
    X = df.drop(columns=['cb_person_default_on_file'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8, test_size=0.2)
    
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

    # Combine transformers into a preprocessor
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
    ])

    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test