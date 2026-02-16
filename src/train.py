import joblib

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data_preprocessing import preprocess_data

def train(df: pd.DataFrame, model_path='models/best_model.pkl') -> None:
    """
    Train multiple models on the credit risk dataset and save the best one.
    Args:
        df (pd.DataFrame): The input DataFrame to train on.
     Returns:
        None
     Saves:
        best_model.pkl: The best performing model based on F1 score.
     Prints:
        The name and F1 score of the best model.
    """ 

    X_train, X_test, y_train, y_test = preprocess_data(df)

    models = {
        'LogReg': LogisticRegression(solver='liblinear'),
        'RF': RandomForestClassifier(),
        'XGB': XGBClassifier(),
    }

    grid_params = {
        'LogReg': {
            'penalty': ['l1', 'l2'],
            'C': [0.1, 1, 10],
        },
        'RF': {
            'n_estimators': [200, 300, 500],
            'max_depth': [3, 5, 7],
        },
        'XGB': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 1.0],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'gamma': [0, 0.1, 0.3],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [1, 1.5, 2],
        }
    }

    best_score = 0
    best_model = None
    best_model_name = None

    for name, model in models.items():
        grid = GridSearchCV(model, grid_params[name], cv=5, scoring='f1', verbose=0, n_jobs=-1)
        grid.fit(X_train, y_train)

        if best_score > grid.best_score_:
            best_score = grid.best_score
            best_model = grid.best_estimator_
            best_model_name = name

    joblib.dump(best_model, model_path)
    print(f'Saved best model: {best_model_name} with F1: {best_score}')
