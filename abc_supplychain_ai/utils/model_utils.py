# 📁 FILE: utils/model_utils.py

import joblib
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier

def load_prophet_model(path='models/prophet_model.pkl') -> Prophet:
    """
    Load a saved Prophet forecasting model from disk.

    Parameters:
        path (str): File path to the saved Prophet model (.pkl)

    Returns:
        Prophet: Loaded Prophet model instance
    """
    return joblib.load(path)


def load_rf_model(path='models/rf_classifier.pkl') -> RandomForestClassifier:
    """
    Load a saved Random Forest classification model from disk.

    Parameters:
        path (str): File path to the saved RandomForestClassifier model (.pkl)

    Returns:
        RandomForestClassifier: Loaded RandomForest model
    """
    return joblib.load(path)


def save_model(model, path: str):
    """
    Save any scikit-learn or Prophet model to disk using Joblib.

    Parameters:
        model: Trained model object (e.g., Prophet, RandomForestClassifier)
        path (str): Destination file path to save model (.pkl)
    """
    joblib.dump(model, path)
