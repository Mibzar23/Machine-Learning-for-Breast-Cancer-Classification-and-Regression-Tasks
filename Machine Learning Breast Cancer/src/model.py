# model.py

import json
import os

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor


def load_configurations(config_filename="configurations.json", folder="configs"):

    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    config_path = os.path.join(project_root, folder, config_filename)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


# CLASSIFICATION MODELS (Target: pCR)
def get_classification_models(config_filename="configurations.json", folder="configs"):
    """
    Initialize and return classification models and their hyperparameter grids.
    """
    configs = load_configurations(config_filename=config_filename, folder=folder)["classification"]

    model_builders = {
        "Random Forest": RandomForestClassifier,
        "SVM": SVC,
        "XGBoost": XGBClassifier,
        "Decision Tree": DecisionTreeClassifier,
        "KNN": KNeighborsClassifier,
        "Naive Bayes": GaussianNB
    }

    models = {}
    param_grids = {}

    for name, builder in model_builders.items():
        model_cfg = configs[name]
        models[name] = builder(**model_cfg.get("params", {}))
        param_grids[name] = model_cfg.get("param_grid", {})

    return models, param_grids


# REGRESSION MODELS (Target: RelapseFreeSurvival)
def get_regression_models(config_filename="configurations.json", folder="configs"):
    """
    Initialize and return regression models and their hyperparameter grids.
    """
    configs = load_configurations(config_filename=config_filename, folder=folder)["regression"]

    model_builders = {
        "SVM": SVR,
        "XGBoost": XGBRegressor,
        "Random Forest": RandomForestRegressor,
        "KNN": KNeighborsRegressor
    }

    models = {}
    param_grids = {}

    for name, builder in model_builders.items():
        model_cfg = configs[name]
        models[name] = builder(**model_cfg.get("params", {}))
        param_grids[name] = model_cfg.get("param_grid", {})

    return models, param_grids