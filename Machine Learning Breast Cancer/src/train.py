# train.py

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_preprocessor(X_train, scale_pca=True):
    """
    Build a ColumnTransformer that standardizes continuous/ordinal clinical columns,
    scales or passes through PCA image columns based on 'scale_pca', and leaves binary 
    columns untouched.
    """

    binary_cols = [
        col for col in X_train.columns
        if set(X_train[col].unique()).issubset({0, 1, 0.0, 1.0})
    ]
    pca_cols = [
        col for col in X_train.columns
        if col.startswith("Image_PC")
    ]
    clinical_scale_cols = [
        col for col in X_train.columns
        if col not in binary_cols and col not in pca_cols
    ]

    pca_transformer = StandardScaler() if scale_pca else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("clinical_scaler", StandardScaler(), clinical_scale_cols),
            ("pca_scaler", pca_transformer, pca_cols),
            ("bin_passthrough", "passthrough", binary_cols)
        ],
        remainder="passthrough"
    )

def train_and_tune_models(
    models,
    param_grids,
    X_train,
    y_train,
    task="classification",
    cv_splits=5,
    scoring=None,
    scale_pca=True,
    random_state=23
):
    """
    Wrap each model in a ColumnTransformer + Estimator pipeline and tune hyperparameters 
    using GridSearchCV on the training set.

    Parameters:
    - scale_pca: True (scale PCs) or False (keep original PC variance).
    """
    
    if task not in ["classification", "regression"]:
        raise ValueError("task must be either 'classification' or 'regression'.")

    if not isinstance(scale_pca, bool):
        raise ValueError("scale_pca must be True or False.")

    y_train_1d = y_train.squeeze()

    if task == "classification":
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        scoring = scoring or "roc_auc"
    else:
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        scoring = scoring or "neg_root_mean_squared_error"

    best_models = {}
    cv_summary = {}

    print(
        f"\n{task.upper()} HYPERPARAMETER TUNING "
        f"({cv_splits}-Fold CV | Metric: {scoring} | scale_pca: {scale_pca})"
    )

    for name, estimator in models.items():
        preprocessor = _build_preprocessor(X_train, scale_pca=scale_pca)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])

        raw_grid = param_grids.get(name, {})
        pipeline_grid = {f"model__{param}": values for param, values in raw_grid.items()}

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=pipeline_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True
        )

        grid_search.fit(X_train, y_train_1d)

        clean_params = {
            k.replace("model__", ""): v for k, v in grid_search.best_params_.items()
        }

        best_models[name] = grid_search.best_estimator_
        cv_summary[name] = {
            "best_cv_score": grid_search.best_score_,
            "best_params": clean_params
        }

        print(f"[{name:14s}] Best CV {scoring}: {grid_search.best_score_:.4f} | Best Params: {clean_params}")

    return best_models, cv_summary