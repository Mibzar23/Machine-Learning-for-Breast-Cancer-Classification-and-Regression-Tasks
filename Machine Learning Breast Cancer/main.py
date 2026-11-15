# main.py

from sklearn.model_selection import train_test_split
from src import data_loader, data_preprocessing, evaluation, model, train

# Load dataset.
cancer_df = data_loader.load_data(
    file_name='cancer_dataset.xls', 
    index_col=0
)

#============================================================================================
# PART ONE: DATA PREPROCESSING.
#============================================================================================

# Handles separation of targets from features and encodes missing values to "pandas" format.
preprocessor = data_preprocessing.DataPreprocessor(
    raw_data=cancer_df,
    has_target=True
)

# Encode binary features.
binary_features = [
    'ER',
    'PgR',
    'HER2',
    'TrippleNegative',
    'LNStatus'
]

preprocessor.binary_encode(binary_columns=binary_features)

# Encode nominal categorical features.
preprocessor.one_hot_encode(nominal_columns=['HistologyType'])

# Drop 'Gene' column due to the number of missing values in it.
preprocessor.raw_data = preprocessor.raw_data.drop(columns=['Gene'])
print("Column 'Gene' has been dropped due to the number of missing values in it.")

# Impute missing values.
preprocessor.fill_missing_values()

# Convert 'Age' to integer.
preprocessor.age_as_integer()

# Target columns (pCR and RelapseFreeSurvival).
y = preprocessor.target_column

# Features columns (all other columns).
X = preprocessor.raw_data

# Save preprocessed dataset.
data_loader.save_data(
    df=X,
    file_name='preprocessed_cancer_features.xlsx',
    folder='processed'
)
data_loader.save_data(
    df=y,
    file_name='preprocessed_cancer_targets.xlsx',
    folder='processed'
)

#============================================================================================
# PART TWO: FEATURE MANIPULATION
#============================================================================================

# Load preprocessed features and targets with matching indices.
cancer_features = data_loader.load_data(
    file_name='preprocessed_cancer_features.xlsx', 
    folder='processed',
    index_col=None
)
cancer_targets = data_loader.load_data(
    file_name='preprocessed_cancer_targets.xlsx',
    folder='processed',
    index_col=None
)

# Extract both targets.
y_classification = cancer_targets['pCR (outcome)'].astype(int)
y_regression = cancer_targets['RelapseFreeSurvival (outcome)'].astype(float)

# Split into Train (80%) and Test (20%) once so both tasks share the exact same patients.
(
    X_train, X_test,
    y_train_classification, y_test_classification,
    y_train_regression, y_test_regression
) = train_test_split(
    cancer_features,
    y_classification,
    y_regression,
    test_size=0.2,
    random_state=23,
    stratify=y_classification
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Define clinical and image feature lists.
clinical_features = [
    'Age',
    'ER',
    'PgR',
    'HER2',
    'TrippleNegative',
    'ChemoGrade',
    'Proliferation',
    'HistologyType_1.0',
    'HistologyType_2.0',
    'LNStatus',
    'TumourStage'
]

image_features = [col for col in X_train.columns if col not in clinical_features]

# Classification pipeline (target: pCR (outcome)).
columns_to_drop_classification = [
    'TrippleNegative',
    'ChemoGrade',
    'HistologyType_1.0',
    'HistologyType_2.0'
]

# Apply feature selection (column drop + 13-component PCA) to train and test sets.
X_train_classification, X_test_classification = data_preprocessing.feature_manipulation(
    features_df=X_train,
    test_df=X_test,
    task='classification',
    columns_to_drop=columns_to_drop_classification,
    image_features=image_features
)

# Regression pipeline (target: RelapseFreeSurvival (outcome)).
columns_to_drop_regression = [
    'ER',
    'PgR',
    'TrippleNegative',
    'LNStatus',
    'HistologyType_1.0',
    'HistologyType_2.0'
]

# Apply feature selection (column drop + 13-component PCA) to train and test sets.
X_train_regression, X_test_regression = data_preprocessing.feature_manipulation(
    features_df=X_train,
    test_df=X_test,
    task='regression',
    columns_to_drop=columns_to_drop_regression,
    image_features=image_features
)

#============================================================================================
# PART THREE: MACHINE LEARNING METHOD APPLICATION
#============================================================================================

# Classification pipeline (target: pCR (outcome)).
clf_models, clf_grids = model.get_classification_models()

best_clf_models, clf_cv_summary = train.train_and_tune_models(
    models=clf_models,
    param_grids=clf_grids,
    X_train=X_train_classification,
    y_train=y_train_classification,
    task='classification',
    scoring='roc_auc',
    scale_pca=True,
    random_state=23
)

print("CLASSIFICATION MODELS TRAINING COMPLETED.")

# Regression pipeline (target: RelapseFreeSurvival (outcome)).
reg_models, reg_grids = model.get_regression_models()

best_reg_models, reg_cv_summary = train.train_and_tune_models(
    models=reg_models,
    param_grids=reg_grids,
    X_train=X_train_regression,
    y_train=y_train_regression,
    task='regression',
    scoring='neg_root_mean_squared_error',
    scale_pca=True,
    random_state=23
)

print("REGRESSION MODELS TRAINING COMPLETED.")

#============================================================================================
# PART FOUR: MODEL EVALUATION & EXPORT TO RESULTS FOLDER
#============================================================================================

# Classification evaluation.
evaluation.save_cv_summary(clf_cv_summary, task='classification')

clf_test_metrics = evaluation.plot_classification_2x2(
    trained_models=best_clf_models,
    X_test=X_test_classification,
    y_test=y_test_classification
)

clf_stability_raw, clf_stability_summary = evaluation.evaluate_seed_stability(
    best_models=best_clf_models,
    X_train=X_train_classification,
    y_train=y_train_classification,
    X_test=X_test_classification,
    y_test=y_test_classification,
    task='classification'
)

print("CLASSIFICATION MODELS EVALUATION AND EXPORTATION COMPLETED.")

# Regression evaluation.
evaluation.save_cv_summary(reg_cv_summary, task='regression')

reg_test_metrics = evaluation.plot_regression_2x2(
    trained_models=best_reg_models,
    X_test=X_test_regression,
    y_test=y_test_regression
)

reg_stability_raw, reg_stability_summary = evaluation.evaluate_seed_stability(
    best_models=best_reg_models,
    X_train=X_train_regression,
    y_train=y_train_regression,
    X_test=X_test_regression,
    y_test=y_test_regression,
    task='regression'
)

print("REGRESSION MODELS EVALUATION AND EXPORTATION COMPLETED.")