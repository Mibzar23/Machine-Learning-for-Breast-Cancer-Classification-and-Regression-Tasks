# data_preprocessing.py

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, raw_data, has_target=True):
        # Copy to avoid mutating the original dataframe outside the class.
        self.raw_data = raw_data.copy()
        
        # Replace all '999' with standard numpy NaN.
        self.raw_data.replace(999, np.nan, inplace=True)

        # Conditional logic to interpret if the data is labeled or for prediction only.
        if has_target:
            # Define target columns.
            targets = [
                'pCR (outcome)',
                'RelapseFreeSurvival (outcome)'
            ]

            # Drop row if there are missing values in target columns.
            self.raw_data = self.raw_data.dropna(subset=targets)

            self.target_column = self.raw_data[targets]

            self.raw_data.drop(
                targets,
                axis=1,
                inplace=True
            )
        else:
            self.target_column = None
            print("Target columns are not defined. The data is assumed to be unlabeled.")

    def binary_encode(self, binary_columns):
        """
        Binary encode the specified columns in the dataset using numeric/NaN handling.
        """
        for column in binary_columns:
            if column in self.raw_data.columns:

                # Check the number of unique values in the column.
                if self.raw_data[column].nunique() > 2:

                    # If the column has more than two unique values, check if the minimum value is 0.
                    if self.raw_data[column].min() == 0:
                        encoded = np.where(
                            self.raw_data[column].isna(), 
                            np.nan, 
                            np.where(self.raw_data[column] == 0, 0, 1)
                        )
                        self.raw_data[column] = encoded.astype(float)

                    else:
                        print(f"Column '{column}' has more than two unique values and must be one-hot encoded.")

                elif self.raw_data[column].nunique() == 2:

                    # If the column has exactly two unique values, check if the minimum value is 0 or greater than 0.
                    if self.raw_data[column].min() > 0:
                        print(f"Column '{column}' has two unique values, but the lowest is greater than '0'. It is recommended to one-hot encode it.")

                    # If the column has exactly two unique values and the minimum value is 0, check if the maximum value is 1.
                    elif self.raw_data[column].min() == 0 and self.raw_data[column].max() == 1:
                        self.raw_data[column] = self.raw_data[column].astype(float)
                        print(f"Column '{column}' is already binary encoded.")

                    # If the column has exactly two unique values and the minimum value is 0, but the maximum value is not 1, encode it to binary.
                    elif self.raw_data[column].min() == 0 and self.raw_data[column].max() != 1:
                        encoded = np.where(
                            self.raw_data[column].isna(), 
                            np.nan, 
                            np.where(self.raw_data[column] == 0, 0, 1)
                        )
                        self.raw_data[column] = encoded.astype(float)
                        print(f"Column '{column}' has been binary encoded.")
                    
                    else:
                        print(f"Column '{column}' has a negative value. Review the data manually.")
                
                elif self.raw_data[column].nunique() == 1:
                    print(f"Column '{column}' has only one unique value.")
                
                else:
                    raise ValueError(f"Column '{column}' only has missing values.")
            
            else:
                raise ValueError(f"Column '{column}' does not exist in the dataset.")

        return self.raw_data

    def one_hot_encode(self, nominal_columns):
        """
        One-hot encode nominal columns using float representation for NaNs.
        """
        print("Starting one-hot encoding. This step is recommended after binary encoding.")

        for column in nominal_columns:
            if column in self.raw_data.columns:

                # Check the number of unique values in the column.
                if self.raw_data[column].nunique() == 2:

                    # If the column has exactly two unique values, check any value is different from 0.
                    if self.raw_data[column].all() != 0:
                        col_series = self.raw_data[column]
                        dummies = pd.get_dummies(col_series, prefix=column, dtype=float)

                        # If there are NaN values in the original column, set the corresponding rows in the dummy DataFrame to NaN.
                        if col_series.isna().any():
                            dummies.loc[col_series.isna()] = np.nan

                        col_idx = self.raw_data.columns.get_loc(column)
                        self.raw_data = self.raw_data.drop(columns=[column])

                        for i, dummy_col in enumerate(dummies.columns):
                            self.raw_data.insert(col_idx + i, dummy_col, dummies[dummy_col])

                        print(f"Column '{column}' has been one-hot encoded.")

                    else:
                        print(f"Column '{column}' has two unique values, but lowest is '0'. Use binary encode instead.")

                # If the column has more than two unique values, perform one-hot encoding.
                elif self.raw_data[column].nunique() > 2:
                    col_series = self.raw_data[column]
                    dummies = pd.get_dummies(col_series, prefix=column, dtype=float)

                    if col_series.isna().any():
                        dummies.loc[col_series.isna()] = np.nan

                    col_idx = self.raw_data.columns.get_loc(column)
                    self.raw_data = self.raw_data.drop(columns=[column])

                    for i, dummy_col in enumerate(dummies.columns):
                        self.raw_data.insert(col_idx + i, dummy_col, dummies[dummy_col])

                    print(f"Column '{column}' has been one-hot encoded.")

                else:
                    print(f"Column '{column}' has only one unique value.")

            else:
                raise ValueError(f"Column '{column}' does not exist in the dataset.")

        return self.raw_data
    
    def age_as_integer(self):
        """
        Change 'Age' to integer format at the final stage.
        """
        if 'Age' not in self.raw_data.columns:
            raise ValueError("Could not find 'Age' column.")

        self.raw_data['Age'] = self.raw_data['Age'].round(0).astype(int)
        return self.raw_data

    def fill_missing_values(self):
            """
            Fill missing values in the dataset using SimpleImputer.
            """

            clinical_features = [
                'Age', 
                'ER', 
                'PgR', 
                'HER2', 
                'TrippleNegative', 
                'ChemoGrade', 
                'Proliferation', 
                'HistologyType', 
                'LNStatus', 
                'TumourStage'
            ]

            features_to_use = []
            for column in clinical_features:
                if column in self.raw_data.columns:
                    features_to_use.append(column)
                else:
                    dummy_cols = [c for c in self.raw_data.columns if c.startswith(f"{column}_")]
                    features_to_use.extend(dummy_cols)

            if not features_to_use:
                raise ValueError("No clinical features found in the dataset for imputation.")

            # Create a temporary float copy for the imputer.
            temp_df = self.raw_data[features_to_use].astype(float)
            
            # Initialize SimpleImputer (using most_frequent).
            imputer = SimpleImputer(strategy='most_frequent')
            
            imputed_array = imputer.fit_transform(temp_df)

            # Convert back to a DataFrame.
            imputed_df = pd.DataFrame(
                imputed_array, 
                columns=features_to_use, 
                index=temp_df.index
            )

            # Cast discrete features back to integers, leaving 'Age' as float.
            for col in features_to_use:
                if col != 'Age':
                    imputed_df[col] = imputed_df[col].round(0).astype('Int64')

            # Assign back to raw_data.
            self.raw_data[features_to_use] = imputed_df

            print("Missing values successfully imputed using SimpleImputer.")

def feature_manipulation(features_df=None, test_df=None, task=None, columns_to_drop=None, image_features=None):
    """
    Perform task-specific preprocessing on the training DataFrame and optionally
    transform the test DataFrame using the training scaler and PCA parameters.
    """

    if features_df is None:
        raise ValueError("features_df must be provided.")
    
    if task not in ['classification', 'regression']:
        raise ValueError("task must be either 'classification' or 'regression'.")

    train_df = features_df.copy()
    test_out = test_df.copy() if test_df is not None else None

    # Drop specified insignificant columns.
    if columns_to_drop:
        train_df = train_df.drop(columns=columns_to_drop, errors='ignore')
        if test_out is not None:
            test_out = test_out.drop(columns=columns_to_drop, errors='ignore')
        print(f"[{task.upper()}] Dropped columns: {columns_to_drop}")

    # Standardize and perform PCA on image features (fit on train, transform both).
    if image_features:
        valid_img_cols = [col for col in image_features if col in train_df.columns]
        
        if valid_img_cols:
            scaler = StandardScaler()
            pca = PCA(n_components=13, svd_solver='full', random_state=23)
            pc_cols = [f'Image_PC{i+1}' for i in range(13)]

            # Fit and transform on training set.
            train_img = train_df[valid_img_cols].astype(float).to_numpy(copy=True)
            train_pcs = pca.fit_transform(scaler.fit_transform(train_img))
            train_pca_df = pd.DataFrame(train_pcs, columns=pc_cols, index=train_df.index)
            train_df = pd.concat([train_df.drop(columns=valid_img_cols), train_pca_df], axis=1)

            # Transform test set using the training scaler and PCA.
            if test_out is not None:
                test_img = test_out[valid_img_cols].astype(float).to_numpy(copy=True)
                test_pcs = pca.transform(scaler.transform(test_img))
                test_pca_df = pd.DataFrame(test_pcs, columns=pc_cols, index=test_out.index)
                test_out = pd.concat([test_out.drop(columns=valid_img_cols), test_pca_df], axis=1)

            print(f"[{task.upper()}] Reduced {len(valid_img_cols)} image features to 13 PCs.")

    if test_out is not None:
        return train_df, test_out
    
    return train_df