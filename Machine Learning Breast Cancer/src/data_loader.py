# data_loader.py

import pandas as pd

from pathlib import Path

def load_data(file_name, folder="external", index_col=0):
    """Reads an Excel file and returns a pandas DataFrame."""

    # Parent folder.
    parent_folder = Path(__file__).resolve().parents[1]

    # Raw databases source.
    data_folder = parent_folder / "data" / folder

    # File path.
    file_path = data_folder / file_name

    # Load with pandas.
    try:
        df = pd.read_excel(
            file_path,                 # be sure to use format: 'dataset_name.xls' or 'dataset_name.xlsx'.
            index_col=index_col)
        print(f"Successfully loaded: {file_path}")

        return df

    except FileNotFoundError:
        print(f"Could not find '{file_name}' inside the folder: {data_folder}")

        return None

    except Exception as exception:
        print(f"An error occurred: {exception}")

        return None

def save_data(df, file_name, folder="processed"):
    """Saves a pandas DataFrame to an Excel file."""

    # Parent folder.
    parent_folder = Path(__file__).resolve().parents[1]

    # Source.
    data_folder = parent_folder / "data" / folder

    # Ensure the target directory exists (prevents FileNotFoundError).
    data_folder.mkdir(parents=True, exist_ok=True)

    # File path.
    file_path = data_folder / file_name

    # Save dataframe to Excel.
    try:
        df.to_excel(file_path, index=False)  
        print(f"Successfully saved to: {file_path}")

    except Exception as exception:
        print(f"An error occurred: {exception}")