import os
import json
import zipfile

def set_kaggle_credentials(json_path):
    """Set Kaggle credentials from a JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Kaggle JSON file not found at {json_path}")

    with open(json_path) as file:
        data = json.load(file)

    os.environ['KAGGLE_USERNAME'] = data['username']
    os.environ['KAGGLE_KEY'] = data['key']

def download_dataset(user, dataset, datapath):
    from kaggle.api.kaggle_api_extended import KaggleApi

    """Download dataset from Kaggle."""
    if not os.path.exists(datapath):
        os.makedirs(datapath)
        print(f"The directory {datapath} is created.")

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(f"{user}{dataset}", path=datapath, unzip=True)
    print(f"Dataset {dataset} downloaded to {datapath}")

def main():
    parent_directory = os.path.dirname(os.getcwd())
    json_path = os.path.join(parent_directory, 'kaggle.json')

    set_kaggle_credentials(json_path)

    USER = 'msambare/'
    DATASET = 'fer2013'
    DATAPATH = os.path.join(parent_directory, 'data')

    download_dataset(USER, DATASET, DATAPATH)

if __name__ == "__main__":
    main()
