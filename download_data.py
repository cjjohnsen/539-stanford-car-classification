import requests
import zipfile
import os
import pandas as pd

# Link to dataset: https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder/code
# I used the network tab in dev tools to get the URL of the zip file, this might need to be updated when you run it as it will expire
SC_ZIP_URL = 'https://storage.googleapis.com/kaggle-data-sets/31559/46697/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com@kaggle-161607.iam.gserviceaccount.com/20231206/auto/storage/goog4_request&X-Goog-Date=20231206T004837Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=333cca27e5001450a7efef3778a7c75d7ca86ad45bb1a9011e5105f36b81242f1195ae8eba602f3013f5079f4192879dc2bd23b445a4dc83f6db359fb2defd01ab158ad083b7b35e1566029606a270749b0e9bf4d3a00c46570be9934038f6242abd9da3a0c82b1f73928d1b8b3d9d4aba54fd1daa3a7c2da20b62803c64390c20f05086ee42de52c1336e09e82441ebe86f286dafe3a21df1ec17c5a7e81360a5f4957a35ea763e1b539d5089b57ef8c2c7b9ccdbe9c1e04018bf28820e8a644eb2b683eafe48f183c94edd2fe8ab7ecccd880708f3cdf2ad01de7a86a5ebb9dcd001fabdb7c45f7c7ff899c048484f67bee18d18a7c13158f689ecf75c3f63'

def download_and_extract_zip(url, target_folder):
    """
    Download a ZIP file from the given URL and extract it to the specified target folder.
    """
    # Check if the target folder exists, if not, create it
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Download the file
    print('Downloading zip file...')
    response = requests.get(url)
    if response.status_code == 200:
        zip_path = os.path.join(target_folder, "downloaded.zip")
        with open(zip_path, "wb") as file:
            file.write(response.content)

        # Extract the ZIP file
        print('Extracting.')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_folder)

        os.remove(zip_path)

        classes = get_classes_by_make_and_year(target_folder)

        # The following was written for a Linux enviornment
        os.system(f'mkdir {target_folder}/car_data_new')
        os.system(f'mkdir {target_folder}/car_data_new/train && mkdir {target_folder}/car_data_new/test')

        for c in classes:
            os.system(f'mkdir \'{target_folder}/car_data_new/train/{c}\' && mkdir \'{target_folder}/car_data_new/test/{c}\'')
        
        for c in os.listdir(f'{target_folder}/car_data/car_data/train'):
            newc = c.split(' ')[0] + ' ' + c.split(' ')[-1]
            for f in os.listdir(f'{target_folder}/car_data/car_data/train/{c}'):
                os.system(f'mv \'{target_folder}/car_data/car_data/train/{c}/{f}\' \'{target_folder}/car_data_new/train/{newc}/{f}\'')

        for c in os.listdir(f'{target_folder}/car_data/car_data/test'):
            newc = c.split(' ')[0] + ' ' + c.split(' ')[-1]
            for f in os.listdir(f'{target_folder}/car_data/car_data/test/{c}'):
                os.system(f'mv \'{target_folder}/car_data/car_data/test/{c}/{f}\' \'{target_folder}/car_data_new/test/{newc}/{f}\'')

        os.system(f'rm -rf {target_folder}/car_data')
        os.system(f'mv {target_folder}/car_data_new {target_folder}/car_data')
        print('Done downloading and processing data.')
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")


def get_classes_by_make_and_year(target_folder):
    if not os.path.exists(target_folder + '/names.csv'):
        print(f'ERROR: Could not find class csv ({target_folder}/names.csv)')
        exit(1)
    new_classes = []
    classes = pd.read_csv(target_folder + '/names.csv', header=None).iloc[:, 0].values
    for name in classes:
        new_name = name.split(' ')[0] + ' ' + name.split(' ')[-1]
        if new_name not in new_classes:
            new_classes.append(new_name)
    return new_classes