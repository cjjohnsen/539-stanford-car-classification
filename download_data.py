import requests
import zipfile
import os

SC_ZIP_URL = 'https://storage.googleapis.com/kaggle-data-sets/30084/38348/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20231116%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20231116T193711Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=9f1f2650d538d0c32643d9686309a0c25c2932cc40f27b19f0330f92325fe110756743e788b05d312e9b3d749b30bfe555a339b3def2fd2b5d82f60bd046d8190f8e0141b7dd5a44cc310e0c5d649f74486763da9a56dac7f96c5dcc73046d87e900f601527f7f3ea13e1db1addc65744d7a2caadf16f6a1927d50464421abdaf5ab2a23503e0cdcec2eb33e172a7f159de335ba1f43ee382dcd4d345697c1a5b32fc0e19498cd4a7fd3bc6a544540a4f932ddf887d2532c673ca04a1b7e35e03c6f3c172e91ef506c64bbae538576a691410a0a436ea9cf8dbc09af6d306db7ee87bfdff1c7fcee5ffb79ad9f6dc16b0957809817d2ddcae7a4ec395ed30c71'

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

        # this section is written for Windows, might need to edit if you are on Mac/Linux
        os.system('move data data_temp')
        os.system('mkdir data')
        os.system('move data_temp\\cars_test\\cars_test data\\.')
        os.system('move data_temp\\cars_train\\cars_train data\\.')
        os.system('move data_temp\\cars_annos.mat data\\.')
        os.system('rmdir /s /q data_temp')
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
