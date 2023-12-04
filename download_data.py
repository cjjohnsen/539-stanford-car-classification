import requests
import zipfile
import os

SC_ZIP_URL = 'https://storage.googleapis.com/kaggle-data-sets/30084/38348/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20231204%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20231204T164851Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=3922ed0cfed4e42b6f2c3b16e2081b4e76fab71c1ad4bcc6a30056508fb4ff60cb5fbd6e65e3328a8cfb9d12ab8a60675816bd96a2c23ec5e1801681952ffde33157d4c937034d00fe80332fdaf7bb969c41d6653330aeb6fa209854ae9d42268ae632126e1e77d37acab0e9ae3b879517048d3977f552ff270c8bfc8936fac3d87aa4be6ef81901e5f859bef16109c25ee3e92699e3450c4caadeaff3b82ef4d94aa207bee64313a0825b08165565968a6405d40121ec54f47baf4f21ef8b428ed053691500dfc434dff5bdd29173ce404e0b1758def9b7d0c42c1b2d74fcfded6b9da750dfeb4d5a045d53713453e2237da6582ae8731ba0d7986f4e7f35d8'

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
