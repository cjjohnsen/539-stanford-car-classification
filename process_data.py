import torch
import scipy.io
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
from download_data import download_and_extract_zip, SC_ZIP_URL

def load_stanford_cars_dataset(data_folder):
    """
    Load the Stanford Cars Dataset from the given folder.
    """
    print('Loading data...')

    if not os.path.exists('./data/cars_test') or not os.path.exists('./data/cars_train') or not os.path.exists('./data/cars_annos.mat'):
        download_and_extract_zip(SC_ZIP_URL, './data')

    # Paths to the dataset files
    train_folder = os.path.join(data_folder, "cars_train")
    test_folder = os.path.join(data_folder, "cars_test")
    annotations_path = os.path.join(data_folder, "cars_annos.mat")

    # Load annotations
    annotations = scipy.io.loadmat(annotations_path)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(227, 227), scale=(0.5, 1)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Preparing training and testing data
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    i = 0
    training_c = 0
    testing_c = 0
    for annotation in annotations['annotations'][0]:
        if i % 200 == 0: print(f'{i/16000*100:.2f}%', end="\r")

        # get data from entry
        image_path = annotation[0][0]
        label = annotation[5][0][0]
        label = torch.tensor([label])
        testing = int(annotation[-1])

        # get expected img name
        if testing: testing_c += 1
        else: training_c += 1
        img_name = f'{training_c:05d}.jpg' if not testing else f'{testing_c:05d}.jpg'

        # append data to correct dataset
        if not testing:
            corrected_path = os.path.join(train_folder, img_name)
            image = Image.open(corrected_path).convert('RGB')
            tensor_image = train_transform(image)
            train_images.append(tensor_image)
            train_labels.append(label)
        else:
            corrected_path = os.path.join(test_folder, img_name)
            image = Image.open(corrected_path).convert('RGB')
            tensor_image = test_transform(image)
            test_images.append(tensor_image)
            test_labels.append(label)
        i += 1

    # Converting lists to tensors
    print('\nConverting data to tensors')
    print('Converting 1/4...', end='\r')
    train_images = torch.stack(train_images)
    print('Converting 2/4...', end='\r')
    train_labels = torch.stack(train_labels)
    print('Converting 3/4...', end='\r')
    test_images = torch.stack(test_images)
    print('Converting 4/4...', end='\r')
    test_labels = torch.stack(test_labels)

    print('\nDone loading data. Saving.')

    if not os.path.exists('./data/tensors'):
        os.makedirs('./data/tensors')

    torch.save(train_images, './data/tensors/train_images.pt')
    torch.save(train_labels, './data/tensors/train_labels.pt')
    torch.save(test_images, './data/tensors/test_images.pt')
    torch.save(test_labels, './data/tensors/test_labels.pt')
    
    print('Tensors have been created and saved.')
    return train_images, train_labels, test_images, test_labels


