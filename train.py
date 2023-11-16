import torch
import scipy.io
from torchvision import models
from torch.utils.data import DataLoader
import os
from process_data import load_stanford_cars_dataset

if not os.path.exists('./data/tensors/train_images.pt') or not os.path.exists('./data/tensors/train_labels.pt') or not os.path.exists('./data/tensors/test_images.pt') or not os.path.exists('./data/tensors/test_labels.pt'):
    print('One or more data tensors not found, tensors will be loaded.')
    load_stanford_cars_dataset('./data')

# load dataset from stored tensors
# make sure to run download_data.py and process_data.py (in that order)
# before running this script
train_images = torch.load('./data/tensors/train_images.pt')
train_labels = torch.load('./data/tensors/train_labels.pt')
test_images = torch.load('./data/tensors/test_images.pt')
test_labels = torch.load('./data/tensors/test_labels.pt')

# Map old classes (make, model, and year) to new classes (make and year)
classes = []
mappings = {}

def create_mappings():
    annotations_path = "./data/cars_annos.mat"
    annotations = scipy.io.loadmat(annotations_path)

    for i, class_name in enumerate(annotations["class_names"][0]):
        name = class_name[0]
        new_class_name = name.split(' ')[0] + ' ' + name.split(' ')[-1]
        if new_class_name not in classes:
            classes.append(new_class_name)
            mappings[i] = len(classes)-1
        else:
            mappings[i] = classes.index(new_class_name)

    for i, label in enumerate(train_labels):
        train_labels[i] = torch.tensor([mappings[int(label) - 1] + 1])

    for i, label in enumerate(test_labels):
        test_labels[i] = torch.tensor([mappings[int(label) - 1] + 1])


def get_label(class_no):
    return classes[class_no-1]

create_mappings()

model = models.alexnet(pretrained=False)        # Could change in the future based on training speed/accuracy
