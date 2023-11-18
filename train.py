import torch
import scipy.io
from torchvision import models
from torch.utils.data import DataLoader
import os
from process_data import load_stanford_cars_dataset
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
        train_labels[i] = torch.tensor([mappings[int(label) - 1]])

    for i, label in enumerate(test_labels):
        test_labels[i] = torch.tensor([mappings[int(label) - 1]])

create_mappings()

model = models.alexnet(weights=None)        # Could change in the future based on training speed/accuracy
model.classifier[6] = nn.Linear(4096, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataset = TensorDataset(train_images, train_labels)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

num_epochs = 200 # This can be adjusted

print('Starting training')

model_path = './data/model.pth'
if os.path.exists(model_path):
    print("Loading saved model...")
    model.load_state_dict(torch.load(model_path))
else:
    print("Training new model...")
    losses = []
    # Loop over the dataset multiple times
    for epoch in range(num_epochs):
        running_loss = 0.0

        # Iterate over the data
        for i, data in enumerate(train_loader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.reshape((-1)))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
        # Print loss
        print(f'Epoch {epoch + 1}: loss={running_loss:.3f}')
        losses.append(running_loss)

    # save graph and model every 25 epochs
    if (epoch+1) % 25 == 0:
        plt.figure()
        plt.plot(losses, marker='o', linestyle='-', color='blue')
        plt.title('Training Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.xticks(range(len(losses)), range(1, len(losses) + 1))  # Set epoch numbers starting from 1
        plt.savefig(f'./data/train_loss_{epoch}.png', bbox_inches='tight')

        torch.save(model.state_dict(), f'./data/model_{epoch}.pth')
        print('Finished training new model.')

print('Evaluating...')
model.eval()

# Load test data
test_dataset = TensorDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

# Initialize lists to monitor test accuracy and loss
test_loss = 0.0
class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))

# No gradient is needed for evaluation
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # Forward pass
        outputs = model(images)
        labels = labels.reshape((-1))
        loss = criterion(outputs, labels)
        test_loss += loss.item()*images.size(0)
        _, predicted = torch.max(outputs, 1)
        
        # Compare predictions to true label
        correct = (predicted == labels).squeeze()
        for i in range(batch_size):
            label = labels[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

# Calculate and print avg test loss
test_loss = test_loss / len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(len(classes)):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
