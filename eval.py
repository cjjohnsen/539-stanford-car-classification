from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch
from torchvision import models
import os
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from download_data import SC_ZIP_URL, download_and_extract_zip, get_classes_by_make_and_year
from load import get_data_loaders

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

root = './data'
test_root = f'{root}/car_data/test'
pretrained = True

model_path = f'{root}/models/model.pth'
if not os.path.exists(model_path):
    print(f"Did not find model at path {model_path}")
    exit(1)

if not os.path.exists(test_root):
    download_and_extract_zip(SC_ZIP_URL, root)

classes = get_classes_by_make_and_year(root)
n_class = len(classes)

batch_size = 64
workers = 4
_, test_loader = get_data_loaders(root, batch_size=batch_size, num_workers=workers)

# if pretrained: 
#     model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
# else: 
#     model = models.alexnet(weights=None)

# model.classifier[6] = nn.Linear(4096, n_class)

# if pretrained:
#     for p in model.parameters():
#         p.requires_grad = False
#     for p in model.classifier.parameters():
#         p.requires_grad = True
if pretrained: 
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
else: 
    model = models.resnet18(weights=None)

# model.classifier[6] = nn.Linear(4096, n_class)
model.fc = nn.Linear(512, n_class, bias=False)

if pretrained:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
    # for p in model.classifier.parameters():
        p.requires_grad = True
    for p in model.layer4.parameters():
       p.requires_grad =True

model = model.to(device)
criterion = nn.CrossEntropyLoss()

print("Loading saved model...")
model.load_state_dict(torch.load(model_path))

print('Evaluating...')
model.eval()

test_loss = 0.0
class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))
preds = []
true = []

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images).to(device)
        labels = labels.reshape((-1))
        for i in labels: true.append(int(i))
        loss = criterion(outputs, labels)
        test_loss += loss.item()*images.size(0)
        _, predicted = torch.max(outputs, 1)
        for i in predicted: preds.append(int(i))
        
        # Compare predictions to true label
        correct = (predicted == labels).squeeze()
        for i in range(len(labels)):    # This was iterating over batch_size before, not sure why?
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

cm = confusion_matrix(true, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig('./confusion_matrix.png')
