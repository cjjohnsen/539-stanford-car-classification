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
if(device == 'cuda:0'): print('CUDA available, training will be on GPU.')
else: print('Training on CPU')

pretrained = True

root = './data'
train_root = f'{root}/car_data/train'
test_root = f'{root}/car_data/test'

if not os.path.exists(train_root) or not os.path.exists(test_root):
    download_and_extract_zip(SC_ZIP_URL, root)

classes = get_classes_by_make_and_year(root)

batch_size = 32
workers = 4
train_loader, test_loader = get_data_loaders(root, batch_size=batch_size, num_workers=workers)

n_class = len(classes)

if pretrained: 
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
else: 
    model = models.efficientnet_b3(weights=None)

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
print(model)
model = model.to(device)

updated_paras = list(filter(lambda p: p.requires_grad, model.parameters()))
#print(updated_paras)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(updated_paras, lr=0.002, weight_decay=5e-3)
#optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=5e-4)
#optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
#optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)

num_epochs = 1000 # This can be adjusted
save_model_every = 10

def save_loss_plot(losses, test_losses, epoch, file_path='./data'):
    plt.figure()
    plt.plot(losses, color='blue', label='Train Loss')
    plt.plot(test_losses, color='red', label='Test Loss')
    plt.title('Training and Test Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(file_path, f'train_test_loss_{epoch+1}.png'), bbox_inches='tight')
    plt.close()

def save_accuracy_plot(accuracies, test_accuracies, epoch, file_path='./data'):
    plt.figure()
    plt.plot(accuracies, color='blue', label='Train Accuracy')
    plt.plot(test_accuracies, color='red', label='Test Accuracy')
    plt.title('Training and Test Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(file_path, f'train_test_accuracy_{epoch+1}.png'), bbox_inches='tight')
    plt.close()

print('Starting training')

save_path = f'{root}/models'
model_path = f'{save_path}/model.pth'

if not os.path.exists(save_path):
    os.system(f'mkdir {save_path}')


if os.path.exists(model_path):
    print("Loading saved model...")
    model.load_state_dict(torch.load(model_path))
else:
    print("Training new model...")
    losses = []
    test_losses = []
    accuracies = []
    test_accuracies = []
    # Loop over the dataset multiple times

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,40,80], gamma=0.5)
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0

        # Training phase
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1))
            acc = (outputs.argmax(-1) == labels).float().mean()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            running_acc += acc.item()

        # Print training loss
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)

        avg_acc = running_acc / len(train_loader)
        accuracies.append(avg_acc)

        # Validation phase
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.view(-1))
                test_loss += loss.item()

                acc = (outputs.argmax(-1) == labels).float().mean()
                test_acc += acc.detach().item()

        # Calculate and print test loss
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        avg_test_acc = test_acc / len(test_loader)
        test_accuracies.append(avg_test_acc)

        print(f'Epoch {epoch + 1}: train loss={avg_loss:.3f}, test loss={avg_test_loss:.3f}, train acc={avg_acc:.3f}, test acc={avg_test_acc:.3f}')

        # Save graph and model periodically
        if (epoch + 1) % save_model_every == 0:
            save_loss_plot(losses, test_losses, epoch, save_path)
            save_accuracy_plot(accuracies, test_accuracies, epoch, save_path)
            torch.save(model.state_dict(), os.path.join(save_path, f'model_{epoch+1}.pth'))
            print('Saved checkpoint model.')

        scheduler.step()

    print('Finished training new model.')

print('Evaluating...')
model.eval()

# Initialize lists to monitor test accuracy and loss
test_loss = 0.0
class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))

# No gradient is needed for evaluation
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images).to(device)
        labels = labels.reshape((-1))
        loss = criterion(outputs, labels)
        test_loss += loss.item()*images.size(0)
        _, predicted = torch.max(outputs, 1)
        
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