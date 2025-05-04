import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
from model import VGG16

# this vgg16 model hasnt been tested. but the idea is to eventually train
# this model and compile it with my compiler. i will lower the dag, after some optimizations to llvm ir and/or c

# im guessing the llvm language is all need or something similar

# taken from pytorch docs
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


annotation = 'annotations.csv'
training_dir = 'driving_dataset/'

annotation_testing = 'annotations_testing.csv'
testing_dir = 'testing_data/'

driving_dataset = CustomDataset(annotation, img_dir)
train_dataloader = DataLoader(driving_dataset, batach_size=64, shuffle=True)

testing_dataset = CustomDataset(annotation_testing, training_dir)
testing_dataloader = DataLoader(testing_dataset, batach_size=64, shuffle=True)

# models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 100
num_epochs = 20
batch_size = 16
learning_rate = 0.005

model = VGG16(num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  


total_step = len(training_data_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        
        images = images.to(device)
        labels = labels.to(device)
        
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
    
        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))
