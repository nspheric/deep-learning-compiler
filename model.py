import torch
import torch.nn as nn

# taken from here; https://www.digitalocean.com/community/tutorials/vgg-from-scratch-pytorch
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.batch1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batch3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.batch4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.batch5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()
        
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batch6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU()
        
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batch7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU()
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.batch8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU()
        
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batch9 = nn.BatchNorm2d(512)
        self.relu9 = nn.ReLU()
        
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batch10 = nn.BatchNorm2d(512)
        self.relu10 = nn.ReLU()
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batch11 = nn.BatchNorm2d(512)
        self.relu11 = nn.ReLU()

        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batch12 = nn.BatchNorm2d(512)
        self.relu12 = nn.ReLU()

        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batch13 = nn.BatchNorm2d(512)
        self.relu13 = nn.ReLU()
        self.max5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(7 * 7 * 512, 4096)
        self.relu_fc1 = nn.ReLU()
        
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu_fc2 = nn.ReLU()
        
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.relu1(self.batch1(self.conv1(x)))
        x = self.relu2(self.batch2(self.conv2(x)))
        x = self.max1(x)
        
        x = self.relu3(self.batch3(self.conv3(x)))
        x = self.relu4(self.batch4(self.conv4(x)))
        x = self.max2(x)
        
        x = self.relu5(self.batch5(self.conv5(x)))
        x = self.relu6(self.batch6(self.conv6(x)))
        x = self.relu7(self.batch7(self.conv7(x)))
        x = self.max3(x)
        
        x = self.relu8(self.batch8(self.conv8(x)))
        x = self.relu9(self.batch9(self.conv9(x)))
        x = self.relu10(self.batch10(self.conv10(x)))
        x = self.max4(x)

        x = self.relu11(self.batch11(self.conv11(x)))
        x = self.relu12(self.batch12(self.conv12(x)))
        x = self.relu13(self.batch13(self.conv13(x)))
        x = self.max5(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(self.fc1(self.relu_fc1(x)))
        x = self.dropout2(self.fc2(self.relu_fc2(x)))
        x = self.fc3(x)

        return x
        
