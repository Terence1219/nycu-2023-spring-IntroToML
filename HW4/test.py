import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        #self.resnet18 = models.resnet18(weights='IMAGENET1K_V1')
        self.resnet18 = models.resnet18()
        self.fc = nn.Linear(self.resnet18.fc.in_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 10)
        self.resnet18.fc = self.fc
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.resnet18(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

class simpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #124*124*6
        self.conv2 = nn.Conv2d(6, 12, 5) #58*58*12
        self.conv3 = nn.Conv2d(12, 16, 4) #26*26*16
        self.conv4 = nn.Conv2d(16, 20, 4) #10*10*20
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(20 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #62*62*5
        x = self.pool(F.relu(self.conv2(x))) #29*29*5
        x = self.pool(F.relu(self.conv3(x))) #13*13*5
        x = self.pool(F.relu(self.conv4(x))) #5*5*20
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize([128, 128]),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 定義數據目錄
    test_dir = '.\\training'

    batch_size = 512
    # 載入數據集
    test_data = ImageFolder(test_dir,transform=transform)   
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=True)

    classes = ['butterfly','cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

    '''
    # get some random training images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # show images
    
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    imshow(torchvision.utils.make_grid(images))
    '''
    
    model_name = 'Resnet'
    if model_name == 'Resnet':
        net = ResNet().to(device)
    else:
        net = simpleCNN().to(device)

    load_path = '.\\resnet18_pretrained_dropout\\animals_ResNet.pth'
    net.load_state_dict(torch.load(load_path))
    
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    y_pred = []   
    y_true = []  

    net.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            y_pred.extend(predictions.view(-1).detach().cpu().numpy())             
            y_true.extend(labels.view(-1).detach().cpu().numpy())     
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    acclist = []
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        acclist.append(accuracy)
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    
    plt.bar(list(correct_pred.keys()),acclist)
    plt.xlabel('classes')
    plt.ylabel('accuracy(%)')
    ax=plt.gca()
    ax.set_title('accuracy for each class')
    plt.savefig(f'{model_name}_acc_each.png')

    #confusion_matrix
    cf_matrix = confusion_matrix(y_true, y_pred)                                
    df_cm = pd.DataFrame(cf_matrix, classes, classes)     
    plt.figure(figsize = (9,6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.savefig("confusion_matrix.png")

    