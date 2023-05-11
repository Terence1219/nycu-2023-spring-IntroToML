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
from tqdm import tqdm


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
    #plt.imshow(np.transpose(np.squeeze(npimg[0]),(1, 2, 0)))
    plt.show()

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize([128, 128]),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 256

    # 定義數據目錄
    data_dir = '.\\training'

    # 載入數據集
    dataset = ImageFolder(data_dir,transform=transform)
    
    train_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.2])

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    classes = ['butterfly','cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

    ''' 
    # get some random training images
    dataiter = iter(testloader)
    for images, labels in dataiter:
        imshow(images)
    # show images
    
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    imshow(torchvision.utils.make_grid(images))
    '''
    #model_name = 'simpleCNN'
    model_name = 'Resnet'
    if model_name == 'Resnet':
        net = ResNet().to(device)
    else:
        net = simpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001) 

    best_acc = 0
    save_path = f'./animals_{model_name}.pth'
    epochs = 50
    train_acc = []
    train_loss = []
    val_acc = []

    for epoch in range(epochs):  # loop over the dataset multiple times

        train_correct = 0
        train_total = 0
        train_loss_sum = 0
        net.train()
        for i, data in enumerate(tqdm(trainloader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss_sum += loss.item()
        acc = round(100 *train_correct / train_total,2)
        loss = np.mean(train_loss_sum)
        print('Epoch {number} training accuracy:{acc}%, loss:{loss}'.format(number=epoch+1, acc=acc, loss=loss))
        train_acc.append(acc)
        train_loss.append(loss)

        test_correct = 0
        test_total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        net.eval()
        with torch.no_grad():
            for data in tqdm(testloader):
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
            acc = round(100 *test_correct / test_total,2)
            print('Validation accuracy:{acc}%'.format(acc=acc))
            val_acc.append(acc)
            if acc > best_acc:
                tqdm.write('Model saved.')
                torch.save(net.state_dict(), save_path)
                best_acc = acc
    
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    print('Ground Truth: ',' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    imshow(torchvision.utils.make_grid(images[0:4]))

    outputs = net(images.to(device))
    _, predicted = torch.max(outputs.data, 1)
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'for j in range(4)))
    
    x = np.arange(1,epochs+1)
    plt.figure()
    plt.plot(x,train_acc,label='train')
    plt.plot(x,val_acc,label='val')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    ax=plt.gca()
    ax.set_title('accuracy')
    plt.savefig(f'animals_{model_name}_acc.png')

    x = np.arange(1,epochs+1)
    plt.figure()
    plt.plot(x,train_loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    ax=plt.gca()
    ax.set_title('Training loss')
    plt.savefig(f'animals_{model_name}_loss.png')
    
    