# Federated Learning on Fashion MNIST using CNN
Implement federated learning on Fashion MNIST dataset.

# Step-by-step
## Step 1: Import and model specification
- Initialize CNN hyperparameters such as <code>learning rate, batch size, numpy seed.</code>
- Create 2 workers Alice and Bob

## Step 2: Load data
- Load Fashion MNIST dataset using <code>torchvision.datasets</code>.
![FashionMNIST](https://github.com/yasirabd/SPAI/blob/master/Project%202/assets/fashion-mnist.png "FashionMNIST")
- Create <code>federated_train_loader</code> using <code>FederatedDataLoader</code>. Don't forget to transform dataset!
- Create <code>test_loader</code> using <code>torch.utils.DataLoader</code>. Don't forget to transform dataset!
- Visualize one data

![Shoes](https://github.com/yasirabd/SPAI/blob/master/Project%202/assets/shoes.png "Shoes")

## Step 3: Convolutional Neural Network
- Create architecture Class <code>CNN</code>.  
```python
class CNN(nn.Module):
    def __init__(self,num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

## Step 4: Training
- Train Neural network and validate with test set after completion of training every epoch

## My Notes
- <code>BatchNorm</code> is still not compatible on Federated Learning. So, it is still imposible using pre-trained model such as resnet, VGG, etc.
- For some reason, the training suddenly stopped at 48%. I don't know if this is a bug or my code has an error. But, so far we already implementing Federated Learning on Fashion MNIST using CNN.
