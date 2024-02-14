import torch

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np

class BigModel(nn.Module):
    def __init__(self):
        super(BigModel, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256,10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset= datasets.MNIST('./data', train=True, download = True, transform = transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)

def train(model, dataloader, criterion, optimizer, device='cpu', num_epochs=10 ):
    print("train")
    model.train()
    model.to(device)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs.view(inputs.size(0), -1))
            loss = criterion(outputs, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")
    return model

big_model = BigModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(big_model.parameters(), lr=1e-3)
big_model = train(big_model, train_loader, criterion, optimizer, device='cuda', num_epochs=2)


torch.save(big_model.state_dict(), 'big_model.pth')

# prune to a smaller model
def prune_network(model, pruning_rate=0.5, method='global'):
    print("start prune")
    for name, param in model.named_parameters():
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                if method == 'global':
                    threshold = np.percentile(abs(tensor), pruning_rate * 100)
                else:
                    threshold = np.percentile(abs(tensor), pruning_rate * 100, axis=1, keepdims=True)
                mask = abs(tensor) > threshold
                param.data = torch.FloatTensor(tensor* mask.astype(float)).to(param.device)
    
big_model.load_state_dict(torch.load('big_model.pth'))
prune_network(big_model, pruning_rate=0.5, method='global')

torch.save(big_model.state_dict(), 'big_model_pruned.pth')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(big_model.parameters(), lr=1e-4)

# 较低学习率微调
fintuned_model = train(big_model, train_loader, criterion, optimizer, device='cuda', num_epochs=10)


torch.save(fintuned_model.state_dict(), 'big_model_pruned_finetuned.pth')
            

"""
root@f66a1cb54d6c:/code/2/shouxie_ai_prune/3.1# python goPruning.py 
train
Epoch 1, Loss: 0.2102482723040399
Epoch 2, Loss: 0.08687407039711549
start prune
train
Epoch 1, Loss: 0.03393178170451076
Epoch 2, Loss: 0.0226753912089186
Epoch 3, Loss: 0.016753726374405847
Epoch 4, Loss: 0.012751432718119092
Epoch 5, Loss: 0.009728807850362562
Epoch 6, Loss: 0.007285257001767954
Epoch 7, Loss: 0.005482732687491863
Epoch 8, Loss: 0.004318303400444442
Epoch 9, Loss: 0.003367406661038567
Epoch 10, Loss: 0.002207980164883974 
"""