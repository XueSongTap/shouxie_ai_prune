import torch

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
# https://www.bilibili.com/video/BV1TM4y1b7Gq/?spm_id_from=333.788&vd_source=7371452b85fe4d187885825b04f8393a
class BigModel(nn.Module):
    def __init__(self):
        super(BigModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size = 3, padding = 1) 
        self.fc = nn.Linear(16 * 28 * 28, 10)
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)  # 调整输入维度
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

def train(model, dataloader, criterion, optimizer, device='gpu', num_epochs=10 ):
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

if __name__ == "__main__":
    transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset= datasets.MNIST('./data', train=True, download = True, transform = transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)

    big_model = BigModel()
    criterion = nn.CrossEntropyLoss()
    optimzier = optim.Adam(big_model.parameters(), lr=1e-3)
    big_model = train(big_model, train_loader, criterion, optimzier, device='cuda', num_epochs=3)

    torch.save(big_model.state_dict(), 'big_model.pth')

    dummy_input = torch.randn(1, 1, 28, 28).to('cuda')
    
    torch.onnx.export(big_model, dummy_input, "big_model.onnx")
