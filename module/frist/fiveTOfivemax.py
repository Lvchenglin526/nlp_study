import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 定义一个简单的全连接神经网络
class MaxNumberNet(nn.Module):
    def __init__(self):
        super(MaxNumberNet, self).__init__()
        self.fc1 = nn.Linear(5, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 生成随机的数据集
def generate_data(num_samples=1000):
    inputs = torch.rand((num_samples, 5))
    targets = torch.argmax(inputs, dim=1)
    one_hot_targets = torch.zeros((num_samples, 5)).scatter_(1, targets.unsqueeze(1), 1)
    return inputs, one_hot_targets

# 训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

# 测试函数
def test_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            _, actual = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == actual).sum().item()
    print(f'Accuracy of the network on the test set: {100 * correct / total}%')

# 主程序
if __name__ == "__main__":
    # 生成训练和测试数据
    train_inputs, train_labels = generate_data(1000)
    test_inputs, test_labels = generate_data(200)

    # 创建DataLoader
    train_dataset = TensorDataset(train_inputs, train_labels)
    test_dataset = TensorDataset(test_inputs, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = MaxNumberNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer)

    # 测试模型
    test_model(model, test_loader)

    # 示例预测
    example_input = torch.tensor([[0.1, 0.4, 0.3, 0.2, 0.5]])
    with torch.no_grad():
        output = model(example_input)
        _, predicted_class = torch.max(output.data, 1)
    print(f"The maximum number is at index: {predicted_class.item()}")