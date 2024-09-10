import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

#进行数据转换操作,
#先进行Totensor，转换为张量形式
#再进行Normalize归一化，Normalize((mean,),(std)),output=(input-mean)/std
# 灰度图像仅有一个通道，所以这里只用给定一个mean和std
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))])

#加载训练数据集和测试数据集
#root:数据集的根目录； 数据集若已存在则直接加载，若不存在且download=True,则下载数据集并存到root路径
#train:True为加载训练集，否则为测试集
#transform:通常是定义的torchvision.transforms中的转换操作或定义的函数
#
train_dataset = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_dataset = datasets.MNIST(root="./data",train=False,download=True,transform=transform)

# 使用 DataLoader 加载数据并定义批量大小
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 定义第一个卷积层：输入1个通道，输出10个通道，卷积核大小为5x5
        #卷积核的个数决定了输出通道数，这里输出10个通道说明有10个卷积核
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # 定义第二个卷积层：输入10个通道，输出20个通道，卷积核大小为5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 定义一个Dropout层，用于防止过拟合
        self.dropout = nn.Dropout2d()
        # 定义全连接层，输入特征数为320，输出特征数为50
        self.fc1 = nn.Linear(320, 50)
        # 定义第二个全连接层，输出10类
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # 第一层卷积 -> ReLU -> 最大池化
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 第二层卷积 -> ReLU -> Dropout -> 最大池化
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        # 展平操作，将数据展平成一维
        x = x.view(-1, 320)
        # 第一层全连接 -> ReLU
        x = F.relu(self.fc1(x))
        # 第二层全连接，输出10个类别的得分
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# 定义训练函数
def train(model, device, train_loader, optimizer, epoch):
    #设置为训练模式

    model.train()
    # batch_idx表示当前批次的索引，data为输入数据，target为标签（目标）
    for batch_idx, (data, target) in enumerate(train_loader):
        #放到gpu上
        data, target = data.to(device), target.to(device)
        #清空梯度
        optimizer.zero_grad()
        #前向传播，输入数据传入模型，计算模型输出
        output = model(data)
        #计算损失值
        loss = F.nll_loss(output, target)
        #反向传播，计算损失
        loss.backward()
        #更新模型的参数
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 定义测试函数
def test(model, device, test_loader):
    #进入评估模式
    model.eval()
    test_loss = 0
    correct = 0
    # 禁用梯度计算
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # reduction = 'sum'表示损失不会按批次平均，而是累加每个样本的损失值，
            # 这样最终计算的损失就是整个测试机的损失
            test_loss += F.nll_loss(output, target, reduction='sum').item() #累加损失
            pred = output.argmax(dim=1, keepdim=True)   #获取最大值的索引，即模型预测的类别
            correct += pred.eq(target.view_as(pred)).sum().item()   #统计预测正确的数量

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({accuracy:.0f}%)\n')


# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型和优化器
model = CNNModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 开始训练
for epoch in range(1, 11):  # 训练10个epoch
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)


# 保存训练好的模型
torch.save(model.state_dict(), "mnist_cnn_model.pth")


# 加载一些测试数据进行可视化
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

# 取出部分预测结果
with torch.no_grad():
    output = model(example_data.to(device))

# 可视化结果
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title(f"Prediction: {output.argmax(dim=1, keepdim=True)[i].item()}")
    plt.xticks([])
    plt.yticks([])
plt.show()
