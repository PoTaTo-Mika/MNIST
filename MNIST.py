from logging import shutdown
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import subprocess
import sys
import os
from PIL import Image

# 定义模型保存路径
model_path = 'D:\PythonProject\MNIST\data\MNIST\model'
model_filename = 'mnist_model.pth'
model_save_path = os.path.join(model_path, model_filename)

# 定义图片所在文件夹路径
image_folder_path = 'D:\PythonProject\MNIST\data\MNIST\infer'

# 执行下载数据集操作
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 执行预处理操作
batch_size = 64 # 规定批大小
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 搭建神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义第一个卷积层
        # 1个输入通道（图像为灰度图），32个输出通道，卷积核大小为3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # 定义第二个卷积层
        # 32个输入通道，64个输出通道，卷积核大小为3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 定义一个全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 最后一个全连接层输出10个类别（MNIST数据集的10个数字）
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 通过第一个卷积层后，应用ReLU激活函数，然后进行最大池化
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 通过第二个卷积层，再次应用ReLU激活函数和最大池化
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # 将特征图展平
        x = x.view(-1, 64 * 7 * 7)
        # 通过第一个全连接层，应用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 通过第二个全连接层得到最终的输出
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 实例化网络
model = Net()

# 打印我们的模型
print(model)

# 损失函数定义
criterion = nn.NLLLoss()

# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# 设置训练的周期数
epochs = 10

a = input("train or infer?\n")
if a=='train':
# 开始训练
 for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model.forward(images)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        
        # 更新权重
        optimizer.step()
        
        # 打印统计信息
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader)}")
    
# 测试网络性能
 correct = 0
 total = 0
 with torch.no_grad():  # 测试阶段不需要计算梯度
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

 print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# 检查文件夹是否存在，如果不存在，则创建
 if not os.path.exists(model_path):
    os.makedirs(model_path)

# 保存模型状态字典
 torch.save(model.state_dict(), model_save_path)
 print(f"Model saved to {model_save_path}")

if a=='infer':
 print("infer mode is on\n")
# 创建模型实例
 model = Net()

# 加载模型状态字典
 model.load_state_dict(torch.load('D:\PythonProject\MNIST\data\MNIST\model/mnist_model.pth'))

# 将模型设置为评估模式
 model.eval()

 def preprocess_image(image_path):
    #转换格式
    transform = transforms.Compose([
        transforms.Grayscale(),  # 转换为灰度图
        transforms.Resize((28, 28)),  # 调整大小以匹配训练时的输入尺寸
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])
    # 加载图片
    image = Image.open(image_path).convert('L')  # 使用.convert('L')确保图像是灰度的
    # 应用预处理
    image_tensor = transform(image)
    return image_tensor
# 遍历文件夹中的所有图片
 for image_name in os.listdir(image_folder_path):
    image_path = os.path.join(image_folder_path, image_name)
    if image_path.endswith('.png'):  # 确保处理的是PNG图片
        image_tensor = preprocess_image(image_path).unsqueeze(0)  # 添加batch维度
        with torch.no_grad():  # 推理阶段不需要计算梯度
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            print(f'Image: {image_name}, Predicted Digit: {predicted.item()}')

