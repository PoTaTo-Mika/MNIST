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

# ����ģ�ͱ���·��
model_path = 'D:\PythonProject\MNIST\data\MNIST\model'
model_filename = 'mnist_model.pth'
model_save_path = os.path.join(model_path, model_filename)

# ����ͼƬ�����ļ���·��
image_folder_path = 'D:\PythonProject\MNIST\data\MNIST\infer'

# ִ���������ݼ�����
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# ִ��Ԥ�������
batch_size = 64 # �涨����С
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# �������
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # �����һ�������
        # 1������ͨ����ͼ��Ϊ�Ҷ�ͼ����32�����ͨ��������˴�СΪ3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # ����ڶ��������
        # 32������ͨ����64�����ͨ��������˴�СΪ3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # ����һ��ȫ���Ӳ�
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # ���һ��ȫ���Ӳ����10�����MNIST���ݼ���10�����֣�
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # ͨ����һ��������Ӧ��ReLU�������Ȼ��������ػ�
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # ͨ���ڶ�������㣬�ٴ�Ӧ��ReLU����������ػ�
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # ������ͼչƽ
        x = x.view(-1, 64 * 7 * 7)
        # ͨ����һ��ȫ���Ӳ㣬Ӧ��ReLU�����
        x = F.relu(self.fc1(x))
        # ͨ���ڶ���ȫ���Ӳ�õ����յ����
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ʵ��������
model = Net()

# ��ӡ���ǵ�ģ��
print(model)

# ��ʧ��������
criterion = nn.NLLLoss()

# �����Ż���
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# ����ѵ����������
epochs = 10

a = input("train or infer?\n")
if a=='train':
# ��ʼѵ��
 for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        # �����ݶ�
        optimizer.zero_grad()
        
        # ǰ�򴫲�
        outputs = model.forward(images)
        
        # ������ʧ
        loss = criterion(outputs, labels)
        
        # ���򴫲�
        loss.backward()
        
        # ����Ȩ��
        optimizer.step()
        
        # ��ӡͳ����Ϣ
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader)}")
    
# ������������
 correct = 0
 total = 0
 with torch.no_grad():  # ���Խ׶β���Ҫ�����ݶ�
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

 print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# ����ļ����Ƿ���ڣ���������ڣ��򴴽�
 if not os.path.exists(model_path):
    os.makedirs(model_path)

# ����ģ��״̬�ֵ�
 torch.save(model.state_dict(), model_save_path)
 print(f"Model saved to {model_save_path}")

if a=='infer':
 print("infer mode is on\n")
# ����ģ��ʵ��
 model = Net()

# ����ģ��״̬�ֵ�
 model.load_state_dict(torch.load('D:\PythonProject\MNIST\data\MNIST\model/mnist_model.pth'))

# ��ģ������Ϊ����ģʽ
 model.eval()

 def preprocess_image(image_path):
    #ת����ʽ
    transform = transforms.Compose([
        transforms.Grayscale(),  # ת��Ϊ�Ҷ�ͼ
        transforms.Resize((28, 28)),  # ������С��ƥ��ѵ��ʱ������ߴ�
        transforms.ToTensor(),  # ת��ΪTensor
        transforms.Normalize((0.5,), (0.5,))  # ��һ��
    ])
    # ����ͼƬ
    image = Image.open(image_path).convert('L')  # ʹ��.convert('L')ȷ��ͼ���ǻҶȵ�
    # Ӧ��Ԥ����
    image_tensor = transform(image)
    return image_tensor
# �����ļ����е�����ͼƬ
 for image_name in os.listdir(image_folder_path):
    image_path = os.path.join(image_folder_path, image_name)
    if image_path.endswith('.png'):  # ȷ���������PNGͼƬ
        image_tensor = preprocess_image(image_path).unsqueeze(0)  # ���batchά��
        with torch.no_grad():  # ����׶β���Ҫ�����ݶ�
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            print(f'Image: {image_name}, Predicted Digit: {predicted.item()}')

