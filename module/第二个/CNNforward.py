#coding:utf8

import torch
import torch.nn as nn
import numpy as np


"""
手动实现简单的神经网络
使用pytorch实现CNN
手动实现CNN
对比
"""
#一个二维卷积（使用封装好的函数）
class TorchCNN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel):
        super(TorchCNN, self).__init__()
        self.layer = nn.Conv2d(in_channel, out_channel, kernel, bias=False)

    def forward(self, x):
        return self.layer(x)

'''
in_channel    输入通道数，灰度图像通常为1，RGB通常为3
out_channel   输出通道数，表示卷积层输出的特征图个数
kernel        卷积核大小，维度为kernel*kernel
'''

#自定义CNN模型（diy）
class DiyModel:
    def __init__(self, input_height, input_width, weights, kernel_size):
        self.height = input_height
        self.width = input_width
        self.weights = weights
        self.kernel_size = kernel_size

    def forward(self, x):
        output = []
        for kernel_weight in self.weights:
            kernel_weight = kernel_weight.squeeze().numpy() #shape : 2x2
            kernel_output = np.zeros((self.height - kernel_size + 1, self.width - kernel_size + 1))
            for i in range(self.height - kernel_size + 1):
                for j in range(self.width - kernel_size + 1):
                    window = x[i:i+kernel_size, j:j+kernel_size]
                    kernel_output[i, j] = np.sum(kernel_weight * window) # np.dot(a, b) != a * b
            output.append(kernel_output)
        return np.array(output)


x = np.array([[0.1,   0.2,  0.3, 0.4],
              [-3,     -4,   -5,  -6],
              [5.1,   6.2,  7.3, 8.4],
              [-0.7, -0.8, -0.9, -1]])  #网络输入

#torch实验
in_channel = 1
out_channel = 3
kernel_size = 2
torch_model = TorchCNN(in_channel, out_channel, kernel_size)  # 初始化模型，输入输出通道和卷积核大小
'''
输出通道为3，所以生成卷积核的时候，为3*2*2

'''
print('_________________')
print(torch_model.state_dict())  # 完成初始化之后会随机生成一个权重
print('_________________')
torch_w = torch_model.state_dict()["layer.weight"]  # 保存生成的权重
print('_________________')
print(torch_w.numpy().shape)  # 打印权重形状（所谓的权重实际上就是卷积核的值）
torch_x = torch.FloatTensor([[x]])
output = torch_model.forward(torch_x)
output = output.detach().numpy()
print(output, output.shape, "torch模型预测结果\n")  # 将X输入，打印模型预测值
print("---------------")
diy_model = DiyModel(x.shape[0], x.shape[1], torch_w, kernel_size)
output = diy_model.forward(x)
print(output, "diy模型预测结果")
