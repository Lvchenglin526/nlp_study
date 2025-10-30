#coding:utf8

import torch
import torch.nn as nn
import numpy as np


"""
手动实现简单的神经网络
使用pytorch实现RNN
手动实现RNN
对比
"""

class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TorchRNN, self).__init__()  # 利用super关键字，来继承父类nn.Module的属性和方法
        self.layer = nn.RNN(input_size, hidden_size, bias=False, batch_first=True)  # bias：是否使用偏置，默认为True。
        pass  # batch_first：输入数据的形式，若为True，则输入张量的第一个维度是批量大小；
        pass  # 若为False，则第二个维度是批量大小，默认为False

    def forward(self, x):
        return self.layer(x)

#自定义RNN模型
class DiyModel:
    def __init__(self, w_ih, w_hh, hidden_size):
        self.w_ih = w_ih
        self.w_hh = w_hh
        self.hidden_size = hidden_size

    def forward(self, x):
        ht = np.zeros((self.hidden_size))
        output = []
        for xt in x:
            ux = np.dot(self.w_ih, xt)
            wh = np.dot(self.w_hh, ht)
            ht_next = np.tanh(ux + wh)
            output.append(ht_next)
            ht = ht_next
        return np.array(output), ht


x = np.array([[1, 2, 3],
              [3, 4, 5],
              [5, 6, 7],
              [4, 9, 7]])  #网络输入,三个1*4(过了embadding的四个字)

#torch实验
hidden_size = 5
torch_model = TorchRNN(3, hidden_size)  # 四个字转化成四个不同的向量，向量的维度就是这个字的特征数
# torch_model = TorchRNN(input_size=3, hidden_size=hidden_size)

print(torch_model.state_dict(),'\n')
print("---------------")
w_ih = torch_model.state_dict()["layer.weight_ih_l0"]
w_hh = torch_model.state_dict()["layer.weight_hh_l0"]
print("---------------")
print(w_ih, w_ih.shape,'\n')
print(w_hh, w_hh.shape,'\n')
#
torch_x = torch.FloatTensor([x])
print("############")
print(torch_x.shape)  # 1*4*3
output, h = torch_model.forward(torch_x)
print(output.shape)  # 1*4*5
print(h.shape)  # 1*1*5
print(h)
print(output.detach().numpy(), "torch模型预测结果")
print(h.detach().numpy(), "torch模型预测隐含层结果")
print("---------------")
diy_model = DiyModel(w_ih, w_hh, hidden_size)
output, h = diy_model.forward(x)
print(output, "diy模型预测结果")
print(h, "diy模型预测隐含层结果")
'''
OrderedDict({'layer.weight_ih_l0': tensor([[ 0.3374,  0.3177,  0.4238],
        [ 0.2000,  0.2314, -0.3408],
        [-0.1932,  0.2714,  0.2700],
        [ 0.2254,  0.0542,  0.2194],
        [-0.1512,  0.2005,  0.3726]]), 'layer.weight_hh_l0': tensor([[ 0.1689,  0.0023,  0.1573,  0.1637,  0.3921],
        [ 0.0667, -0.4198, -0.2286, -0.4147, -0.0314],
        [ 0.3477,  0.2143, -0.1810, -0.3378, -0.2276],
        [ 0.1045,  0.0613,  0.0705,  0.4110,  0.3410],
        [-0.2215, -0.1432,  0.1477, -0.2508,  0.2952]])}) 

---------------
---------------
tensor([[ 0.3374,  0.3177,  0.4238],
        [ 0.2000,  0.2314, -0.3408],
        [-0.1932,  0.2714,  0.2700],
        [ 0.2254,  0.0542,  0.2194],
        [-0.1512,  0.2005,  0.3726]]) torch.Size([5, 3]) 

tensor([[ 0.1689,  0.0023,  0.1573,  0.1637,  0.3921],
        [ 0.0667, -0.4198, -0.2286, -0.4147, -0.0314],
        [ 0.3477,  0.2143, -0.1810, -0.3378, -0.2276],
        [ 0.1045,  0.0613,  0.0705,  0.4110,  0.3410],
        [-0.2215, -0.1432,  0.1477, -0.2508,  0.2952]]) torch.Size([5, 5]) 

############
torch.Size([1, 4, 3])
torch.Size([1, 4, 5])
torch.Size([1, 1, 5])
tensor([[[1.0000, 0.0515, 0.9959, 0.9990, 0.9990]]], grad_fn=<StackBackward0>)
[[[ 0.97777116 -0.34477255  0.8208225   0.7581894   0.87811   ]
  [ 0.99993455 -0.46040517  0.9081708   0.99169254  0.9773319 ]
  [ 0.9999993  -0.368244    0.9691906   0.9991284   0.99572736]
  [ 0.9999998   0.05148468  0.995852    0.999049    0.9990489 ]]] torch模型预测结果
[[[0.9999998  0.05148468 0.995852   0.999049   0.9990489 ]]] torch模型预测隐含层结果
---------------
[[ 0.97777115 -0.34477256  0.82082249  0.7581894   0.87811001]
 [ 0.99993456 -0.46040515  0.90817082  0.99169255  0.97733185]
 [ 0.99999928 -0.36824399  0.96919057  0.99912838  0.99572737]
 [ 0.9999998   0.0514847   0.995852    0.99904901  0.99904891]] diy模型预测结果
[0.9999998  0.0514847  0.995852   0.99904901 0.99904891] diy模型预测隐含层结果
'''