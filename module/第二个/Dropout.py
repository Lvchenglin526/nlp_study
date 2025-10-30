#coding:utf8

# import torch
# import torch.nn as nn
# import numpy as np


"""
基于pytorch的网络编写
测试dropout层
"""

import torch

x = torch.Tensor([1,2,3,4,5,6,7,8,9])
dp_layer = torch.nn.Dropout(0.9)(x)
# dp_x = dp_layer(x)


'''
dp_layer = torch.nn.Dropout(0.9)
这一步为先实例化dropout层，0.9为设置的类属性

dp_layer = torch.nn.Dropout(0.9)(x)
调用实例化之后的对象，调用类方法进行数据处理
'''

