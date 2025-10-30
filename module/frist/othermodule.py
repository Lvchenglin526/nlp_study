import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
'''
用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。
'''
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5) # 线性层, 输出分5类
        self.activation = torch.softmax # 给模型增加非线性能力，添加激活函数
        self.loss = nn.CrossEntropyLoss() # loss函数使用交叉熵,交叉熵内部有softmax,不用再加激活函数

    def forward(self, x, y_true=None):
        y_pred = self.linear(x) # (batchSize, inputSize) -->w=(inputSize, 5)--> (batchSize, 5)
        if y_true is not None:
            loss = self.loss(y_pred, y_true) # 计算损失值,交叉熵内部有softmax,不用再加激活函数
            return loss
        else:
            y_pred = self.activation(y_pred, 1) # (batch_size, 5) -> (batch_size, 5)
            return y_pred # 返回预测值

# 生成一个样本
# 随机生成一个五维向量，返回(该五维向量, 最大值的下标)
def build_sample():
    x = np.random.random(5)
    # 获取最大值的索引
    max_index = np.argmax(x)
    return x, max_index

# 随机生成一批样本
def build_dataset(sample_num):
    X = []
    Y = []
    for i in range(sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)# mse 加[]，ce不加[]
    print(f'X: {X}, Y: {Y}')
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct_num, wrong_num = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_pred, y_true in zip(y_pred, y):
            if torch.argmax(y_pred)==int(y_true):
                correct_num += 1
            else:
                wrong_num += 1
    print(f'正确预测个数{correct_num}, 准确率{correct_num/(correct_num+wrong_num)}')
    return correct_num/(correct_num+wrong_num)

def main():
    # 配置参数
    epoch_num = 10 # 训练轮数
    batch_size = 20 # 每次训练的样本数
    train_sample_size = 5000 # 每轮训练的样本总数
    input_size = 5 # 输入向量维度
    learning_rate = 0.001 # 学习率
    # 训练模型
    model = TorchModel(input_size)
    # 构建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 读取训练集
    train_x, train_y = build_dataset(train_sample_size)
    # 训练过程
    for epoch in range(epoch_num):
        model.train() # 标记开始训练
        loss_list = []
        for batch_index in range(train_sample_size // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            loss = model(x, y) # 计算loss， model.forward
            loss.backward() # 计算梯度
            optimizer.step() # 更新权重
            optimizer.zero_grad() # 梯度置0
            loss_list.append(loss.item())
        print(f'第{epoch + 1}轮，平均loss：{np.mean(loss_list)}')
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(loss_list))])
    # 保存模型
    torch.save(model.state_dict(), 'model.pth')
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label='acc')
    plt.plot(range(len(log)), [l[1] for l in log], label='loss')
    plt.legend()
    plt.show()
    return

if __name__ == '__main__':
    main()