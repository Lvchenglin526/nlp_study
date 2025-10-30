#coding:utf8
import torch
import torch.nn as nn

'''
embedding层的处理
只是根据输入的字符集字数总数和量化后向量维度生成num_embeddings * embedding_dim的向量
'''

num_embeddings = 8  # 通常对于nlp任务，此参数为字符集字符总数
embedding_dim = 5   # 每个字符向量化后的向量维度
embedding_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
''''生成8*5字符矩阵（padding_idx（可选）：指定填充符的索引，其对应的嵌入向量会被设为0）'''   # 这就是为什么要把pad放在第一位
print("随机初始化权重")  # 随机填充数值
print(embedding_layer.weight)
# print(type(embedding_layer))
print("################")

#构造字符表
vocab = {
    "[pad]" : 0,
    "你" : 1,
    "好" : 2,
    "中" : 3,
    "国" : 4,
    "欢" : 5,
    "迎" : 6,
    "[unk]":7
}
# print(type(vocab))  # 字符表的类型为字典

#中国欢迎你 -> 中 国 欢 迎 你 -> [3,4,5,6,1] -> embedding_layer([3,4,5,6,1]) -> 5*5 矩阵
#你好中国 -> 你 好 中 国 -> [1,2,3,4] -> embedding_layer([1,2,3,4]) -> 4*5 矩阵



# 为了让不同长度的训练样本能够放在同一个batch中，需要将所有样本补齐或截断到相同长度
# 限制最大长度为5
# padding 补齐,[pad]占位符,[nuk]无法识别
# [1,2,3,0,0]
# [1,2,3,4,0]
# [1,2,3,4,5]
# 截断
# [1,2,3,4,5,6,7] -> [1,2,3,4,5]
# 少截断，多补0

def str_to_sequence(string, vocab):
    seq = [vocab.get(s, vocab["[unk]"]) for s in string][:5]  # S从字符串中获取key，没有就用vocab["[unk]"]代替，所以seq为列表
    if len(seq) < 5:
        seq += [vocab["[pad]"]] * (5 - len(seq))
    return seq

string1 = "abcddf"  # 字符表中不包含
string2 = "ddcc"
string3 = "feda"

sequence1 = str_to_sequence(string1, vocab)
sequence2 = str_to_sequence(string2, vocab)
sequence3 = str_to_sequence(string3, vocab)

print(sequence1)
print("################")
print(sequence2)
print("################")
print(sequence3)
print("################")

x = torch.LongTensor([sequence1, sequence2, sequence3])
# x = torch.LongTensor([sequence1])
print(x)
embedding_out = embedding_layer(x)
print(embedding_out)



