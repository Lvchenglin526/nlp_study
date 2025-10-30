import torch

# 创建一个示例张量
tensor = torch.tensor([[10, 64, 25, 66], [78, 34, 12, 54]])

# 使用 argmax 并设置 keepdim=False (默认行为)
argmax_without_keepdim = torch.argmax(tensor, dim=1)

# 使用 argmax 并设置 keepdim=True
argmax_with_keepdim = torch.argmax(tensor, dim=1, keepdim=True)

print("Original Tensor:\n", tensor)
print("\nArgmax without keepdim:\n", argmax_without_keepdim)
print("\nArgmax with keepdim:\n", argmax_with_keepdim)
'''
                                             dim = None
Original Tensor:
 tensor([[10, 64, 25, 66],
        [78, 34, 12, 54]])

Argmax without keepdim:
 tensor(4)

Argmax with keepdim:
 tensor([[4]])
 
                                             dim = 0
 Original Tensor:
 tensor([[10, 64, 25, 66],
        [78, 34, 12, 54]])

Argmax without keepdim:
 tensor([1, 0, 0, 0])

Argmax with keepdim:
 tensor([[1, 0, 0, 0]])
 
 
                                              dim = 1
 Original Tensor:
 tensor([[10, 64, 25, 66],
        [78, 34, 12, 54]])

Argmax without keepdim:
 tensor([3, 0])

Argmax with keepdim:
 tensor([[3],
        [0]])
'''