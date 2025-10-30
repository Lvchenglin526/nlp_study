import numpy

# 创建一个示例二维数组
tensor = ([[10, 64, 25, 66], [78, 34, 12, 54]])


argmax_without1_keepdim = numpy.argmax(tensor, axis=None)

argmax_without_keepdim = numpy.argmax(tensor, axis=0)

argmax_with_keepdim = numpy.argmax(tensor, axis=1)

print("Original Tensor:\n", tensor)
print(type(tensor))
print("\nArgmax without keepdim:\n", argmax_without1_keepdim)
print(type(argmax_without1_keepdim))
print("\nArgmax without keepdim:\n", argmax_without_keepdim)
print(type(argmax_without_keepdim))
print("\nArgmax with keepdim:\n", argmax_with_keepdim)
print(type(argmax_with_keepdim))
'''
Original Tensor:
 [[10, 64, 25, 66], [78, 34, 12, 54]]
<class 'list'>

Argmax without keepdim:
 4
<class 'numpy.int64'>

Argmax without keepdim:
 [1 0 0 0]
<class 'numpy.ndarray'>

Argmax with keepdim:
 [3 0]
<class 'numpy.ndarray'>

Process finished with exit code 0
'''