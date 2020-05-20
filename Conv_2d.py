import torch
import torch.nn as nn

conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2, bias=False)

X = torch.FloatTensor([[[
    [4, 2, -1],
    [-6, 0, 5],
    [3, 2, 2]]]])

conv.weight.data = torch.FloatTensor([[[
    [0, 1, 2],
    [1, -1, 0],
    [1, 0, -2]
]]])

res = conv(X).data[0, 0]
print(res)