import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(5, 5),
    nn.ReLU(),
    nn.Linear(5, 5),
    nn.ReLU(),
    nn.Linear(5, 5),
    nn.Softmax(),
)

import pdb; pdb.set_trace()