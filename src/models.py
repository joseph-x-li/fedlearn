import torch.nn as nn

def femnistmodel():
    model = nn.Sequential(
        nn.Conv2d(1, 32, 5, padding=(2, 2)),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(32, 64, 5, padding=(2, 2)),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Flatten(),
        nn.Linear(7 * 7 * 64, 2048),
        nn.ReLU(),
        nn.Linear(2048, 62), # 62 classes: 10 numbers, 26 lowercase, 26 uppercase
    )
    return model