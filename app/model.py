import torch
import torch.nn as nn
import string

CHARS = string.ascii_letters + string.digits  # a-zA-Z0-9
CHAR_TO_INDEX = {char: idx for idx, char in enumerate(CHARS)}
INDEX_TO_CHAR = {idx: char for char, idx in CHAR_TO_INDEX.items()}
NUM_CLASSES = len(CHARS)
CAPTCHA_LENGTH = 5

class CaptchaCNN(nn.Module):
    def __init__(self):
        super(CaptchaCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 10 * 25, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, CAPTCHA_LENGTH * NUM_CLASSES)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(-1, CAPTCHA_LENGTH, NUM_CLASSES)

def decode_output(output_tensor):
    pred = torch.argmax(output_tensor, dim=2)
    return [''.join(INDEX_TO_CHAR[c.item()] for c in seq) for seq in pred]
