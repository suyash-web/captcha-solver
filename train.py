import os
import string
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch.optim as optim

CHARS = string.ascii_letters + string.digits  # a-zA-Z0-9
CHAR_TO_INDEX = {char: idx for idx, char in enumerate(CHARS)}
INDEX_TO_CHAR = {idx: char for char, idx in CHAR_TO_INDEX.items()}
NUM_CLASSES = len(CHARS)
CAPTCHA_LENGTH = 5

def encode_label(text):
    return [CHAR_TO_INDEX[c] for c in text]

def decode_output(output_tensor):
    pred = torch.argmax(output_tensor, dim=2)
    return [''.join(INDEX_TO_CHAR[c.item()] for c in seq) for seq in pred]

class CaptchaDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        # self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".png") and len(f.split(".")[0]) == CAPTCHA_LENGTH]
        self.image_files = [f for f in os.listdir(image_dir) if len(f.split(".")[0]) == CAPTCHA_LENGTH]
        self.image_files = self.image_files[:10000]
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((40, 100)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        file = self.image_files[idx]
        label_text = file.split(".")[0]
        if len(label_text) == 5:
            label = torch.tensor(encode_label(label_text), dtype=torch.long)
            img_path = os.path.join(self.image_dir, file)
            image = Image.open(img_path)
            return self.transform(image), label

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

def train_model(image_dir, batch_size=64, lr=0.001):
    dataset = CaptchaDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = CaptchaCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    _loss = 999
    # for epoch in range(epochs):
    epoch = 1
    while _loss > 0.8:
        model.train()
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = 0
            for i in range(CAPTCHA_LENGTH):
                loss += criterion(outputs[:, i, :], labels[:, i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch}] - Loss: {total_loss/len(dataloader):.4f}")
        epoch += 1
        _loss = total_loss/len(dataloader)
    
    torch.save(model.state_dict(), "path/to/model.pth")
    print("Model trained and saved.")

    return model

def test_model(model, image_dir, num_samples=10):
    model.eval()
    dataset = CaptchaDataset(image_dir)
    for i in range(num_samples):
        img, label = dataset[i]
        img = img.unsqueeze(0).to(device)
        pred = model(img)
        pred_text = decode_output(pred)[0]
        true_text = ''.join(INDEX_TO_CHAR[c.item()] for c in label)
        print(f"True value: {true_text} | Predicted value: {pred_text}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model = train_model("path/to/dataset", batch_size=64, lr=0.001)
test_model(trained_model, "path/to/dataset", num_samples=10)
