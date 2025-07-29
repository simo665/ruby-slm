import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
from tqdm import tqdm  # <--- for progress bars
from model import RubyMini
import os

class TokenDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def load_training_data(file_path: str = "dataset/dataset.npy"):
    return np.load(file_path)

def get_inputs_targets():
    data = load_training_data()
    print("🔎 Max token ID in dataset:", )
    # print data max token length
    seq_length = 128
    inputs, targets = [], []

    for i in range(0, len(data) - seq_length):
        inputs.append(data[i : i + seq_length])
        targets.append(data[i + 1 : i + seq_length + 1])

    return np.array(inputs), np.array(targets)

def count_model_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def print_gpu_info():
    try:
        os.system("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv")
    except:
        print("Unable to fetch GPU info")

def train():
    inputs, targets = get_inputs_targets()

    dataset = TokenDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Using device: {device}")
    
    vocab_size = 50000
    model = RubyMini(vocab_size).to(device)

    total_params, trainable_params = count_model_params(model)
    print(f"🧠 Model params: {trainable_params} / {total_params} trainable")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    num_epochs = 5
    total_start = time.time()

    for epoch in range(num_epochs):
        print(f"\n🌀 Epoch {epoch + 1}/{num_epochs}")
        epoch_start = time.time()

        model.train()
        total_loss = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch")

        for inputs_batch, targets_batch in progress:
            inputs_batch, targets_batch = inputs_batch.to(device), targets_batch.to(device)

            optimizer.zero_grad()
            outputs = model(inputs_batch)

            loss = loss_fn(outputs.view(-1, vocab_size), targets_batch.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        print(f"✅ Epoch {epoch+1} finished. Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

        # Save checkpoint
        os.makedirs("weight_data", exist_ok=True)
        torch.save(model.state_dict(), f"weight_data/ruby_mini2_epoch_{epoch + 1}.pth")

        print_gpu_info()

    total_time = time.time() - total_start
    print(f"\n🎉 Training completed in {total_time:.2f} seconds.")

if __name__ == "__main__":
    train()
