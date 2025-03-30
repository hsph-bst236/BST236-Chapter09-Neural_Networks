import torch
import torch.nn as nn
import torch.optim as optim
from model import TinyVGG
import wandb
from dataset import CustomCIFAR
# Initialize WandB only once at the start.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#wandb.init(project="pytorch-cnn-cv-tutorial", name="cifar10-tinyvgg-run")

# Define your model, loss, optimizer, etc.
model = TinyVGG().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

data = CustomCIFAR(subset_size = 100)
train_loader, val_loader = data.get_train_val_loaders(batch_size=50, validation_split=0.2)

epochs = 10

# Training loop
for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")
    
    # Log the average loss for this epoch.
    #wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss})