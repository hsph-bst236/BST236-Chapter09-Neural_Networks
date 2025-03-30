import torch
import torch.nn as nn
import torch.optim as optim
from dataset import CustomCIFAR
from model import TinyVGG
from utils import visualize_model_layers, compute_confusion_matrix

from config import train_config, config_TinyVGG
import datetime
import os

from torch.utils.tensorboard import SummaryWriter
import wandb

def main():
    # Initialize WandB project for visualization.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{train_config['run_name']}_graph_{timestamp}"
    run = wandb.init(project=train_config["project_name"], 
               name=run_name,
               config={**train_config, **config_TinyVGG})
    
    # Initialize TensorBoard logger
    logger = SummaryWriter(log_dir='logs')
    
    # Create directory for saving feature visualizations
    vis_dir = os.path.join('logs', 'feature_maps')
    os.makedirs(vis_dir, exist_ok=True)

    # Set device and hyperparameters.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = train_config["epochs"]
    batch_size = train_config["batch_size"]
    learning_rate = train_config["learning_rate"]

    # Get data loaders.
    data = CustomCIFAR(subset_size = 100)
    train_loader, val_loader = data.get_train_val_loaders(batch_size=batch_size, validation_split=0.2)


    # Instantiate model, loss function and optimizer.
    model = TinyVGG(input_channels=config_TinyVGG["input_channels"], 
                    num_classes=config_TinyVGG["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Watch the model to track architecture, gradients, and parameters
    # run.watch(model, log="all", log_graph=True, log_freq=1)

    # Try to add model graph to tensorboard
    try:
        # Get a sample batch to trace the model
        dataiter = iter(train_loader)
        images, _ = next(dataiter)
        images = images.to(device)
            
        # Add graph using the actual batch data
        logger.add_graph(model, images)
        print("Successfully added model graph to TensorBoard")
    except Exception as e:
        print(f"Failed to add model graph to TensorBoard: {e}")
        print("Continuing training without model graph visualization")


    def train():
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Zero the parameter gradients.
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}")  
            # Log to WandB
            run.log({"train_loss": avg_loss}) 
            # Log to TensorBoard
            logger.add_scalar(tag='Loss/train', scalar_value=avg_loss, global_step=epoch)

            evaluate(epoch)
            
            

    def evaluate(epoch):
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item()

        avg_loss = running_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {avg_loss:.4f}")   
        # Log to WandB
        run.log({"val_loss": avg_loss}) 
        # Log to TensorBoard
        logger.add_scalar(tag='Loss/val', scalar_value=avg_loss, global_step=epoch)
        
        # Visualize feature maps at the end of each epoch and save to file
        feature_fig = visualize_model_layers(model, val_loader)
        
        # Compute and visualize confusion matrix - now simpler to call
        conf_matrix_fig = compute_confusion_matrix(model, val_loader, device)
        
        # Log figures to TensorBoard
        logger.add_figure(tag='Feature Maps/Epoch', figure=feature_fig, global_step=epoch)
        logger.add_figure(tag='Confusion Matrix/Epoch', figure=conf_matrix_fig, global_step=epoch)
        
        # Log figures to WandB
        # run.log({
        #     "Feature Maps/Epoch": feature_fig,
        #     "Confusion Matrix/Epoch": conf_matrix_fig
        # })

        model.train()
    
    train()
    run.finish()

if __name__ == "__main__":
    main()
