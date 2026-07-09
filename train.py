import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np

# =========================================================
# Multimodal Deep Learning Training Pipeline
# =========================================================

def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001, save_path="efficientnet_weights.pth"):
    """
    Trains an EfficientNet-B4 classification model on a medical dataset.
    Expects dataset organized in standard PyTorch ImageFolder format:
        data_dir/
            train/
                Pathology_A/
                Pathology_B/
                Normal/
            val/
                Pathology_A/
                Pathology_B/
                Normal/
    """
    print(f"Initializing Multimodal Training pipeline for dataset: {data_dir}")
    
    # -----------------------------------------------------
    # 1. Transforms & Augmentations
    # -----------------------------------------------------
    # Medical images benefit from color jitter and mild rotations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # -----------------------------------------------------
    # 2. Dataloaders
    # -----------------------------------------------------
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    
    print(f"Pathology Classes Detected: {class_names}")
    print(f"Training dataset size: {dataset_sizes['train']} images")
    print(f"Validation dataset size: {dataset_sizes['val']} images")

    # -----------------------------------------------------
    # 3. Hardware Device Mapping (MPS for Mac / CUDA for Linux/Windows)
    # -----------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using High-End CUDA GPU Accelerator")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS (Metal Performance Shaders) Accelerator")
    else:
        device = torch.device("cpu")
        print("Using CPU (Warning: Deep learning training on CPU is extremely slow)")

    # -----------------------------------------------------
    # 4. Initialize Pretrained EfficientNet Backbone
    # -----------------------------------------------------
    # Loading pretrained weights from ImageNet
    print("Loading Pretrained EfficientNet-B4 Backbone...")
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    
    # Modify final classifier layer to match our pathology output classes
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_features, num_classes)
    )
    model = model.to(device)

    # -----------------------------------------------------
    # 5. Define Criterion & Optimizer
    # -----------------------------------------------------
    # CrossEntropy is standard for multi-class classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler: decays learning rate by 0.1 every 7 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # -----------------------------------------------------
    # 6. Training Loop
    # -----------------------------------------------------
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Track history only in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize only if training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # Statistics calculation
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]
            
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            # Deep copy the best performing weights
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'class_names': class_names,
                    'accuracy': best_acc
                }, save_path)
                print(f"--> Saved New Best Model Weights to {save_path} (Acc: {best_acc:.4f})")

    print(f"\nTraining complete. Best Validation Accuracy: {best_acc:.4f}")

# =========================================================
# Live Grad-CAM Generation Feature (Reference Implementation)
# =========================================================

class RealGradCAM:
    """
    Computes real Gradient-weighted Class Activation Mapping (Grad-CAM)
    on a trained PyTorch model.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture forward activations and backward gradients
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # Backward pass for target class
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Pool gradients across width and height dimensions
        # Shape: [channels, height, width]
        pooled_gradients = torch.mean(self.gradients[0], dim=[1, 2])
        
        # Weight the activations by pooled gradients
        for i in range(self.activations.shape[1]):
            self.activations[0, i, :, :] *= pooled_gradients[i]
            
        # Compute mean across channels
        heatmap = torch.mean(self.activations[0], dim=0).cpu().numpy()
        
        # Apply ReLU to keep positive activations
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize heatmap to [0, 1] range
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
            
        return heatmap

if __name__ == "__main__":
    # Example execution:
    # Set data_dir to your local downloaded Kaggle directories
    # train_model(data_dir="path/to/downloaded/dataset", num_epochs=5)
    pass
