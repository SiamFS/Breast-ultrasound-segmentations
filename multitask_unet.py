import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import argparse

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ConvBlock(nn.Module):
    """
    Convolutional Block with double convolution and batch normalization
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class MultiTaskUNet(nn.Module):
    """
    Multi-Task U-Net with segmentation and classification heads
    """
    def __init__(self, in_channels=1, out_channels=1, num_classes=3):
        super(MultiTaskUNet, self).__init__()
        
        # Encoder path
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        
        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)
        
        # Decoder path
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(1024, 512)  # 512 + 512 = 1024 due to skip connection
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(512, 256)   # 256 + 256 = 512 due to skip connection
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)   # 128 + 128 = 256 due to skip connection
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(128, 64)    # 64 + 64 = 128 due to skip connection
        
        # Final segmentation layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Classification head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),  # Increased first layer size for better feature extraction
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),       # Increased dropout for better regularization
            nn.Linear(512, 256),   # Added extra layer for deeper classification network
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Max pooling operation
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Initialize weights for better convergence
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Encoder path with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.maxpool(e1))
        e3 = self.enc3(self.maxpool(e2))
        e4 = self.enc4(self.maxpool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.maxpool(e4))
        
        # Classification head
        c = self.classifier(self.pool(b))
        
        # Standard U-Net decoder path
        d1 = self.dec1(torch.cat([self.up1(b), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), e1], dim=1))
        
        # Final segmentation
        seg = torch.sigmoid(self.final_conv(d4))
        
        return seg, c

class BreastUltrasoundDataset(Dataset):
    """
    Dataset class for Breast Ultrasound Images
    """
    def __init__(self, images, masks, labels, transform=None):
        self.images = images
        self.masks = masks
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return {
            'image': torch.FloatTensor(image),
            'mask': torch.FloatTensor(mask),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_image(path, size):
    """Load and preprocess image from path with error handling"""
    try:
        image = cv2.imread(path)
        if image is None:
            print(f"Warning: Could not read image {path}")
            return None
        
        image = cv2.resize(image, (size, size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image / 255.0  # Normalize
        return image
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def load_data(root_path, size=256, use_processed=True):
    """
    Load dataset with images and masks
    
    Args:
        root_path: Path to dataset
        size: Size to resize images to
        use_processed: Whether to use the preprocessed data from data_preparation.py
    
    Returns:
        images, masks, labels arrays
    """
    # Check if the dataset has been preprocessed
    processed_dir = os.path.join(os.path.dirname(root_path.split('*')[0]), 'processed')
    
    if use_processed and os.path.exists(processed_dir):
        print(f"Using preprocessed dataset from {processed_dir}")
        return load_processed_data(processed_dir, size)
    else:
        print("Preprocessed dataset not found. Using original dataset and preprocessing on-the-fly.")
        return load_raw_data(root_path, size)

def load_processed_data(processed_dir, size=256):
    """Load data from the preprocessed directory created by data_preparation.py"""
    train_images = []
    train_masks = []
    train_labels = []
    
    val_images = []
    val_masks = []
    val_labels = []
    
    # Map class names to integer labels
    class_map = {'normal': 0, 'benign': 1, 'malignant': 2}
    
    # Load training and validation data separately
    for split in ['train', 'val']:
        split_images = []
        split_masks = []
        split_labels = []
        
        for class_name in ['normal', 'benign', 'malignant']:
            class_dir = os.path.join(processed_dir, split, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist. Skipping.")
                continue
            
            print(f"Loading {split} data from {class_dir}")
            
            # Get all image files (excluding masks)
            image_files = [f for f in glob(os.path.join(class_dir, '*.png')) if '_mask' not in f]
            
            for img_path in tqdm(image_files, desc=f"Loading {split}/{class_name}"):
                img_name = os.path.basename(img_path).split('.')[0]
                mask_path = os.path.join(class_dir, f"{img_name}_mask.png")
                
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask not found for {img_path}. Skipping.")
                    continue
                
                # Load and preprocess image and mask
                image = load_image(img_path, size)
                if image is None:
                    continue
                
                mask = load_image(mask_path, size)
                if mask is None:
                    continue
                
                # Binarize mask
                mask = (mask > 0.5).astype(np.float32)
                
                split_images.append(image)
                split_masks.append(mask)
                split_labels.append(class_map[class_name])
        
        # Add to corresponding lists
        if split == 'train':
            train_images.extend(split_images)
            train_masks.extend(split_masks)
            train_labels.extend(split_labels)
        elif split == 'val':
            val_images.extend(split_images)
            val_masks.extend(split_masks)
            val_labels.extend(split_labels)
    
    if len(train_images) == 0:
        raise ValueError("No valid training images loaded. Please check your dataset.")
    
    if len(val_images) == 0:
        raise ValueError("No valid validation images loaded. Please check your dataset.")
    
    # Convert to numpy arrays and add channel dimension
    train_images = np.array(train_images)[:, np.newaxis, :, :]  # (N, 1, H, W)
    train_masks = np.array(train_masks)[:, np.newaxis, :, :]    # (N, 1, H, W)
    train_labels = np.array(train_labels)
    
    val_images = np.array(val_images)[:, np.newaxis, :, :]  # (N, 1, H, W)
    val_masks = np.array(val_masks)[:, np.newaxis, :, :]    # (N, 1, H, W)
    val_labels = np.array(val_labels)
    
    print(f"Loaded {len(train_images)} training images, {len(train_masks)} masks, {len(train_labels)} labels")
    print(f"Training class distribution: {np.bincount(train_labels)}")
    
    print(f"Loaded {len(val_images)} validation images, {len(val_masks)} masks, {len(val_labels)} labels")
    print(f"Validation class distribution: {np.bincount(val_labels)}")
    
    return (train_images, train_masks, train_labels), (val_images, val_masks, val_labels)

def load_raw_data(root_path, size=256):
    """Load dataset with images and masks, handling multiple masks per image"""
    images = []
    masks = []
    labels = []
    
    # Map class names to integer labels
    class_map = {'normal': 0, 'benign': 1, 'malignant': 2}
    
    # Track combined masks
    mask_tracker = {}
    
    # Get all image paths
    all_paths = sorted(glob(root_path))
    if not all_paths:
        raise FileNotFoundError(f"No files found at {root_path}. Please check your dataset path.")
    
    # First, process all masks
    print("Processing mask files...")
    for path in tqdm(all_paths):
        if '_mask' in path:
            img_name = path.split('/')[-1].split('\\')[-1].split('_mask')[0]
            img_class = None
            
            # Determine class from path
            if 'normal' in path.lower():
                img_class = 'normal'
            elif 'benign' in path.lower():
                img_class = 'benign'
            elif 'malignant' in path.lower():
                img_class = 'malignant'
                
            if img_name not in mask_tracker:
                mask_tracker[img_name] = {'mask': None, 'class': img_class}
            
            mask = load_image(path, size)
            if mask is None:
                continue
                
            # Threshold to binary (0 or 1)
            mask = (mask > 0.5).astype(np.float32)
            
            if mask_tracker[img_name]['mask'] is None:
                mask_tracker[img_name]['mask'] = mask
            else:
                # Combine with previous mask
                mask_tracker[img_name]['mask'] = np.logical_or(
                    mask_tracker[img_name]['mask'], mask).astype(np.float32)
    
    # Now process all images
    print("Processing image files...")
    for path in tqdm(all_paths):
        if '_mask' not in path and (path.endswith('.png') or path.endswith('.jpg')):
            img_name = path.split('/')[-1].split('\\')[-1].split('.')[0]
            
            # Determine class from path
            img_class = None
            if 'normal' in path.lower():
                img_class = 'normal'
            elif 'benign' in path.lower():
                img_class = 'benign'
            elif 'malignant' in path.lower():
                img_class = 'malignant'
            
            # Skip if it's an invalid file or we don't have a mask for it
            if img_name not in mask_tracker or img_class is None:
                continue
                
            image = load_image(path, size)
            if image is None:
                continue
                
            images.append(image)
            masks.append(mask_tracker[img_name]['mask'])
            labels.append(class_map[img_class])
    
    if len(images) == 0:
        raise ValueError("No valid images loaded. Please check your dataset.")
    
    # Convert to numpy arrays and add channel dimension
    images = np.array(images)[:, np.newaxis, :, :]  # (N, 1, H, W)
    masks = np.array(masks)[:, np.newaxis, :, :]    # (N, 1, H, W)
    labels = np.array(labels)
    
    print(f"Loaded {len(images)} images, {len(masks)} masks, {len(labels)} labels")
    print(f"Class distribution: {np.bincount(labels)}")
    
    return images, masks, labels

# ------------------- SEGMENTATION METRICS -------------------

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Calculate Dice coefficient
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Small constant to avoid division by zero
    
    Returns:
        Dice coefficient
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f)
    
    # Handle empty masks case
    if union == 0:
        return torch.tensor(1.0, device=y_true.device)
        
    return (2. * intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred, smooth=1e-6):
    """Calculate Dice loss"""
    return 1 - dice_coefficient(y_true, y_pred, smooth)

def mean_iou(y_pred, y_true, threshold=0.5, smooth=1e-6):
    """
    Calculate mean IoU by computing IoU for each image separately and then averaging
    
    Args:
        y_pred: Predicted masks (B, 1, H, W)
        y_true: Ground truth masks (B, 1, H, W)
        threshold: Threshold to binarize predictions
        smooth: Small constant to avoid division by zero
        
    Returns:
        Mean IoU across all images in the batch
    """
    batch_size = y_pred.size(0)
    
    # Apply threshold to get binary predictions
    y_pred_binary = (y_pred > threshold).float()
    
    # Initialize IoU sum
    iou_sum = 0.0
    
    # Calculate IoU for each image in the batch
    for i in range(batch_size):
        # Get current image and mask
        pred = y_pred_binary[i].view(-1)
        true = y_true[i].view(-1)
        
        # Calculate intersection and union
        intersection = (pred * true).sum()
        union = pred.sum() + true.sum() - intersection
        
        # Handle edge case where both prediction and target are empty
        if union == 0:
            iou_sum += 1.0  # Consider perfect IoU if both are empty
            continue
        
        # Calculate IoU for this image
        iou = (intersection + smooth) / (union + smooth)
        
        # Add to sum
        iou_sum += iou.item()
    
    # Return mean IoU
    return iou_sum / batch_size

def pixel_accuracy(y_pred, y_true, threshold=0.5):
    """
    Calculate pixel-wise accuracy
    
    Args:
        y_pred: Predicted masks
        y_true: Ground truth masks
        threshold: Threshold to binarize predictions
    
    Returns:
        Pixel-wise accuracy
    """
    y_pred_binary = (y_pred > threshold).float()
    correct = torch.eq(y_pred_binary, y_true).float().sum()
    total = float(torch.numel(y_true))
    
    # Avoid division by zero
    if total == 0:
        return torch.tensor(1.0, device=y_true.device)
    
    return correct / total

# ------------------- CLASSIFICATION METRICS -------------------

def calculate_classification_metrics(true_labels, pred_labels, num_classes=3):
    """
    Calculate classification metrics: accuracy, precision, recall, F1
    
    Args:
        true_labels: Ground truth labels
        pred_labels: Predicted labels
        num_classes: Number of classes
    
    Returns:
        Dictionary with classification metrics
    """
    # Convert to numpy arrays if they are tensors
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()
    
    # Calculate classification metrics
    try:
        # Handle the case where a class may have no samples
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted', zero_division=0
        )
        
        # Calculate accuracy (correctly classified / total)
        accuracy = np.mean(true_labels == pred_labels)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    except Exception as e:
        print(f"Warning: Error calculating classification metrics: {e}")
        # Return default values
        return {
            'accuracy': np.mean(true_labels == pred_labels),
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

# ------------------- TRAINING FUNCTION -------------------

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0, path='checkpoint.pth'):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement
            min_delta (float): Minimum change in monitored value to qualify as improvement
            path (str): Path to save the checkpoint
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path
        self.val_loss_min = float('inf')
    
    def __call__(self, val_loss, model, optimizer, epoch, metrics):
        """
        Call the early stopping class to check if training should stop
        
        Args:
            val_loss (float): Validation loss
            model (nn.Module): Model to save
            optimizer (torch.optim): Optimizer state to save
            epoch (int): Current epoch
            metrics (dict): Validation metrics
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, metrics)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, metrics)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, optimizer, epoch, metrics):
        """Save model when validation loss decreases"""
        if val_loss < self.val_loss_min:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
            self.val_loss_min = val_loss
            
            # Create directory for save path if it doesn't exist
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': metrics
            }, self.path)

def plot_training_progress(history, save_path, epoch):
    """Plot training progress up to the current epoch and save as image"""
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot mIoU
    plt.subplot(2, 3, 2)
    plt.plot(history['train_miou'], label='Train mIoU')
    plt.plot(history['val_miou'], label='Val mIoU')
    plt.title('Mean IoU')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Plot Dice
    plt.subplot(2, 3, 3)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.plot(history['val_dice'], label='Val Dice')
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Plot Pixel Accuracy
    plt.subplot(2, 3, 4)
    plt.plot(history['train_pixel_acc'], label='Train Pixel Acc')
    plt.plot(history['val_pixel_acc'], label='Val Pixel Acc')
    plt.title('Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot Classification Accuracy
    plt.subplot(2, 3, 5)
    plt.plot(history['train_cls_acc'], label='Train Cls Acc')
    plt.plot(history['val_cls_acc'], label='Val Cls Acc')
    plt.title('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot F1 Score
    plt.subplot(2, 3, 6)
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['val_f1'], label='Val F1')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'training_progress_epoch_{epoch+1}.png'))
    plt.close()

def train_model(model, train_loader, val_loader, device, num_epochs=30, save_path='models', patience=3, metrics_file='metrics.csv'):
    """Train the multi-task U-Net model with early stopping and metrics tracking"""
    # Create directory for saving models
    os.makedirs(save_path, exist_ok=True)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, path=os.path.join(save_path, 'best_model.pth'))
    
    # Loss functions
    criterion_seg = dice_loss  # Using Dice loss instead of BCE for better segmentation
    criterion_cls = nn.CrossEntropyLoss()
    
    # Weights for multi-task learning - Increased weight for segmentation to improve IoU
    seg_weight = 0.85  # Increased to give even more weight to segmentation
    cls_weight = 0.15  # Reduced to focus more on segmentation
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay for regularization
    
    # Learning rate scheduler - Changed to CosineAnnealingLR for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Initialize metrics tracking file
    with open(os.path.join(save_path, metrics_file), 'w') as f:
        # Write header
        f.write("epoch,train_loss,train_miou,train_dice,train_pixel_acc,train_cls_acc,train_precision,train_recall,train_f1,")
        f.write("val_loss,val_miou,val_dice,val_pixel_acc,val_cls_acc,val_precision,val_recall,val_f1\n")
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_miou': [], 'val_miou': [],
        'train_dice': [], 'val_dice': [],
        'train_pixel_acc': [], 'val_pixel_acc': [],
        'train_cls_acc': [], 'val_cls_acc': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_f1': [], 'val_f1': [],
        'lr': []
    }
    
    # Set scaler for mixed precision training if available
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train mode
        model.train()
        train_loss = 0.0
        train_miou_sum = 0.0
        train_dice_sum = 0.0
        train_pixel_acc_sum = 0.0
        train_samples = 0
        
        # Store all class predictions and labels for accuracy, precision, recall calculation
        all_cls_preds = []
        all_cls_true = []
        
        for batch in tqdm(train_loader, desc="Training"):
            # Get data
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            labels = batch['label'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Use mixed precision training if available
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    # Forward pass
                    seg_outputs, cls_outputs = model(images)
                    
                    # Calculate losses
                    seg_loss = criterion_seg(seg_outputs, masks)
                    cls_loss = criterion_cls(cls_outputs, labels)
                    
                    # Combined loss
                    loss = seg_weight * seg_loss + cls_weight * cls_loss
                
                # Scales loss and calls backward()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward pass
                seg_outputs, cls_outputs = model(images)
                
                # Calculate losses
                seg_loss = criterion_seg(seg_outputs, masks)
                cls_loss = criterion_cls(cls_outputs, labels)
                
                # Combined loss
                loss = seg_weight * seg_loss + cls_weight * cls_loss
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            
            # Update training metrics
            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            
            # Calculate mean IoU for this batch
            batch_miou = mean_iou(seg_outputs, masks)
            train_miou_sum += batch_miou * batch_size
            
            # Calculate Dice coefficient for this batch
            batch_dice = dice_coefficient(masks, (seg_outputs > 0.5).float()).item()
            train_dice_sum += batch_dice * batch_size
            
            # Calculate pixel accuracy for this batch
            batch_pixel_acc = pixel_accuracy(seg_outputs, masks).item()
            train_pixel_acc_sum += batch_pixel_acc * batch_size
            
            # Store class predictions and labels for later calculation
            _, cls_preds = torch.max(cls_outputs, 1)
            all_cls_preds.append(cls_preds.cpu())
            all_cls_true.append(labels.cpu())
            
            train_samples += batch_size
        
        # Calculate average training metrics
        train_loss /= train_samples
        train_miou = train_miou_sum / train_samples
        train_dice = train_dice_sum / train_samples
        train_pixel_acc = train_pixel_acc_sum / train_samples
        
        # Concatenate all class predictions and true labels
        all_cls_preds = torch.cat(all_cls_preds)
        all_cls_true = torch.cat(all_cls_true)
        
        # Calculate classification metrics
        train_cls_metrics = calculate_classification_metrics(all_cls_true, all_cls_preds)
        train_cls_acc = train_cls_metrics['accuracy']
        train_precision = train_cls_metrics['precision']
        train_recall = train_cls_metrics['recall']
        train_f1 = train_cls_metrics['f1']
        
        # Validation mode
        model.eval()
        val_loss = 0.0
        val_miou_sum = 0.0
        val_dice_sum = 0.0
        val_pixel_acc_sum = 0.0
        val_samples = 0
        
        # Store all class predictions and labels for accuracy, precision, recall calculation
        all_cls_preds = []
        all_cls_true = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Get data
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                seg_outputs, cls_outputs = model(images)
                
                # Calculate losses
                seg_loss = criterion_seg(seg_outputs, masks)
                cls_loss = criterion_cls(cls_outputs, labels)
                
                # Combined loss
                loss = seg_weight * seg_loss + cls_weight * cls_loss
                
                # Update validation metrics
                batch_size = images.size(0)
                val_loss += loss.item() * batch_size
                
                # Calculate mean IoU for this batch
                batch_miou = mean_iou(seg_outputs, masks)
                val_miou_sum += batch_miou * batch_size
                
                # Calculate Dice coefficient for this batch
                batch_dice = dice_coefficient(masks, (seg_outputs > 0.5).float()).item()
                val_dice_sum += batch_dice * batch_size
                
                # Calculate pixel accuracy for this batch
                batch_pixel_acc = pixel_accuracy(seg_outputs, masks).item()
                val_pixel_acc_sum += batch_pixel_acc * batch_size
                
                # Store class predictions and labels for later calculation
                _, cls_preds = torch.max(cls_outputs, 1)
                all_cls_preds.append(cls_preds.cpu())
                all_cls_true.append(labels.cpu())
                
                val_samples += batch_size
        
        # Calculate average validation metrics
        val_loss /= val_samples
        val_miou = val_miou_sum / val_samples
        val_dice = val_dice_sum / val_samples
        val_pixel_acc = val_pixel_acc_sum / val_samples
        
        # Concatenate all class predictions and true labels
        all_cls_preds = torch.cat(all_cls_preds)
        all_cls_true = torch.cat(all_cls_true)
        
        # Calculate classification metrics
        val_cls_metrics = calculate_classification_metrics(all_cls_true, all_cls_preds)
        val_cls_acc = val_cls_metrics['accuracy']
        val_precision = val_cls_metrics['precision']
        val_recall = val_cls_metrics['recall']
        val_f1 = val_cls_metrics['f1']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_miou'].append(train_miou)
        history['val_miou'].append(val_miou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_pixel_acc'].append(train_pixel_acc)
        history['val_pixel_acc'].append(val_pixel_acc)
        history['train_cls_acc'].append(train_cls_acc)
        history['val_cls_acc'].append(val_cls_acc)
        history['train_precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Metrics for early stopping
        metrics = {
            'miou': val_miou,
            'dice': val_dice,
            'pixel_acc': val_pixel_acc,
            'cls_acc': val_cls_acc,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1
        }
        
        # Early stopping
        early_stopping(val_loss, model, optimizer, epoch, metrics)
        
        # Save metrics to file
        with open(os.path.join(save_path, metrics_file), 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{train_miou:.4f},{train_dice:.4f},{train_pixel_acc:.4f},{train_cls_acc:.4f},{train_precision:.4f},{train_recall:.4f},{train_f1:.4f},")
            f.write(f"{val_loss:.4f},{val_miou:.4f},{val_dice:.4f},{val_pixel_acc:.4f},{val_cls_acc:.4f},{val_precision:.4f},{val_recall:.4f},{val_f1:.4f}\n")
        
        # Generate and save training progress plot
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            plot_training_progress(history, save_path, epoch)
        
        # Print metrics
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f} seconds")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)
        
        # Print segmentation metrics
        print("SEGMENTATION METRICS:")
        print(f"Train - mIoU: {train_miou:.4f}, Dice: {train_dice:.4f}, Pixel Acc: {train_pixel_acc:.4f}")
        print(f"Val   - mIoU: {val_miou:.4f}, Dice: {val_dice:.4f}, Pixel Acc: {val_pixel_acc:.4f}")
        print("-" * 50)
        
        # Print classification metrics
        print("CLASSIFICATION METRICS:")
        print(f"Train - Accuracy: {train_cls_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Accuracy: {val_cls_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        print("-" * 50)
        
        # Check if early stopping triggered
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    
    # Generate final training progress plot
    plot_training_progress(history, save_path, epoch)
    
    return history, model

def evaluate_model(model, dataloader, device, save_path='results'):
    """
    Evaluate model and save results
    
    Args:
        model: The trained model
        dataloader: DataLoader with test/validation data
        device: Device to run inference on
        save_path: Path to save evaluation results
    """
    os.makedirs(save_path, exist_ok=True)
    
    model.eval()
    val_miou_sum = 0.0
    val_dice_sum = 0.0
    val_pixel_acc_sum = 0.0
    val_samples = 0
    
    # Store all class predictions and labels for accuracy, precision, recall calculation
    all_cls_preds = []
    all_cls_true = []
    
    # Create confusion matrix
    class_names = ['Normal', 'Benign', 'Malignant']
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get data
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            seg_outputs, cls_outputs = model(images)
            
            # Update evaluation metrics
            batch_size = images.size(0)
            
            # Calculate mean IoU for this batch
            batch_miou = mean_iou(seg_outputs, masks)
            val_miou_sum += batch_miou * batch_size
            
            # Calculate Dice coefficient for this batch
            batch_dice = dice_coefficient(masks, (seg_outputs > 0.5).float()).item()
            val_dice_sum += batch_dice * batch_size
            
            # Calculate pixel accuracy for this batch
            batch_pixel_acc = pixel_accuracy(seg_outputs, masks).item()
            val_pixel_acc_sum += batch_pixel_acc * batch_size
            
            # Store class predictions and labels for later calculation
            _, cls_preds = torch.max(cls_outputs, 1)
            all_cls_preds.append(cls_preds.cpu())
            all_cls_true.append(labels.cpu())
            
            # Update confusion matrix
            for i in range(batch_size):
                true_class = labels[i].item()
                pred_class = cls_preds[i].item()
                confusion[true_class, pred_class] += 1
            
            val_samples += batch_size
    
    # Calculate average validation metrics
    val_miou = val_miou_sum / val_samples
    val_dice = val_dice_sum / val_samples
    val_pixel_acc = val_pixel_acc_sum / val_samples
    
    # Concatenate all class predictions and true labels
    all_cls_preds = torch.cat(all_cls_preds).numpy()
    all_cls_true = torch.cat(all_cls_true).numpy()
    
    # Calculate classification metrics
    val_cls_metrics = calculate_classification_metrics(all_cls_true, all_cls_preds)
    val_cls_acc = val_cls_metrics['accuracy']
    val_precision = val_cls_metrics['precision']
    val_recall = val_cls_metrics['recall']
    val_f1 = val_cls_metrics['f1']
    
    # Calculate per-class metrics
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        all_cls_true, all_cls_preds, labels=range(num_classes), average=None
    )
    
    # Print evaluation results
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print("\nSEGMENTATION METRICS:")
    print(f"Mean IoU: {val_miou:.4f}")
    print(f"Dice Coefficient: {val_dice:.4f}")
    print(f"Pixel Accuracy: {val_pixel_acc:.4f}")
    
    print("\nCLASSIFICATION METRICS:")
    print(f"Accuracy: {val_cls_acc:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall: {val_recall:.4f}")
    print(f"F1 Score: {val_f1:.4f}")
    
    print("\nCLASS-WISE CLASSIFICATION METRICS:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}:")
        print(f"  Precision: {class_precision[i]:.4f}")
        print(f"  Recall: {class_recall[i]:.4f}")
        print(f"  F1 Score: {class_f1[i]:.4f}")
        print(f"  Support: {class_support[i]}")
    
    print("\nCONFUSION MATRIX:")
    print(confusion)
    
    # Save results to file
    with open(os.path.join(save_path, 'evaluation_results.txt'), 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("SEGMENTATION METRICS:\n")
        f.write(f"Mean IoU: {val_miou:.4f}\n")
        f.write(f"Dice Coefficient: {val_dice:.4f}\n")
        f.write(f"Pixel Accuracy: {val_pixel_acc:.4f}\n\n")
        
        f.write("CLASSIFICATION METRICS:\n")
        f.write(f"Accuracy: {val_cls_acc:.4f}\n")
        f.write(f"Precision: {val_precision:.4f}\n")
        f.write(f"Recall: {val_recall:.4f}\n")
        f.write(f"F1 Score: {val_f1:.4f}\n\n")
        
        f.write("CLASS-WISE CLASSIFICATION METRICS:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}:\n")
            f.write(f"  Precision: {class_precision[i]:.4f}\n")
            f.write(f"  Recall: {class_recall[i]:.4f}\n")
            f.write(f"  F1 Score: {class_f1[i]:.4f}\n")
            f.write(f"  Support: {class_support[i]}\n")
        
        f.write("\nCONFUSION MATRIX:\n")
        for i in range(num_classes):
            f.write(f"{confusion[i]}\n")
    
    # Create and save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()
    
    return {
        'miou': val_miou,
        'dice': val_dice,
        'pixel_acc': val_pixel_acc,
        'cls_acc': val_cls_acc,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1,
        'confusion': confusion
    }

def predict_on_new_images(model, image_path, output_dir, device, image_size=256):
    """Make predictions on new unseen images for live demonstration"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input is a file or directory
    if os.path.isfile(image_path):
        # Single file mode
        image_files = [image_path]
    elif os.path.isdir(image_path):
        # Directory mode
        image_files = glob(os.path.join(image_path, "*.png")) + glob(os.path.join(image_path, "*.jpg"))
    else:
        # Check if the path might be a pattern
        image_files = glob(image_path)
    
    if not image_files:
        print(f"No images found at {image_path}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Set model to evaluation mode
    model.eval()
    
    class_names = ['Normal', 'Benign', 'Malignant']
    
    results_data = []
    
    with torch.no_grad():
        for img_path in image_files:
            print(f"Processing {img_path}...")
            
            try:
                # Load and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Could not read image {img_path}")
                    continue
                    
                img_orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # For display
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (image_size, image_size))
                img = img / 255.0  # Normalize
                
                # Add batch and channel dimensions
                img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(device)
                
                # Forward pass
                seg_output, cls_output = model(img_tensor)
                
                # Process outputs
                seg_output = seg_output.squeeze().cpu().numpy()
                seg_binary = (seg_output > 0.5).astype(np.float32)
                
                pred_class = torch.argmax(cls_output, dim=1).item()
                pred_class_name = class_names[pred_class]
                
                # Calculate confidence
                cls_probs = F.softmax(cls_output, dim=1).squeeze().cpu().numpy()
                confidence = cls_probs[pred_class]
                
                # Calculate mask coverage
                mask_coverage = np.mean(seg_binary) * 100
                
                # Create visualization
                plt.figure(figsize=(15, 5))
                
                # Original image
                plt.subplot(1, 3, 1)
                plt.imshow(img_orig)
                plt.title('Original Image')
                plt.axis('off')
                
                # Predicted mask
                plt.subplot(1, 3, 2)
                plt.imshow(seg_output, cmap='gray')
                plt.title('Predicted Mask')
                plt.axis('off')
                
                # Overlay
                plt.subplot(1, 3, 3)
                plt.imshow(cv2.resize(img_orig, (image_size, image_size)))
                plt.imshow(seg_binary, alpha=0.5, cmap='jet')
                plt.title(f'Prediction: {pred_class_name}\nConfidence: {confidence:.2f}')
                plt.axis('off')
                
                # Save figure
                output_path = os.path.join(output_dir, os.path.basename(img_path).split('.')[0] + '_prediction.png')
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
                
                # Store results data
                results_data.append({
                    'filename': os.path.basename(img_path),
                    'prediction': pred_class_name,
                    'confidence': confidence,
                    'mask_coverage': mask_coverage,
                    'class_probs': {name: prob for name, prob in zip(class_names, cls_probs)}
                })
                
                # Save prediction details
                text_path = os.path.join(output_dir, os.path.basename(img_path).split('.')[0] + '_prediction.txt')
                with open(text_path, 'w') as f:
                    f.write(f"Predicted Class: {pred_class_name}\n")
                    f.write(f"Class Probabilities:\n")
                    for i, cls_name in enumerate(class_names):
                        f.write(f"  {cls_name}: {cls_probs[i]:.4f}\n")
                    
                    # Segmentation metrics
                    f.write(f"Mask Coverage: {mask_coverage:.2f}%\n")
                
                print(f"Prediction saved to {output_path} and {text_path}")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Save summary results to a single file
    summary_path = os.path.join(output_dir, 'summary_results.txt')
    with open(summary_path, 'w') as f:
        f.write("BREAST ULTRASOUND ANALYSIS RESULTS\n")
        f.write("=================================\n\n")
        
        for result in results_data:
            f.write(f"File: {result['filename']}\n")
            f.write(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.4f})\n")
            f.write(f"Mask Coverage: {result['mask_coverage']:.2f}%\n")
            f.write("Class Probabilities:\n")
            for name, prob in result['class_probs'].items():
                f.write(f"  - {name}: {prob:.4f}\n")
            f.write("\n" + "-"*50 + "\n\n")
            
    print(f"Summary results saved to {summary_path}")
    
    # Return results_data for potential further processing
    return results_data

def load_test_dataset(processed_dir, size=256):
    """Load the test dataset from the preprocessed directory"""
    test_images = []
    test_masks = []
    test_labels = []
    
    # Map class names to integer labels
    class_map = {'normal': 0, 'benign': 1, 'malignant': 2}
    
    split = 'test'
    
    for class_name in ['normal', 'benign', 'malignant']:
        class_dir = os.path.join(processed_dir, split, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist. Skipping.")
            continue
        
        print(f"Loading {split} data from {class_dir}")
        
        # Get all image files (excluding masks)
        image_files = [f for f in glob(os.path.join(class_dir, '*.png')) if '_mask' not in f]
        
        for img_path in tqdm(image_files, desc=f"Loading {split}/{class_name}"):
            img_name = os.path.basename(img_path).split('.')[0]
            mask_path = os.path.join(class_dir, f"{img_name}_mask.png")
            
            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found for {img_path}. Skipping.")
                continue
            
            # Load and preprocess image and mask
            image = load_image(img_path, size)
            if image is None:
                continue
            
            mask = load_image(mask_path, size)
            if mask is None:
                continue
            
            # Binarize mask
            mask = (mask > 0.5).astype(np.float32)
            
            test_images.append(image)
            test_masks.append(mask)
            test_labels.append(class_map[class_name])
    
    if len(test_images) == 0:
        raise ValueError("No valid test images loaded. Please check your dataset.")
    
    # Convert to numpy arrays and add channel dimension
    test_images = np.array(test_images)[:, np.newaxis, :, :]  # (N, 1, H, W)
    test_masks = np.array(test_masks)[:, np.newaxis, :, :]    # (N, 1, H, W)
    test_labels = np.array(test_labels)
    
    print(f"Loaded {len(test_images)} test images, {len(test_masks)} masks, {len(test_labels)} labels")
    print(f"Test class distribution: {np.bincount(test_labels)}")
    
    return test_images, test_masks, test_labels

def main(args):
    # Configuration
    IMAGE_SIZE = args.image_size
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    DATASET_PATH = args.dataset_path
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Check if the dataset has been preprocessed by data_preparation.py
    processed_dir = os.path.join(os.path.dirname(DATASET_PATH.split('*')[0]), 'processed')
    
    if os.path.exists(processed_dir):
        print(f"Using preprocessed dataset from {processed_dir}")
        # Load the preprocessed data - now keeps train and val separate
        (train_images, train_masks, train_labels), (val_images, val_masks, val_labels) = load_processed_data(processed_dir, size=IMAGE_SIZE)
        
        # Create datasets with the separate train and val data
        train_dataset = BreastUltrasoundDataset(train_images, train_masks, train_labels)
        val_dataset = BreastUltrasoundDataset(val_images, val_masks, val_labels)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        # Load test data for final evaluation
        test_images, test_masks, test_labels = load_test_dataset(processed_dir, size=IMAGE_SIZE)
        test_dataset = BreastUltrasoundDataset(test_images, test_masks, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    else:
        print("Preprocessed dataset not found. Processing data on-the-fly...")
        # Load data
        print("Loading dataset...")
        try:
            images, masks, labels = load_data(DATASET_PATH, size=IMAGE_SIZE, use_processed=False)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please make sure the dataset path is correct and contains breast ultrasound images.")
            return
        
        # Split the data
        X_train, X_val, y_train, y_val, labels_train, labels_val = train_test_split(
            images, masks, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        
        # Create datasets
        train_dataset = BreastUltrasoundDataset(X_train, y_train, labels_train)
        val_dataset = BreastUltrasoundDataset(X_val, y_val, labels_val)
        test_dataset = val_dataset  # Use validation set as test set for consistency
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    model = MultiTaskUNet(in_channels=1, out_channels=1, num_classes=3)
    model = model.to(device)
    
    print("Model summary:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    history, _ = train_model(model, train_loader, val_loader, device, NUM_EPOCHS, 'models', patience=args.patience)
    
    # Load best model for evaluation
    best_model = MultiTaskUNet(in_channels=1, out_channels=1, num_classes=3)
    checkpoint = torch.load('models/best_model.pth', map_location=device)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model = best_model.to(device)
    
    # Evaluate model using test data
    print("Evaluating best model on test data...")
    evaluate_model(best_model, test_loader, device, save_path='results')
    
    print("Training and evaluation completed!")

def live_demo(args):
    """Run live demonstration on new images"""
    # Configuration
    IMAGE_SIZE = args.image_size
    MODEL_PATH = 'models/best_model.pth'
    
    # Determine device (prefer CPU for live demo to ensure compatibility)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Ask for the input image path
    print("Live Demonstration Mode")
    image_path = input("Enter the path to test images (can be a directory, file, or pattern): ")
    
    # Create output directory
    output_dir = 'live_demo_results'
    
    # Load model
    model = MultiTaskUNet(in_channels=1, out_channels=1, num_classes=3)
    
    try:
        # Load model
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        print(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Make predictions
    results = predict_on_new_images(model, image_path, output_dir, device, IMAGE_SIZE)
    
    if results:
        print(f"Predictions for {len(results)} images saved to {output_dir}")
    else:
        print("No valid predictions were made")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-Task U-Net for Breast Ultrasound Segmentation and Classification')
    
    # Add mode argument
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'live_demo'], 
                        help='Mode: train or live_demo')
    
    # Add other arguments
    parser.add_argument('--dataset_path', type=str, default="Dataset_BUSI_with_GT/*/*",
                        help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for training')
    parser.add_argument('--patience', type=int, default=3, 
                        help='Patience for early stopping')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size to resize images to')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        main(args)
    elif args.mode == 'live_demo':
        live_demo(args)