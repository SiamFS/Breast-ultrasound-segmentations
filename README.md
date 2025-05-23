# Breast Ultrasound Image Segmentation and Classification

A deep learning project for breast ultrasound image segmentation and classification using a Classic Multi-task U-Net architecture. The model simultaneously performs segmentation of breast lesions and classification into normal, benign, or malignant categories.

## Features

- Automated preprocessing of breast ultrasound images
- Segmentation of breast lesions in ultrasound images
- Classification of breast lesions as normal, benign, or malignant
- Multi-task learning approach with classic U-Net architecture
- Comprehensive performance metrics
- Live demo mode for testing on new images

## System Configuration

This project was developed and tested on the following hardware:
- CPU: Intel i5-13600KF
- GPU: NVIDIA RTX 4070
- RAM: 32GB
- OS: Windows 11

Training with this configuration takes approximately 2-3 hours for 50 epochs with the full dataset.

## Setup Instructions

1. Clone the repository
   ```bash
   git clone https://github.com/SiamFS/breast-ultrasound-segmentation.git
   cd breast-ultrasound-segmentation
   ```

2. Install requirements
   ```bash
   pip install -r requirements.txt
   ```

3. **Important**: Download the dataset and model files (see below)

## Large Files (Dataset and Model)

Due to GitHub file size limitations, the dataset and trained model are not included in this repository. You'll need to download them separately:

### Download Links
- **Dataset**: Download the BUSI Dataset from [Original Source](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) or [Kaggle Link](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- **Pre-trained Model**: [best_model.pth](https://drive.google.com/file/d/1qBA3834GsQeYD6fc1ZL-u0XotzZnOEWQ/view?usp=sharing)

### File Placement
- Place the dataset in the root directory as `Dataset_BUSI_with_GT/`
- Place the model file at `models/best_model.pth`

## GPU Setup

For optimal performance, we recommend using a GPU. Our model uses PyTorch with CUDA support:

a) **Install CUDA and cuDNN**:
   - Download and install [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - Download and install [cuDNN](https://developer.nvidia.com/cudnn)

b) **Install PyTorch with CUDA support**:
   ```bash
   # For CUDA 11.8 (adjust based on your CUDA version)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

c) **Verify GPU installation**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device count: {torch.cuda.device_count()}")
   print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
   ```

**Note**: The model will fall back to CPU if no GPU is available, but training will be significantly slower.

## Dataset

This project is designed to work with the Breast Ultrasound Images Dataset (BUSI). The dataset should be structured as follows:

```
Dataset_BUSI_with_GT/
├── benign/
│   ├── benign (1).png
│   ├── benign (1)_mask.png
│   ├── ...
├── malignant/
│   ├── malignant (1).png
│   ├── malignant (1)_mask.png
│   ├── ...
└── normal/
    ├── normal (1).png
    ├── ...
```

The dataset contains 780 images with corresponding masks for benign and malignant cases.

## Running the Code

Follow these steps in sequence to train and use the model:

### Step 1: Data Preparation

First, run the data preparation script to preprocess the dataset:

```bash
python data_preparation.py --dataset_path Dataset_BUSI_with_GT
```

This will:
- Resize all images to 256×256
- Create binary masks from mask images
- Handle multiple masks per image
- Remove duplicates
- Split the dataset into training (80%), validation (10%), and test (10%) sets
- Generate useful visualizations of your dataset

You can verify an already processed dataset with:
```bash
python data_preparation.py --dataset_path Dataset_BUSI_with_GT --verify
```

### Step 2: Training the Model

Train the multi-task U-Net model:

```bash
python multitask_unet.py --mode train --dataset_path Dataset_BUSI_with_GT --epochs 50 --batch_size 16
```

Optional arguments:
- `--image_size`: Size to resize images (default: 256)
- `--patience`: Early stopping patience (default: 3)

Training results and model checkpoints will be saved in the `models/` directory.
Evaluation results will be saved in the `results/` directory.

### Step 3: Live Demo

Run the model on new images:

```bash
python multitask_unet.py --mode live_demo
```

When prompted, enter the path to your test images. This can be:
- A single image file: `path/to/image.png`
- A directory: `path/to/images/`
- A pattern: `path/to/images/*.png`

Results will be saved in the `live_demo_results/` directory.

**Important Notes:**
- Make sure to run the scripts in the correct order as they depend on each other
- The training process can take several hours depending on your hardware
- GPU acceleration is highly recommended for reasonable training times

## Model Architecture

The model is a Classic Multi-Task U-Net with:
- Encoder pathway with convolutional blocks and max pooling
- Bottleneck layer
- Decoder pathway with skip connections (following the classic U-Net design)
- Segmentation head for lesion segmentation
- Classification head for normal/benign/malignant classification

This classic U-Net architecture is extended with a multi-task approach, allowing it to perform both pixel-wise segmentation and image-level classification simultaneously.

## Performance Metrics

The model is evaluated using:

### Segmentation Metrics:
- Mean IoU (Intersection over Union)
- Dice Coefficient
- Pixel Accuracy

### Classification Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Project Structure

```
├── data_preparation.py       # Data preprocessing script
├── multitask_unet.py         # Model definition, training, and evaluation
├── models/                   # Saved models
├── results/                  # Evaluation results
├── live_demo_results/        # Results from live demo
├── requirements.txt          # Required packages
└── README.md                 # This file
```

## Citation

If you use this code for your research, please cite our work:

```
@software{breast_ultrasound_segmentation,
  author = {Ferdous, Siam},
  title = {Breast Ultrasound Image Segmentation and Classification with Multi-task U-Net},
  year = {2025},
  url = {https://github.com/SiamFS/breast-ultrasound-segmentation}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
