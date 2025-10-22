# Breast Ultrasound Image Segmentation and Classification

## Project Overview

This project implements a deep learning solution for automated breast cancer detection and analysis using ultrasound imaging. The system employs a multi-task U-Net architecture that simultaneously performs two critical tasks: segmenting breast lesions in ultrasound images and classifying them into three categories - normal, benign, or malignant. This dual approach enables comprehensive analysis of breast ultrasound scans, providing both localization and diagnostic information that can assist radiologists in early breast cancer detection.

The implementation is based on a classic U-Net architecture that has been extended with multi-task learning capabilities, allowing the model to learn both segmentation and classification tasks simultaneously. This approach is particularly powerful in medical imaging applications where both localization of abnormalities and their characterization are equally important for clinical decision-making. The model processes grayscale ultrasound images and outputs both a segmentation mask highlighting the lesion boundaries and a classification prediction indicating the likelihood of malignancy.

## Why This Project Matters

Breast cancer remains one of the most common causes of death among women worldwide, with early detection being crucial for improving survival rates. Traditional ultrasound examination relies heavily on the expertise of radiologists, which can be subjective and time-consuming. This project addresses these challenges by developing an automated system that can analyze breast ultrasound images with high accuracy, potentially serving as a second opinion tool or assisting less experienced practitioners.

The significance of this work extends beyond mere automation. By providing both segmentation and classification in a single framework, the system offers a comprehensive diagnostic tool that can help reduce diagnostic variability between different radiologists and institutions. The multi-task learning approach ensures that the segmentation and classification tasks inform each other, leading to more robust and clinically relevant predictions. Furthermore, the system's ability to process images quickly makes it suitable for integration into clinical workflows where rapid assessment is often required.

## Dataset Information

The project utilizes the Breast Ultrasound Images Dataset (BUSI), a comprehensive collection of breast ultrasound images specifically designed for computer-aided diagnosis research. The dataset was collected from 600 female patients aged between 25 and 75 years old at baseline in 2018. It contains 780 ultrasound images with an average resolution of 500×500 pixels, categorized into three classes:

- **Normal**: 133 images showing healthy breast tissue
- **Benign**: 437 images with benign lesions (non-cancerous growths)
- **Malignant**: 210 images with malignant lesions (cancerous tumors)

Each image in the benign and malignant categories includes corresponding ground truth segmentation masks that outline the lesion boundaries. The dataset represents real clinical scenarios and includes various imaging conditions, making it suitable for developing robust deep learning models.

The BUSI dataset is particularly valuable for medical image analysis research because it provides both classification labels and pixel-level segmentation masks, enabling the development of comprehensive computer-aided diagnosis systems. The images were acquired using different ultrasound machines and settings, reflecting the variability encountered in real clinical practice. This diversity helps ensure that models trained on this dataset can generalize to different imaging conditions and equipment types.

## Data Preprocessing and Statistics

The preprocessing pipeline transforms the raw dataset into a format suitable for deep learning training. After preprocessing, the dataset contains 771 unique images (9 duplicates were removed). The images are resized to 256×256 pixels for computational efficiency while maintaining diagnostic quality. The dataset is split into training (80%), validation (10%), and test (10%) sets with the following distribution:

- **Training Set**: 616 images (346 benign, 167 malignant, 103 normal)
- **Validation Set**: 77 images (43 benign, 21 malignant, 13 normal)
- **Test Set**: 78 images (44 benign, 21 malignant, 13 normal)

The preprocessing also handles multiple masks per image (17 cases found) by combining them into single binary masks, ensuring consistent data format for the segmentation task.

The preprocessing steps are carefully designed to preserve the diagnostic information while preparing the data for efficient neural network training. Image resizing uses bicubic interpolation to maintain image quality, and the train-validation-test split is stratified to ensure balanced class distributions across all sets. The duplicate removal process uses perceptual hashing to identify visually similar images, preventing data leakage and ensuring the model's ability to generalize to new cases.

## Model Architecture and Performance

### Architecture
The model implements a classic Multi-Task U-Net architecture with:
- **Encoder**: Four convolutional blocks with max pooling for feature extraction
- **Bottleneck**: Additional convolutional layers for deep feature processing
- **Decoder**: Four upsampling blocks with skip connections for precise segmentation
- **Segmentation Head**: Final convolutional layer producing binary lesion masks
- **Classification Head**: Fully connected layers for three-class classification

The multi-task design allows simultaneous optimization of both segmentation and classification objectives, with configurable loss weights (85% segmentation, 15% classification) to balance the two tasks.

The U-Net architecture was chosen for its proven effectiveness in medical image segmentation tasks. The encoder progressively reduces spatial dimensions while increasing feature depth, capturing both local and global image features. The decoder reconstructs the spatial information through transposed convolutions and skip connections, which concatenate features from the encoder at corresponding scales. This design allows the network to leverage both high-resolution local features and abstract global context for accurate segmentation.

### Performance Metrics

The model achieves strong performance on both tasks:

**Segmentation Performance:**
- Mean Intersection over Union (IoU): 0.5387
- Dice Coefficient: 0.6320
- Pixel Accuracy: 0.9600

**Classification Performance:**
- Overall Accuracy: 0.7179
- Precision: 0.7475
- Recall: 0.7179
- F1 Score: 0.7069

**Class-wise Classification Results:**
- **Normal**: Precision 1.0000, Recall 0.3846, F1 0.5556 (13 samples)
- **Benign**: Precision 0.7115, Recall 0.8409, F1 0.7708 (44 samples)
- **Malignant**: Precision 0.6667, Recall 0.6667, F1 0.6667 (21 samples)

The model shows particularly strong performance in identifying benign lesions and achieves perfect precision for normal cases, demonstrating its potential clinical utility.

The performance metrics indicate that the model performs well on the segmentation task, with high pixel accuracy suggesting good background discrimination. The classification performance shows strong results for benign cases, which is clinically important as these represent the majority of breast lesions. The conservative approach to normal case classification (high precision, lower recall) is preferable in screening scenarios where missing a potential malignancy is more costly than false positives.

## System Requirements

**Hardware Configuration (Tested):**
- CPU: Intel i5-13600KF
- GPU: NVIDIA RTX 4070
- RAM: 32GB
- OS: Windows 11


## Setup and Installation

### **IMPORTANT: Dataset and Model File Locations**

Before running any commands, you must place the downloaded files in the correct locations:

**Dataset Location:**
- Download the BUSI dataset from: [Original Source](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)
- **Extract and place the dataset in:** `Dataset_BUSI_with_GT/` (in the root directory of this project)
- The folder should contain subfolders: `benign/`, `malignant/`, `normal/`

**Model Location:**
- Download the pre-trained model from: [Google Drive Link](https://drive.google.com/file/d/1qBA3834GsQeYD6fc1ZL-u0XotzZnOEWQ/view?usp=sharing)
- **Place the model file at:** `models/best_model.pth`

**Example Project Structure After Setup:**
```
breast-ultrasound-segmentations/
├── Dataset_BUSI_with_GT/          # ← PLACE DATASET HERE
│   ├── benign/
│   ├── malignant/
│   └── normal/
├── models/
│   └── best_model.pth            # ← PLACE MODEL HERE
├── data_preparation.py
├── multitask_unet.py
└── README.md
```

### 1. Clone Repository
```bash
git clone https://github.com/SiamFS/breast-ultrasound-segmentations.git
cd breast-ultrasound-segmentations
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset and Model
Due to GitHub file size limitations, download these separately:

**Dataset:**
- Primary: [Original BUSI Dataset](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)
- Alternative: [Kaggle Mirror](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

** File Placement:** See the **IMPORTANT** section at the top of this setup guide for exact file locations.

## Usage Instructions

** Reminder:** Ensure dataset is in `Dataset_BUSI_with_GT/` and model is in `models/best_model.pth` before proceeding.

### Step 1: Data Preparation
```bash
python data_preparation.py --dataset_path Dataset_BUSI_with_GT
```

This script:
- Resizes images to 256×256 pixels
- Creates binary segmentation masks
- Handles multiple masks per image
- Removes duplicate images
- Splits data into train/validation/test sets (80/10/10)
- Generates dataset visualizations

**Verify processed dataset:**
```bash
python data_preparation.py --dataset_path Dataset_BUSI_with_GT --verify
```

### Step 2: Train Model
```bash
python multitask_unet.py --mode train --dataset_path Dataset_BUSI_with_GT --epochs 50 --batch_size 16
```

**Optional arguments:**
- `--image_size`: Image dimension (default: 256)
- `--patience`: Early stopping patience (default: 3)

Training results save to `models/` and evaluation metrics to `results/`.

### Step 3: Live Demo
```bash
python multitask_unet.py --mode live_demo
```

When prompted, provide:
- Single image: `path/to/image.png`
- Directory: `path/to/images/`
- Pattern: `path/to/images/*.png`

Results save to `live_demo_results/` with predictions, confidence scores, and visualizations.

## Project Structure

```
├── data_preparation.py          # Data preprocessing and augmentation
├── multitask_unet.py           # Model architecture, training, evaluation
├── models/                     # Saved models and training metrics
│   ├── best_model.pth         # Pre-trained model weights
│   └── metrics.csv            # Training progress metrics
├── results/                    # Evaluation results and plots
│   └── evaluation_results.txt # Final performance metrics
├── live_demo_results/          # Live prediction results
├── test_images/               # Test images for demonstration
├── requirements.txt           # Python dependencies
├── data_result.txt            # Data preprocessing summary
└── README.md                  # This documentation
```

## Dependencies

Core libraries include:
- **PyTorch**: Deep learning framework with CUDA support
- **OpenCV**: Image processing and computer vision
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **Scikit-learn**: Machine learning utilities
- **Tqdm**: Progress bars

## Methodology

### Multi-Task Learning Approach

The project employs a multi-task learning paradigm where a single neural network simultaneously learns to:
1. **Segment lesions**: Pixel-level binary classification to identify lesion boundaries
2. **Classify lesions**: Image-level multi-class classification (normal/benign/malignant)

This approach offers several advantages:
- **Parameter efficiency**: Shared feature extraction reduces model complexity
- **Improved generalization**: Joint learning captures correlations between tasks
- **Computational efficiency**: Single forward pass for both predictions
- **Clinical relevance**: Provides comprehensive diagnostic information

The multi-task learning framework is particularly well-suited for medical imaging applications where both localization and characterization of abnormalities are clinically important. By sharing the encoder layers between segmentation and classification tasks, the model can learn rich feature representations that are beneficial for both objectives. The segmentation task provides spatial context that can improve classification accuracy, while the classification task provides semantic guidance that can refine segmentation boundaries.


### Data Processing Pipeline

1. **Image Loading**: PNG format with variable resolutions (average 500×500px)
2. **Grayscale Conversion**: RGB to single-channel grayscale
3. **Resize**: Bicubic interpolation to 256×256 pixels
4. **Mask Processing**: Binary thresholding and morphological operations
5. **Normalization**: Pixel values scaled to [0,1] range
6. **Channel Addition**: Single-channel images expanded to 3D tensors (1×256×256)

The data processing pipeline is designed to maintain image quality while preparing data for efficient neural network processing. Grayscale conversion reduces computational complexity while preserving the diagnostic information in ultrasound images. The resizing operation uses bicubic interpolation to minimize artifacts that could affect segmentation accuracy. Mask processing ensures binary segmentation masks are properly formatted, and normalization standardizes the input range for stable training.

### Model Architecture Details

**Convolutional Block Structure:**
```
Conv2D (3×3, padding=1) → BatchNorm → ReLU → Conv2D (3×3, padding=1) → BatchNorm → ReLU
```

**U-Net Encoder Path:**
- Block 1: 64 filters, input → 128×128
- Block 2: 128 filters, 128×128 → 64×64
- Block 3: 256 filters, 64×64 → 32×32
- Block 4: 512 filters, 32×32 → 16×16

**Bottleneck:**
- Block 5: 1024 filters, 16×16 → 16×16

**U-Net Decoder Path:**
- Up1: TransposeConv2D, 1024 → 512 filters, 16×16 → 32×32
- Block 6: 512 filters (concatenated with encoder features)
- Up2: TransposeConv2D, 512 → 256 filters, 32×32 → 64×64
- Block 7: 256 filters (concatenated with encoder features)
- Up3: TransposeConv2D, 256 → 128 filters, 64×64 → 128×128
- Block 8: 128 filters (concatenated with encoder features)
- Up4: TransposeConv2D, 128 → 64 filters, 128×128 → 256×256
- Block 9: 64 filters (concatenated with encoder features)

**Segmentation Head:**
- Final Conv2D: 64 → 1 filter, 1×1 kernel
- Sigmoid activation for binary segmentation

**Classification Head:**
- Global Average Pooling: 1024 features
- FC1: 1024 → 512 neurons, ReLU, Dropout(0.5)
- FC2: 512 → 256 neurons, ReLU, Dropout(0.3)
- FC3: 256 → 3 neurons (no activation, logits)

The U-Net architecture follows the classic design with symmetric encoder-decoder paths and skip connections. Each convolutional block consists of two 3×3 convolutions with batch normalization and ReLU activation, providing rich feature extraction at each scale. The skip connections preserve spatial information from the encoder, enabling precise localization in the decoder. The bottleneck layer processes the most abstract features before reconstruction. The segmentation head produces pixel-level predictions, while the classification head aggregates global features for image-level classification.

### Training Strategy

**Optimization:**
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-5)
- **Learning Rate Schedule**: Cosine Annealing (T_max=50, eta_min=1e-6)
- **Early Stopping**: Patience=3 epochs, monitored on validation loss

**Regularization Techniques:**
- Batch Normalization in all convolutional blocks
- Dropout (0.5, 0.3) in classification head
- Weight decay (L2 regularization)
- Early stopping to prevent overfitting

**Mixed Precision Training:**
- Automatic mixed precision (AMP) for faster training
- Gradient scaling to prevent underflow

The training strategy employs several techniques to ensure stable and efficient learning. Adam optimizer with weight decay provides adaptive learning rates and regularization. Cosine annealing gradually reduces the learning rate, allowing fine-tuning in later epochs. Early stopping prevents overfitting by monitoring validation loss. Mixed precision training accelerates computation while maintaining accuracy.

### Evaluation Metrics

**Segmentation Metrics:**
- **IoU (Intersection over Union)**: Measures overlap between predicted and ground truth masks
- **Dice Coefficient**: Harmonic mean of precision and recall for segmentation
- **Pixel Accuracy**: Percentage of correctly classified pixels

**Classification Metrics:**
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed class-wise performance

The evaluation metrics provide comprehensive assessment of both segmentation and classification performance. IoU and Dice coefficient measure spatial accuracy of lesion segmentation, while pixel accuracy assesses overall image-level correctness. Classification metrics evaluate diagnostic accuracy across different lesion types, with the confusion matrix providing detailed insights into classification errors.

## Implementation Details

### Code Structure

**`data_preparation.py`:**
- Dataset loading and preprocessing
- Duplicate detection using perceptual hashing
- Train/validation/test splitting with stratification
- Data integrity verification
- Visualization generation

**`multitask_unet.py`:**
- Model architecture definition
- Training loop with progress tracking
- Evaluation and metrics calculation
- Live demo functionality
- Results visualization and saving

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image Size | 256×256 | Input resolution |
| Batch Size | 16 | Training batch size |
| Learning Rate | 0.001 | Initial learning rate |
| Epochs | 50 | Maximum training epochs |
| Patience | 3 | Early stopping patience |
| Segmentation Weight | 0.85 | Loss weight for segmentation |
| Classification Weight | 0.15 | Loss weight for classification |
| Dropout (FC1) | 0.5 | Dropout in first FC layer |
| Dropout (FC2) | 0.3 | Dropout in second FC layer |
| Weight Decay | 1e-5 | L2 regularization strength |

## Training Procedure

### Data Preparation Phase
1. Download and extract BUSI dataset
2. Run preprocessing: `python data_preparation.py --dataset_path Dataset_BUSI_with_GT`
3. Verify preprocessing: `python data_preparation.py --dataset_path Dataset_BUSI_with_GT --verify`

### Model Training Phase
1. Initialize model with Xavier/Kaiming weight initialization
2. Set up data loaders with appropriate batch sizes
3. Configure optimizer, scheduler, and loss functions
4. Train for up to 50 epochs with early stopping
5. Save best model based on validation loss
6. Generate training progress plots

### Evaluation Phase
1. Load best model checkpoint
2. Evaluate on held-out test set
3. Calculate comprehensive metrics
4. Generate confusion matrix and performance plots
5. Save detailed results to files

## Results Analysis

### Segmentation Performance Analysis

The multi-task U-Net model demonstrates moderate segmentation performance with an Intersection over Union (IoU) score of 0.5387 and a Dice coefficient of 0.6320. These metrics indicate that the model can identify approximately 54% of the true lesion boundaries correctly, which is a reasonable performance for medical image segmentation tasks where precise boundary detection is challenging due to the irregular shapes and varying contrast of breast lesions in ultrasound images.

The high pixel accuracy of 0.9600 reveals an interesting characteristic of the model's performance: it excels at correctly identifying background pixels (non-lesion areas) but struggles with the precise delineation of lesion boundaries. This is a common phenomenon in medical imaging segmentation where the majority of pixels belong to the background class, making pixel accuracy a less informative metric compared to region-based metrics like IoU and Dice. The model's conservative approach to lesion detection prioritizes precision over recall, meaning it tends to under-segment lesions rather than including excessive background pixels within the predicted lesion masks.

### Classification Performance Analysis

The classification component of the multi-task model achieves an overall accuracy of 71.79%, successfully categorizing breast ultrasound images into normal, benign, and malignant classes. The model's performance shows strong discriminative ability, particularly for benign cases with an F1-score of 0.7708, indicating balanced precision and recall for this class. The perfect precision (1.0000) for normal cases suggests the model is highly confident when predicting normal tissue, though the low recall (0.3846) indicates that some normal cases are being misclassified as abnormal.

This classification behavior reflects a clinically acceptable trade-off: in breast cancer screening, it is generally preferable to have false positives (normal cases flagged as abnormal) rather than false negatives (malignant cases missed). The model's conservative approach to normal classification ensures that potentially suspicious cases receive further clinical evaluation, which aligns with the sensitivity requirements of medical diagnostic systems.

**Class-wise Performance Insights:**
- **Normal Class**: The high precision (1.0000) demonstrates the model's ability to correctly identify truly normal tissue when it makes a normal prediction. However, the low recall suggests that some normal images exhibit features that the model associates with abnormalities, possibly due to imaging artifacts or subtle findings that require expert interpretation.
- **Benign Class**: This class shows the best overall performance (F1=0.7708), likely benefiting from the largest training set size (437 images) and clearer distinguishing features compared to malignant lesions.
- **Malignant Class**: With balanced precision (0.6667) and recall (0.6667), the model demonstrates reasonable performance for the most critical class in clinical decision-making, though there remains room for improvement in detecting all malignant cases.

### Training Dynamics and Model Convergence

The training process reveals important insights into the model's learning behavior and convergence patterns. The segmentation metrics show steady improvement throughout the training epochs, with both IoU and Dice coefficients gradually increasing as the model learns to better capture the complex spatial relationships within ultrasound images. This progressive improvement suggests that the U-Net architecture effectively learns hierarchical features from the encoder-decoder structure, with skip connections helping preserve spatial information during upsampling.

The classification accuracy stabilizes after approximately 10 epochs, reaching a plateau that indicates the model has learned the discriminative features necessary for tissue classification. The use of early stopping at epoch 12, triggered by validation loss monitoring, prevents overfitting and ensures the model generalizes well to unseen data. The absence of significant overfitting is evidenced by the close alignment between training and validation performance curves, suggesting that the regularization techniques (dropout, batch normalization) and data augmentation strategies are effective.

The multi-task learning approach appears beneficial, as both segmentation and classification tasks improve simultaneously. The shared encoder likely learns generic ultrasound image features that are useful for both tasks, while the task-specific decoders specialize in their respective objectives. This joint optimization may contribute to the model's overall robustness and clinical utility.

## Limitations and Challenges

### Dataset Limitations
- **Class Imbalance**: Normal class underrepresented (133 vs 437 benign)
- **Limited Sample Size**: Only 780 images from single institution
- **No External Validation**: Tested only on BUSI dataset
- **Annotation Quality**: Manual segmentation may have inter-observer variability

### Technical Limitations
- **Resolution Constraints**: 256×256 may lose fine details
- **Single Modality**: Ultrasound only, no multimodal fusion
- **Binary Segmentation**: Cannot distinguish lesion subtypes
- **Computational Requirements**: Requires GPU for practical training times

### Clinical Limitations
- **Not FDA Approved**: Not validated for clinical use
- **Operator Dependent**: Ultrasound quality varies with technician skill
- **Population Bias**: Trained on specific demographic (Egyptian patients)
- **No Temporal Analysis**: Cannot track lesion changes over time

## Future Improvements

Potential enhancements include:
- Integration with larger datasets
- Cross-validation for robust evaluation
- Attention mechanisms in U-Net architecture
- Ensemble methods for improved accuracy
- Real-time processing optimization
- Clinical validation studies
