import os
import numpy as np
import pandas as pd
import cv2
import shutil
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import datetime
from sklearn.model_selection import train_test_split
import hashlib

# Global configuration
IMAGE_SIZE = 256
RANDOM_SEED = 42
LOG_FILE = "data_result.txt"

def calculate_image_hash(img):
    """Calculate a perceptual hash of an image for better duplicate detection"""
    # Resize image to 8x8
    img_small = cv2.resize(img, (8, 8))
    # Convert to grayscale if not already
    if len(img_small.shape) > 2:
        img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    # Compute the mean value
    mean_val = np.mean(img_small)
    # Create binary hash
    hash_img = (img_small > mean_val).flatten().astype(int)
    # Convert binary hash to hex string
    hex_hash = ''.join([str(i) for i in hash_img])
    return hex_hash

def preprocess_dataset(dataset_path='Dataset_BUSI_with_GT'):
    print("\nPreprocessing dataset...")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} does not exist.")
        return
    
    # Create processed directory
    processed_dir = os.path.join(dataset_path, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Use pandas to track all images
    data = []
    
    # Find all images and masks
    for class_name in ['benign', 'malignant', 'normal']:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory {class_dir} not found.")
            continue
        
        # Create directory for processed data
        processed_class_dir = os.path.join(processed_dir, class_name)
        os.makedirs(processed_class_dir, exist_ok=True)
        
        # Get all image files
        image_files = glob(os.path.join(class_dir, '*.png'))
        image_files.extend(glob(os.path.join(class_dir, '*.jpg')))
        image_files = [f for f in image_files if '_mask' not in f]
        
        for img_path in tqdm(image_files, desc=f"Processing {class_name} images"):
            img_name = os.path.basename(img_path).split('.')[0]
            
            # Find all mask files for this image
            mask_files = glob(os.path.join(class_dir, f"{img_name}_mask*.png"))
            
            # Load and resize image with error handling
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Cannot read image {img_path}. Skipping.")
                    continue
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
                
            # Calculate image hash for duplicate detection
            # Using a perceptual hash for better similarity detection
            img_hash = calculate_image_hash(img)
            
            # Resize image with error handling
            try:
                img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            except Exception as e:
                print(f"Error resizing image {img_path}: {e}")
                continue
            
            # Process masks
            has_mask = len(mask_files) > 0
            multiple_masks = len(mask_files) > 1
            
            if has_mask:
                # Combine masks
                combined_mask = None
                
                for mask_path in mask_files:
                    try:
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask is None:
                            print(f"Warning: Cannot read mask {mask_path}. Skipping.")
                            continue
                        
                        # Binarize and resize
                        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                        mask_resized = cv2.resize(mask_binary, (IMAGE_SIZE, IMAGE_SIZE))
                        
                        if combined_mask is None:
                            combined_mask = mask_resized
                        else:
                            combined_mask = cv2.bitwise_or(combined_mask, mask_resized)
                    except Exception as e:
                        print(f"Error processing mask {mask_path}: {e}")
                        continue
                
                # If all mask processing failed, create empty mask
                if combined_mask is None:
                    combined_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
                    has_mask = False
            else:
                # Create empty mask
                combined_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            
            # Save processed image and mask
            img_out_path = os.path.join(processed_class_dir, f"{img_name}.png")
            mask_out_path = os.path.join(processed_class_dir, f"{img_name}_mask.png")
            
            cv2.imwrite(img_out_path, img_resized)
            cv2.imwrite(mask_out_path, combined_mask)
            
            # Verify the files were saved correctly
            if not os.path.exists(img_out_path) or not os.path.exists(mask_out_path):
                print(f"Warning: Failed to save {img_out_path} or {mask_out_path}")
                continue
                
            # Verify file integrity by reading them back
            try:
                check_img = cv2.imread(img_out_path)
                check_mask = cv2.imread(mask_out_path, cv2.IMREAD_GRAYSCALE)
                if check_img is None or check_mask is None:
                    print(f"Warning: Verification failed for {img_out_path} or {mask_out_path}")
                    continue
            except Exception as e:
                print(f"Error verifying saved files for {img_name}: {e}")
                continue
            
            # Add to dataframe
            data.append({
                'class': class_name,
                'image_name': img_name,
                'original_path': img_path,
                'processed_path': img_out_path,
                'mask_path': mask_out_path,
                'has_mask': has_mask,
                'multiple_masks': multiple_masks,
                'image_hash': img_hash
            })
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        print("Error: No valid images processed. Check your dataset.")
        return None, None
    
    # Initialize is_duplicate column with False for all rows
    df['is_duplicate'] = False
    
    # Find duplicates by hash - group by hash to find similar images
    hash_counts = df['image_hash'].value_counts()
    duplicate_hashes = hash_counts[hash_counts > 1].index.tolist()
    
    duplicates_df = pd.DataFrame()
    for hash_val in duplicate_hashes:
        dups = df[df['image_hash'] == hash_val].copy()
        # Keep the first one
        keep_idx = dups.index[0]
        # Mark the rest as duplicates
        dup_indices = dups.index[1:]
        df.loc[dup_indices, 'is_duplicate'] = True
        # Add to duplicates dataframe
        duplicates_df = pd.concat([duplicates_df, df.loc[dup_indices]])
    
    # Remove duplicates, keeping first occurrence
    df = df[~df['is_duplicate']].copy()
    
    # Create train/val/test splits (80/10/10)
    df['split'] = 'none'  # Initialize column
    
    for cls in df['class'].unique():
        class_df = df[df['class'] == cls]
        
        # Get indices for this class
        class_indices = class_df.index.tolist()
        
        # Split indices
        train_indices, temp_indices = train_test_split(
            class_indices, train_size=0.8, random_state=RANDOM_SEED
        )
        
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=0.5, random_state=RANDOM_SEED
        )
        
        # Assign splits
        df.loc[train_indices, 'split'] = 'train'
        df.loc[val_indices, 'split'] = 'val'
        df.loc[test_indices, 'split'] = 'test'
    
    # Copy files to split directories - with progress tracking
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        
        for cls in split_df['class'].unique():
            # Create directory
            split_dir = os.path.join(processed_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            
            # Get files for this class and split
            class_split_df = split_df[split_df['class'] == cls]
            
            print(f"Copying {len(class_split_df)} {cls} images to {split} directory...")
            for _, row in tqdm(class_split_df.iterrows(), total=len(class_split_df)):
                # Get image name
                img_name = row['image_name']
                
                # Copy image and mask
                src_img = row['processed_path']
                src_mask = row['mask_path']
                
                dst_img = os.path.join(processed_dir, split, cls, f"{img_name}.png")
                dst_mask = os.path.join(processed_dir, split, cls, f"{img_name}_mask.png")
                
                try:
                    shutil.copy2(src_img, dst_img)
                    shutil.copy2(src_mask, dst_mask)
                    
                    # Verify files were copied correctly
                    if not os.path.exists(dst_img) or not os.path.exists(dst_mask):
                        print(f"Warning: Failed to copy {img_name} to {split}/{cls}")
                        continue
                        
                    # Check file integrity
                    check_img = cv2.imread(dst_img)
                    check_mask = cv2.imread(dst_mask, cv2.IMREAD_GRAYSCALE)
                    if check_img is None or check_mask is None:
                        print(f"Warning: Verification failed for copied files {img_name}")
                        continue
                except Exception as e:
                    print(f"Error copying {img_name} to {split}/{cls}: {e}")
                    continue
    
    # Calculate statistics
    stats = {
        'initial': {
            'benign': len(df[df['class'] == 'benign']),
            'malignant': len(df[df['class'] == 'malignant']),
            'normal': len(df[df['class'] == 'normal']),
        },
        'issues': {
            'no_mask': len(df[df['has_mask'] == False]),
            'multiple_masks': len(df[df['multiple_masks'] == True]),
            'duplicates': len(duplicates_df),
        },
        'splits': {
            'train': df[df['split'] == 'train']['class'].value_counts().to_dict(),
            'val': df[df['split'] == 'val']['class'].value_counts().to_dict(),
            'test': df[df['split'] == 'test']['class'].value_counts().to_dict(),
        }
    }
    
    # Add totals
    stats['initial']['total'] = stats['initial']['benign'] + stats['initial']['malignant'] + stats['initial']['normal']
    
    for split in ['train', 'val', 'test']:
        total = sum(stats['splits'][split].values())
        stats['splits'][split]['total'] = total
    
    # Write results to log file
    with open(LOG_FILE, 'w') as f:
        f.write("INITIAL DATASET\n")
        f.write(f"Benign: {stats['initial']['benign']} images\n")
        f.write(f"Malignant: {stats['initial']['malignant']} images\n")
        f.write(f"Normal: {stats['initial']['normal']} images\n")
        f.write(f"Total: {stats['initial']['total']} images\n\n")
        
        f.write("PREPROCESSING STEPS\n")
        f.write(f"Images without masks: {stats['issues']['no_mask']}\n")
        f.write(f"Images with multiple masks: {stats['issues']['multiple_masks']}\n")
        f.write(f"Duplicate images removed: {stats['issues']['duplicates']}\n")
        f.write(f"Images resized to: {IMAGE_SIZE}x{IMAGE_SIZE}\n\n")
        
        f.write("DATA SPLIT (80/10/10)\n")
        f.write("Training set:\n")
        for cls, count in stats['splits']['train'].items():
            if cls != 'total':
                f.write(f"  {cls.capitalize()}: {count} images\n")
        f.write(f"  Total: {stats['splits']['train']['total']} images\n\n")
        
        f.write("Validation set:\n")
        for cls, count in stats['splits']['val'].items():
            if cls != 'total':
                f.write(f"  {cls.capitalize()}: {count} images\n")
        f.write(f"  Total: {stats['splits']['val']['total']} images\n\n")
        
        f.write("Test set:\n")
        for cls, count in stats['splits']['test'].items():
            if cls != 'total':
                f.write(f"  {cls.capitalize()}: {count} images\n")
        f.write(f"  Total: {stats['splits']['test']['total']} images\n\n")
        
        # List duplicate images
        f.write("\nDUPLICATE IMAGES:\n")
        for hash_val in duplicate_hashes:
            dups = df[df['image_hash'] == hash_val]
            kept = dups.iloc[0]['original_path']
            
            # Duplicates from the duplicates_df
            removed_dups = duplicates_df[duplicates_df['image_hash'] == hash_val]['original_path'].tolist()
            
            f.write(f"Kept: {kept}\n")
            for dup in removed_dups:
                f.write(f"  Removed duplicate: {dup}\n")
    
    # Print summary
    print("\nDataset preprocessing completed!")
    print(f"Initial dataset: {stats['initial']['total']} images")
    print(f"Final dataset: {len(df)} images after removing {stats['issues']['duplicates']} duplicates")
    print(f"Training set: {stats['splits']['train']['total']} images")
    print(f"Validation set: {stats['splits']['val']['total']} images")
    print(f"Test set: {stats['splits']['test']['total']} images")
    print(f"Images resized to: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Detailed results saved to '{LOG_FILE}'")
    
    # Create visualizations
    visualize_dataset(processed_dir)
    
    return df, stats

def visualize_dataset(processed_dir):
    """Create visualizations of the processed dataset"""
    # Class distribution
    plt.figure(figsize=(10, 6))
    
    # Count images per class
    classes = ['benign', 'malignant', 'normal']
    counts = []
    
    for class_name in classes:
        image_files = glob(os.path.join(processed_dir, '*', class_name, '*.png'))
        image_files = [f for f in image_files if '_mask' not in f]
        counts.append(len(image_files))
    
    plt.bar(classes, counts, color=['blue', 'red', 'green'])
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    
    for i, count in enumerate(counts):
        plt.text(i, count + 5, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    
    # Split distribution
    plt.figure(figsize=(12, 6))
    
    splits = ['train', 'val', 'test']
    split_counts = []
    
    for split in splits:
        image_files = glob(os.path.join(processed_dir, split, '*', '*.png'))
        image_files = [f for f in image_files if '_mask' not in f]
        split_counts.append(len(image_files))
    
    plt.bar(splits, split_counts, color=['blue', 'green', 'red'])
    plt.title('Data Split Distribution (80/10/10)')
    plt.xlabel('Split')
    plt.ylabel('Number of Images')
    
    for i, count in enumerate(split_counts):
        plt.text(i, count + 5, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig('split_distribution.png')
    
    # Show sample images
    plt.figure(figsize=(15, 10))
    
    for i, class_name in enumerate(['benign', 'malignant', 'normal']):
        # Get a random image from training set
        img_files = glob(os.path.join(processed_dir, 'train', class_name, '*.png'))
        img_files = [f for f in img_files if '_mask' not in f]
        
        if not img_files:
            continue
            
        img_path = np.random.choice(img_files)
        img_name = os.path.basename(img_path).split('.')[0]
        mask_path = os.path.join(os.path.dirname(img_path), f"{img_name}_mask.png")
        
        # Load image and mask with error handling
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Cannot read sample image {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Cannot read sample mask {mask_path}")
                continue
        except Exception as e:
            print(f"Error loading sample image/mask: {e}")
            continue
        
        # Show image, mask and overlay
        plt.subplot(3, 3, i*3 + 1)
        plt.imshow(img)
        plt.title(f'{class_name.capitalize()} - Image')
        plt.axis('off')
        
        plt.subplot(3, 3, i*3 + 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f'{class_name.capitalize()} - Mask')
        plt.axis('off')
        
        plt.subplot(3, 3, i*3 + 3)
        plt.imshow(img)
        plt.imshow(mask, alpha=0.5, cmap='jet')
        plt.title(f'{class_name.capitalize()} - Overlay')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    print("Visualizations saved to 'class_distribution.png', 'split_distribution.png', and 'sample_images.png'")

def verify_dataset_integrity(processed_dir):
    """Verify the integrity of the processed dataset"""
    print("\nVerifying dataset integrity...")
    
    # Check if processed directory exists
    if not os.path.exists(processed_dir):
        print(f"Error: Processed directory {processed_dir} does not exist.")
        return False
    
    # Check if all required directories exist
    for split in ['train', 'val', 'test']:
        for class_name in ['benign', 'malignant', 'normal']:
            split_class_dir = os.path.join(processed_dir, split, class_name)
            if not os.path.exists(split_class_dir):
                print(f"Warning: Directory {split_class_dir} does not exist.")
                continue
            
            # Count images and masks
            images = [f for f in glob(os.path.join(split_class_dir, '*.png')) if '_mask' not in f]
            masks = glob(os.path.join(split_class_dir, '*_mask.png'))
            
            print(f"  {split}/{class_name}: {len(images)} images, {len(masks)} masks")
            
            # Check if every image has a corresponding mask
            for img_path in images:
                img_name = os.path.basename(img_path).split('.')[0]
                mask_path = os.path.join(split_class_dir, f"{img_name}_mask.png")
                
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask missing for {img_path}")
                    continue
                
                # Verify file integrity
                try:
                    img = cv2.imread(img_path)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        print(f"Warning: Cannot read image {img_path}")
                        continue
                        
                    if mask is None:
                        print(f"Warning: Cannot read mask {mask_path}")
                        continue
                        
                    # Check image dimensions
                    if img.shape[0] != IMAGE_SIZE or img.shape[1] != IMAGE_SIZE:
                        print(f"Warning: Image {img_path} has incorrect dimensions: {img.shape[:2]}")
                        continue
                        
                    if mask.shape[0] != IMAGE_SIZE or mask.shape[1] != IMAGE_SIZE:
                        print(f"Warning: Mask {mask_path} has incorrect dimensions: {mask.shape[:2]}")
                        continue
                except Exception as e:
                    print(f"Error verifying {img_path} and {mask_path}: {e}")
                    continue
    
    print("Dataset verification completed.")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess Breast Ultrasound Images for UNet')
    parser.add_argument('--dataset_path', type=str, default='Dataset_BUSI_with_GT',
                      help='Path to the dataset directory')
    parser.add_argument('--verify', action='store_true', 
                      help='Verify the integrity of an already processed dataset')
    
    args = parser.parse_args()
    
    # Create directory structure
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    if args.verify:
        # Verify existing processed dataset
        processed_dir = os.path.join(args.dataset_path, 'processed')
        verify_dataset_integrity(processed_dir)
    else:
        # Preprocess dataset
        preprocess_dataset(args.dataset_path)