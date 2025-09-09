import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path
import zipfile
import kaggle

def setup_kaggle_credentials():
    """
    Setup Kaggle credentials for dataset download
    Make sure you have kaggle.json in ~/.kaggle/ directory
    """
    print("Please ensure your kaggle.json file is in ~/.kaggle/ directory")
    print("You can download it from: https://www.kaggle.com/settings -> API -> Create New API Token")

def download_dataset():
    """
    Download the tuberculosis dataset from Kaggle
    """
    try:
        # Create data directory
        os.makedirs("data", exist_ok=True)
        
        # Download dataset using kaggle API
        # Replace with the actual dataset identifier from your Kaggle link
        kaggle.api.dataset_download_files('tawsifurrahman/tuberculosis-tb-chest-xray-dataset', 
                                         path='data/', unzip=True)
        print("Dataset downloaded successfully!")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please download the dataset manually from Kaggle and extract to 'data/' folder")

def load_and_explore_dataset(data_path="data"):
    """
    Load and explore the tuberculosis dataset
    """
    print(f"Exploring dataset in {data_path}...")
    
    # List all files and folders in the data directory
    for root, dirs, files in os.walk(data_path):
        level = root.replace(data_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")
    
    return data_path

def preprocess_images(image_path, target_size=(224, 224)):
    """
    Preprocess individual images
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        return image
    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return None

def create_dataset_structure(source_path, output_path="processed_data"):
    """
    Create organized dataset structure for training
    """
    # Create output directories
    train_dir = os.path.join(output_path, "train")
    val_dir = os.path.join(output_path, "validation")
    test_dir = os.path.join(output_path, "test")
    
    # Create class directories
    classes = ["Normal", "Tuberculosis"]
    
    for split in [train_dir, val_dir, test_dir]:
        for cls in classes:
            os.makedirs(os.path.join(split, cls), exist_ok=True)
    
    print(f"Created dataset structure in {output_path}")
    
    # Look for existing folder structure in source
    normal_path = None
    tb_path = None
    
    # Search for image folders
    for root, dirs, files in os.walk(source_path):
        for dir_name in dirs:
            if "normal" in dir_name.lower() or "healthy" in dir_name.lower():
                normal_path = os.path.join(root, dir_name)
            elif "tb" in dir_name.lower() or "tuberculosis" in dir_name.lower():
                tb_path = os.path.join(root, dir_name)
    
    # If structured folders found, use them
    if normal_path and tb_path:
        process_class_folder(normal_path, "Normal", train_dir, val_dir, test_dir)
        process_class_folder(tb_path, "Tuberculosis", train_dir, val_dir, test_dir)
    else:
        # If no structured folders, look for Excel file with labels
        excel_files = []
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if file.endswith(('.xlsx', '.xls', '.csv')):
                    excel_files.append(os.path.join(root, file))
        
        if excel_files:
            process_with_excel_labels(source_path, excel_files[0], train_dir, val_dir, test_dir)
        else:
            print("No structured folders or Excel file found. Manual organization required.")
    
    return output_path

def process_class_folder(class_path, class_name, train_dir, val_dir, test_dir):
    """
    Process images from a class folder and split them
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(class_path) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No images found in {class_path}")
        return
    
    print(f"Processing {len(image_files)} images for class {class_name}")
    
    # Split the data: 70% train, 20% validation, 10% test
    train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.33, random_state=42)
    
    # Copy files to respective directories
    copy_files(class_path, train_files, os.path.join(train_dir, class_name))
    copy_files(class_path, val_files, os.path.join(val_dir, class_name))
    copy_files(class_path, test_files, os.path.join(test_dir, class_name))
    
    print(f"Class {class_name}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

def process_with_excel_labels(source_path, excel_path, train_dir, val_dir, test_dir):
    """
    Process images using Excel file labels - FIXED VERSION
    """
    try:
        # Read Excel file
        if excel_path.endswith('.csv'):
            df = pd.read_csv(excel_path)
        else:
            df = pd.read_excel(excel_path)
        
        print("Excel file columns:", df.columns.tolist())
        print("Sample data:")
        print(df.head())
        
        # FIXED: Use the correct column names for your dataset
        filename_col = 'fname'  # Your dataset uses 'fname' for filenames
        label_col = 'target'    # Your dataset uses 'target' for labels
        
        print(f"Using filename column: {filename_col}")
        print(f"Using label column: {label_col}")
        
        # Check unique labels in your dataset
        unique_labels = df[label_col].unique()
        print(f"Unique labels found: {unique_labels}")
        
        # Counters for tracking
        normal_count = 0
        tb_count = 0
        processed_count = 0
        
        # Process each image according to its label
        for _, row in df.iterrows():
            filename = row[filename_col]
            label = row[label_col]
            
            # Map labels to class names - FIXED for your dataset
            if label == 'no_tb':  # Your dataset uses 'no_tb' for normal cases
                class_name = "Normal"
                normal_count += 1
            elif label == 'tb':   # Your dataset uses 'tb' for tuberculosis cases
                class_name = "Tuberculosis" 
                tb_count += 1
            else:
                print(f"Unknown label '{label}' for file {filename}, skipping...")
                continue
            
            # Find the actual image file
            image_path = find_image_file(source_path, filename)
            if image_path:
                # Determine split randomly (70% train, 20% validation, 10% test)
                rand = np.random.random()
                if rand < 0.7:
                    dest_dir = os.path.join(train_dir, class_name)
                elif rand < 0.9:
                    dest_dir = os.path.join(val_dir, class_name)
                else:
                    dest_dir = os.path.join(test_dir, class_name)
                
                # Copy file
                dest_file = os.path.join(dest_dir, filename)
                shutil.copy2(image_path, dest_file)
                processed_count += 1
            else:
                print(f"Image file not found: {filename}")
        
        print(f"\nDataset processing completed!")
        print(f"Total processed: {processed_count} images")
        print(f"Normal cases: {normal_count}")
        print(f"TB cases: {tb_count}")
        
    except Exception as e:
        print(f"Error processing Excel file: {e}")
        import traceback
        traceback.print_exc()

def find_image_file(source_path, filename):
    """
    Find image file in source directory - ENHANCED VERSION
    """
    # First, try exact match
    for root, dirs, files in os.walk(source_path):
        if filename in files:
            return os.path.join(root, filename)
    
    # If not found, try different extensions
    name_without_ext = os.path.splitext(filename)[0]
    for root, dirs, files in os.walk(source_path):
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            test_filename = f"{name_without_ext}{ext}"
            if test_filename in files:
                return os.path.join(root, test_filename)
            # Also try uppercase extensions
            test_filename_upper = f"{name_without_ext}{ext.upper()}"
            if test_filename_upper in files:
                return os.path.join(root, test_filename_upper)
    
    return None

def copy_files(source_dir, file_list, dest_dir):
    """
    Copy files from source to destination directory
    """
    for filename in file_list:
        source_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(dest_dir, filename)
        try:
            shutil.copy2(source_file, dest_file)
        except Exception as e:
            print(f"Error copying {filename}: {e}")

def generate_dataset_summary(dataset_path):
    """
    Generate summary of the processed dataset
    """
    summary = {}
    total_images = 0
    
    for split in ["train", "validation", "test"]:
        split_path = os.path.join(dataset_path, split)
        split_summary = {}
        
        if os.path.exists(split_path):
            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if os.path.isdir(class_path):
                    image_count = len([f for f in os.listdir(class_path) 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
                    split_summary[class_name] = image_count
                    total_images += image_count
        
        summary[split] = split_summary
    
    print("\n=== Dataset Summary ===")
    print(f"Total images: {total_images}")
    for split, classes in summary.items():
        print(f"\n{split.capitalize()}:")
        for class_name, count in classes.items():
            print(f"  {class_name}: {count} images")
    
    return summary

def verify_dataset_structure():
    """
    Verify the processed dataset structure is correct
    """
    print("\n=== Verifying Dataset Structure ===")
    
    base_path = "processed_data"
    required_structure = {
        "train": ["Normal", "Tuberculosis"],
        "validation": ["Normal", "Tuberculosis"], 
        "test": ["Normal", "Tuberculosis"]
    }
    
    all_good = True
    
    for split, classes in required_structure.items():
        split_path = os.path.join(base_path, split)
        if not os.path.exists(split_path):
            print(f"❌ Missing directory: {split_path}")
            all_good = False
            continue
            
        for class_name in classes:
            class_path = os.path.join(split_path, class_name)
            if not os.path.exists(class_path):
                print(f"❌ Missing directory: {class_path}")
                all_good = False
            else:
                image_count = len([f for f in os.listdir(class_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
                if image_count > 0:
                    print(f"✅ {class_path}: {image_count} images")
                else:
                    print(f"⚠️  {class_path}: No images found")
                    all_good = False
    
    if all_good:
        print("✅ Dataset structure is ready for training!")
    else:
        print("❌ Dataset structure has issues. Please check the errors above.")
    
    return all_good

def main():
    """
    Main function to prepare the tuberculosis dataset
    """
    print("=== Tuberculosis Dataset Preparation ===")
    
    # Step 1: Setup Kaggle credentials
    setup_kaggle_credentials()
    
    # Step 2: Download dataset (comment out if already downloaded)
    # download_dataset()
    
    # Step 3: Load and explore dataset
    data_path = load_and_explore_dataset("data")
    
    # Step 4: Create organized dataset structure
    processed_path = create_dataset_structure(data_path)
    
    # Step 5: Generate dataset summary
    summary = generate_dataset_summary(processed_path)
    
    # Step 6: Verify dataset structure
    verify_dataset_structure()
    
    print(f"\nDataset preparation completed!")
    print(f"Processed data saved to: {processed_path}")
    print("\nYou can now use this organized dataset for training your tuberculosis detection model.")
    print("\nNext steps:")
    print("1. Run: python model_training.py (to train a custom model)")
    print("2. Run: streamlit run app.py (to start the web application)")

if __name__ == "__main__":
    main()