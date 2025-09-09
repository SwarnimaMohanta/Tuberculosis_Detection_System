import os
import gdown

def download_model():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Replace 'YOUR_FILE_ID' with your actual Google Drive file ID
    file_id = "1YwMsA30W4rZOkVh-Gkkhe4XHxF695IMp"
    filename = "best_tb_model.h5"
    file_path = os.path.join(model_dir, filename)

    if not os.path.exists(file_path):
        print(f"📥 Downloading {filename}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_path, quiet=False)
        print(f"✅ {filename} downloaded successfully!")
    else:
        print(f"✅ {filename} already exists, skipping download.")

if __name__ == "__main__":
    download_model()
