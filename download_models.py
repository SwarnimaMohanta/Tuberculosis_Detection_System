import os
import gdown

def download_model():
    # Folder where the model(s) will be saved
    model_dir = "document/best_tb_models"
    os.makedirs(model_dir, exist_ok=True)

    # ----------- OPTION 1: Single .h5 file -----------
    file_id = "1M6ymQ4HS2iNlMHtH9yd7-G5jGY-YqMoH"  # Replace with your file ID
    filename = "best_tb_model.h5"
    file_path = os.path.join(model_dir, filename)

    if not os.path.exists(file_path):
        print(f"📥 Downloading single model file {filename}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_path, quiet=False)
        print(f"✅ {filename} downloaded successfully!")
    else:
        print(f"✅ {filename} already exists, skipping download.")

    # ----------- OPTION 2: Entire folder -----------
    folder_url = "https://drive.google.com/drive/folders/16DRpYK04wHZFfWkcI2O-q1VPfymqHiRS?usp=drive_link"  # Replace with your folder URL
    # Check if the folder is already downloaded by checking for at least one file
    folder_contents = os.listdir(model_dir)
    if not folder_contents or (len(folder_contents) == 1 and folder_contents[0] == filename):
        print(f"📥 Downloading entire folder from Google Drive...")
        gdown.download_folder(folder_url, output=model_dir, quiet=False, use_cookies=False, remaining_ok=True)
        print(f"✅ Folder downloaded successfully!")
    else:
        print(f"✅ Folder already exists, skipping download.")

if __name__ == "__main__":
    download_model()
