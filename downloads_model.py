import os
import gdown

DRIVE_FOLDER_URL = os.getenv("DRIVE_FOLDhttps://drive.google.com/drive/u/0/folders/1-Zg5HuGE4ooSQpFILv61chdPoG8YbP8eER_URL")

print("DEBUG VALUE:", DRIVE_FOLDER_URL)  # 👈 for debugging

if not DRIVE_FOLDER_URL:
    raise Exception("DRIVE_FOLDER_URL not found")

gdown.download_folder(DRIVE_FOLDER_URL, quiet=False, use_cookies=False)