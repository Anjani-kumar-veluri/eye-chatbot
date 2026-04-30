import os
import gdown

DRIVE_FOLDER_URL = os.getenv("https://drive.google.com/drive/u/0/folders/1-Zg5HuGE4ooSQpFILv61chdPoG8YbP8e")

if not DRIVE_FOLDER_URL:
    raise Exception("DRIVE_FOLDER_URL not found")

gdown.download_folder(DRIVE_FOLDER_URL, output=".", quiet=False, use_cookies=False)