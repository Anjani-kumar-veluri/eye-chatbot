import gdown

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1-Zg5HuGE4ooSQpFILv61chdPoG8YbP8e"

gdown.download_folder(
    DRIVE_FOLDER_URL,
    output=".",
    quiet=False,
    use_cookies=False
)