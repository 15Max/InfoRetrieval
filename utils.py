import requests
import os
import gzip
import shutil


URL_LIST = [
    "https://snap.stanford.edu/data/wiki-topcats.txt.gz",
    "https://snap.stanford.edu/data/wiki-topcats-categories.txt.gz",
    "https://snap.stanford.edu/data/wiki-topcats-page-names.txt.gz"
]

FILE_NAMES = [
    "wiki-topcats.txt",
    "wiki-topcats-categories.txt",
    "wiki-topcats-page-names.txt"
]


def download_and_extract(urls, save_dir):
    """
    Downloads .gz files from URLs, saves them, and extracts their content.
  
    Args:
        urls (list): List of URLs to download
        save_dir (str): Directory to save the downloaded and extracted files
    """
    os.makedirs(save_dir, exist_ok=True)

    for url in urls:
        gz_name = url.split('/')[-1]               # e.g. file.txt.gz
        gz_path = os.path.join(save_dir, gz_name) # path to save .gz
        extracted_name = gz_name[:-3]             # remove .gz extension
        extracted_path = os.path.join(save_dir, extracted_name)

        try:
            print(f"Downloading {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Save .gz file
            with open(gz_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved .gz to: {gz_path}")

            # Extract .gz
            print(f"Extracting {gz_path}...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(extracted_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Extracted to: {extracted_path}")

            # Delete original .gz
            os.remove(gz_path)
            print(f"Deleted original .gz: {gz_path}")


        except Exception as e:
            print(f"Error handling {url}: {e}")

def ensure_files_exist(file_list, directory):
    """
    Checks if all files in file_list exist in directory.

    Args:
        file_list (list): List of file names to check
        directory (str): Directory where files should be located

    Returns:
        bool: True if all files are present, False otherwise
    """
    missing = []
    for fname in file_list:
        path = os.path.join(directory, fname)
        if not os.path.isfile(path):
            missing.append(fname)

    if missing:
        print(f"Missing files: {missing}")
        return False
        
    
    else:
        print(f"All files needed are present in {directory}.")
    
        return True
