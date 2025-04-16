from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from PIL import Image
from io import BytesIO

import polars as pl

import requests
import tempfile
import zstandard
import os
import re

def download_process_zst(url:str) -> pl.DataFrame:    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download
        response = requests.get(url)
        zst_path = os.path.join(temp_dir, "data.zst")
        with open(zst_path, 'wb') as f:
            f.write(response.content)

        # Decompress
        json_path = os.path.join(temp_dir, "data.jsonl")
        with open(zst_path, 'rb') as compressed_file:
            with open(json_path, 'wb') as decompressed_file:
                dctx = zstandard.ZstdDecompressor()
                dctx.copy_stream(compressed_file, decompressed_file)

        # File is in JSONL format
        df = pl.read_ndjson(
            json_path,
            infer_schema_length=10000,
            ignore_errors=True
        )
        
    return df

def download_thumbnail(url: str, path: str, max_retries=5):
    # Configure retry strategy with backoff
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,  # Exponential backoff: 1, 2, 4, 8, 16 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["GET"]
    )
    
    # Create a session with the retry strategy
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
    session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
    
    # Add headers to make request more browser-like
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        'Referer': 'https://imgur.com/'
    }
    
    # Use the session to get the image with retries, redirects and headers
    response = session.get(url, headers=headers, allow_redirects=True, timeout=30)
    response.raise_for_status()
    
    # Check image dimensions before writing to disk
    image = Image.open(BytesIO(response.content))
    width, height = image.size
    
    # Throw error if image is the imgur NOT FOUND image
    if width == 161 and height == 81:
        raise RuntimeError("Image not found")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Write image to file
    with open(path, 'wb') as f:
        f.write(response.content)

def remove_url_args(url:str):
    return re.sub(r'\?.*$', '', url)

def is_img_url(url:str):
    return url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.svg'))


def extract_all_floats(cmt_text:str) -> list:
    return [float(match[0]) for match in re.findall(r"[+-]?(\d+(\.\d+)?)", cmt_text)]
    
def first_float_extraction(cmt_text:str) -> float:
    return next(filter(lambda x: 0 <= x <= 10, extract_all_floats(cmt_text)), None)

def mean_of_floats_extraction(cmt_text:str):
    floats = extract_all_floats(cmt_text)
    return sum(floats) / len(floats) if floats else None