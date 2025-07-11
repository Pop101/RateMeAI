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

def extract_all_floats(cmt_text:str) -> list:
    return [float(match[0]) for match in re.findall(r"[+-]?(\d+(\.\d+)?)", cmt_text)]
    
def first_float_extraction(cmt_text:str) -> float:
    return next(filter(lambda x: 0 <= x <= 10, extract_all_floats(cmt_text)), None)

def mean_of_floats_extraction(cmt_text:str):
    floats = extract_all_floats(cmt_text)
    return sum(floats) / len(floats) if floats else None