import os
import zstandard
import polars as pl
import requests
import json
import tempfile

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

print("Downloading and processing data...")
df = download_process_zst("https://the-eye.eu/redarcs/files/truerateme_submissions.zst")
print(df.columns)

# select only needed columns
df = df.select([
    "id", "author", "created_utc", "subreddit",         # metadata
    "title", "selftext", "media_embed", "media", "url", # content
])
print(df.head(5))