import os
import polars as pl
import re

from modules.reddit_tools import (
    download_process_zst,
    download_thumbnail,
    remove_url_args,
    is_img_url,
    mean_of_floats_extraction
)


print("Downloading and processing data...")
df_posts = download_process_zst("https://the-eye.eu/redarcs/files/truerateme_comments.zst")

df_posts = df_posts.select([
    "id", "author", "created_utc", "subreddit",         # metadata
    "title", "selftext", "media_embed", "media", "url", # content
])

# filter empty, removed, and delted posts
df_posts = df_posts.filter(
    (pl.col("media_embed").is_not_null()) &
    (pl.col("media").is_not_null()) &
    (pl.col("url").is_not_null()) &
    (pl.col("selftext") != "[removed]") &
    (pl.col("selftext") != "[deleted]")
)

# Download all thumbnails in df, updating rows with the local path upon download
# note that we write every step to parquet to avoid losing data
# also we skip rows if file already exists
if 'local_thumbnail_path' not in df_posts.columns:
    df_posts = df_posts.with_columns(pl.lit("").alias('local_thumbnail_path'))

df_view = df_posts.clone()
for idx, row in enumerate(df_view.iter_rows(named=True)):
    # Extract thumbnail url
    try:
        thumbnail_url = row['media']['oembed']['thumbnail_url']
    except (KeyError, TypeError):
        continue
    if thumbnail_url == None: continue
    
    clean_url = remove_url_args(thumbnail_url)
    if not is_img_url(clean_url):  continue
    
    ext = os.path.splitext(clean_url)[1]
    local_path = os.path.join("thumbnails", f"{row['id']}{ext}")
    
    if os.path.exists(local_path): continue
    
    try:
        download_thumbnail(clean_url, local_path)
        df_posts[idx, 'local_thumbnail_path'] = local_path
        df_posts.write_parquet("reddit_posts.parquet")
    except:
        print(f"Failed to download {clean_url}")
        continue

print("All thumbnails downloaded and paths updated in dataframe.")


print("Downloading and processing comments...")
df_comments = download_process_zst("https://the-eye.eu/redarcs/files/truerateme_comments.zst")

extraction_method = mean_of_floats_extraction

top_level_comments = df_comments.filter(
    pl.col("parent_id") == pl.col("link_id")
)

extract_id = lambda x: re.sub(r"t\d_", "", x)

top_level_comments = top_level_comments.with_columns(
    pl.col("link_id").map_elements(extract_id, pl.String).alias("thread"),             # Extract post ID
    pl.col("body").map_elements(extraction_method, pl.Float32).alias("rating")         # Extract rating
)

rated_comments = top_level_comments.filter(
    pl.col("rating").is_not_null() & 
    (pl.col("rating") >= 0) & 
    (pl.col("rating") <= 10)
)

# Now, we join comments onto posts
reddit_posts = pl.read_parquet("reddit_posts.parquet")

rated_posts = reddit_posts.join(
    rated_comments,
    left_on  = "id",
    right_on = "thread",
    how      = "inner"
).with_columns(
    zeroed_score = pl.when(pl.col("score") < 0).then(0).otherwise(pl.col("score"))
).group_by('id').agg(
    mean_rating     = pl.col("rating").mean().alias("mean_rating"),
    median_rating   = pl.col("rating").median().alias("median_rating"),
    rating_stdev    = pl.col("rating").std().fill_null(0).alias("rating_stdev"),
    weighted_rating = (pl.col("rating") * pl.col("zeroed_score")).sum() / pl.col("zeroed_score").sum(),
    rating_count    = pl.col("rating").count().alias("rating_count"),
)

# Join ratings back onto posts
reddit_posts = reddit_posts.join(
    rated_posts,
    left_on  = "id",
    right_on = "id",
    how      = "left"
).filter(
    (pl.col("rating_count") > 0) &
    (pl.col("local_thumbnail_path") != "")
)

reddit_posts.write_parquet("reddit_posts_rated.parquet")