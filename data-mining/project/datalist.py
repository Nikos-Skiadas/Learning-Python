from huggingface_hub import list_repo_files

repo_id = "McAuley-Lab/Amazon-Reviews-2023"

files = list_repo_files(repo_id, repo_type="dataset")

# Keep only parquet files
parquet_files = [f for f in files if f.endswith(".parquet")]

# Extract category prefixes (folder names)
categories = sorted(set(f.split("/")[0] for f in parquet_files))

print("Available folders on HF:")
for c in categories:
    print(c)
