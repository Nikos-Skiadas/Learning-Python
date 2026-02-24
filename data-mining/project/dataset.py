from __future__ import annotations


import argparse
import pathlib

import datasets
from huggingface_hub import list_repo_files  # List files inside the HF dataset repo.


def get_categories() -> list[str]:
	# Ask Hugging Face for every file path in the Amazon Reviews 2023 dataset repo.
	files = list_repo_files(
		"McAuley-Lab/Amazon-Reviews-2023",
		repo_type = "dataset",
	)

	# Keep only raw_meta parquet folders, remove the "raw_meta_" prefix, deduplicate, and sort.
	return sorted(
		{
			file_path.split("/")[0].removeprefix("raw_meta_")
			for file_path in files
			if file_path.startswith("raw_meta_") and file_path.endswith(".parquet")
		}
	)


def get_category(
	name: str,
	root: str | pathlib.Path,
	trim: int | None = None,
):
	root = pathlib.Path(root)
	root.mkdir(
		parents = True,
		exist_ok = True,
	)

	try:
		dataset = datasets.load_dataset("parquet",
			data_files=f"hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw_meta_{name}/*.parquet",
			split = "train",
		).remove_columns(
			[
				"features",
				"description",
				"images",
				"videos",
				"categories",
				"details",
			]
		)

	except ValueError:
		print(f"Dataset for category '{name}' not found.")

		return

	if trim is not None:
		dataset = dataset.select(range(trim))

	dataset.to_csv((root / name).with_suffix(".csv"))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--name", default = None)
	parser.add_argument("--root", default = "data-mining/data")
	parser.add_argument("--trim", type = int)
	args = parser.parse_args()

	if args.name is not None:
		get_category(
			args.name,
			args.root,
			args.trim,
		)

	else:
		# Discover all available category names automatically.
		categories = get_categories()
		# Loop through each discovered category.
		for category in categories:
			# Show progress so you know which category is being fetched.
			print(f"Fetching category: {category}")
			# Download, process, and save this category as CSV.
			get_category(
				category,
				args.root,
				args.trim,
			)
