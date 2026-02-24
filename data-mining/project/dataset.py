from __future__ import annotations


import argparse
import pathlib
from itertools import islice

import datasets
from huggingface_hub import list_repo_files  # List files inside the HF dataset repo.


def get_categories() -> list[str]:
	paths = [
		pathlib.Path(path) for path in list_repo_files("McAuley-Lab/Amazon-Reviews-2023",
			repo_type = "dataset",
		)
	]  # ask HuggingFace for every file path in the Amazon Reviews 2023 dataset repository

#	Keep only `raw_meta` parquet folders, remove the `raw_meta_` prefix, deduplicate, and sort:
	return sorted({path.parts[0].removeprefix("raw_meta_")
		for path in paths if path.suffix == ".parquet" and path.parts and path.parts[0].startswith("raw_meta_")})


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
			streaming = True,
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

		raise

	if trim is not None:
		datasets.Dataset.from_list(list(islice(dataset, trim))).to_csv((root / name.lower()).with_suffix(".csv"))

	return dataset


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--name", default = None)
	parser.add_argument("--root", default = "./data")
	parser.add_argument("--trim", type = int)

	args = parser.parse_args()

	if args.name is not None:
		get_category(
			args.name,
			args.root,
			args.trim,
		)

	else:
		categories = get_categories()  # discover all available category names automatically

	#	Loop through each discovered category:
		for category in categories:
			print(f"Fetching category: {category}")  # show progress so you know which category is being fetched

		#	Download, process, and save this category as CSV:
			get_category(category,
				args.root,
				args.trim,
			)
