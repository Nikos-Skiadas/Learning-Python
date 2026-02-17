from __future__ import annotations


import argparse
import pathlib

import datasets


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
		...  # fetch all categories and save them to disk
