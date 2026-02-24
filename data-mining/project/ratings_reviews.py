from __future__ import annotations


import argparse

import numpy
import pandas


pandas.set_option("display.min_rows", 50)  # show all rows when printing DataFrames.


def rating_weight(average_rating: float, rating_number: int) -> float:
	return average_rating / numpy.log(rating_number + 1)


def extract_low_rating_high_volume(data: pandas.DataFrame) -> pandas.DataFrame:
	refined = data[["average_rating", "rating_number"]].copy()
	refined["rating_weight"] = refined.apply(lambda row: rating_weight(row["average_rating"], row["rating_number"]), axis = 1)

	return refined.sort_values("rating_weight")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Extract ratings and reviews from the Amazon Reviews 2023 dataset.")
	parser.add_argument("data", help = "Path to the input CSV file containing the raw metadata.")

	args = parser.parse_args()

	data = pandas.read_csv(args.data)
	refined_data = extract_low_rating_high_volume(data)

	print(refined_data)
