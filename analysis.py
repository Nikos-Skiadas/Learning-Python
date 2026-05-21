from __future__ import annotations


import pandas


def load(paramsB: float, experiment: str, /, *columns: str) -> pandas.Series | pandas.DataFrame:
	data = pandas.read_csv(f"runs_hw3_full/runs/qwen-{paramsB}b_{experiment}.generations.csv",
		index_col = "Id",
	)
	if not columns: return data
	head, *rest = columns
	if not rest: return data[head]
	return data[list(columns)]
