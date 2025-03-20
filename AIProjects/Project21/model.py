from __future__ import annotations


import math
import os
import typing

import numpy
import pandas
import sklearn
import sklearn.feature_extraction
import sklearn.linear_model


def load_data(
	split: typing.Literal[
		"train",
		"val",
		"test",
	],
	root: str = "dataset",
	index: str = "ID",
):
	return pandas.read_csv(os.path.join(f"{root}", f"{split}_dataset.csv"),
		index_col = index,
	)


class Vectorizer(sklearn.feature_extraction.text.TfidfVectorizer):

	def __init__(self, *,
		max_features: int | None = None,
		ngram_lower: int = 1,
		ngram_upper: int = 1,
	):
		super().__init__(
		#	input = 'content',
			encoding = 'utf-8',
		#	decode_error = 'strict',
			strip_accents = 'unicode',
			lowercase = True,
		#	preprocessor = None,
		#	tokenizer = None,
		#	analyzer = 'word',
			stop_words = 'english',  # dataset messages are in english so remove english stop words
		#	token_pattern = '(?u)\\b\\w\\w+\\b',
			ngram_range = (
				ngram_lower,
				ngram_upper,
			),  # NOTE: maybe tunable
		#	max_df = 1.0,
		#	min_df = 1,
			max_features = max_features, # NOTE: tunable
		#	vocabulary = None,
		#	binary = False,
		#	dtype = numpy.float64,
		#	norm = 'l2',
		#	use_idf = True,
		#	smooth_idf = True,
		#	sublinear_tf = False,
		)


class Model(sklearn.linear_model.LogisticRegression):

	def __init__(self, *,
		penalty: typing.Literal[
			'l1',
			'l2', 'elasticnet'
		] | None = 'l2',
		C: float = 1.0,
		l1_ratio: float | None = None,
	):
		super().__init__(
			penalty = penalty, # NOTE: tunable
		#	dual = False,
		#	tol = 0.0001,
			C = C, # NOTE: tunable
			fit_intercept = True,
		#	intercept_scaling = 1,
		#	class_weight = None,
			random_state = 42,  # seed
		#	solver = 'lbfgs',
		#	max_iter = 100,
		#	verbose = 0,
		#	warm_start = False,
		#	n_jobs = None,
			l1_ratio = l1_ratio,  # NOTE: tunable
		)


if __name__ == "__main__":
	train_data = load_data("train")
	val_data   = load_data("val"  )
	test_data  = load_data("test" )

#	Initialize and train vectorizer:
	vectorizer = Vectorizer()
	train_X = vectorizer.fit_transform(train_data.Text)
	train_y = train_data.Label

#	Initialize and train model:
	logistic_regressor = Model().fit(
		train_X,
		train_y,
	)

#	Inference of model on test data:
	test_X = vectorizer.transform(test_data.Text)
	test_y = pandas.Series(logistic_regressor.predict(test_X),
		index = test_data.index,  # align with test data index
		name = train_y.name,  # recover the "Label" column name
	)
	test_y.to_csv("submission.csv")
