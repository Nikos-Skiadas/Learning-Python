from __future__ import annotations


import os
import re
import string
import typing

import numpy
import pandas

import sklearn
import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

import nltk
import nltk.stem
import nltk.corpus


nltk.download('wordnet'  )  # for lemmatization
nltk.download('stopwords')  # for stop word removal


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
		index_col = index,  # put IDs on index
		encoding = "utf-8",
	)


class Preprocessor:

	def __call__(self, text: str) -> str:
		text = re.sub(r"@\w+"   , "", text)  # mentions
		text = re.sub(r"#\w+"   , "", text)  # hashtags
		text = re.sub(r"\S+@\S+", "", text)  # emails

		return text.translate(str.maketrans("", "", string.punctuation))  # punctuation


class Tokenizer:

	def __init__(self):
		self.lemmatizer = nltk.WordNetLemmatizer()
		self.stemmer = nltk.stem.PorterStemmer()
		self.tokenizer = nltk.tokenize.TweetTokenizer(
			preserve_case = False,
			reduce_len = True,
			match_phone_numbers = True,
			strip_handles = True,
		)

	def __call__(self, text: str):
		return [self.stemmer.stem(self.lemmatizer.lemmatize(token))
			for token in self.tokenizer.tokenize(text) if token and not token.isdigit()]


class Vectorizer(sklearn.feature_extraction.text.TfidfVectorizer):

	def __init__(self, *,
		max_features: int | None = None,  # regularize by reduing curse of dimensionality
		ngram_range: tuple[
			int,
			int,
		] = (
			1,
			1,
		),
	):
		super().__init__(
		#	input = "content",
			encoding = "utf-8",
		#	decode_error = "strict",
			strip_accents = "unicode",
			lowercase = True,
			preprocessor = Preprocessor(),
			tokenizer = Tokenizer(),
		#	analyzer = "word",
			stop_words = "english",  # messages seem to be in english
		#	token_pattern = "(?u)\\b\\w\\w+\\b",
		#	ngram_range = ngram_range,  # NOTE: maybe tunable
		#	max_df = 1.0,
		#	min_df = 1,
			max_features = max_features, # NOTE: tunable
		#	vocabulary = None,
		#	binary = False,
		#	dtype = numpy.float64,
		#	norm = "l2",
		#	use_idf = True,
		#	smooth_idf = True,
		#	sublinear_tf = False,
		)


class Model(sklearn.linear_model.LogisticRegression):

	def __init__(self, *,
		penalty: typing.Literal[
			"l1",
			"l2", "elasticnet"
		] | None = "l2",
		C: float = 1.0,
		l1_ratio: float | None = None,
	):
		super().__init__(
			penalty = penalty, # NOTE: tunable
		#	dual = False,
		#	tol = 0.0001,
			C = C, # NOTE: tunable
			fit_intercept = True,  # use bias
		#	intercept_scaling = 1,
		#	class_weight = None,
			random_state = 42,  # seed
		#	solver = "lbfgs",
		#	max_iter = 100,
		#	verbose = 0,
		#	warm_start = False,
			n_jobs = -1,  # parallelization
			l1_ratio = l1_ratio,  # NOTE: tunable
		)


class Classifier:

	def __init__(self, *,
		vectorizer: Vectorizer,
		model: Model,
	):
		self.vectorizer = vectorizer
		self.model = model


	def fit(self, train_data: pandas.DataFrame):
		X = self.vectorizer.fit_transform(train_data.Text)
		y = train_data.Label

	#	Fit model:
		_ = self.model.fit(X, y)

		return self

	def predict(self, test_data: pandas.DataFrame,
		save: str | None = None,
	) -> pandas.Series:
		X = self.vectorizer.transform(test_data.Text)
		y = pandas.Series(self.model.predict(X),
			index = test_data.index,  # align with test data index
			name = test_data.Label.name,  # recover the "Label" column name
		)

	#	Optionally save perdictions for submission:
		if save is not None:
			y.to_csv(save)

		return y

	def evaluate(self, val_data: pandas.DataFrame):
		y_true = val_data.Label
		y_pred = self.predict(val_data)

		return {
			"accuracy"  : sklearn.metrics. accuracy_score(y_true, y_pred),
			"presiction": sklearn.metrics.precision_score(y_true, y_pred),
			"recall"    : sklearn.metrics.   recall_score(y_true, y_pred),
			"f1"        : sklearn.metrics.       f1_score(y_true, y_pred),
		}

	def tune(self,
		train_data: pandas.DataFrame,
		val_data: pandas.DataFrame,
	**param_grid: list):
		data = pandas.concat(
			[
				train_data,
				val_data,
			]
		)

	#	Concatenate train and val split because `sklearn` is a ballbuster for fixed splits:
		X = self.vectorizer.fit_transform(data.Text)  # NOTE: can we steal terms from the validation split?
		y = data.Label

	#	Initialize a tuner with given parameter grid and splits on the stock model:
		tuner = sklearn.model_selection.GridSearchCV(self.model, param_grid,
			scoring = "accuracy",
			n_jobs = -1,  # parallelization
			refit = False,  # we will do our own refitting (on the whole data) thank you very much
			cv = sklearn.model_selection.PredefinedSplit([-1] * len(train_data) + [0] * len(val_data)),  # use our splits
		#	verbose = 0,
		#	pre_dispatch = "2*n_jobs",
		#	error_score = numpy.nan,
		#	return_train_score = False,
		)

	#	Fit and tune model:
		_ = tuner.fit(X, y)

	#	Refit model with the best parameters on the whole data (why throw the val split now that we finished tunning?):
		self.model.set_params(**tuner.best_params_)
		self.model.fit(X, y)

		return self


if __name__ == "__main__":
	train_data = load_data("train")
	val_data   = load_data("val"  )
	test_data  = load_data("test" )

	classifier = Classifier(
		vectorizer = Vectorizer(),
		model = Model(),
	)
	_ = classifier.tune(train_data, val_data,
		penalty = [
			"l1",
			"l2",
			"elasticnet",
		],
		C = list(numpy.arange(0., 1., .5)),
		l1_ratio = list(numpy.arange(0., 1., .5))
	)

	print(classifier.evaluate(train_data))
	print(classifier.evaluate(val_data))
