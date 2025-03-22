from __future__ import annotations


import os
import re
import string
import typing
import warnings; warnings.filterwarnings("ignore",
	category = UserWarning,
)

import numpy
from rich import print
import pandas

import sklearn
import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

import nltk
import nltk.stem
import nltk.corpus
import sklearn.pipeline


nltk.download('wordnet'  )  # for lemmatization
nltk.download('stopwords')  # for stop word removal


def load_data(
	split: typing.Literal[
		"train",
		"val",
		"test",
	],
	root: str = "",
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
			ngram_range = ngram_range,  # NOTE: maybe tunable
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
		C: float = 1.0,
	):
		super().__init__(
		#	penalty = None,
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
		#	l1_ratio = None,
		)


class Classifier:

	def __init__(self, *,
		vectorizer: Vectorizer,
		model: Model,
	):
		self.pipeline = sklearn.pipeline.Pipeline(
			[
				("vectorizer", vectorizer),
				("model", model),
			]
		)


	def fit(self, train_data: pandas.DataFrame):
		X = train_data.Text
		y = train_data.Label

	#	Fit model:
		self.pipeline.fit(X, y)

		return self

	def predict(self, test_data: pandas.DataFrame,
		save: str | None = None,
	) -> pandas.Series:
		X = test_data.Text
		y = pandas.Series(self.pipeline.predict(X),
			index = test_data.index,  # align with test data index
			name = "Label",  # recover the "Label" column name
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
		X = data.Text
		y = data.Label

	#	Initialize a tuner with given parameter grid and splits on the stock model:
		tuner = sklearn.model_selection.GridSearchCV(self.pipeline, param_grid,
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
		tuner.fit(X, y)  # type: ignore

	#	Refit model with the best parameters on the whole data (why throw the val split now that we finished tunning?):
		self.pipeline.set_params(**tuner.best_params_)  # set the best parameters for the model
		self.pipeline.fit(X, y)  # refit it with the best parameters on the whole dataset available for training

		return self


if __name__ == "__main__":
	train_data = load_data("train")
	val_data   = load_data("val"  )
	test_data  = load_data("test" )

	classifier = Classifier(
		vectorizer = Vectorizer(),
		model = Model(),
	)
	classifier.tune(train_data, val_data,
		vectorizer__max_features = [
			128,
			256,
			512,
		],
		vectorizer__ngram_range = [
			(1, 1),
			(1, 2),
			(2, 2),
		],
		model__C = list(numpy.linspace(.5, 1., 2)),
	)

#	Export predictions:
	predictions = classifier.predict(test_data,
		save = "submission.csv",
	)
