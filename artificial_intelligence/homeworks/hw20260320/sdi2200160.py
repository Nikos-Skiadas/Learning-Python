from __future__ import annotations


import numpy
import pandas

import sklearn.preprocessing
import sklearn.feature_extraction.text


from ..protocols import Preprocessor, Encoder, Bicoder, Model, Scorer
from ..pipelines import Classifier
from ..data import data

class QEvasionClassifier(
	Classifier[
		str, numpy.ndarray,
		numpy.ndarray, int,
	]
):
	...


source_encoder = sklearn.feature_extraction.text.TfidfVectorizer(
	encoding = "utf-8",
	decode_error = "replace",
	strip_accents = "unicode",
	lowercase = True,
#	preprocessor = None,  # for question-answer pairs
#	tokenizer = None,  # useful with Word2Vec
	stop_words = "english",
	token_pattern = r"(?u)\b\w[\w']*\b",  # to include contractions like "don't"
#	ngram_range = (
#		1,
#		2,
#	),  # unigrams and bigrams
	max_df = 0.95,  # ignore terms that appear in more than 95% of documents
#	min_df = 5   ,  # ignore terms that appear in fewer than 5 documents
#	max_features = 10000,  # limit to top 10000 features
	sublinear_tf = True,  # use sublinear term frequency scaling
)
target_bicoder = sklearn.preprocessing.LabelEncoder()


test = data["test"].to_pandas()
assert isinstance(test, pandas.DataFrame)
test["question_answer"] = test["question"] + " | " + test["interview_answer"]
source_encoder.fit(test["question_answer"])
print(len(source_encoder.get_feature_names_out()))
