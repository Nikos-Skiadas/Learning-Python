from __future__ import annotations


import typing

import numpy
import sklearn.base

from .protocols import Preprocessor, Encoder, Bicoder, Model, Scorer


Float = float | numpy.float16 | numpy.float32


class Classifier[
	DecodedSource,
	EncodedSource,
	EncodedTarget,
	DecodedTarget,
](
	sklearn.base.BaseEstimator,
	sklearn.base.ClassifierMixin,
):

	def __init__(self,
		preprocessor: Preprocessor[DecodedSource],
		model: Model[EncodedSource, EncodedTarget],
		source_encoder: Encoder[DecodedSource, EncodedSource],
		target_bicoder: Bicoder[DecodedTarget, EncodedTarget],
		scorers: dict[str, Scorer[EncodedTarget, Float]] | None = None,
	) -> None:
		# BaseEstimator uses __init__ params for get_params/set_params
		self.preprocessor = preprocessor
		self.model = model
		self.source_encoder = source_encoder
		self.target_bicoder = target_bicoder
		self.scorers = scorers or {}


	def compile(self, **scorers: Scorer[EncodedTarget, Float]) -> typing.Self:
		self.scorers = scorers

		return self

	def preprocess(self, source: DecodedSource) -> DecodedSource:
		return self.preprocessor(source)

	def fit(self,
		source: DecodedSource,
		target: DecodedTarget, /
	) -> typing.Self:
		source = self.preprocess(source)

		self.source_encoder.fit(source, target)
		self.target_bicoder.fit(        target)

		self.model.fit(
			self.source_encoder.transform(source),
			self.target_bicoder.transform(target),
		)

		return self

	def forward(self, source: DecodedSource) -> EncodedTarget:
		return self.model.predict(
			self.source_encoder.transform(
				self.preprocess(source)
			)
		)

	def predict(self, source: DecodedSource) -> DecodedTarget:
		return self.target_bicoder.inverse_transform(
			self.forward(
				self.preprocess(source)
			)
		)

	def score(self,
		source: DecodedSource,
		target: DecodedTarget, /,
	) -> dict[str, Float]:
		true = self.target_bicoder.transform(target)
		pred = self.forward(source)

		return {
			name: scorer(
				true,
				pred,
			) for name, scorer in self.scorers.items()
		}
