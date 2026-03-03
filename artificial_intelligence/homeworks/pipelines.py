from __future__ import annotations


import typing

from .protocols import Preprocessor, Encoder, Bicoder, Model, Scorer


class Classifier[
	DecodedSource,
	EncodedSource,
	EncodedTarget,
	DecodedTarget,
]:

	def __init__(self, *preprocessors: Preprocessor[DecodedSource],
		model: Model[EncodedSource, EncodedTarget],
		source_encoder: Encoder[DecodedSource, EncodedSource],
		target_bicoder: Bicoder[DecodedTarget, EncodedTarget],
	) -> None:
		self.transforms = preprocessors
		self.model = model

		self.source_encoder = source_encoder
		self.target_bicoder = target_bicoder


	def compile(self, **scorers: Scorer[EncodedTarget, float]) -> typing.Self:
		self.scorers = scorers

		return self

	def preprocess(self, source: typing.Collection[DecodedSource]) -> typing.Collection[DecodedSource]:
		for preprocessor in self.transforms:
			source = preprocessor(source)

		return source

	def fit(self,
		source: typing.Collection[DecodedSource],
		target: typing.Collection[DecodedTarget], /
	) -> typing.Self:
		source = self.preprocess(source)

		self.source_encoder.fit(source, target)
		self.target_bicoder.fit(        target)

		self.model.fit(
			self.source_encoder.transform(source),
			self.target_bicoder.transform(target),
		)

		return self

	def forward(self, source: typing.Collection[DecodedSource]) -> typing.Collection[EncodedTarget]:
		return self.model.predict(
			self.source_encoder.transform(
				self.preprocess(source)
			)
		)

	def predict(self, source: typing.Collection[DecodedSource]) -> typing.Collection[DecodedTarget]:
		return self.target_bicoder.inverse_transform(
			self.forward(
				self.preprocess(source)
			)
		)

	def score(self,
		source: typing.Collection[DecodedSource],
		target: typing.Collection[DecodedTarget], /,
	**metrics: Scorer[EncodedTarget, float]) -> dict[str, float]:
		true = self.target_bicoder.transform(target)
		pred = self.forward(source)

		return {
			name: scorer(
				true,
				pred,
			) for name, scorer in metrics.items()
		}
