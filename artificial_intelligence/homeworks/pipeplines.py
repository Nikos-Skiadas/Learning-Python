from __future__ import annotations


import typing

from .protocols import Preprocessor, Encoder, Bicoder, Model, Scorer


class Classifier[
	DecodedSource,
	EncodedSource,
	EncodedTarget,
	DecodedTarget,
]:

	def __init__(self, *,
		model: Model[EncodedSource, EncodedTarget],
		source_encoder: Encoder[DecodedSource, EncodedSource],
		target_bicoder: Bicoder[DecodedTarget, EncodedTarget],
		transform: Preprocessor[DecodedSource] | None = None,
	) -> None:
		self.transform = transform
		self.model = model

		self.source_encoder = source_encoder
		self.target_bicoder = target_bicoder


	def compile(self, **scorers: Scorer[EncodedTarget, float]) -> typing.Self:
		self.scorers = scorers

		return self

	def fit(self,
		source: DecodedSource,
		target: DecodedTarget, /
	) -> typing.Self:
		if self.transform is not None:
			source = self.transform(source)

		self.source_encoder.fit(source, target)
		self.target_bicoder.fit(        target)

		self.model.fit(
			self.source_encoder.transform(source),
			self.target_bicoder.transform(target),
		)

		return self

	def forward(self, source: DecodedSource) -> EncodedTarget:
		if self.transform is not None:
			source = self.transform(source)

		return self.model.predict(self.source_encoder.transform(source))

	def predict(self, source: DecodedSource) -> DecodedTarget:
		if self.transform is not None:
			source = self.transform(source)

		return self.target_bicoder.inverse_transform(self.forward(source))

	def score(self,
		source: DecodedSource,
		target: DecodedTarget, /,
	**metrics: Scorer[EncodedTarget, float]) -> dict[str, float]:
		true = self.target_bicoder.transform(target)
		pred = self.forward(source)

		return {
			name: scorer(
				true,
				pred,
			) for name, scorer in metrics.items()
		}
