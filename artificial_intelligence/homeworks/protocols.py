from __future__ import annotations


import typing


@typing.runtime_checkable
class Preprocessor[Decoded](typing.Protocol):

	def __call__(self, *X: Decoded) -> Decoded:
		...


@typing.runtime_checkable
class Encoder[Decoded, Encoded](typing.Protocol):

	def fit(self, decoded: Decoded, signal: typing.Any | None = None, /) -> typing.Self:
		...

	def transform(self, decoded: Decoded, /) -> Encoded:
		...


@typing.runtime_checkable
class Bicoder[Decoded, Encoded](Encoder[Decoded, Encoded], typing.Protocol):

	def inverse_transform(self, encoded: Encoded, /) -> Decoded:
		...


@typing.runtime_checkable
class Model[Source, Target](typing.Protocol):

	def fit(self, source: Source, target: Target, /) -> typing.Self:
		...

	def predict(self, source: Source, /) -> Target:
		...


@typing.runtime_checkable
class Scorer[Target, Result](typing.Protocol):

	def __call__(self, target: Target, prediction: Target, /) -> Result:
		...
