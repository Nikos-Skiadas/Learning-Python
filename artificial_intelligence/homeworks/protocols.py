from __future__ import annotations


import typing


@typing.runtime_checkable
class Preprocessor[Decoded](typing.Protocol):

	def __call__(self, source: Decoded) -> Decoded:
		...


@typing.runtime_checkable
class Scorer[Target, Result](typing.Protocol):

	def __call__(self,
		true: Target,
		pred: Target, /
	) -> Result:
		...


@typing.runtime_checkable
class Encoder[Decoded, Encoded](typing.Protocol):

	def fit(self, source: Decoded, signal: typing.Any | None = None, /) -> typing.Self:
		...

	def transform(self, source: Decoded, /) -> Encoded:
		...


@typing.runtime_checkable
class Bicoder[Decoded, Encoded](Encoder[Decoded, Encoded], typing.Protocol):

	def inverse_transform(self, target: Encoded, /) -> Decoded:
		...


@typing.runtime_checkable
class Model[Source, Target](typing.Protocol):

	def fit(self,
		source: Source,
		target: Target, /
	) -> typing.Self:
		...

	def predict(self, source: Source, /) -> Target:
		...
