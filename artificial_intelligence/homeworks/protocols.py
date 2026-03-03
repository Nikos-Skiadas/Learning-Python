from __future__ import annotations


import typing


@typing.runtime_checkable
class Preprocessor[Decoded](typing.Protocol):

	def __call__(self, *sources: typing.Collection[Decoded]) -> typing.Collection[Decoded]:
		...


@typing.runtime_checkable
class Scorer[Target, Result](typing.Protocol):

	def __call__(self,
		true: typing.Collection[Target],
		pred: typing.Collection[Target], /
	) -> Result:
		...


@typing.runtime_checkable
class Encoder[Decoded, Encoded](typing.Protocol):

	def fit(self, source: typing.Collection[Decoded], signal: typing.Any | None = None, /) -> typing.Self:
		...

	def transform(self, source: typing.Collection[Decoded], /) -> typing.Collection[Encoded]:
		...


@typing.runtime_checkable
class Bicoder[Decoded, Encoded](Encoder[Decoded, Encoded], typing.Protocol):

	def inverse_transform(self, target: typing.Collection[Encoded], /) -> typing.Collection[Decoded]:
		...


@typing.runtime_checkable
class Model[Source, Target](typing.Protocol):

	def fit(self,
		source: typing.Collection[Source],
		target: typing.Collection[Target], /
	) -> typing.Self:
		...

	def predict(self, source: typing.Collection[Source], /) -> typing.Collection[Target]:
		...
