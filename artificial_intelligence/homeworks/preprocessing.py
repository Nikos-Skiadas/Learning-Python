"""Composable preprocessing utilities."""


from __future__ import annotations


from .protocols import Preprocessor


class ChainPreprocessor[Decoded]:
	"""Chains multiple preprocessors sequentially.

	Example:
		preprocessor = ChainPreprocessor(CleanText(), Lemmatize())
		result = preprocessor(source)
	"""

	def __init__(self, *preprocessors: Preprocessor[Decoded]) -> None:
		self.preprocessors = preprocessors

	def __call__(self, source: Decoded) -> Decoded:
		for preprocessor in self.preprocessors:
			source = preprocessor(source)
		return source


class IdentityPreprocessor[Decoded]:
	"""No-op preprocessor that returns input unchanged.

	Useful as a default when no preprocessing is needed.
	"""

	def __call__(self, source: Decoded) -> Decoded:
		return source
