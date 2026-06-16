"""Parse LLM generations into valid clarity-classification labels."""


from __future__ import annotations


import dataclasses
import json
import re
import typing

import pandas

from .prompting import CLARITY_LABELS, DEFAULT_LABEL_SPECS, LabelSpec


LABEL_FIELD_NAMES: tuple[str, ...] = (
	"label",
	"final_label",
	"predicted_label",
	"prediction",
	"class",
	"category",
)

RATIONALE_FIELD_NAMES: tuple[str, ...] = (
	"rationale",
	"reason",
	"explanation",
	"analysis",
)


@dataclasses.dataclass(frozen = True)
class ParsedGeneration:
	"""Structured parse result for one generated answer."""

	raw_text: str
	label: str | None
	valid: bool
	parse_method: str
	rationale: str | None = None
	message: str = ""


class LabelNormalizer:
	"""Normalize aliases and formatting variants to canonical labels."""

	def __init__(self, labels: tuple[LabelSpec, ...] = DEFAULT_LABEL_SPECS) -> None:
		self.labels = labels
		self.alias_to_label: dict[str, str] = {}
		for label in labels:
			self._register(label.name, label.name)
			self._register(label.name.replace("-", " "), label.name)
			self._register(label.name.replace("-", ""), label.name)
			for alias in label.aliases:
				self._register(alias, label.name)

	def normalize(self, value: typing.Any, /) -> str | None:
		text = _normalization_key(str(value))
		if not text:
			return None

		if text in self.alias_to_label:
			return self.alias_to_label[text]

		text = _strip_label_noise(text)
		if text in self.alias_to_label:
			return self.alias_to_label[text]

		return None

	def find_mentions(self, text: str, /) -> list[tuple[int, str]]:
		"""Return canonical label mentions found in text, ordered by position."""
		mentions: list[tuple[int, str]] = []
		for alias_key, label in self.alias_to_label.items():
			pattern = r"(?<!\w)" + re.escape(alias_key).replace(r"\ ", r"[\s-]+") + r"(?!\w)"
			for match in re.finditer(pattern, _normalization_key(text)):
				mentions.append((match.start(), label))

		mentions.sort(key = lambda item: item[0])

		return mentions

	def _register(self, alias: str, label: str, /) -> None:
		key = _normalization_key(alias)
		if key:
			self.alias_to_label[key] = label


class GenerationParser:
	"""Robust parser for JSON, labeled lines, and plain-label generations."""

	def __init__(
		self,
		labels: tuple[LabelSpec, ...] = DEFAULT_LABEL_SPECS,
		default_label: str | None = None,
	) -> None:
		self.normalizer = LabelNormalizer(labels)
		self.labels = tuple(label.name for label in labels)
		self.default_label = default_label

	def parse(self, text: typing.Any, /) -> ParsedGeneration:
		raw_text = "" if text is None else str(text).strip()

		for parsed_object in reversed(list(_iter_json_objects(raw_text))):
			parsed = self._parse_json_object(parsed_object, raw_text)
			if parsed.valid:
				return parsed

		parsed = self._parse_json_field_fragment(raw_text)
		if parsed.valid:
			return parsed

		parsed = self._parse_labeled_lines(raw_text)
		if parsed.valid:
			return parsed

		parsed = self._parse_plain_mentions(raw_text)
		if parsed.valid:
			return parsed

		if self.default_label is not None:
			return ParsedGeneration(
				raw_text = raw_text,
				label = self.default_label,
				valid = False,
				parse_method = "default",
				message = "No valid label found; default label used.",
			)

		return ParsedGeneration(
			raw_text = raw_text,
			label = None,
			valid = False,
			parse_method = "invalid",
			message = "No valid clarity label found.",
		)

	def parse_many(self, texts: typing.Iterable[typing.Any], /) -> list[ParsedGeneration]:
		return [self.parse(text) for text in texts]

	def predictions(
		self,
		texts: typing.Iterable[typing.Any],
		/,
		invalid_label: str | None = None,
	) -> pandas.Series:
		labels = [
			parsed.label if parsed.label is not None else invalid_label
			for parsed in self.parse_many(texts)
		]

		return pandas.Series(labels, name = "Predicted")

	def records_frame(self, texts: typing.Iterable[typing.Any], /) -> pandas.DataFrame:
		return pandas.DataFrame(
			dataclasses.asdict(parsed)
			for parsed in self.parse_many(texts)
		)

	def _parse_json_object(self, parsed_object: typing.Any, raw_text: str, /) -> ParsedGeneration:
		if not isinstance(parsed_object, dict):
			return ParsedGeneration(raw_text, None, False, "json", message = "JSON value was not an object.")

		label: str | None = None
		for key in LABEL_FIELD_NAMES:
			if key in parsed_object:
				label = self.normalizer.normalize(parsed_object[key])
				break

		if label is None:
			return ParsedGeneration(raw_text, None, False, "json", message = "JSON object did not contain a valid label.")

		rationale = None
		for key in RATIONALE_FIELD_NAMES:
			if key in parsed_object and parsed_object[key] is not None:
				rationale = str(parsed_object[key]).strip()
				break

		return ParsedGeneration(
			raw_text = raw_text,
			label = label,
			valid = True,
			parse_method = "json",
			rationale = rationale,
		)

	def _parse_json_field_fragment(self, raw_text: str, /) -> ParsedGeneration:
		"""Recover labels from truncated JSON-like generations.

		Small models often emit a correct leading ``"label": "..."`` field and then
		run out of tokens while writing the rationale. Treat that as a valid parse
		because the classification decision itself is already explicit.
		"""
		field_names = "|".join(re.escape(name) for name in LABEL_FIELD_NAMES)
		pattern = re.compile(
			rf"""(?is)["']?(?:{field_names})["']?\s*:\s*["']([^"'\n\r}}]+)["']"""
		)
		for match in reversed(list(pattern.finditer(raw_text))):
			label = self.normalizer.normalize(match.group(1))
			if label is not None:
				return ParsedGeneration(
					raw_text = raw_text,
					label = label,
					valid = True,
					parse_method = "json_field_fragment",
					message = "Recovered label from a JSON-like field in a truncated generation.",
				)

		return ParsedGeneration(raw_text, None, False, "json_field_fragment", message = "No JSON-like label field found.")

	def _parse_labeled_lines(self, raw_text: str, /) -> ParsedGeneration:
		pattern = re.compile(
			r"(?im)^\s*(?:final\s+)?(?:label|answer|prediction|class|category)\s*[:=-]\s*(.+?)\s*$"
		)
		for match in reversed(list(pattern.finditer(raw_text))):
			label = self.normalizer.normalize(match.group(1))
			if label is not None:
				return ParsedGeneration(
					raw_text = raw_text,
					label = label,
					valid = True,
					parse_method = "labeled_line",
				)

		return ParsedGeneration(raw_text, None, False, "labeled_line", message = "No labeled line matched.")

	def _parse_plain_mentions(self, raw_text: str, /) -> ParsedGeneration:
		mentions = self.normalizer.find_mentions(raw_text)
		if not mentions:
			return ParsedGeneration(raw_text, None, False, "mention", message = "No label mention found.")

		unique_labels = {label for _, label in mentions}
		if len(unique_labels) == 1:
			return ParsedGeneration(
				raw_text = raw_text,
				label = next(iter(unique_labels)),
				valid = True,
				parse_method = "single_mention",
			)

		last_position, last_label = mentions[-1]
		normalized_text = _normalization_key(raw_text)
		last_window = normalized_text[max(0, last_position - 40):last_position + 120]
		if re.search(r"\b(final|therefore|answer|label|prediction)\b", last_window):
			return ParsedGeneration(
				raw_text = raw_text,
				label = last_label,
				valid = True,
				parse_method = "final_mention",
			)

		return ParsedGeneration(
			raw_text = raw_text,
			label = None,
			valid = False,
			parse_method = "ambiguous_mentions",
			message = f"Multiple labels mentioned: {sorted(unique_labels)}",
		)


def submission_frame(
	predictions: typing.Iterable[str | None],
	/,
	index: typing.Iterable[typing.Hashable] | None = None,
	invalid_label: str = "Ambivalent",
) -> pandas.DataFrame:
	"""Build the Kaggle submission shape: Id, Predicted."""
	values = [
		prediction if prediction in CLARITY_LABELS else invalid_label
		for prediction in predictions
	]
	frame = pandas.DataFrame({"Predicted": values})
	frame.index.name = "Id"
	if index is not None:
		frame.index = list(index)
		frame.index.name = "Id"

	return frame


def _iter_json_objects(text: str, /) -> typing.Iterator[typing.Any]:
	decoder = json.JSONDecoder()
	for start in [match.start() for match in re.finditer(r"\{", text)]:
		try:
			parsed, _ = decoder.raw_decode(text[start:])
		except json.JSONDecodeError:
			continue

		yield parsed


def _normalization_key(text: str, /) -> str:
	text = text.casefold()
	text = re.sub(r"[`\"'*_{}\[\]().,:;!?]", " ", text)
	text = re.sub(r"[/_]+", " ", text)
	text = re.sub(r"\s+", " ", text)

	return text.strip()


def _strip_label_noise(text: str, /) -> str:
	text = re.sub(r"^(the\s+)?(final\s+)?(label|answer|prediction|class|category)\s+(is|=|:)?\s+", "", text)
	text = re.sub(r"\s+(because|since|as)\s+.*$", "", text)

	return text.strip()
