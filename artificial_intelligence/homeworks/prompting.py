"""Prompt construction utilities for clarity classification."""


from __future__ import annotations


import dataclasses
import random
import re
import typing

import pandas
import sklearn.base


CLARITY_LABELS: tuple[str, ...] = (
	"Clear Reply",
	"Ambivalent",
	"Clear Non-Reply",
)


@dataclasses.dataclass(frozen = True)
class LabelSpec:
	"""Canonical label name, aliases, and task-facing definition."""

	name: str
	description: str
	aliases: tuple[str, ...] = ()


@dataclasses.dataclass(frozen = True)
class EvasionSpec:
	"""Fine-grained evasion subtype and its parent clarity label."""

	name: str
	clarity_label: str
	description: str


DEFAULT_LABEL_SPECS: tuple[LabelSpec, ...] = (
	LabelSpec(
		name = "Clear Reply",
		description = "The answer explicitly, directly, and specifically addresses the main question, even if it includes extra context.",
		aliases = ("direct reply", "clear answer", "direct answer", "explicit reply", "explicit answer", "explicit"),
	),
	LabelSpec(
		name = "Ambivalent",
		description = "The answer is partly responsive but incomplete, implicit, vague, hedged, mixed, conditional, deflective, or mostly topic-adjacent rather than explicitly answering.",
		aliases = ("ambivalent reply", "ambiguous", "partial reply", "unclear", "implicit", "dodging", "general", "deflection", "partial", "partial half answer", "half answer"),
	),
	LabelSpec(
		name = "Clear Non-Reply",
		description = "The answer clearly avoids, redirects, refuses, or otherwise provides no substantive answer to the main question.",
		aliases = ("clear not reply", "non-reply", "non reply", "not reply", "evasion", "declining", "declining to answer", "ignorance", "claims ignorance", "clarification"),
	),
)


EVASION_DESCRIPTIONS: dict[str, str] = {
	"Explicit": "Directly and explicitly answers the question.",
	"Implicit": "Gives an implied or indirect answer, but not a fully explicit one.",
	"Dodging": "Moves around the question while still touching related material.",
	"General": "Responds with broad or generic statements instead of the requested specific answer.",
	"Deflection": "Shifts emphasis toward another issue while keeping some connection to the question.",
	"Partial": "Answers only part of a multi-part or specific question.",
	"Partial/half-answer": "Answers only part of the question or gives a half-answer that leaves the requested point unresolved.",
	"Declining": "Openly refuses or declines to answer.",
	"Declining to answer": "Openly refuses or declines to answer.",
	"Ignorance": "States lack of knowledge or inability to answer.",
	"Claims ignorance": "States lack of knowledge or inability to answer.",
	"Clarification": "Asks for clarification or challenges the question instead of answering it.",
}


EVASION_SPECS: tuple[EvasionSpec, ...] = (
	EvasionSpec(
		name = "Explicit",
		clarity_label = "Clear Reply",
		description = EVASION_DESCRIPTIONS["Explicit"],
	),
	EvasionSpec(
		name = "Implicit",
		clarity_label = "Ambivalent",
		description = EVASION_DESCRIPTIONS["Implicit"],
	),
	EvasionSpec(
		name = "Dodging",
		clarity_label = "Ambivalent",
		description = EVASION_DESCRIPTIONS["Dodging"],
	),
	EvasionSpec(
		name = "General",
		clarity_label = "Ambivalent",
		description = EVASION_DESCRIPTIONS["General"],
	),
	EvasionSpec(
		name = "Deflection",
		clarity_label = "Ambivalent",
		description = EVASION_DESCRIPTIONS["Deflection"],
	),
	EvasionSpec(
		name = "Partial",
		clarity_label = "Ambivalent",
		description = EVASION_DESCRIPTIONS["Partial"],
	),
	EvasionSpec(
		name = "Declining",
		clarity_label = "Clear Non-Reply",
		description = EVASION_DESCRIPTIONS["Declining"],
	),
	EvasionSpec(
		name = "Ignorance",
		clarity_label = "Clear Non-Reply",
		description = EVASION_DESCRIPTIONS["Ignorance"],
	),
	EvasionSpec(
		name = "Clarification",
		clarity_label = "Clear Non-Reply",
		description = EVASION_DESCRIPTIONS["Clarification"],
	),
)


@dataclasses.dataclass(frozen = True)
class PromptExample:
	"""One labeled demonstration example for in-context learning."""

	question: str
	answer: str
	label: str
	evasion_label: str | None = None
	rationale: str | None = None
	source_index: typing.Hashable | None = None

	@classmethod
	def from_record(
		cls,
		record: typing.Mapping[str, typing.Any] | pandas.Series,
		/,
		question_key: str = "question",
		answer_key: str = "interview_answer",
		label_key: str = "clarity_label",
		evasion_key: str = "evasion_label",
		rationale_key: str | None = None,
		source_index: typing.Hashable | None = None,
	) -> typing.Self:
		rationale = None
		if rationale_key is not None and rationale_key in record:
			rationale = clean_text(record[rationale_key])

		return cls(
			question = clean_text(record.get(question_key, "")),
			answer = clean_text(record.get(answer_key, "")),
			label = clean_text(record.get(label_key, "")),
			evasion_label = clean_text(record.get(evasion_key, "")) or None,
			rationale = rationale,
			source_index = source_index,
		)


@dataclasses.dataclass(frozen = True)
class PromptConfig:
	"""Prompt-format switches used by PromptBuilder."""

	name: str = "zero_shot"
	task_description: str = "Classify how clearly the interview answer responds to the question."
	include_label_definitions: bool = True
	include_decision_rules: bool = True
	include_evasion_taxonomy: bool = True
	include_output_schema: bool = True
	include_context: bool = False
	context_keys: tuple[str, ...] = ()
	reasoning: typing.Literal["none", "concise", "step_by_step", "self_check"] = "none"
	use_json: bool = True
	evasion_taxonomy_style: typing.Literal["descriptive", "bare"] = "descriptive"
	include_example_evasion: bool = True
	include_example_rationales: bool = False
	max_question_chars: int | None = None
	max_answer_chars: int | None = None
	max_context_chars: int | None = 500
	example_header: str = "Examples"
	instance_header: str = "Instance to classify"


def clean_text(value: typing.Any, /) -> str:
	"""Return a single-line, whitespace-normalized text value."""
	if pandas.isna(value):
		return ""

	return " ".join(str(value).split())


def canonical_clarity_label(value: typing.Any, /) -> str:
	text = clean_text(value)
	normalized = re.sub(r"[-_]+", " ", text).casefold()
	normalized = re.sub(r"\s+", " ", normalized).strip()

	if normalized in {"clear reply", "explicit"}:
		return "Clear Reply"
	if normalized in {"ambivalent", "ambivalent reply"}:
		return "Ambivalent"
	if normalized in {"clear non reply", "clear not reply", "clear non-reply", "clear not-reply"}:
		return "Clear Non-Reply"

	return text


def canonical_evasion_label(value: typing.Any, /) -> str:
	text = clean_text(value)
	text = re.sub(r"^\d+(?:\.\d+)*\s*", "", text)
	for known in EVASION_DESCRIPTIONS:
		if text.casefold() == known.casefold():
			return known

	return text


def infer_evasion_specs(
	source: pandas.DataFrame,
	/,
	label_key: str = "clarity_label",
	evasion_key: str = "evasion_label",
	fallback: tuple[EvasionSpec, ...] = EVASION_SPECS,
) -> tuple[EvasionSpec, ...]:
	"""Infer the observed high-level/fine-grained label taxonomy from data."""
	if label_key not in source or evasion_key not in source:
		return fallback

	pairs: set[tuple[str, str]] = set()
	for _, record in source[[label_key, evasion_key]].fillna("").iterrows():
		label = canonical_clarity_label(record[label_key])
		evasion = canonical_evasion_label(record[evasion_key])
		if label and evasion:
			pairs.add((label, evasion))

	if not pairs:
		return fallback

	order = {label: i for i, label in enumerate(CLARITY_LABELS)}

	return tuple(
		EvasionSpec(
			name = evasion,
			clarity_label = label,
			description = EVASION_DESCRIPTIONS.get(evasion, f"Fine-grained subtype observed in the dataset under {label}."),
		)
		for label, evasion in sorted(pairs, key = lambda pair: (order.get(pair[0], len(order)), pair[1]))
	)


def truncate_text(text: str, max_chars: int | None, /) -> str:
	if max_chars is None or len(text) <= max_chars:
		return text

	if max_chars <= 3:
		return text[:max_chars]

	marker = " ... "
	available = max_chars - len(marker)
	if available <= 0:
		return text[:max_chars]

	head = max(1, available // 2)
	tail = max(1, available - head)

	return text[:head].rstrip() + marker + text[-tail:].lstrip()


class PromptBuilder:
	"""Build zero-shot, few-shot, context-aware, and CoT-style prompts."""

	def __init__(
		self,
		config: PromptConfig | None = None,
		labels: tuple[LabelSpec, ...] = DEFAULT_LABEL_SPECS,
		evasion_specs: tuple[EvasionSpec, ...] = EVASION_SPECS,
		question_key: str = "question",
		answer_key: str = "interview_answer",
	) -> None:
		self.config = config or PromptConfig()
		self.labels = labels
		self.evasion_specs = evasion_specs
		self.question_key = question_key
		self.answer_key = answer_key

	@property
	def label_names(self) -> tuple[str, ...]:
		return tuple(label.name for label in self.labels)

	def build(
		self,
		record: typing.Mapping[str, typing.Any] | pandas.Series,
		/,
		examples: typing.Sequence[PromptExample] = (),
		context: typing.Mapping[str, typing.Any] | None = None,
	) -> str:
		"""Build a complete prompt for one question-answer record."""
		parts = [
			self.config.task_description,
			self._label_instruction(),
		]

		if self.config.include_label_definitions:
			parts.append(self._label_block())

		if self.config.include_decision_rules:
			parts.append(self._decision_rules_block())

		if self.config.include_evasion_taxonomy:
			parts.append(self._evasion_taxonomy_block())

		if examples:
			parts.append(self._examples_block(examples))

		context_block = self._context_block(record, context)
		if context_block:
			parts.append(context_block)

		parts.append(self._instance_block(record))
		parts.append(self._reasoning_block())

		if self.config.include_output_schema:
			parts.append(self._output_block())

		return "\n\n".join(part for part in parts if part)

	def build_many(
		self,
		source: pandas.DataFrame,
		/,
		examples: typing.Sequence[PromptExample] = (),
	) -> list[str]:
		return [
			self.build(record, examples = examples)
			for _, record in source.iterrows()
		]

	def _label_instruction(self) -> str:
		labels = ", ".join(self.label_names)

		return f"Use exactly one of these labels: {labels}."

	def _label_block(self) -> str:
		lines = ["Label definitions:"]
		for label in self.labels:
			lines.append(f"- {label.name}: {label.description}")

		return "\n".join(lines)

	def _decision_rules_block(self) -> str:
		return "\n".join([
			"Decision rules:",
			"- Judge the answer relative to the exact question, not by general topical relevance.",
			"- Choose Clear Reply only when the main question is answered explicitly, directly, and specifically.",
			"- Before choosing Clear Reply, rule out the Ambivalent subtypes; if the answer is implicit, partial, generic, deflective, or dodging, choose Ambivalent.",
			"- Choose Clear Non-Reply only when the answer gives no substantive answer and mainly redirects, refuses, or evades.",
			"- Choose Ambivalent for the middle cases: partial answer, mixed answer, hedging, conditional answer, broad/general answer, or answer that touches the topic but leaves the main question unresolved.",
			"- If the answer contains both responsive material and substantial evasion or uncertainty, choose Ambivalent rather than Clear Reply.",
			"- Do not reward length: a long answer can still be Ambivalent or Clear Non-Reply.",
		])

	def _evasion_taxonomy_block(self) -> str:
		if self.config.evasion_taxonomy_style == "bare":
			lines = ["Label taxonomy:"]
			for label in self.label_names:
				subtypes = [spec.name for spec in self.evasion_specs if spec.clarity_label == label]
				if subtypes:
					lines.append(f"- {label}: {', '.join(subtypes)}")

			return "\n".join(lines)

		lines = [
			"Fine-grained evasion taxonomy:",
			"- First judge whether the answer resembles one of these response/evasion subtypes, then map it to the parent clarity label.",
			"- Output only the parent clarity label, not the subtype.",
		]
		for label in self.label_names:
			subtypes = [spec for spec in self.evasion_specs if spec.clarity_label == label]
			if not subtypes:
				continue

			lines.append(f"- {label}:")
			for spec in subtypes:
				lines.append(f"  - {spec.name}: {spec.description}")

		return "\n".join(lines)

	def _examples_block(self, examples: typing.Sequence[PromptExample], /) -> str:
		lines = [f"{self.config.example_header}:"]

		for i, example in enumerate(examples, start = 1):
			lines.extend([
				f"Example {i}:",
				f"Question: {truncate_text(example.question, self.config.max_question_chars)}",
				f"Answer: {truncate_text(example.answer, self.config.max_answer_chars)}",
				f"Label: {example.label}",
			])
			if self.config.include_example_evasion and example.evasion_label:
				lines.append(f"Evasion subtype: {example.evasion_label}")
			if self.config.include_example_rationales and example.rationale:
				lines.append(f"Rationale: {example.rationale}")
			lines.append("")

		while lines and lines[-1] == "":
			lines.pop()

		return "\n".join(lines)

	def _context_block(
		self,
		record: typing.Mapping[str, typing.Any] | pandas.Series,
		context: typing.Mapping[str, typing.Any] | None,
	) -> str:
		if not self.config.include_context:
			return ""

		items: list[tuple[str, str]] = []
		for key in self.config.context_keys:
			if key in record:
				items.append((key, clean_text(record[key])))

		if context is not None:
			for key, value in context.items():
				items.append((key, clean_text(value)))

		lines = [
			f"- {key}: {truncate_text(value, self.config.max_context_chars)}"
			for key, value in items
			if value
		]

		if not lines:
			return ""

		return "Additional context:\n" + "\n".join(lines)

	def _instance_block(self, record: typing.Mapping[str, typing.Any] | pandas.Series, /) -> str:
		question = truncate_text(clean_text(record.get(self.question_key, "")), self.config.max_question_chars)
		answer = truncate_text(clean_text(record.get(self.answer_key, "")), self.config.max_answer_chars)

		return "\n".join([
			f"{self.config.instance_header}:",
			f"Question: {question}",
			f"Answer: {answer}",
		])

	def _reasoning_block(self) -> str:
		match self.config.reasoning:
			case "none":
				return "Decide the label from the answer's responsiveness to the question. Do not output explanations."
			case "concise":
				return "Briefly justify the decision using evidence from the question-answer pair."
			case "step_by_step":
				return (
					"Analyze the response step by step: whether it addresses the asked content, "
					"whether it is specific or vague, and whether it redirects away from the question."
				)
			case "self_check":
				return (
					"First decide the most likely label, then check that it is one of the allowed labels "
					"and that the answer is judged relative to the question rather than in isolation."
				)

		raise ValueError(f"Unknown reasoning mode: {self.config.reasoning}")

	def _output_block(self) -> str:
		label_hint = " | ".join(self.label_names)

		if not self.config.use_json:
			return (
				"Output constraints:\n"
				f"- Your entire response must be exactly one of these labels: {label_hint}.\n"
				"- Do not output JSON.\n"
				"- Do not output explanations, markdown, punctuation, or any extra text."
			)

		if self.config.reasoning == "none":
			return (
				f'Return valid JSON only: {{"label": "<{label_hint}>"}}. '
				"Do not include markdown, comments, reasoning, or any extra keys."
			)

		return (
			f'Return valid JSON only: {{"label": "<{label_hint}>", "rationale": "<brief justification>"}}. '
			"Do not include markdown, comments, or any extra keys."
		)


class FewShotSampler(sklearn.base.BaseEstimator):
	"""Select labeled demonstrations from the training split."""

	def __init__(
		self,
		k: int = 0,
		k_per_label: int | None = None,
		strategy: typing.Literal["balanced", "random", "length_matched"] = "balanced",
		seed: int = 42,
		question_key: str = "question",
		answer_key: str = "interview_answer",
		label_key: str = "clarity_label",
		rationale_key: str | None = None,
		evasion_key: str = "evasion_label",
		labels: tuple[str, ...] = CLARITY_LABELS,
	) -> None:
		self.k = k
		self.k_per_label = k_per_label
		self.strategy = strategy
		self.seed = seed
		self.question_key = question_key
		self.answer_key = answer_key
		self.label_key = label_key
		self.rationale_key = rationale_key
		self.evasion_key = evasion_key
		self.labels = labels
		self._examples: list[PromptExample] = []

	def fit(self, source: pandas.DataFrame, signal: typing.Any | None = None, /) -> typing.Self:
		frame = source.copy()
		if signal is not None and self.label_key not in frame:
			frame[self.label_key] = list(signal)

		self._examples = [
			PromptExample.from_record(
				record,
				question_key = self.question_key,
				answer_key = self.answer_key,
				label_key = self.label_key,
				evasion_key = self.evasion_key,
				rationale_key = self.rationale_key,
				source_index = index,
			)
			for index, record in frame.iterrows()
			if clean_text(record.get(self.label_key, ""))
		]

		return self

	def select(
		self,
		record: typing.Mapping[str, typing.Any] | pandas.Series | None = None,
		/,
		exclude_index: typing.Hashable | None = None,
	) -> list[PromptExample]:
		"""Return examples for one prompt, optionally matched to a target record."""
		candidates = [
			example for example in self._examples
			if exclude_index is None or example.source_index != exclude_index
		]
		if not candidates:
			return []

		match self.strategy:
			case "balanced":
				return self._balanced(candidates)
			case "random":
				return self._random(candidates)
			case "length_matched":
				if record is None:
					return self._balanced(candidates)

				return self._length_matched(candidates, record)

		raise ValueError(f"Unknown few-shot strategy: {self.strategy}")

	def _balanced(self, candidates: list[PromptExample], /) -> list[PromptExample]:
		rng = random.Random(self.seed)
		per_label = self.k_per_label
		if per_label is None:
			per_label = max(1, self.k // max(1, len(self.labels))) if self.k else 1

		selected: list[PromptExample] = []
		for label in self.labels:
			label_examples = [example for example in candidates if example.label == label]
			rng.shuffle(label_examples)
			selected.extend(label_examples[:per_label])

		if self.k:
			selected = selected[:self.k]

		return selected

	def _random(self, candidates: list[PromptExample], /) -> list[PromptExample]:
		rng = random.Random(self.seed)
		k = self.k or len(self.labels)

		return rng.sample(candidates, k = min(k, len(candidates)))

	def _length_matched(
		self,
		candidates: list[PromptExample],
		record: typing.Mapping[str, typing.Any] | pandas.Series,
		/,
	) -> list[PromptExample]:
		target_length = _qa_length(
			clean_text(record.get(self.question_key, "")),
			clean_text(record.get(self.answer_key, "")),
		)

		per_label = self.k_per_label
		if per_label is None:
			per_label = max(1, self.k // max(1, len(self.labels))) if self.k else 1

		selected: list[PromptExample] = []
		for label in self.labels:
			label_examples = [example for example in candidates if example.label == label]
			label_examples.sort(key = lambda example: abs(_qa_length(example.question, example.answer) - target_length))
			selected.extend(label_examples[:per_label])

		if self.k:
			selected = selected[:self.k]

		return selected


class PromptEncoder(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
	"""sklearn-style encoder that turns rows into prompts."""

	def __init__(
		self,
		builder: PromptBuilder | None = None,
		fewshot_sampler: FewShotSampler | None = None,
		exclude_self_from_fewshot: bool = False,
	) -> None:
		self.builder = builder or PromptBuilder()
		self.fewshot_sampler = fewshot_sampler
		self.exclude_self_from_fewshot = exclude_self_from_fewshot

	def fit(self, source: pandas.DataFrame, signal: typing.Any | None = None, /) -> typing.Self:
		if self.fewshot_sampler is not None:
			self.fewshot_sampler.fit(source, signal)

		return self

	def transform(self, source: pandas.DataFrame, /) -> list[str]:
		prompts: list[str] = []
		for index, record in source.iterrows():
			examples: list[PromptExample] = []
			if self.fewshot_sampler is not None:
				exclude_index = index if self.exclude_self_from_fewshot else None
				examples = self.fewshot_sampler.select(record, exclude_index = exclude_index)

			prompts.append(self.builder.build(record, examples = examples))

		return prompts


def _qa_length(question: str, answer: str, /) -> int:
	return len(question.split()) + len(answer.split())
