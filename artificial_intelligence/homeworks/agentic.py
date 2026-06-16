"""D3-style agentic prompting helpers for response clarity classification."""

from __future__ import annotations

import dataclasses
import typing

import pandas

from .generation import DecodingConfig, TextGenerator
from .parsing import GenerationParser
from .prompting import (
	CLARITY_LABELS,
	EVASION_SPECS,
	EvasionSpec,
	clean_text,
	truncate_text,
)


@dataclasses.dataclass(frozen = True)
class D3PromptSettings:
	"""Prompt-shaping controls for the four D3 agents."""

	max_question_chars: int | None = 300
	max_answer_chars: int | None = 900
	max_intermediate_chars: int | None = 1200
	include_evasion_taxonomy: bool = True
	require_json: bool = True


@dataclasses.dataclass(frozen = True)
class AgentPrompt:
	"""One agent prompt with an analysis-friendly role name."""

	agent: str
	prompt: str


class D3PromptBuilder:
	"""Construct prompts for the required D3 agents.

	D3 decomposes response clarity into question intent, answer content, gap/evasion
	analysis, and final decision. The first three agents are not allowed to emit a
	final clarity label; only the Decision Agent does.
	"""

	def __init__(
		self,
		settings: D3PromptSettings | None = None,
		evasion_specs: tuple[EvasionSpec, ...] = EVASION_SPECS,
		question_key: str = "question",
		answer_key: str = "interview_answer",
	) -> None:
		self.settings = settings or D3PromptSettings()
		self.evasion_specs = evasion_specs
		self.question_key = question_key
		self.answer_key = answer_key

	def question_intent_prompt(self, record: typing.Mapping[str, typing.Any] | pandas.Series, /) -> AgentPrompt:
		question = self._question(record)
		return AgentPrompt(
			agent = "question_intent",
			prompt = "\n\n".join([
				"You are the Question Intent Agent in a D3 response-clarity pipeline.",
				"Your task is to identify what the journalist is asking for. Do not classify the answer.",
				f"Question:\n{question}",
				"Return valid JSON only with these keys: "
				'"asked_for", "required_information", "question_focus". '
				"Keep each value brief and concrete.",
			]),
		)

	def answer_content_prompt(self, record: typing.Mapping[str, typing.Any] | pandas.Series, /) -> AgentPrompt:
		question = self._question(record)
		answer = self._answer(record)
		return AgentPrompt(
			agent = "answer_content",
			prompt = "\n\n".join([
				"You are the Answer Content Agent in a D3 response-clarity pipeline.",
				"Your task is to extract what the answer actually says. Do not classify the answer.",
				f"Question:\n{question}",
				f"Answer:\n{answer}",
				"Return valid JSON only with these keys: "
				'"explicit_claims", "relevant_details", "hedges_or_qualifiers", "omitted_points". '
				"Use short phrases or compact lists.",
			]),
		)

	def gap_evasion_prompt(
		self,
		record: typing.Mapping[str, typing.Any] | pandas.Series,
		/,
		question_intent: str,
		answer_content: str,
	) -> AgentPrompt:
		question = self._question(record)
		answer = self._answer(record)
		parts = [
			"You are the Gap and Evasion Agent in a D3 response-clarity pipeline.",
			"Compare the question intent with the answer content. Do not output the final clarity label.",
			self._taxonomy_block(),
			f"Question:\n{question}",
			f"Answer:\n{answer}",
			"Question Intent Agent output:\n" + self._intermediate(question_intent),
			"Answer Content Agent output:\n" + self._intermediate(answer_content),
			"Return valid JSON only with these keys: "
			'"matched_requirements", "missing_requirements", "evasion_patterns", '
			'"responsiveness_summary", "likely_subtype".',
		]
		return AgentPrompt(agent = "gap_evasion", prompt = "\n\n".join(part for part in parts if part))

	def decision_prompt(
		self,
		record: typing.Mapping[str, typing.Any] | pandas.Series,
		/,
		question_intent: str,
		answer_content: str,
		gap_evasion: str,
	) -> AgentPrompt:
		question = self._question(record)
		answer = self._answer(record)
		labels = " | ".join(CLARITY_LABELS)
		parts = [
			"You are the Decision Agent in a D3 response-clarity pipeline.",
			"Use the previous agents' outputs to assign exactly one final clarity label.",
			"Decision rules:",
			"- Clear Reply: the answer explicitly and specifically answers the main question.",
			"- Ambivalent: the answer is partial, implicit, generic, hedged, deflective, or only partly responsive.",
			"- Clear Non-Reply: the answer clearly refuses, redirects, asks for clarification, claims ignorance, or gives no substantive answer.",
			"- Judge the answer relative to the exact question. Do not reward length or topical fluency.",
			self._taxonomy_block(),
			f"Question:\n{question}",
			f"Answer:\n{answer}",
			"Question Intent Agent output:\n" + self._intermediate(question_intent),
			"Answer Content Agent output:\n" + self._intermediate(answer_content),
			"Gap and Evasion Agent output:\n" + self._intermediate(gap_evasion),
			"Output exactly one compact JSON object on one line. Do not include rationale, markdown, or extra keys.",
			f'Return only: {{"label": "<{labels}>"}}.',
		]
		return AgentPrompt(agent = "decision", prompt = "\n\n".join(part for part in parts if part))

	def direct_decision_prompt(self, record: typing.Mapping[str, typing.Any] | pandas.Series, /) -> AgentPrompt:
		"""Single-agent comparator using the same label guidance as the D3 decision agent."""
		question = self._question(record)
		answer = self._answer(record)
		labels = " | ".join(CLARITY_LABELS)
		parts = [
			"Classify the interview answer's responsiveness to the question.",
			f"Use exactly one label: {labels}.",
			self._taxonomy_block(),
			"Decision rules:",
			"- Clear Reply only if the answer explicitly and specifically answers the main question.",
			"- Ambivalent for partial, implicit, generic, hedged, deflective, or partly responsive answers.",
			"- Clear Non-Reply for refusals, redirections, clarification requests, ignorance claims, or no substantive answer.",
			f"Question:\n{question}",
			f"Answer:\n{answer}",
			"Output exactly one compact JSON object on one line. Do not include rationale, markdown, or extra keys.",
			f'Return only: {{"label": "<{labels}>"}}.',
		]
		return AgentPrompt(agent = "direct_decision", prompt = "\n\n".join(part for part in parts if part))

	def _question(self, record: typing.Mapping[str, typing.Any] | pandas.Series, /) -> str:
		return truncate_text(clean_text(record.get(self.question_key, "")), self.settings.max_question_chars)

	def _answer(self, record: typing.Mapping[str, typing.Any] | pandas.Series, /) -> str:
		return truncate_text(clean_text(record.get(self.answer_key, "")), self.settings.max_answer_chars)

	def _intermediate(self, value: str, /) -> str:
		return truncate_text(clean_text(value), self.settings.max_intermediate_chars)

	def _taxonomy_block(self) -> str:
		if not self.settings.include_evasion_taxonomy:
			return ""

		lines = ["Dataset label taxonomy:"]
		for label in CLARITY_LABELS:
			subtypes = [spec.name for spec in self.evasion_specs if spec.clarity_label == label]
			if subtypes:
				lines.append(f"- {label}: {', '.join(subtypes)}")

		return "\n".join(lines)


class D3AgenticClassifier:
	"""Run the four-agent D3 pipeline and parse the final decision."""

	def __init__(
		self,
		generator: TextGenerator,
		builder: D3PromptBuilder | None = None,
		parser: GenerationParser | None = None,
	) -> None:
		self.generator = generator
		self.builder = builder or D3PromptBuilder()
		self.parser = parser or GenerationParser()

	def fit(self, source: pandas.DataFrame, target: typing.Any | None = None, /) -> typing.Self:
		return self

	def generate_frame(
		self,
		source: pandas.DataFrame,
		/,
		include_prompts: bool = True,
	) -> pandas.DataFrame:
		source = source.copy()

		intent_prompts = [self.builder.question_intent_prompt(record).prompt for _, record in source.iterrows()]
		intent_generations = self.generator.generate(intent_prompts)

		content_prompts = [self.builder.answer_content_prompt(record).prompt for _, record in source.iterrows()]
		content_generations = self.generator.generate(content_prompts)

		gap_prompts = [
			self.builder.gap_evasion_prompt(record, intent, content).prompt
			for (_, record), intent, content in zip(source.iterrows(), intent_generations, content_generations, strict = True)
		]
		gap_generations = self.generator.generate(gap_prompts)

		decision_prompts = [
			self.builder.decision_prompt(record, intent, content, gap).prompt
			for (_, record), intent, content, gap in zip(
				source.iterrows(),
				intent_generations,
				content_generations,
				gap_generations,
				strict = True,
			)
		]
		decision_generations = self.generator.generate(decision_prompts)
		parsed = self.parser.parse_many(decision_generations)

		frame = pandas.DataFrame({
			"question_intent_generation": intent_generations,
			"answer_content_generation": content_generations,
			"gap_evasion_generation": gap_generations,
			"decision_generation": decision_generations,
			"Predicted": [item.label for item in parsed],
			"valid": [item.valid for item in parsed],
			"parse_method": [item.parse_method for item in parsed],
			"parse_message": [item.message for item in parsed],
			"rationale": [item.rationale for item in parsed],
		}, index = source.index)
		frame.index.name = source.index.name or "Id"

		if include_prompts:
			frame.insert(0, "question_intent_prompt", intent_prompts)
			frame.insert(2, "answer_content_prompt", content_prompts)
			frame.insert(4, "gap_evasion_prompt", gap_prompts)
			frame.insert(6, "decision_prompt", decision_prompts)

		return frame

	def predict(self, source: pandas.DataFrame, /) -> pandas.Series:
		return self.generate_frame(source, include_prompts = False)["Predicted"].rename("Predicted")


class DirectAgenticComparator:
	"""Single-agent comparator using the D3 builder's direct decision prompt."""

	def __init__(
		self,
		generator: TextGenerator,
		builder: D3PromptBuilder | None = None,
		parser: GenerationParser | None = None,
	) -> None:
		self.generator = generator
		self.builder = builder or D3PromptBuilder()
		self.parser = parser or GenerationParser()

	def fit(self, source: pandas.DataFrame, target: typing.Any | None = None, /) -> typing.Self:
		return self

	def generate_frame(
		self,
		source: pandas.DataFrame,
		/,
		include_prompts: bool = True,
	) -> pandas.DataFrame:
		prompts = [self.builder.direct_decision_prompt(record).prompt for _, record in source.iterrows()]
		generations = self.generator.generate(prompts)
		parsed = self.parser.parse_many(generations)

		frame = pandas.DataFrame({
			"decision_generation": generations,
			"Predicted": [item.label for item in parsed],
			"valid": [item.valid for item in parsed],
			"parse_method": [item.parse_method for item in parsed],
			"parse_message": [item.message for item in parsed],
			"rationale": [item.rationale for item in parsed],
		}, index = source.index)
		frame.index.name = source.index.name or "Id"

		if include_prompts:
			frame.insert(0, "decision_prompt", prompts)

		return frame

	def predict(self, source: pandas.DataFrame, /) -> pandas.Series:
		return self.generate_frame(source, include_prompts = False)["Predicted"].rename("Predicted")
