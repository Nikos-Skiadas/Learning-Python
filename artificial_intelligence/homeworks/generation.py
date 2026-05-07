"""Prompted generation helpers for instruction-tuned causal LMs."""


from __future__ import annotations


import dataclasses
import pathlib
import random
import typing

import numpy
import pandas

from .parsing import GenerationParser
from .prompting import PromptEncoder


@dataclasses.dataclass(frozen = True)
class DecodingConfig:
	"""Generation settings that should be logged with each experiment."""

	max_new_tokens: int = 48
	do_sample: bool = False
	temperature: float = 0.0
	top_p: float = 1.0
	repetition_penalty: float = 1.0


@typing.runtime_checkable
class TextGenerator(typing.Protocol):
	"""Protocol for prompt -> text generation backends."""

	def generate(self, prompts: typing.Sequence[str], /) -> list[str]:
		...


def seed_everything(seed: int = 42) -> None:
	"""Seed Python, NumPy, and PyTorch if available."""
	random.seed(seed)
	numpy.random.seed(seed)

	try:
		import torch
	except ImportError:
		return

	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


class StaticGenerator:
	"""Small deterministic generator useful for parser and notebook smoke tests."""

	def __init__(self, output: str = '{"label": "Ambivalent"}') -> None:
		self.output = output

	def generate(self, prompts: typing.Sequence[str], /) -> list[str]:
		return [self.output for _ in prompts]


class HuggingFaceCausalLMGenerator:
	"""Generate continuations with AutoModelForCausalLM.

	The implementation is intentionally thin so that Kaggle notebooks can still
	control model names, device maps, dtypes, and decoding settings explicitly.
	"""

	def __init__(
		self,
		model_name: str,
		decoding: DecodingConfig | None = None,
		batch_size: int = 1,
		device_map: str | dict[str, typing.Any] | None = "auto",
		torch_dtype: str | typing.Any = "auto",
		trust_remote_code: bool = True,
		use_chat_template: bool = True,
		system_message: str | None = None,
	) -> None:
		self.model_name = model_name
		self.decoding = decoding or DecodingConfig()
		self.batch_size = batch_size
		self.device_map = device_map
		self.torch_dtype = torch_dtype
		self.trust_remote_code = trust_remote_code
		self.use_chat_template = use_chat_template
		self.system_message = system_message
		self._tokenizer = None
		self._model = None

	def generate(self, prompts: typing.Sequence[str], /) -> list[str]:
		self._load()
		assert self._tokenizer is not None
		assert self._model is not None

		import torch

		outputs: list[str] = []
		for start in range(0, len(prompts), self.batch_size):
			batch_prompts = [
				self._format_prompt(prompt)
				for prompt in prompts[start:start + self.batch_size]
			]
			encoded = self._tokenizer(
				batch_prompts,
				return_tensors = "pt",
				padding = True,
			)
			encoded = {
				key: value.to(self._model.device)
				for key, value in encoded.items()
			}
			input_width = encoded["input_ids"].shape[1]

			with torch.no_grad():
				generated = self._model.generate(
					**encoded,
					pad_token_id = self._tokenizer.pad_token_id,
					eos_token_id = self._tokenizer.eos_token_id,
					**self._generation_kwargs(),
				)

			for sequence in generated:
				continuation = sequence[input_width:]
				outputs.append(
					self._tokenizer.decode(
						continuation,
						skip_special_tokens = True,
					).strip()
				)

		return outputs

	def _load(self) -> None:
		if self._model is not None and self._tokenizer is not None:
			return

		import transformers

		self._tokenizer = transformers.AutoTokenizer.from_pretrained(
			self.model_name,
			trust_remote_code = self.trust_remote_code,
		)
		if self._tokenizer.pad_token_id is None:
			self._tokenizer.pad_token = self._tokenizer.eos_token

		self._model = transformers.AutoModelForCausalLM.from_pretrained(
			self.model_name,
			device_map = self.device_map,
			torch_dtype = self.torch_dtype,
			trust_remote_code = self.trust_remote_code,
		)
		self._model.eval()

	def _format_prompt(self, prompt: str, /) -> str:
		tokenizer = self._tokenizer
		if not self.use_chat_template or tokenizer is None or not getattr(tokenizer, "chat_template", None):
			return prompt

		messages: list[dict[str, str]] = []
		if self.system_message:
			messages.append({"role": "system", "content": self.system_message})
		messages.append({"role": "user", "content": prompt})

		return tokenizer.apply_chat_template(
			messages,
			tokenize = False,
			add_generation_prompt = True,
		)

	def _generation_kwargs(self) -> dict[str, typing.Any]:
		config = self.decoding
		kwargs: dict[str, typing.Any] = {
			"max_new_tokens": config.max_new_tokens,
			"do_sample": config.do_sample,
			"repetition_penalty": config.repetition_penalty,
		}
		if config.do_sample:
			kwargs["temperature"] = config.temperature
			kwargs["top_p"] = config.top_p

		return kwargs


class PromptedGenerationClassifier:
	"""End-to-end prompted classifier: rows -> prompts -> generations -> labels."""

	def __init__(
		self,
		prompt_encoder: PromptEncoder,
		generator: TextGenerator,
		parser: GenerationParser | None = None,
	) -> None:
		self.prompt_encoder = prompt_encoder
		self.generator = generator
		self.parser = parser or GenerationParser()

	def fit(self, source: pandas.DataFrame, target: typing.Any | None = None, /) -> typing.Self:
		self.prompt_encoder.fit(source, target)

		return self

	def generate_frame(
		self,
		source: pandas.DataFrame,
		/,
		include_prompts: bool = True,
	) -> pandas.DataFrame:
		prompts = self.prompt_encoder.transform(source)
		generations = self.generator.generate(prompts)
		parsed = self.parser.parse_many(generations)

		frame = pandas.DataFrame({
			"generation": generations,
			"Predicted": [item.label for item in parsed],
			"valid": [item.valid for item in parsed],
			"parse_method": [item.parse_method for item in parsed],
			"parse_message": [item.message for item in parsed],
			"rationale": [item.rationale for item in parsed],
		}, index = source.index)
		frame.index.name = source.index.name or "Id"

		if include_prompts:
			frame.insert(0, "prompt", prompts)

		return frame

	def predict(self, source: pandas.DataFrame, /) -> pandas.Series:
		frame = self.generate_frame(source, include_prompts = False)

		return frame["Predicted"].rename("Predicted")

	def save_run(
		self,
		source: pandas.DataFrame,
		path: str | pathlib.Path,
		/,
		include_prompts: bool = True,
	) -> pandas.DataFrame:
		frame = self.generate_frame(source, include_prompts = include_prompts)
		path = pathlib.Path(path)
		path.parent.mkdir(parents = True, exist_ok = True)
		frame.to_csv(path)

		return frame

