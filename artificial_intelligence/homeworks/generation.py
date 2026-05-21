"""Prompted generation helpers for instruction-tuned Hugging Face models."""


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

	def asdict(self) -> dict[str, int | float | bool]:
		return dataclasses.asdict(self)


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


class HuggingFaceGenerator:
	"""Generate continuations with Hugging Face chat/instruction models.

	`backend="auto"` first tries the image-text-to-text API used by current
	Qwen3.5 model cards, then falls back to AutoModelForCausalLM for text-only
	instruction models.
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
		backend: typing.Literal["auto", "causal-lm", "image-text-to-text"] = "auto",
	) -> None:
		self.model_name = model_name
		self.decoding = decoding or DecodingConfig()
		self.batch_size = batch_size
		self.device_map = device_map
		self.torch_dtype = torch_dtype
		self.trust_remote_code = trust_remote_code
		self.use_chat_template = use_chat_template
		self.system_message = system_message
		self.backend = backend
		self._tokenizer = None
		self._processor = None
		self._model = None
		self._loaded_backend: str | None = None

	def generate(self, prompts: typing.Sequence[str], /) -> list[str]:
		self._load()
		assert self._model is not None
		assert self._loaded_backend is not None

		match self._loaded_backend:
			case "image-text-to-text":
				return self._generate_image_text_to_text(prompts)
			case "causal-lm":
				return self._generate_causal_lm(prompts)

		raise ValueError(f"Unsupported loaded backend: {self._loaded_backend}")

	def _generate_causal_lm(self, prompts: typing.Sequence[str], /) -> list[str]:
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
			encoded = self._move_batch(encoded)
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

	def _generate_image_text_to_text(self, prompts: typing.Sequence[str], /) -> list[str]:
		assert self._processor is not None
		assert self._model is not None

		import torch

		outputs: list[str] = []
		for start in range(0, len(prompts), self.batch_size):
			batch_messages = [
				self._messages(prompt, structured_content = True)
				for prompt in prompts[start:start + self.batch_size]
			]
			batch_text = [
				self._processor.apply_chat_template(
					messages,
					tokenize = False,
					add_generation_prompt = True,
				)
				for messages in batch_messages
			]
			encoded = self._processor(
				text = batch_text,
				return_tensors = "pt",
				padding = True,
			)
			encoded = self._move_batch(encoded)
			input_width = encoded["input_ids"].shape[1]

			with torch.no_grad():
				generated = self._model.generate(
					**encoded,
					**self._generation_token_kwargs(),
					**self._generation_kwargs(),
				)

			continuations = generated[:, input_width:]
			outputs.extend(
				text.strip()
				for text in self._processor.batch_decode(
					continuations,
					skip_special_tokens = True,
				)
			)

		return outputs

	def _load(self) -> None:
		if self._model is not None:
			return

		import transformers

		if self.backend in {"auto", "image-text-to-text"}:
			try:
				self._load_image_text_to_text(transformers)
			except (AttributeError, OSError, ValueError) as error:
				if self.backend == "image-text-to-text":
					raise
				print(f"Image-text-to-text load failed for {self.model_name}: {error}")

		if self._model is None and self.backend in {"auto", "causal-lm"}:
			self._load_causal_lm(transformers)

	def _load_causal_lm(self, transformers) -> None:
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
		self._loaded_backend = "causal-lm"

	def _load_image_text_to_text(self, transformers) -> None:
		model_cls = getattr(transformers, "AutoModelForImageTextToText")
		self._processor = transformers.AutoProcessor.from_pretrained(
			self.model_name,
			trust_remote_code = self.trust_remote_code,
		)
		self._ensure_processor_padding_token()
		self._model = model_cls.from_pretrained(
			self.model_name,
			device_map = self.device_map,
			torch_dtype = self.torch_dtype,
			trust_remote_code = self.trust_remote_code,
		)
		self._sync_generation_config_token_ids()
		self._model.eval()
		self._loaded_backend = "image-text-to-text"

	def _format_prompt(self, prompt: str, /) -> str:
		tokenizer = self._tokenizer
		if not self.use_chat_template or tokenizer is None or not getattr(tokenizer, "chat_template", None):
			return prompt

		messages = self._messages(prompt, structured_content = False)

		return tokenizer.apply_chat_template(
			messages,
			tokenize = False,
			add_generation_prompt = True,
		)

	def _messages(self, prompt: str, /, structured_content: bool) -> list[dict[str, typing.Any]]:
		messages: list[dict[str, typing.Any]] = []
		if self.system_message:
			content: typing.Any = self.system_message
			if structured_content:
				content = [{"type": "text", "text": self.system_message}]
			messages.append({"role": "system", "content": content})

		content = prompt
		if structured_content:
			content = [{"type": "text", "text": prompt}]
		messages.append({"role": "user", "content": content})

		return messages

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

	def _generation_token_kwargs(self) -> dict[str, typing.Any]:
		"""Pass explicit special-token ids so Transformers does not warn per batch."""
		eos_token_id = self._token_id("eos_token_id")
		pad_token_id = self._token_id("pad_token_id")
		if pad_token_id is None:
			pad_token_id = eos_token_id

		kwargs: dict[str, typing.Any] = {}
		if pad_token_id is not None:
			kwargs["pad_token_id"] = pad_token_id
		if eos_token_id is not None:
			kwargs["eos_token_id"] = eos_token_id

		return kwargs

	def _ensure_processor_padding_token(self) -> None:
		tokenizer = self._processor_tokenizer()
		if tokenizer is None or getattr(tokenizer, "pad_token_id", None) is not None:
			return

		eos_token = getattr(tokenizer, "eos_token", None)
		if eos_token is not None:
			tokenizer.pad_token = eos_token

	def _sync_generation_config_token_ids(self) -> None:
		if self._model is None:
			return

		generation_config = getattr(self._model, "generation_config", None)
		if generation_config is None:
			return

		eos_token_id = self._token_id("eos_token_id")
		pad_token_id = self._token_id("pad_token_id")
		if pad_token_id is None:
			pad_token_id = eos_token_id
		if getattr(generation_config, "pad_token_id", None) is None and pad_token_id is not None:
			generation_config.pad_token_id = pad_token_id
		if getattr(generation_config, "eos_token_id", None) is None and eos_token_id is not None:
			generation_config.eos_token_id = eos_token_id

	def _token_id(self, name: typing.Literal["eos_token_id", "pad_token_id"]) -> typing.Any:
		tokenizer = self._tokenizer or self._processor_tokenizer()
		value = getattr(tokenizer, name, None) if tokenizer is not None else None
		if value is not None:
			return value

		if self._model is None:
			return None

		generation_config = getattr(self._model, "generation_config", None)
		return getattr(generation_config, name, None)

	def _processor_tokenizer(self) -> typing.Any:
		if self._processor is None:
			return None

		return getattr(self._processor, "tokenizer", None)

	def _move_batch(self, batch):
		device = getattr(self._model, "device", None)
		if hasattr(batch, "to") and device is not None:
			return batch.to(device)

		if device is None:
			return batch

		return {
			key: value.to(device) if hasattr(value, "to") else value
			for key, value in batch.items()
		}


class HuggingFaceCausalLMGenerator(HuggingFaceGenerator):
	"""Backward-compatible causal-LM-only generator."""

	def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
		kwargs["backend"] = "causal-lm"
		super().__init__(*args, **kwargs)


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
