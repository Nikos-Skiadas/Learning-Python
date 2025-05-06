from __future__ import annotations


import logging; logging.basicConfig(level = logging.INFO)

import torch; torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
import transformers


tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")


text = "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
tokens = tokenizer(text,
	return_tensors = "pt",
)

model = transformers.BertModel.from_pretrained("bert-base-uncased",
	output_hidden_states = True,
)
model.eval()

with torch.no_grad():
	output = model(**tokens)
	hidden_states = output[2]

token_embeddings = torch.stack(hidden_states,
	dim = 0,
).squeeze(
	dim = 1,
).permute(
	1,
	0,
	2,
)

sentence_embedding = torch.sum(token_embeddings,
	dim = 1,
).mean(
	dim = 0,
)


# The sentence embedding is the mean of the token embeddings, encapsulating the code above.


class BertSentenceEmbedder:

	def __init__(self, *,
		model_name = "bert-base-uncased",
	):
		self.tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
		self.model = transformers.BertModel.from_pretrained(model_name,
			output_hidden_states = True,
		)
		self.model.eval()

	def __call__(self, texts: list[str] | str) -> torch.Tensor:
		return self.encode(texts)


	def encode(self, texts: list[str] | str) -> torch.Tensor:
		if isinstance(texts, str):
			texts = [texts]

		tokens = self.tokenizer(texts,
			padding = True,
			truncation = True,
			return_tensors = "pt",
		)

		with torch.no_grad():
			output = self.model(**tokens)
			hidden_states = output.hidden_states  # tuple: (layer, batch, token, hidden)

		token_embeddings = torch.stack(hidden_states,
			dim = 0,
		)  # (layers, batch, tokens, dim)

		return token_embeddings.sum(
			 dim = 0,
		).mean(
			 dim = 1,
		)  # one vector per sentence
