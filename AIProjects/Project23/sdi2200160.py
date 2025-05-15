from __future__ import annotations


import logging; logging.basicConfig(level = logging.INFO)
import torch; torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
import transformers


class BertSentenceEmbedder:

    def __init__(self, *,
        model_name = "bert-base-uncased",  # Default to the standard uncased BERT
    ):
        self.tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
        self.model = transformers.BertModel.from_pretrained(model_name,
            output_hidden_states = True,
        )
        self.model.eval()  # Evaluation mode

    def __call__(self, texts: list[str] | str) -> torch.Tensor:
        return self.encode(texts)

    def encode(self, texts: list[str] | str) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize text input
        tokens = self.tokenizer(texts,
            padding = True,  # Pad sequences to the same length
            truncation = True,  # Truncate if too long for BERT
            return_tensors = "pt",  # Return as PyTorch tensors
        )

        # Inference without gradients
        with torch.no_grad():
            output = self.model(**tokens)
            hidden_states = output.hidden_states  # (layers, batch, tokens, hidden_dim)

        token_embeddings = torch.stack(hidden_states,
            dim = 0,  # Stack to shape: (layers, batch, tokens, dim)
        )

        # Return sentence embedding: (batch, dim) — one vector per input text
        return token_embeddings.sum(
             dim = 0,  # Sum across layers
        ).mean(
             dim = 1,  # Mean across tokens
        )
