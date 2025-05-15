from __future__ import annotations

import logging; logging.basicConfig(level = logging.INFO)

# Import PyTorch and set the default device to GPU if available
import torch; torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

# Import the HuggingFace Transformers library
import transformers

tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

# Define an ambiguous sentence with polysemy (word “bank” appears in different contexts)
text = "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."

# Tokenize the input text (convert text into input IDs and attention masks for BERT)
tokens = tokenizer(text,
    return_tensors = "pt",  # Return PyTorch tensors
)

# Load the pretrained BERT model with output of all hidden layers
model = transformers.BertModel.from_pretrained("bert-base-uncased",
    output_hidden_states = True,
)
model.eval()  # Set model to evaluation mode (no dropout, etc.)

# Disable gradient calculations (not training)
with torch.no_grad():
    output = model(**tokens)
    hidden_states = output[2]  # Tuple of hidden states from all layers

# Stack the hidden states to form a tensor: (layers, batch, tokens, hidden_dim)
token_embeddings = torch.stack(hidden_states,
    dim = 0,
).squeeze(
    dim = 1,  # Remove batch dimension since it's 1
).permute(
    1, 0, 2,  # Change to shape: (tokens, layers, hidden_dim)
)

# Aggregate embeddings: sum across layers and then mean across tokens
sentence_embedding = torch.sum(token_embeddings,
    dim = 1,  # Sum over layers
).mean(
    dim = 0,  # Mean over tokens
)

# --------------------------------------------
# Define a reusable class for sentence embeddings using BERT
class BertSentenceEmbedder:

    def __init__(self, *,
        model_name = "bert-base-uncased",  # Default to the standard uncased BERT
    ):
        # Initialize tokenizer and model
        self.tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
        self.model = transformers.BertModel.from_pretrained(model_name,
            output_hidden_states = True,
        )
        self.model.eval()  # Evaluation mode

    def __call__(self, texts: list[str] | str) -> torch.Tensor:
        # Allow instance to be called like a function
        return self.encode(texts)

    def encode(self, texts: list[str] | str) -> torch.Tensor:
        # Convert single string input to list
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize text input
        tokens = self.tokenizer(texts,
            padding = True,         # Pad sequences to the same length
            truncation = True,      # Truncate if too long for BERT
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
