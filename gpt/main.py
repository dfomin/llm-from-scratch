from typing import Literal

import tiktoken
import torch
import typer
from tokenizers import ByteLevelBPETokenizer

from gpt.generate import generate_text_simple
from gpt.gpt import GPTModel


def main(lang: Literal["en", "ru"] = "en"):
    if lang == "en":
        GPT_CONFIG_124M = {
            "vocab_size": 50257,  # Vocabulary size
            "context_length": 1024,  # Context length
            "emb_dim": 768,  # Embedding dimension
            "n_heads": 12,  # Number of attention heads
            "n_layers": 12,  # Number of layers
            "drop_rate": 0.1,  # Dropout rate
            "qkv_bias": False,  # Query-Key-Value bias
        }

        tokenizer = tiktoken.get_encoding("gpt2")
        start_context = "Hello, I am"
        encoded = tokenizer.encode(start_context)
    else:
        GPT_CONFIG_124M = {
            "vocab_size": 50264,  # Vocabulary size
            "context_length": 1024,  # Context length
            "emb_dim": 768,  # Embedding dimension
            "n_heads": 12,  # Number of attention heads
            "n_layers": 12,  # Number of layers
            "drop_rate": 0.1,  # Dropout rate
            "qkv_bias": False,  # Query-Key-Value bias
        }

        tokenizer = ByteLevelBPETokenizer(
            "./rugpt3small/vocab.json", "./rugpt3small/merges.txt"
        )
        start_context = "Как твои дела?"
        encoded = tokenizer.encode(start_context).ids

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # disable dropout

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50 * '='}\n{22 * ' '}IN\n{50 * '='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        indices=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"],
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50 * '='}\n{22 * ' '}OUT\n{50 * '='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)


def cli():
    typer.run(main)
