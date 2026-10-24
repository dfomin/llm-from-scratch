import torch
from torch import Tensor

from gpt.gpt import GPTModel


def generate_text_simple(
    model: GPTModel, indices: Tensor, max_new_tokens: int, context_size: int
) -> Tensor:
    # indices is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        indices_context = indices[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(indices_context)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        indices_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        indices = torch.cat((indices, indices_next), dim=1)  # (batch, n_tokens+1)

    return indices
