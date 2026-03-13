import tiktoken
import torch

tokenizer = tiktoken.get_encoding("gpt2")

def text_to_token_ids(text, device):

    encoded = tokenizer.encode(text)
    return torch.tensor(encoded).unsqueeze(0).to(device)

def token_ids_to_text(token_ids):

    text = tokenizer.decode(token_ids.squeeze(0).tolist())
    text = text.replace("<|endoftext|>", "")

    return text