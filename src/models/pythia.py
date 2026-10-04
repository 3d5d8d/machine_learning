from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPTNeoXForCausalLM

MODEL_ID = "EleutherAI/pythia-70m"
REVISION = "step143000"


class Pythia(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    @classmethod
    def from_pretrained(cls, cache_dir="models/hf_cache", device="cpu"):
        model = GPTNeoXForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, cache_dir=cache_dir,
            torch_dtype=torch.float32, attn_implementation="eager"
        )
        return cls(model).to(device).eval()

    def forward(self, idx, targets=None):
        logits = self.model(input_ids=idx, use_cache=False, return_dict=True).logits
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=32):
        self.eval()
        return self.model.generate(
            input_ids=idx, attention_mask=torch.ones_like(idx),
            max_new_tokens=max_new_tokens, do_sample=False, use_cache=False,
            pad_token_id=self.config.eos_token_id
        )


class FixedAttention(nn.Module):
    def __init__(self, attention, A):
        super().__init__()
        self.query_key_value = attention.query_key_value
        self.dense = attention.dense
        self.num_attention_heads = attention.num_attention_heads
        self.head_size = attention.head_size
        self.register_buffer("A", A.detach().to(self.dense.weight).clone())

    def forward(self, hidden_states, output_attentions=False, **kwargs):
        batch_size, length, hidden_size = hidden_states.shape
        heads, head_size = self.num_attention_heads, self.head_size
        weight = self.query_key_value.weight.view(heads, 3, head_size, hidden_size)
        weight_v = weight[:, 2].reshape(hidden_size, hidden_size)
        bias = self.query_key_value.bias
        bias_v = bias.view(heads, 3, head_size)[:, 2].reshape(hidden_size)
        value = F.linear(hidden_states, weight_v, bias_v)
        value = value.view(batch_size, length, heads, head_size).transpose(1, 2)
        A = self.A[:, :length, :length].unsqueeze(0)
        output = (A @ value).transpose(1, 2).reshape(batch_size, length, hidden_size)
        result = (self.dense(output), None)
        return result + (A.expand(batch_size, -1, -1, -1),) if output_attentions else result


class FixedPythia(Pythia):
    def __init__(self, normal_model, A):
        super().__init__(deepcopy(normal_model.model))
        for layer, layer_A in zip(self.model.gpt_neox.layers, A):
            layer.attention = FixedAttention(layer.attention, layer_A)
        self.eval()
