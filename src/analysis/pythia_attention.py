import math

import torch
from tqdm import tqdm

from src.data.pythia_data import batches


@torch.no_grad()
def mean_attention(model, records, device, length=2048, batch_size=1):
    model.eval()
    config = model.config
    total = torch.zeros(config.num_hidden_layers, config.num_attention_heads, length, length, dtype=torch.float64)
    count = 0
    for x, _ in tqdm(batches(records, device, length, batch_size),
                     total=math.ceil(len(records) / batch_size), desc="Mean attention"):
        output = model.model(input_ids=x, use_cache=False, output_attentions=True, return_dict=True)
        for layer, A in enumerate(output.attentions):
            total[layer].add_(A.detach().cpu().double().sum(0))
        count += x.shape[0]
        del output
    return total.div_(count).float()
