# adapted from https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
import torch
import numpy as np
from tqdm import tqdm


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def generate_llada(
    model=None,
    tokenizer=None,
    prompt="Oh wow, it's Judy!",
    mask_id=126336,
    steps=128,
    gen_length=128,
    block_length=32,
    temperature=0.0,
):
    prompt_len = input_ids.shape[-1]          
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(torch.device(model.device))

    x = torch.full((1, input_ids.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :input_ids.shape[1]] = input_ids.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in tqdm(range(num_blocks), desc="Blocks", unit="block", position=0):
        block_mask_index = (x[:, input_ids.shape[1] + num_block * block_length: input_ids.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in tqdm(range(steps_per_block), desc="Steps", unit="step", position=1, leave=False):

            mask_index = (x == mask_id)
            # cfg scale = 0
            model.eval()
            with torch.no_grad():
                logits = model(x)

            # gumbel noise for added perplexity
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            # random remasking
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

            x0_p[:, input_ids.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    output_string = tokenizer.decode(x[0, prompt_len:].tolist())

    return output_string
