"""Debug f1041 L15H5 — worst AP-vs-AtP* disagreement on Gemma 2 2B."""
import os, math, sys
os.environ['TRANSFORMERLENS_ALLOW_MPS'] = '1'
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset

print("loading model + SAE...", flush=True)
model = HookedTransformer.from_pretrained('gemma-2-2b', device='mps', dtype=torch.bfloat16)
model.set_use_split_qkv_input(True)
sae = SAE.from_pretrained('gemma-scope-2b-pt-res', 'layer_12/width_16k/average_l0_82', device='mps')
print("loaded.", flush=True)

ds = load_dataset('NeelNanda/pile-10k', split='train')
text = ds[27]['text']
tokens = torch.tensor([model.tokenizer.encode(text)[:30]], device='mps')
last_pos = 15
fid = 1041
L, h_q = 15, 5  # worst disagreement
n_heads = 8; n_kv = 4; d_head = 256
decoder_col = sae.W_dec[fid].detach().to(torch.float32)
tokens_use = tokens[:, :last_pos+1]
seq_len = tokens_use.shape[1]
last_in_use = seq_len - 1

target_token = int(tokens[0, last_pos].item())
tokens_short = tokens_use[:, :-1]
seq_short = tokens_short.shape[1]
last_short = seq_short - 1
print(f"target_token={target_token}, tokens_short len={seq_short}", flush=True)

with torch.no_grad():
    _, c0 = model.run_with_cache(tokens_use, names_filter=['blocks.12.hook_resid_post'])
f_clean = sae.encode(c0['blocks.12.hook_resid_post'][0, last_in_use, :].unsqueeze(0))[0, fid].item()
print(f"f_clean = {f_clean:.2f}", flush=True)

with torch.no_grad():
    _, c = model.run_with_cache(tokens_short, names_filter=[
        f'blocks.{L}.hook_q_input', f'blocks.{L}.attn.hook_rot_k',
        f'blocks.{L}.attn.hook_attn_scores', f'blocks.{L}.attn.hook_pattern'])
clean_rot_k = c[f'blocks.{L}.attn.hook_rot_k'].detach()
clean_scores = c[f'blocks.{L}.attn.hook_attn_scores'][0, h_q, last_short, :].detach().to(torch.float32)
clean_pattern = c[f'blocks.{L}.attn.hook_pattern'][0, h_q, last_short, :].detach().to(torch.float32)

# Clean fwd+bwd for pattern grad
saved = {}
def cap_grad(act, hook):
    act.retain_grad(); saved[hook.name] = act
    return act
with model.hooks(fwd_hooks=[(f'blocks.{L}.attn.hook_pattern', cap_grad)]):
    logits = model(tokens_short)
baseline_logp = torch.log_softmax(logits[0, -1, :], dim=-1)[target_token].item()
metric = torch.log_softmax(logits[0, -1, :], dim=-1)[target_token]
model.zero_grad(); metric.backward()
g_pattern = saved[f'blocks.{L}.attn.hook_pattern'].grad.detach()[0, h_q, last_short, :].to(torch.float32)
print(f"baseline_logp={baseline_logp:+.4f}, |g_pattern|={g_pattern.norm().item():.3f}", flush=True)

# Closed-form
attn = model.blocks[L].attn; ln1 = model.blocks[L].ln1
W_Q = attn.W_Q.detach().to(torch.float32)
clean_q_input = c[f'blocks.{L}.hook_q_input'][0, last_short, h_q, :].detach().to(torch.float32)
delta = (f_clean * decoder_col).to(torch.float32)
with torch.no_grad():
    clean_ln = ln1(clean_q_input.to(model.cfg.dtype)).to(torch.float32)
    pert_ln = ln1((clean_q_input - delta).to(model.cfg.dtype)).to(torch.float32)
delta_ln = pert_ln - clean_ln
dq = delta_ln @ W_Q[h_q]
x = torch.zeros(1, seq_short, n_heads, d_head, device='mps', dtype=dq.dtype)
x[0, last_short, h_q, :] = dq
with torch.no_grad():
    drot_q = attn.apply_rotary(x.to(model.cfg.dtype))[0, last_short, h_q, :].detach().to(torch.float32)
h_kv = h_q // 2
dscores = (drot_q @ clean_rot_k[0, :, h_kv, :].to(torch.float32).T) / math.sqrt(d_head)
patched_pattern_closed = F.softmax(clean_scores + dscores, dim=-1)
dpattern_closed = patched_pattern_closed - clean_pattern

# Actual via perturbed forward
def ablate(act, hook):
    act[:, -1, h_q, :] = act[:, -1, h_q, :] - f_clean * decoder_col.to(act.dtype)
    return act
captured = {}
def cap(act, hook):
    captured[hook.name] = act.detach().clone()
    return act
with torch.no_grad():
    pl = model.run_with_hooks(tokens_short, fwd_hooks=[
        (f'blocks.{L}.hook_q_input', ablate),
        (f'blocks.{L}.attn.hook_pattern', cap)])
    ppl = torch.log_softmax(pl[0, -1, :], dim=-1)[target_token].item()
ap_drop = baseline_logp - ppl
actual_dpattern = captured[f'blocks.{L}.attn.hook_pattern'][0, h_q, last_short, :].to(torch.float32) - clean_pattern

effect_qk_closed = -float(torch.dot(g_pattern, dpattern_closed).item())
effect_qk_actual_dp = -float(torch.dot(g_pattern, actual_dpattern).item())
print(f"\n|actual Δpattern|={actual_dpattern.norm().item():.5f}, |closed Δpattern|={dpattern_closed.norm().item():.5f}", flush=True)
print(f"Δpattern cos: {F.cosine_similarity(actual_dpattern.unsqueeze(0), dpattern_closed.unsqueeze(0)).item():.4f}", flush=True)
print(f"effect_qk (closed Δp): {effect_qk_closed:+.5f}", flush=True)
print(f"effect_qk (actual Δp): {effect_qk_actual_dp:+.5f}", flush=True)
print(f"AP logp_drop:          {ap_drop:+.5f}", flush=True)
print(f"=> linear (g_pat·actualΔp) / nonlinear (AP) = {effect_qk_actual_dp / ap_drop if abs(ap_drop) > 1e-8 else float('inf'):.4f}", flush=True)
print(f"  (if 1.0: pattern path is dominant and linear approximation perfect)", flush=True)
print(f"  (if <<1.0: most of the AP effect is NOT via pattern at this layer — it's V-side or cascading)", flush=True)
