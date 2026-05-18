"""Verify the SAE training environment.

Loads Pythia-160M via TransformerLens, runs one forward pass on a sample text,
extracts the layer-6 residual stream activation, and prints shapes.

If this script runs cleanly, the W2 environment is good — we can move on to
extracting activations on a real corpus and training the SAE.
"""
import torch
import time
from transformer_lens import HookedTransformer


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    print("loading Pythia-160M (via TransformerLens)...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained("pythia-160m", device=device)
    print(f"  loaded in {time.time() - t0:.1f}s")
    print(f"  d_model:    {model.cfg.d_model}")
    print(f"  n_layers:   {model.cfg.n_layers}")
    print(f"  n_heads:    {model.cfg.n_heads}")
    print(f"  d_vocab:    {model.cfg.d_vocab}")

    sample = "The mitochondria is the powerhouse of the cell. Inside its membrane,"
    print(f"\nrunning forward pass on: {sample!r}")

    # We'll hook the residual stream at layer 6 (middle of a 12-layer model)
    target_layer = 6
    hook_name = f"blocks.{target_layer}.hook_resid_post"

    t0 = time.time()
    tokens = model.to_tokens(sample)
    print(f"  tokens shape: {tokens.shape}  (batch=1, seq_len)")
    logits, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    elapsed = time.time() - t0

    activation = cache[hook_name]
    print(f"  forward pass: {elapsed*1000:.0f} ms")
    print(f"  layer-{target_layer} resid_post shape: {activation.shape}")
    print(f"    (batch=1, seq_len, d_model={model.cfg.d_model})")
    print(f"  activation stats:  mean={activation.mean().item():+.3f}  "
          f"std={activation.std().item():.3f}  "
          f"|max|={activation.abs().max().item():.2f}")

    # Show what the next-token prediction is, just for fun
    next_token_id = logits[0, -1].argmax().item()
    next_token = model.to_string(next_token_id)
    print(f"\n  predicted next token: {next_token!r}")

    print("\nSetup verified. Ready to extract activations on a corpus.")


if __name__ == "__main__":
    main()
