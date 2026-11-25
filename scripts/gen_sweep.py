"""Generate the size-sweep cases on the U280 node (uses gen_data.write_case, same format and goldens)."""
import sys
sys.path.insert(0, "scripts")
import gen_data

CASES = [
    # square sizes (instructor: try 256 / 1024, scale to 4096)
    ("sq256", 256, 256), ("sq512", 512, 512), ("sq1024", 1024, 1024),
    ("sq2048", 2048, 2048), ("sq4096", 4096, 4096),
    # real LLM layer shapes (rows = output features, cols = input features)
    ("gpt2_attn_768x768", 768, 768),        # GPT-2 small attention projection
    ("gpt2_mlp_3072x768", 3072, 768),       # GPT-2 small MLP up projection
    ("llama7b_attn_4096x4096", 4096, 4096), # Llama-2-7B q/k/v/o projection
    ("llama7b_ffn_11008x4096", 11008, 4096),# Llama-2-7B FFN gate / up projection
]

for name, M, K in CASES:
    gen_data.write_case(name, M, K, "rand", 1)
print("done")
