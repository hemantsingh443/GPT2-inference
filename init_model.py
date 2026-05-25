import os
import json
import struct
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def main():
    print("Preparing weights directory...")
    os.makedirs("weights", exist_ok=True)

    # 1. Download and load model weights
    print("Loading pretrained GPT-2 model weights from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    state_dict = model.state_dict()

    print("Exporting model weights to binary files...")
    for name, tensor in state_dict.items():
        # Convert to float32 NumPy array
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        # Convert name format (e.g. transformer.h.0.ln_1.weight -> transformer_h_0_ln_1_weight.bin)
        filename = name.replace(".", "_") + ".bin"
        filepath = os.path.join("weights", filename)
        arr.tofile(filepath)
    print(f"Exported {len(state_dict)} weight tensors.")

    # 2. Download and load tokenizer
    print("Loading tokenizer from HuggingFace...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Save the tokenizer backend representation to JSON
    tokenizer_json = json.loads(tokenizer.backend_tokenizer.to_str())
    
    vocab = tokenizer_json["model"]["vocab"]
    merges = tokenizer_json["model"]["merges"]
    
    # Sort vocab by token ID
    vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])

    # Write vocab.bin
    print("Exporting tokenizer vocabulary (vocab.bin)...")
    vocab_path = os.path.join("weights", "vocab.bin")
    with open(vocab_path, "wb") as f:
        f.write(struct.pack("i", len(vocab_sorted)))
        for token_str, token_id in vocab_sorted:
            token_bytes = token_str.encode('utf-8')
            f.write(struct.pack("i", len(token_bytes)))
            f.write(token_bytes)

    # Write merges.bin
    print("Exporting tokenizer merges (merges.bin)...")
    merges_path = os.path.join("weights", "merges.bin")
    with open(merges_path, "wb") as f:
        f.write(struct.pack("i", len(merges)))
        for merge_item in merges:
            parent = merge_item[0]
            child = merge_item[1]
            p_bytes = parent.encode('utf-8')
            c_bytes = child.encode('utf-8')
            f.write(struct.pack("i", len(p_bytes)))
            f.write(p_bytes)
            f.write(struct.pack("i", len(c_bytes)))
            f.write(c_bytes)

    print("Initialization completed successfully! All files are in the 'weights' directory.")

if __name__ == "__main__":
    main()
