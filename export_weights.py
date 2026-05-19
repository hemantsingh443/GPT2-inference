import os # pyrefly: ignore [missing-import]
import numpy as np # pyrefly: ignore [missing-import]
from transformers import GPT2LMHeadModel # pyrefly: ignore [missing-import] 

model = GPT2LMHeadModel.from_pretrained("gpt2") 

os.makedirs("weights", exist_ok=True) 

state_dict = model.state_dict() 

for name, tensor in state_dict.items(): 

    tensor = tensor.detach().cpu().numpy().astype(np.float32) 

    filename = name.replace(".", "_") + ".bin" 

    tensor.tofile(f"weights/{filename}") 

    print(name, tensor.shape)
