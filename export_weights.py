import os
import numpy as np 
from transformers import GPT2LMHeadModel 

model = GPT2LMHeadModel.from_pretrained("gpt2") 

os.makedirs("weights", exist_ok=True) 

state_dict = model.state_dict() 

for name, tensor in state_dict.items(): 

    tensor = tensor.detach().cpu().numpy().astype(np.float32) 

    filename = name.replace(".", "_") + ".bin" 

    tensor.tofile(f"weights/{filename}") 

    print(name, tensor.shape)
