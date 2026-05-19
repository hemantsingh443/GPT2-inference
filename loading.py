# pyrefly: ignore [missing-import]
import torch 
# pyrefly: ignore [missing-import]
from transformers import GPT2LMHeadModel, GPT2Tokenizer  

device = "cuda" 

model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval() 
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  

# prompt = "hello"  

# inputs = tokenizer(prompt, return_tensors="pt").to(device) 

# with torch.no_grad(): 
#     outputs = model(**inputs) 

# logits = outputs.logits 
# print(logits.shape) 

static_dict = model.state_dict() 

for k, v in static_dict.items(): 
    print(k, v.shape)