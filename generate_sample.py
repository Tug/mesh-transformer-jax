from transformers import GPTJForCausalLM, AutoTokenizer
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Running on {torch.cuda.get_device_name(device)}")

model = GPTJForCausalLM.from_pretrained("../out-cedille-150000", revision="float16", torch_dtype=torch.float16)
# model = GPTJForCausalLM.from_pretrained("../out-cedille-150000-fp32", torch_dtype=torch.float32)
#model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

# prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, " \
#          "previously unexplored valley, in the Andes Mountains. Even more surprising to the " \
#          "researchers was the fact that the unicorns spoke perfect English."

# prompt = "Je recherche du travail en tant que développeur web après avoir terminé mes études d'informatique. " \
#          "Voici ma lettre de motivation pour une grande agence lausannoise.\n" \
#          "\n" \
#          "Madame, monsieur,"

prompt = "J'ai écrit le nouveau tube de l'été. Voici les paroles:\n" \
	 "Certains comprennent mal\n" \
	 "Qu'il soit mon pote, mon ami\n" \
	 "'Disent que c'est pas normal\n" \
	 "Que je sois si proche de lui\n"

# input_ids = tokenizer(prompt, return_tensors="pt").input_ids

inputs    = tokenizer(prompt, return_tensors="pt").to(device)
model     = model.to(device)

for i in range(10):
    gen_tokens = model.generate(**inputs, do_sample=True, temperature=0.9, max_length=1024)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    print(gen_text)
