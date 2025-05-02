import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

HOME_DIR = Path().home()
DEFAULT_CONFIG = {
  "model_name": "Qwen/Qwen2.5-Coder-3B-Instruct",
  "temperature": 0.1,
  "cache_dir": str(Path(os.environ.get("TRANSFORMERS_CACHE", HOME_DIR / ".cache" / "huggingface"))),
}

Tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CONFIG["model_name"], cache_dir=DEFAULT_CONFIG["cache_dir"])
Model = AutoModelForCausalLM.from_pretrained(
  DEFAULT_CONFIG["model_name"],
  cache_dir=DEFAULT_CONFIG["cache_dir"],
  torch_dtype="auto",
)

print("tokenizing input text...")
prompt = "write a quick sort algorithm."
messages = [
  {"role": "system", "content": "You are a code assistant."},
  {"role": "user", "content": prompt}
]
text = Tokenizer.apply_chat_template(
  messages,
  tokenize=False,
  add_generation_prompt=True
)
model_inputs = Tokenizer([text], return_tensors="pt").to(Model.device)
print("input text tokenized.")

print("generating text...")
generated_ids = Model.generate(
  **model_inputs,
  max_new_tokens=512
)
generated_ids = [
  output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
print("text generated.")

response = Tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Response:", response)