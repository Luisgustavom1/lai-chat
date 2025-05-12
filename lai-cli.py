import os
import torch
import importlib.util
import re
from typing import List, Dict, Callable
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig

HOME_DIR = Path().home()
DEFAULT_CONFIG = {
  "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
  "temperature": 0.1,
  "cache_dir": str(Path(os.environ.get("TRANSFORMERS_CACHE", HOME_DIR / ".cache" / "huggingface"))),
  "quantization": "8bit",
  "max_context_tokens": 100000,
  "max_new_tokens": 10000,
}
def build_system_prompt(tool_prompts: List[str]) -> str:
    base = (
      "!!! STRICT MODE ACTIVATED !!!\n"
      "You are Qwen2.5 coder, a coding AI that *only* responds with TOOL commands when tools are needed.\n"
      "Deviating from this rule will crash the system.\n"
    )
    tools_section = "TOOLS:\n" + "\n".join(tool_prompts) + "\n\n" if tool_prompts else ""
    rules_section = (
      "RULES:\n"
      "1. **Strict Silence**:\n"
      "   - If a tool is needed, respond *ONLY* with the tool command (e.g., `[LOAD_FILE ./file.py]`).\n"
      "   - NO other text, explanations, or filler (e.g., no 'I am thinking...').\n"
      "2. **Mandatory Tool Use**:\n"
      "   - If the user asks to read a file/URL, use the tool *immediately*.\n"
      "Examples:\n"
      "User: Read ./app.py\n"
      "AI: [LOAD_FILE ./app.py]\n"
      "User: Fetch https://example.com\n"
      "AI: [FETCH_URL https://example.com]\n"
    )
    return base + tools_section + rules_section

def load_tools(helpers_dir: str) -> (Dict[str, Callable], List[str]):
  tools = {}
  tool_prompts = []
  helpers_path = Path(helpers_dir)

  for py_file in helpers_path.glob("*.py"):
    module_name = py_file.stem
    if module_name.startswith("__"): continue
    spec = importlib.util.spec_from_file_location(module_name, py_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attr_name in dir(module):
      if attr_name.startswith("handle_"):
        func = getattr(module, attr_name)
        command = attr_name[7:].upper()
        tools[command] = func
        doc = func.__doc__ or ""
        first_line = doc.strip().splitlines()[0] if doc.strip() else ""
        if first_line:
          tool_prompts.append(f"[{command} args] – {first_line}")

  return tools, tool_prompts

def parse_special_commands(response: str) -> List[tuple]:
    pattern = r'\[([A-Z_]+)\s+([^\]]+)\]'
    return [(m.group(1), m.group(2).strip(), m.start(), m.end()) for m in re.finditer(pattern, response)]

class ChatSession:
  _model = None
  _tokenizer = None

  def __init__(self, tool_prompts) -> None:
    self.history = [{"role": "system", "content": build_system_prompt(tool_prompts)}]

  def start(self):
    self._load_model()

    print(f"[SYSTEM PROMPT]: {self.history[0]['content']}")
    return
  
  def chat(self, prompt: str, tools_functions: Dict[str, Callable]):
    self.history.append({"role": "user", "content": prompt})

    text = self._tokenizer.apply_chat_template(
      self.history,
      tokenize=False,
      add_generation_prompt=True
    )
    model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

    streamer = TextStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

    print(f"I am thinking...")
    generated_ids = self._model.generate(
      **model_inputs,
      streamer=streamer,
      max_new_tokens=DEFAULT_CONFIG["max_new_tokens"],
      num_return_sequences=1,
      eos_token_id=self._tokenizer.eos_token_id,
      pad_token_id=self._tokenizer.eos_token_id,
    )
    generated_ids = [
      output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = self._tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    for cmd_type, cmd_arg, _, _ in parse_special_commands(response):
      if cmd_type in tools_functions:
          result = tools_functions[cmd_type](cmd_arg)
          if result:
            self.history.append({"role": "assistant", "content": result})
            self.history.append({"role": "user", "content": "Please continue you reasoning and analysis using the result of the tool."})
            print(f"[{cmd_type}] processed '{cmd_arg}'")
            return self.chat(prompt, tools_functions)


  def _load_model(self):
    if self._model and self._tokenizer:
      return

    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
      print("SDPA is available and (hopefully) being used!")

    model_config = {}
    try:
      import xformers.ops
      model_config["attention_implementation"] = "flash_attention_2", # Or ""memory_efficient""
    except ImportError:
      print("xFormers is not available. Using default attention implementation.")
    except Exception as e:
      print(f"An unexpected error occurred: {e}")

    model_kwargs = {
      "cache_dir": DEFAULT_CONFIG["cache_dir"],
      "torch_dtype": "auto",
      "config": model_config,
    }

    print(f"[MODEL] loading {DEFAULT_CONFIG['model_name']}...")
    self._tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CONFIG["model_name"], cache_dir=DEFAULT_CONFIG["cache_dir"])
    self._model = AutoModelForCausalLM.from_pretrained(
      DEFAULT_CONFIG["model_name"],
      **model_kwargs,
    )
    print(f"[MODEL] loaded.")

def main():
  tools_functions, tool_prompts = load_tools("./tools")
  session = ChatSession(tool_prompts)

  session.start()

  while True:
    try:
      prompt = input("\n> ")
      if prompt.strip().lower() == "exit":
        print("Exiting chat session. See you soon!")
        break
      if not prompt.strip():
        print("Tell me something.")
        continue
      session.chat(prompt, tools_functions)
    except KeyboardInterrupt:
      print("\nInterrupted by user. Type 'exit' to close chat.")
    except EOFError:
      print("\nEOF (ctrl+d). Exiting...")
      break
    except Exception as e:
      print(f"An error occurred: {e}")

if __name__ == "__main__":
  try:
    main()
  except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit(1)
