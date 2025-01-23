# src/deepseek_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_model():
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-Base")
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V3-Base")
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
