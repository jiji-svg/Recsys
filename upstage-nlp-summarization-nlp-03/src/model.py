import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import HfApi, Repository

def load_model_and_tokenizer(model_config):
    model_id = model_config.model_name
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM", 
    )
    model = get_peft_model(model, peft_config)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def save_adapter(adapter, save_path):
    adapter.save_pretrained(save_path)
    print(f"LoRA adapter saved at {save_path}")

def merge_and_unload(model_id, lora_model_path):
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, lora_model_path, device="auto", torch_dtype=torch.float16)
    model = model.merge_and_unload()
    print(f"Merge and Unload Model")
    return model

def save_model(model, tokenizer, save_path):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved locally at {save_path}")

def upload_model_to_hub(model, tokenizer, save_path, hub_model_id, commit_message):
    api = HfApi()
    repo = Repository(save_path, clone_from=hub_model_id, use_auth_token=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    repo.push_to_hub(commit_message=commit_message)
    print(f"Model and tokenizer uploaded to Hugging Face Hub at {hub_model_id}")