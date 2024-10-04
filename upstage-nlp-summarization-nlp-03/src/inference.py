import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from src.utils import generate_prompt

def load_model_and_tokenizer(model_path):
    # Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        low_cpu_mem_usage=True,  
        quantization_config=bnb_config, 
        device_map="cuda"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def inference(model_path, dataset):
    model, tokenizer = load_model_and_tokenizer(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    pipe_finetuned = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)
    
    prompts = generate_prompt(dataset['test'])

    summaries = []
    for idx, prompt in enumerate(tqdm(prompts)):
        outputs = pipe_finetuned(
            prompt,
            do_sample=True,
            temperature=0.1,
            top_k=30,
            top_p=0.95,
            repetition_penalty=1.1,
        )
        summary = outputs[0]["generated_text"][len(prompt):]

        # if idx % 50 == 0:
        #     print("="*25, "[ 대화 ]", "="*25)
        #     print(dataset['test'][idx]["dialogue"])
        #     print("="*25, "[ 요약 ]", "="*25)
        #     print(summary)
        #     print()

        summaries.append(summary.strip())

    return summaries