import time
import wandb
from transformers import TrainingArguments
from trl import SFTTrainer
from src.model import merge_and_unload, save_adapter, save_model, upload_model_to_hub
from src.utils import prompt_formatter

def train(model, tokenizer, dataset, model_config, train_config, wandb_config):
    wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        name=f"OPEN-SOLAR-KO-10.7B-{str(int(time.time()))}"
    )

    args = TrainingArguments(
        output_dir=train_config.output_dir,
        num_train_epochs=train_config.num_epochs,
        per_device_train_batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        logging_steps=train_config.logging_steps,
        save_strategy=train_config.save_strategy,
        learning_rate=train_config.learning_rate,
        optim=train_config.optim,
        bf16=train_config.bf16,
        fp16=train_config.fp16,
        tf32=train_config.tf32,
        max_grad_norm=train_config.max_grad_norm,
        warmup_ratio=train_config.warmup_ratio,
        lr_scheduler_type=train_config.lr_scheduler_type,
        disable_tqdm=train_config.disable_tqdm,
        weight_decay=train_config.weight_decay,
        report_to='wandb',
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['valid'],
        max_seq_length=1024,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=prompt_formatter,
        args=args,
    )

    # 모델 학습
    trainer.train()

    # wandb 종료
    wandb.finish()
    
    # LoRA adapter 저장
    save_adapter(model, train_config.lora_model_path)

    # LoRA 어댑터와 기존 모델 병합
    model = merge_and_unload(model_config.model_name, train_config.lora_model_path)

    # 모델 저장
    save_model(model, tokenizer, train_config.save_path)

    # 허브에 모델 업로드
    if train_config.push_to_hub:
        upload_model_to_hub(model, tokenizer, train_config.save_path, train_config.hub_model_id, "Upload model")

    return model
