import os
import time
import pandas as pd
import numpy as np
import argparse

os.chdir('/lockard_ai/works/PEFT/')
#from tools.data_loader import get_full_path, load_dataset
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, Trainer, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from peft import LoraConfig, PeftModel, PeftConfig


def generate_prompts_train(example):
    output_texts = []
    for i in range(len(example['user'])):
        messages = [
#             {"role": "system", "content": system_msg},
            {'role': 'user', 'content': "{}\n 앞 문장의 감정은 분노, 기쁨, 불안, 당황, 슬픔, 상처 중에 어떤거야? 반드시 한 단어로 답변해줘.".format(example['user'][i])},
            {'role': 'assistant', 'content': "{}".format(example['assistant'][i])}
        ]
        chat_message = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        chat_message = chat_message.rstrip()
        chat_message = chat_message+"<eos>"
        '''
        print("###################################")
        print('###chat_message###\n', chat_message)
        print("###################################")
        '''
        
        output_texts.append(chat_message)

    return output_texts


parser = argparse.ArgumentParser(description="Process some input argument.")

# Add a positional argument
parser.add_argument(
    'llm_model',  # Name of the argument
    type=str,  # The data type expected
    help='LLMModel path or name e.g., llama3'  # Help message
)
parser.add_argument(
    'dataset',  # Name of the argument
    type=str,  # The data type expected
    help='dataset path'  # Help message
)
parser.add_argument(
    'finetuned_model',  # Name of the argument
    type=str,  # The data type expected
    help='finedtuned model path'  # Help message
)

args = parser.parse_args()
print(f"Received argument: {args.llm_model}, {args.dataset}, {args.finetuned_model}")
#llm_model = args.llm_model
#dataset = args.dataset

if args.dataset:
    dataset_dict = DatasetDict.load_from_disk(args.dataset)
    print(f"dataset path : {args.dataset}")
else:
    dataset_dict = DatasetDict.load_from_disk('./emotional_dataset')

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(args.llm_model, device_map="auto", quantization_config=bnb_config) # 양자화 함
tokenizer = AutoTokenizer.from_pretrained(args.llm_model, add_special_tokens=True)

lora_config = LoraConfig(
    r=6,
    lora_alpha = 8,
    lora_dropout = 0.05,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_dict['train'],
    max_seq_length=512,
    args=TrainingArguments(
        output_dir="outputs",
#        num_train_epochs = 1,
        max_steps=3000,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="paged_adamw_8bit",
#        warmup_steps=0.03,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=100,
        report_to="wandb",
        push_to_hub=False,
    ),
    peft_config=lora_config,
    formatting_func=generate_prompts_train
)

trainer.train()

# model save
ADAPTER_MODEL = args.llm_model+"_lora_adapter"

trainer.model.save_pretrained(ADAPTER_MODEL)

model = PeftModel.from_pretrained(model, ADAPTER_MODEL, device_map='auto', torch_dtype=torch.float16)

model = model.merge_and_unload()
model.save_pretrained(args.finetuned_model)
