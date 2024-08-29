import os
import time
import pandas as pd
import numpy as np
import argparse

os.chdir('/lockard_ai/works/PEFT/')
#from tools.data_loader import get_full_path, load_dataset

# %pip install --upgrade transformers
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, Trainer, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from peft import LoraConfig, PeftModel, PeftConfig

from sklearn.preprocessing import LabelEncoder
import joblib

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score

dataset_dict = DatasetDict.load_from_disk('./emotional_dataset')

def inference_with_pipeline(model, tokenizer, prompt):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512
#         model_kwargs={"torch_dtype": torch.bfloat16},
#pad_token_id=128009
    )

    messages=[
        #{"role": "system", "content": system_msg},
        {'role':'user', 'content': prompt+suffix_prompt}
    ]

    '''
    terminators = [
         pipe.tokenizer.eos_token_id,
         pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
     ]
    '''
    chat_message = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    chat_message += "<start_of_turn>model"
    outputs = pipe(
        chat_message,
        do_sample=True,
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        add_special_tokens=True
    )
    '''
    outputs = pipe(
#messages,
        chat_message,
        max_new_tokens=16,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
    )

    print("##################")
    print('chat_message :\n', chat_message)
    print("##################")

    print("##################")
    print("outputs :\n ", outputs)
    print("##################")
    '''
    return outputs[0]["generated_text"][len(chat_message):]

def chat_with_model(prompt, model, tokenizer, max_length=100):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512
#         model_kwargs={"torch_dtype": torch.bfloat16},
#pad_token_id=128009
    )

    messages=[
        #{"role": "system", "content": system_msg},
        {'role':'user', 'content': prompt}
    ]

    chat_message = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    chat_message += "<start_of_turn>model"
    outputs = pipe(
        chat_message,
        do_sample=True,
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        add_special_tokens=True
    )
    '''
    # Tokenize the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate the model's response
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,  # Ensure padding is done correctly
        no_repeat_ngram_size=2,               # Prevent repetition of phrases
        top_k=50,                             # Use top-k sampling
        top_p=0.95,                           # Use top-p sampling (nucleus sampling)
        temperature=0.2                        # Control the randomness of predictions
    )
    
    # Decode the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    '''
    
    return outputs[0]["generated_text"][len(chat_message):]

def main():
    parser = argparse.ArgumentParser(description="Process some input argument.")

    parser.add_argument(
        'base_model',  # Name of the argument
        type=str,  # The data type expected
        help='LLM base model name'  # Help message
    )

    parser.add_argument(
        'trained_model',  # Name of the argument
        type=str,  # The data type expected
        help='LLM model path'  # Help message
    )

    args = parser.parse_args()
    print(f"base model path : {args.base_model}")
    print(f"trained model path : {args.trained_model}")

    origin_model_path = args.base_model
    if args.trained_model:
        print('#trained_model path : ', args.trained_model)
        trained_model_path = args.trained_model
    else:
        print('#trained_model path : ', args.trained_model)
        print('#model lora path : lora_adapter')
        trained_model_path = './lora_adapter'

    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    print('#######', trained_model_path)
    if args.trained_model:
        print("load raw model with torch.float16")
        trained_model = AutoModelForCausalLM.from_pretrained(trained_model_path, device_map="auto", torch_dtype=torch.bfloat16)
    else:
        print("load with quantization")
        print('bnb_config', bnb_config)
        trained_model = AutoModelForCausalLM.from_pretrained(trained_model_path, device_map="auto", quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(origin_model_path)
    '''
    # If you're using a GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)
    '''
    print("Chat with the model! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["quit", "exit"]:
            break
        
        # Query the model
        response = chat_with_model(user_input, trained_model, tokenizer)
        
        # Display the response
        print(f"Model: {response}")

    '''
    global system_msg, suffix_prompt
    system_msg = 'You are a helpful emotional classification assistant. Always give a one-word answer. \
            You must answer only in Korean. No need for any further explanation. \
            반드시 다음 중 한 단어로 답변해줘 : 분노, 기쁨, 불안, 당황, 슬픔, 상처.'
    suffix_prompt = ' 앞 문장의 감정은 분노, 기쁨, 불안, 당황, 슬픔, 상처 중에 어떤거야? 반드시 한 단어로 답변해줘.'
    '''

if __name__=='__main__':
    main()
