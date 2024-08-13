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
    chat_message += "\n<start_of_turn>model\n"
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
    #return outputs[0]["generated_text"]
    #return outputs[0]["generated_text"][2]['content']


def inference_with_prompt_list(prompt_list, model, tokenizer, save_path=None, Y_val=None):
    label = ['상처', '슬픔', '기쁨', '분노', '불안', '당황']
    answer_list = list()

    start = time.time()
    for i, (q, y) in enumerate(zip(prompt_list, Y_val)):
        answer = inference_with_pipeline(model, tokenizer, q)
        answer = answer.split('\n')[-1]
        if len(answer) > 2:
            answer = answer[:2]

        if answer in label:
            print(i, answer, end=" ")
            answer_list.append(answer)
            if save_path:
                save_oneline_result_to_csv(answer, save_path, y)
        else:
            print("\nWrong answer at {}: {}\n answer : {}".format(i,q,answer))
            answer_list.append('False')
            if save_path:
                save_oneline_result_to_csv('False', save_path, y)

    end = time.time()
    elapsed_time = end - start
    print("\n걸린 시간 : {}초 = {}분".format(elapsed_time, elapsed_time/60))
    print("한 프롬프트당 걸린 시간 : ", elapsed_time/len(prompt_list))

    return answer_list

def save_oneline_result_to_csv(answer, save_path, Y_val):
    if os.path.exists(save_path):
        old_df = pd.read_csv(save_path)
        #new_line = pd.DataFrame([answer], columns=['predict_result'])
        new_line = pd.DataFrame({'Y_pred':[answer], 'Y_val':[Y_val]})
        new_df = pd.concat([old_df,new_line], axis=0)
        new_df.to_csv(save_path, index=False)
    else:
        df = pd.DataFrame({'Y_pred':[answer], 'Y_val':[Y_val]})
        df.to_csv(save_path, index=False)


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

    parser.add_argument(
        'save_path',  # Name of the argument
        type=str,  # The data type expected
        help='inference result save path'  # Help message
    )
    args = parser.parse_args()
    print(f"base model path : {args.base_model}")
    print(f"trained model path : {args.trained_model}")
    print(f"save path : {args.save_path}")

    dataset_dict = DatasetDict.load_from_disk('./emotional_dataset')
    label_encoder = joblib.load('./emotional_dataset_label_encoder.enc')
    print('label :', label_encoder.inverse_transform([0,1,2,3,4,5,6]))
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
        trained_model = AutoModelForCausalLM.from_pretrained(trained_model_path, device_map="auto", torch_dtype=torch.float16)
    else:
        print("load with quantization")
        print('bnb_config', bnb_config)
        trained_model = AutoModelForCausalLM.from_pretrained(trained_model_path, device_map="auto", quantization_config=bnb_config)
        
    trained_tokenizer = AutoTokenizer.from_pretrained(origin_model_path)
    #trained_tokenizer.pad_token = trained_tokenizer.eos_token
    #trained_tokenizer.padding_side = "right"
    X_val = dataset_dict['test']['user']
    Y_val = dataset_dict['test']['assistant']
    '''
    X_val = dataset_dict['train']['user']
    Y_val = dataset_dict['train']['assistant']
    '''
    print("X_val length : ", len(X_val))
    print("Y_val length : ", len(Y_val))

    global system_msg, suffix_prompt
    system_msg = 'You are a helpful emotional classification assistant. Always give a one-word answer. \
            You must answer only in Korean. No need for any further explanation. \
            반드시 다음 중 한 단어로 답변해줘 : 분노, 기쁨, 불안, 당황, 슬픔, 상처.'
    suffix_prompt = ' 앞 문장의 감정은 분노, 기쁨, 불안, 당황, 슬픔, 상처 중에 어떤거야? 반드시 한 단어로 답변해줘.'
    Y_pred =inference_with_prompt_list(X_val, trained_model, trained_tokenizer, save_path=args.save_path, Y_val=Y_val)
    print(f"Y_pred : {Y_pred}")
    encoded_X_val = label_encoder.transform(Y_val)
    encoded_Y_val = label_encoder.transform(Y_pred)

#print((classification_report(encoded_y_train, encoded_y_pred)))



if __name__=='__main__':
    main()
