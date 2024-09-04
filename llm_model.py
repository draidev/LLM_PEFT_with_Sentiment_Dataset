import os
import shutil
import torch
import time
import sys

from dotenv import load_dotenv

from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import BitsAndBytesConfig, AutoTokenizer
import transformers

from vllm import LLM, SamplingParams

from cca_template import CCAPromptTemplate

class LLMModel:
    def __init__(self, conf):
        load_dotenv()
        self.OPENAI_API_KEY = os.getenv("openai_api_key")
        self.model_platform = conf.platform
        self.model_name = conf.model
        self.tokenizer_path = conf.tokenizer
        self.model_tokenizer = None

        if self.model_platform == "ollama":
            self.llm = ChatOllama(
                    model=self.model_name, 
                    temperature=conf.model_parameter.temperature, 
                    num_predict=conf.model_parameter.num_predict,
                    num_ctx=conf.model_parameter.num_ctx,
                    top_p = conf.model_parameter.top_p)

            if self.load_tokenizer():
                print("Success load tokenizer...")
            else:
                print("Fail to load tokenizer...")

        elif self.model_platform == "huggingface":
            self.llm = HuggingFacePipeline.from_model_id(
                model_id=self.model_name, task="text-generation", device=0, pipeline_kwargs={"max_new_tokens": 512},)
            #os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=37000'
        elif self.model_platform == "transformer":
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            self.pipe = transformers.pipeline(
                "text-generation",
                model=self.model_name,
                device_map="auto",
                model_kwargs={"torch_dtype": torch.bfloat16, "quantization_config":self.bnb_config},
                pad_token_id=128009
            )
        elif self.model_platform == "vllm":
            self.llm = LLM(model=self.model_name, quantization="awq_marlin", 
                            gpu_memory_utilization=conf.model_parameter.gpu_memory_utilization, 
                            max_model_len=conf.model_parameter.max_model_len,
                            max_num_batched_tokens=(conf.num_batch*conf.model_parameter.max_model_len), 
                            tensor_parallel_size=conf.model_parameter.num_tensor_parallel,
                            max_num_seqs=conf.num_batch)
        elif self.model_platform == "openai":
            if self.model_name == "openai":
                self.llm = ChatOpenAI(openai_api_key=self.OPENAI_API_KEY, temperature=conf.model_parameter.temperature, max_tokens=conf.model_parameter.max_tokens)
            elif self.model_name == "gpt-3.5":
                self.llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=self.OPENAI_API_KEY, temperature=conf.model_parameter.temperature, max_tokens=conf.model_parameter.max_tokens)
            elif self.model_name== "gpt-4":
                self.llm = ChatOpenAI(model="gpt-4", openai_api_key=self.OPENAI_API_KEY, temperature=conf.model_parameter.temperature, max_tokens=conf.model_parameter.max_tokens)
            elif self.model_name == "gpt-4o":
                self.llm = ChatOpenAI(model="gpt-4o", openai_api_key=self.OPENAI_API_KEY, temperature=conf.model_parameter.temperature, max_tokens=conf.model_parameter.max_tokens)
            elif self.model_name == "gpt-4o-mini":
                self.llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=self.OPENAI_API_KEY, temperature=conf.model_parameter.temperature, max_tokens=conf.model_parameter.max_tokens)
            else:
                self.llm = ChatOpenAI(openai_api_key=self.OPENAI_API_KEY)
        else:
            self.llm = ChatOllama(model=self.model_name)
        return

    def build_prompt(self, cca_template, header, body, passlock, encrypt, bigsize, suspicious):
        email = header + "\n" + body
        self.cca_template = cca_template
        self.prompt_template = PromptTemplate.from_template(self.cca_template.template)
        self.condition_template = PromptTemplate.from_template(self.cca_template.condition)
        self.condition_prompt = self.condition_template.format(passlock="No", encrypt="No", bigsize="No", suspicious="")
        final_prompt = self.prompt_template.format(condition=self.condition_prompt, email=email, format=self.cca_template.format, rule=self.cca_template.rule)
        return final_prompt

    def run(self, conf, prompt):
        start = time.time()
        if self.model_platform == "huggingface":
            torch.cuda.empty_cache()
        if self.model_platform == "transformer":
            messages=[
                {'role':'user', 'content': prompt}
            ]
            outputs = self.pipe(messages,max_new_tokens=128,early_stopping=True,)
            tmp = outputs[0]["generated_text"][-1]
            response = tmp['content']
            torch.cuda.empty_cache()
        elif self.model_platform == "vllm":
            prompts = [prompt]
            sampling_params = SamplingParams(
                    temperature=conf.model_parameter.temperature,
                    top_p=conf.model_parameter.top_p,
                    max_tokens=conf.model_parameter.max_tokens, n=1)
            outputs = self.llm.generate(prompts, sampling_params)
            response = outputs[0].outputs[0].text
        else:
            response = self.llm.invoke(prompt)

        end = time.time()
        elapsed_time = end - start

        return response, elapsed_time

    def count_token(self, prompt):
        encoded_prompt = self.model_tokenizer.tokenize(prompt)
        prompt_token_length = len(encoded_prompt)
        return prompt_token_length


    def contains_tokenizer_file(self):                                                                             
        try:                                                                                                       
            for root, dirs, files in os.walk(self.tokenizer_path):                                                 
                for file in files:                                                                                 
                    if 'tokenizer' in file:                                                                        
                        return True                                                                                
                print("Any tokenizer file not exists in the dir! check dir.")                                      
        except:                                                                                                    
            print("Tokenizer haven't downloaded in local dir yet.")                                                
            return False                                                                                           
                                                                                                                   
    def load_tokenizer(self):                                                                                      
        print("Start load tokenizer...")                                                                           
        try:                                                                                                       
            if self.contains_tokenizer_file():                                                                     
                self.model_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)                          
                return True                                                                                        
            else:                                                                                                  
                self.download_tokenizer()                                                                          
                return True                                                                                        
        except Exception as e:                                                                                     
            print("Tokenizer load ERROR :", e)                                                                     
            return None                                                                                            


    def download_tokenizer(self):                                                                                   
        try:                                                                                                        
            # make dir for saving a tokenizer                                                                       
            tokenizer_creator = self.tokenizer_path.split('/')[0]                                                   
            tokenizer_name = self.tokenizer_path.split('/')[1]                                                      
            dst_tokenizer_path = os.path.join(TOKENIZER_DIR, tokenizer_name)                                        
                                                                                                                    
            if not os.path.exists(dst_tokenizer_path):                                                              
                os.makedirs(dst_tokenizer_path)                                                                     
            else:                                                                                                   
                print("Directory is already exists. No need to mkdir")                                              
                                                                                                                    
            # download model and copy to local dir                                                                  
            self.model_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, cache_dir=TOKENIZER_DIR)      
                                                                                                                    
            src_tokenizer_path = os.path.join(TOKENIZER_DIR, f"models--{tokenizer_creator}--{tokenizer_name}")      
            source_dir = src_tokenizer_path                                                                         
            src_tokenizer_path = os.path.join(src_tokenizer_path, TOKENIZER_SNAPSHOTS)                              
            src_tokenizer_path = os.path.join(src_tokenizer_path, os.listdir(src_tokenizer_path)[0])                
                                                                                                                    
            tokenizer_file_list = os.listdir(src_tokenizer_path)                                                    
            tokenizer_file_list = [f for f in tokenizer_file_list if os.path.isfile(os.path.join(src_tokenizer_path, f))]
 
            for file_name in tokenizer_file_list:                                                                   
                source_file = os.path.join(src_tokenizer_path, file_name)                                           
                if os.path.islink(source_file):
                    abs_path = os.path.realpath(source_file)                                                        
                    shutil.copy2(abs_path, os.path.join(dst_tokenizer_path, file_name))                             
                else:
                    print("Not a link file")                                                                        
                    
            self.delete_directory(source_dir)                                                                       
            
        except Exception as e:                                                                                      
            print("ERROR :", e)                                                                                     
            
    def delete_directory(self, directory_path):                                                                     
        try:
            shutil.rmtree(directory_path)  # Recursively delete the directory and its contents                      
            print(f"Directory {directory_path} and its contents deleted successfully.")
        except FileNotFoundError:
            print(f"Directory {directory_path} not found.")                                                         
        except PermissionError:
            print(f"Permission denied to delete {directory_path}.")                                                 
        except Exception as e:
            print(f"Error occurred while deleting {directory_path}: {e}")                                            

