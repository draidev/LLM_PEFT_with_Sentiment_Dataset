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

            self.model_tokenizer = self.load_tokenizer()
                
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
        for root, dirs, files in os.walk(self.tokenizer_path):
            for file in files:
                if 'tokenizer' in file:
                    return True
        return False

    def load_tokenizer(self):
        print("Start load tokenizer...")
        try:
            if self.contains_tokenizer_file:
                model_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
                print("Success load tokenizer...")
                return model_tokenizer
        except Exception as e:
            print("Tokenizer load ERROR :", e)
            return None

    def download_tokenizer(self):
        if not os.path.exists(self.tokenizer):
            os.makedirs(self.tokenizer_path)

        self.model_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=self.tokenizer_path)
        tokenizer_path = os.path.join(self.tokenizer_path, os.listdir(tokenizer_path)[0])
        tokenizer_path = os.path.join(tokenizer_path, "snapshots")
        tokenizer_path = os.path.join(tokenizer_path, os.listdir(tokenizer_path)[0])
        source_dir = tokenizer_path
        tokenizer_file_list = os.listdir(tokenizer_path)
        tokenizer_file_list = [f for f in file_list if os.path.isfile(os.path.join(tokenizer_path, f))]

        for file_name in tokenizer_file_list:
            source_file = os.path.join(tokenizer_path + file_name)
            if os.path.islilnk(source_file):
                actual_file = os.readlink(source_file)
                print("####actual_file :", actual_file)

                actual_file_path = os.path.join(os.path.dirname(source_dir), actual_file)
                shutil.copy2(actual_file_path, self.tokenizer_path)

            else:
                print("Fail to download tokenizer")

