from __future__ import print_function
import atexit
import errno
import sys
import time
import signal
import json
import logging
import os
import time
import datetime
from dotenv import load_dotenv
from enum import Enum
import html2text
import pprint as pp
import base64

from logging.handlers import TimedRotatingFileHandler
from multiprocessing import Process
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain

from cca_conf import CCA_CONFIG
from cca_email_util import CCAEmailMessageExtracter
from cca_output import CCAOutput, CCA_OUTPUT_MODE, CCAResult
from llm_model import LLMModel
from daemon import Daemon 
from cca_redis import RedisClient
from cca_template import CCAPromptTemplate 

from cca_common import MAIL_TYPE
from lang_detect import LangAnalyzer

class CCAMain():
    def __init__(self, conf, logger, redis, pid, llm_model, cca_output, cca_template, lang_analyzer):
        self.conf = conf
        self.logger = logger
        self.redis = redis
        self.pid = pid
        self.llm_model = llm_model
        self.cca_output = cca_output
        self.cca_template = cca_template
        self.cca_input = None
        self.cca_result = None
        self.elapsed_llm = 0.0
        self.elapsed_token = 0.0
        self.final_tokens = 0
        self.output_tokens = 0
        self.lang_analyzer = lang_analyzer
        self.class_stat = {}
        self.class_stat[MAIL_TYPE.CONFIDENTIAL] = 0
        self.class_stat[MAIL_TYPE.WORK_RELATED] = 0
        self.class_stat[MAIL_TYPE.OTHERS] = 0
        self.class_stat[MAIL_TYPE.NONE] = 0

class CCAInput():
    def __init__(self, filename, header, modified_body, final_prompt):
        self.filename = filename
        self.header = header
        self.modified_body = modified_body
        self.final_prompt = final_prompt

class CCASubscriber(Daemon):
    def __init__(self, pidfile, stdin=os.devnull, stdout=os.devnull, stderr=os.devnull):
        self.logger = None
        self.conf = CCA_CONFIG()
        self.conf.load_conf("./cca_sub.json")

        super().__init__(pidfile, self.conf.is_daemon, self.conf.run_once, stdin, stdout, stderr)

    def setup_logger(self, process_id):
        logger = logging.getLogger(f'cca_sub_{process_id}')
        logger.setLevel(logging.INFO)

        handler = TimedRotatingFileHandler(
                os.path.join(self.conf.log_path, f'cca_sub_{process_id}.log'),
                when='D', interval=1, backupCount=7)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if self.conf.print_console == 1:
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            #formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            #console.setFormatter(formatter)
            logger.addHandler(console)

        self.logger = logger
        return logger

    def run_once(self):
        processes = []

        print(f"MAIN PROCESS : fork {self.conf.num_process} process", flush=True)

        if self.conf.platform == "vllm":
            if self.conf.num_batch > 1:
                self.batched_worker_once(0)
            else:
                self.worker_once(0)
            print("MAIN PROCESS : Done", flush=True)
            return

        # may have scaling issue
        self.worker_once(0)

        time.sleep(2)

        self.init_redis()

        tot_latency = 0.0
        tot_inference = 0

        for p in processes:
            p.join()

        print("MAIN PROCESS : Done", flush=True)

    def run(self):
        processes = []

        print(f"MAIN PROCESS : fork {self.conf.num_process} process", flush=True)

        if self.conf.platform == "vllm":
            if self.conf.num_batch > 1:
                self.batched_worker_daemon(0)
            else:
                self.worker(0)
            return

        for i in range(self.conf.num_process): 
            p = Process(target=self.worker, args=(i,))
            p.daemon = True
            p.start()
            processes.append(p)

        time.sleep(2)

        self.init_redis()

        print("MAIN PROCESS : Wait for Join", flush=True)

        tot_latency = 0.0
        tot_inference = 0

        start = datetime.datetime.now()
        last_sec = datetime.datetime.now()
        last_min = datetime.datetime.now()
        last_count_per_sec = 0
        last_count_per_min = 0
        while True:
            now = datetime.datetime.now()
            if datetime.datetime.now() - last_sec > datetime.timedelta(seconds=1):
                last_sec = now
                last_count_per_sec = tot_inference
                
            if datetime.datetime.now() - last_min > datetime.timedelta(seconds=60):
                last_min = now
                last_count_per_min = tot_inference

            result = self.redis.rpop('cca:result')
            if result is None:
                time.sleep(0.01)
                continue 

            tot_inference = tot_inference + 1

            value = result.split(',')
            tot_latency = tot_latency + float(value[3])

        for p in processes:
            p.join()

        print("MAIN PROCESS : Done", flush=True)

        how_long = datetime.datetime.now() - start

    def init_gpu(self, process_id):
        sched.shutdown()
        return

    def init_redis(self):
        self.redis = RedisClient(self.conf.redis_ip, self.conf.redis_port)
        self.redis.connect()
        return

    def process_batched_works(self, cca_main, redis_prompts, redis_files):
        self.tot_cnt = self.tot_cnt + 1

        self.logger.info(f"\n{self.tot_cnt} REQUEST >>")

        headers, modified_bodies = self.cca_email.batched_message_diet(redis_prompts, redis_files)
        final_prompts = self.cca_email.batched_truncate_token(headers, modified_bodies, redis_files)
        final_prompts = self.llm_model.batched_build_prompt(final_prompts, passlock="No", encrypt="No", bigsize="No", suspicious="No")

        got_term = False
        got_exception = False
        try:
            responses, elapsed_llms = self.llm_model.batched_run(self.conf, final_prompts)
        except KeyboardInterrupt:
            self.logger.info(f"got KeyboardInterrupt")
            got_term = True
            pass
        except:
            self.logger.warning(f"LLM Service is Down!! {self.llm_model.model_platform} / {self.llm_model.model_name} / {redis_files}")
            self.logger.info(f"input : {redis_files} / result : none (exception)")
            got_exception = True
            pass
        else:
            idx = 0
            for response in responses:
                cca_main.elapsed_llm = elapsed_llms[idx]
                redis_file = redis_files[idx]
                header = headers[idx]
                modified_body = modified_bodies[idx]
                redis_prompt = redis_prompts[idx]
                final_prompt = final_prompts[idx]

                _, cca_main.final_tokens = self.llm_model.tokenize_prompt(final_prompt, redis_file)
                cca_main.in_content_language = self.lang_analyzer.detect_language(modified_body.strip())

                cca_main.cca_input = CCAInput(redis_file, header, modified_body, final_prompt)
                cca_main.cca_result = self.cca_output.parse_output(cca_main, response)
                if cca_main.cca_result.validation != "error":
                    self.cca_output.write_csv(cca_main)
                    self.cca_output.write_json(cca_main)
                else:
                    self.error_cnt = self.error_cnt + 1
                    self.cca_output.write_pend(redis_file, redis_prompt)
                self.cca_output.write_log(cca_main, self.tot_cnt, 0, self.error_cnt)
                idx = idx + 1
        return got_term

    def batched_worker_once(self, pid):
        self.tot_cnt = 0
        self.pid = pid

        self.init_engine(self.conf, pid)

        last = time.time()
        redis_prompts = []
        redis_files = []

        files = os.listdir(self.conf.data_path)
        got_term = False
        self.error_cnt = 0
        self.fixed_cnt = 0
        self.logger.info(f"===== STEP 1. PROCESS NEW TASK =====")
        for file in files:
            # flush by timeout
            if time.time() - last > 0.1:
                last = time.time()
                if len(redis_prompts) > 0:
                    got_term = self.process_batched_works(self.cca_main, redis_prompts, redis_files)
                redis_prompts = []
                redis_files = []

            path = os.path.join(self.conf.data_path, file)
            f = open(path, 'r')
            redis_file   = file
            redis_prompt = f.read()
            redis_prompts.append(redis_prompt)
            redis_files.append(redis_file)
            f.close()

            # flush by count
            if len(redis_prompts) >= self.conf.num_batch:
                got_term = self.process_batched_works(self.cca_main, redis_prompts, redis_files)
                redis_prompts = []
                redis_files = []

            if got_term == True:
                break

        self.cca_output.flush_json(self.conf)
        self.cca_output.flush_pend(self.conf)

        self.logger.info(f"===== STEP 2. RE-PROCESS INCOMPLETED TASK =====")
        self.process_pended_work(self.cca_main)

        self.cca_output.stop()
        return

    def batched_worker_daemon(self, pid):
        self.tot_cnt = 0
        self.pid = pid

        self.init_engine(self.conf, pid)

        last = time.time()
        redis_prompts = []
        redis_files = []

        self.logger.info(f"===== STEP 1. PROCESS NEW TASK =====")

        got_term = False
        while True:
            # flush by timeout
            if time.time() - last > 0.1:
                last = time.time()
                if len(redis_prompts) > 0:
                    got_term = self.process_batched_works(self.cca_main, redis_prompts, redis_files)
                redis_prompts = []
                redis_files = []

            redis_prompt = self.redis.rpop('cca:prompt')
            if redis_prompt is None:
                time.sleep(0.01)
                continue

            redis_split  = redis_prompt.split(',', 1)
            redis_file   = redis_split[0]
            redis_prompt = redis_split[1]

            if redis_file != "END":
                redis_prompts.append(redis_prompt)
                redis_files.append(redis_file)

            # flush by count
            if len(redis_prompts) >= self.conf.num_batch:
                got_term = self.process_batched_works(self.cca_main, redis_prompts, redis_files)
                redis_prompts = []
                redis_files = []

            if got_term == True: break

        self.cca_output.flush_json(self.conf)
        self.cca_output.flush_pend(self.conf)

        self.logger.info(f"===== STEP 2. RE-PROCESS INCOMPLETED TASK =====")
        self.process_pended_work(self.cca_main)

        self.cca_output.stop()
        return

    def process_single_work(self, cca_main, redis_file, redis_prompt):
        self.tot_cnt = self.tot_cnt + 1

        self.logger.info(f"\n{self.tot_cnt} REQUEST >>")

        header, modified_body = self.cca_email.message_diet(redis_prompt, redis_file)
        input_message = header + '\n' + modified_body
        input_message, _ = self.cca_email.truncate_token(input_message, redis_file)
        final_prompt = self.llm_model.build_prompt(input_message)

        got_term = False
        got_exception = False
        for retry in range(3):
            try:
                response, cca_main.elapsed_llm = self.llm_model.run(self.conf, final_prompt)
            except KeyboardInterrupt:
                got_term = True
                self.logger.warning(f"got KeyboardInterrupt")
                break
            except Exception as e:
                got_exception = True
                self.logger.info(f"input : {redis_file} / result : none (exception:{sys.exc_info()[0]})")
                self.logger.warning(f"LLM Service is Down!! {self.llm_model.model_platform} / {self.llm_model.model_name} / {redis_file}")
                self.logger.warning(f"ERROR  : {e}")
                break 
            else:
                _, cca_main.final_tokens = self.llm_model.tokenize_prompt(final_prompt, redis_file)
                cca_main.in_content_language = self.lang_analyzer.detect_language(modified_body.strip())
                cca_main.cca_input = CCAInput(redis_file, header, modified_body, final_prompt)
                cca_main.cca_result = self.cca_output.parse_output(cca_main, response)
                if cca_main.cca_result.validation != "error":
                    self.cca_output.write_csv(cca_main)
                    self.cca_output.write_json(cca_main)
                    self.cca_output.write_log(cca_main, self.tot_cnt, retry, self.error_cnt)
                    break;
                else:
                    self.cca_output.write_log(cca_main, self.tot_cnt, retry, self.error_cnt)
                    if retry == 2:
                        self.error_cnt = self.error_cnt + 1
                        self.cca_output.write_pend(redis_file, redis_prompt)
                        return False, got_term

        if got_term == True: return False, got_term
        if got_exception == True: return False, got_term

        return True, got_term

    def process_pended_work(self, cca_main):
        #self.cca_output.print_response = False
        logger = cca_main.logger
        got_term = None

        while True:
            self.error_cnt = 0
            self.tot_cnt = 0
            if self.cca_output.pend_retry_cnt >= 3: break

            json_data = self.cca_output.read_pend(self.conf)
            for json_entry in json_data:
                filename = json_entry['filename']
                prompt_enc = json_entry['prompt']
                prompt = base64.b64decode(prompt_enc)
                prompt = prompt.decode('UTF-8')
                done, got_term = self.process_single_work(self.cca_main, filename, prompt)
                if done == True: self.fixed_cnt = self.fixed_cnt + 1
                if got_term == True: break

            if got_term == True: break
            self.cca_output.flush_json(self.conf)
            self.cca_output.flush_pend(self.conf)
        return

    def init_engine(self, conf, pid):
        self.init_redis()
        self.logger = self.setup_logger(pid)
        self.logger.info(f"\n\n===========\nworker {pid} is started\n")

        self.llm_model = LLMModel(conf, self.logger)
        self.cca_output = CCAOutput(conf, self.logger, self.llm_model, pid, CCA_OUTPUT_MODE.STRUCTURED)
        self.cca_template = CCAPromptTemplate(conf, self.logger)
        self.cca_email = CCAEmailMessageExtracter(conf, self.logger, self.llm_model)
        self.lang_analyzer = LangAnalyzer()
        self.cca_main = CCAMain(conf, self.logger, self.redis, self.pid, self.llm_model, self.cca_output, self.cca_template, self.lang_analyzer)
        template_length = self.llm_model.set_template(self.cca_template, passlock="No", encrypt="No", bigsize="No", suspicious="No", language=self.conf.language)
        return

    def worker_once(self, pid):
        self.tot_cnt = 0
        self.pid = pid 

        self.init_engine(self.conf, pid)

        files = os.listdir(self.conf.data_path)

        last_min = datetime.datetime.now()

        self.logger.info(f"===== STEP 1. PROCESS NEW TASK =====")

        self.error_cnt = 0
        self.fixed_cnt = 0
        for file in files:
            now = datetime.datetime.now()

            if now - last_min > datetime.timedelta(seconds=60):
                self.cca_output.flush_json(self.conf)
                last_min = now

            path = os.path.join(self.conf.data_path, file)
            f = open(path, 'r')
            redis_file   = file
            redis_prompt = f.read()
            f.close()

            _, got_term = self.process_single_work(self.cca_main, redis_file, redis_prompt)
            if got_term == True: break

        self.cca_output.flush_json(self.conf)
        self.cca_output.flush_pend(self.conf)

        self.logger.info(f"===== STEP 2. RE-PROCESS INCOMPLETED TASK =====")
        self.process_pended_work(self.cca_main)

        self.cca_output.stop()
        self.logger.info(f"WORKER LOOP : DONE")

        return

    def worker_daemon(self, pid):
        self.tot_cnt = 0
        self.pid = pid 

        self.init_engine(self.conf, pid)

        got_term = False
        self.error_cnt = 0
        self.fixed_cnt = 0
        self.logger.info(f"===== STEP 1. PROCESS NEW TASK =====")

        while True:
            redis_prompt = self.redis.rpop('cca:prompt')
            if redis_prompt is None:
                time.sleep(0.01)
                continue

            redis_split  = redis_prompt.split(',', 1)
            redis_file   = redis_split[0]
            redis_prompt = redis_split[1]
            
            if redis_file == "END":
                break

            done, got_term = self.process_single_work(self.cca_main, redis_file, redis_prompt)
            if got_term == True: break

        self.cca_output.flush_json(self.conf)
        self.cca_output.flush_pend(self.conf)

        self.logger.info(f"===== STEP 2. RE-PROCESS INCOMPLETED TASK =====")
        self.process_pended_work(self.cca_main)

        self.cca_output.stop()
        self.logger.info(f"WORKER LOOP : DONE")
        return

if __name__ == "__main__":
    daemon = CCASubscriber('/var/lock/cca_sub.pid')

    if len(sys.argv) >= 2:
        # Daemon.start -> Daemon.daemonize
        #              -> CCASubscriber.run    -> Process.start = CCASubscriber.worker
        if 'start' == sys.argv[1]:
            daemon.start()
        elif 'stop' == sys.argv[1]:
            daemon.stop()
            logging.shutdown()
        elif 'restart' == sys.argv[1]:
            daemon.restart()
        else:
            print("Unknown command")
            sys.exit(2)
        sys.exit(0)
    else:
        print("usage: %s start|stop|restart" % sys.argv[0])
        sys.exit(2)


