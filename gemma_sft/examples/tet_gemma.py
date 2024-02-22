# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/8/4 15:11
# @author  : Mo
# @function:



import traceback
import random
import time
import sys
import os

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(path_root)
sys.path.append(path_root)
CUDA_VISIBLE_DEVICES = "-1"
USE_TORCH = "1"
CPU_NUMS = "8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3072"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["USE_TORCH"] = USE_TORCH
os.environ["OMP_NUM_THREADS"] = CPU_NUMS  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = CPU_NUMS  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = CPU_NUMS  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = CPU_NUMS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = CPU_NUMS  # export NUMEXPR_NUM_THREADS=1


from gemma_sft.models.gemma.tokenization_gemma import GemmaTokenizer as LLMTokenizer
from gemma_sft.models.gemma.configuration_gemma import GemmaConfig as LLMConfig
from gemma_sft.models.gemma.modeling_gemma import GemmaForCausalLM as LLMModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


# PATH_PRERAIN_MODEL = "google/google_gemma-2b-it"


tokenizer = LLMTokenizer.from_pretrained(PATH_PRERAIN_MODEL, trust_remote_code=True)
# use bf16
model = LLMModel.from_pretrained(PATH_PRERAIN_MODEL, trust_remote_code=True).bfloat16().eval()

# tokenizer = AutoTokenizer.from_pretrained(PATH_PRERAIN_MODEL, trust_remote_code=True)
# # use bf16
# model = AutoModelForCausalLM.from_pretrained(PATH_PRERAIN_MODEL,
#                                              trust_remote_code=True).bfloat16().eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained(PATH_PRERAIN_MODEL, device_map="auto", trust_remote_code=True, fp16=True).eval()
# use fp32
# model = AutoModelForCausalLM.from_pretrained(PATH_PRERAIN_MODEL,
#                                              device_map="auto",
#                                              trust_remote_code=True).eval()

model.generation_config = GenerationConfig.from_pretrained(PATH_PRERAIN_MODEL,
                                                           trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参


trainable_params = 0
all_param = 0
for name, param in model.named_parameters():
    print((name, param.data.dtype, param.requires_grad))
    num_params = param.numel()
    # if using DS Zero 3 and the weights are initialized empty
    if num_params == 0 and hasattr(param, "ds_numel"):
        num_params = param.ds_numel
    all_param += num_params
    if param.requires_grad:
        trainable_params += num_params
print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")



input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids, max_length=32)
print(tokenizer.decode(outputs[0]))



# # 第一轮对话 1st dialogue turn
# response, history = model.chat(tokenizer, "你好", history=None)
# print(response)
# # 你好！很高兴为你提供帮助。
#
# # 第二轮对话 2nd dialogue turn
# response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
# print(response)
#
# # 第三轮对话 3rd dialogue turn
# response, history = model.chat(tokenizer, "给这个故事起一个标题", history=history)
# print(response)
# # 《奋斗创业：一个年轻人的成功之路》


if __name__ == '__main__':
    import traceback
    import time

    data_dict = {"instruction": "解释为什么下面的分数等于 1/4",
                 "input": "解释为什么下面的分数等于 1/4，4/16。",
                 "output": "分数 4/16 等于 1/4，因为分子和分母都可以被 4 整除。将顶部和底部数字都除以 4 得到分数 1/4。"
                 }
    # input_text = data_dict.get("instruction", "") + "\t" + data_dict.get("input", "")
    # input_text = """<start_of_turn>user\n你知道为什么兔子不喜欢吃窝边草吗？<end_of_turn>\n<start_of_turn>model\n"""
    # input_ids = tokenizer(input_text, return_tensors="pt")
    # outputs = model.generate(**input_ids, max_length=128)
    # print(tokenizer.decode(outputs[0]))
    ques = data_dict.get("input", "")
    chat = [
        {"role": "user", "content": ques},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    outputs = model.generate(**input_ids, max_new_tokens=150)
    print(tokenizer.decode(outputs[0]))

    while True:
        try:
            time_start = time.time()
            history = []
            print("请输入:")
            ques = input()
            print("请稍等...")

            if ques.strip().upper() == "CLEAR":
                history = []
                print("clear ok")
                continue
            else:

                chat = [
                    {"role": "user", "content": ques},
                ]
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                input_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
                outputs = model.generate(**input_ids, max_new_tokens=150)
                print(tokenizer.decode(outputs[0]))
            print(time.time() - time_start)
        except Exception as e:
            print(traceback.print_exc())
            print(str(e))
