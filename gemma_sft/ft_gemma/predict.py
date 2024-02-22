# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/25 21:56
# @author  : Mo
# @function: 推理


import random
import time
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(path_root)
sys.path.append(path_root)
from gemma_sft.ft_gemma.config import CUDA_VISIBLE_DEVICES, USE_TORCH, CPU_NUMS  # from config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3072"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CUDA_VISIBLE_DEVICES
os.environ["USE_TORCH"] = USE_TORCH
os.environ["OMP_NUM_THREADS"] = CPU_NUMS  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = CPU_NUMS  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = CPU_NUMS  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = CPU_NUMS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = CPU_NUMS  # export NUMEXPR_NUM_THREADS=1

from peft import (get_peft_model, LoraConfig)
from transformers import GenerationConfig
import torch

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(path_root)
sys.path.append(path_root)
from gemma_sft.ft_gemma.config import CUDA_VISIBLE_DEVICES, USE_TORCH, CPU_NUMS  # from config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3072"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["USE_TORCH"] = USE_TORCH
os.environ["OMP_NUM_THREADS"] = CPU_NUMS  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = CPU_NUMS  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = CPU_NUMS  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = CPU_NUMS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = CPU_NUMS  # export NUMEXPR_NUM_THREADS=1

from peft import LoraConfig, get_peft_model
from transformers import GenerationConfig
from pydantic import BaseModel
from rouge import Rouge  # pip install rouge
from tqdm import tqdm
import torch
import jieba

from gemma_sft.models.gemma.modeling_gemma import GemmaForCausalLM as LLMModel
from gemma_sft.models.gemma.tokenization_gemma import GemmaTokenizer as LLMTokenizer
from gemma_sft.models.gemma.configuration_gemma import GemmaConfig as LLMConfig
from gemma_sft.ft_gemma.config import PATH_MODEL_PRETRAIN, DATA_PATH, MODEL_SAVE_DIR, REPO_ID
from gemma_sft.ft_gemma.config import MICRO_BATCH_SIZE, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
from gemma_sft.ft_gemma.config import LEARNING_RATE, EPOCHS, SAVE_STEPS, VAL_SET_SIZE, TARGET_MODULES
from gemma_sft.ft_gemma.config import IS_PARALLELIZABLE, MODEL_PARALLEL, USE_CACHE
from gemma_sft.ft_gemma.config import MAX_LENGTH_Q, MAX_LENGTH_A, MAX_LENGTH_QA
from gemma_sft.ft_gemma.config import LORA_DROPOUT, LORA_ALPHA, LORA_R
from gemma_sft.ft_gemma.config import USE_CUDA


# device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
device_map = "auto"
# USE_CUDA = True
print(device_map)
print(ddp)


def load_model_state(model, model_save_dir="./", model_name="pytorch_model.bin", device="cpu"):
    """  仅加载模型参数(推荐使用)  """
    try:
        path_model = os.path.join(model_save_dir, model_name)
        peft_config = LoraConfig.from_pretrained(model_save_dir)
        peft_config.inference_mode = True
        model = get_peft_model(model, peft_config)
        state_dict = torch.load(path_model, map_location=torch.device(device))
        # print(state_dict.keys())
        state_dict = {"base_model.model." + k.replace("_orig_mod.", "")
                      .replace(".lora_A.weight", ".lora_A.default.weight")
                      .replace(".lora_B.weight", ".lora_B.default.weight")
                      : v for k, v in state_dict.items()}
        print(state_dict.keys())
        print("#" * 128)
        ### 排查不存在model.keys的 state_dict.key
        name_dict = {name: 0 for name, param in model.named_parameters()}
        print(name_dict.keys())
        print("#" * 128)
        for state_dict_key in state_dict.keys():
            if state_dict_key not in name_dict:
                print("{} is not exist!".format(state_dict_key))
        model.load_state_dict(state_dict, strict=False)
        # model.to(device)
        print("******model loaded success******")
        print("self.device: {}".format(device))
    except Exception as e:
        print(str(e))
        raise Exception("******load model error******")
    return model
def save_model_state(model, config=None, model_save_dir="./", model_name="pytorch_model.bin"):
    """  仅保存模型参数(推荐使用)  """
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if config:
        config.save_pretrained(model_save_dir)
    # save model
    path_model = os.path.join(model_save_dir, model_name)
    # torch.save(model.state_dict(), path_model)
    grad_params_dict = {k: v.to("cpu") for k, v in model.named_parameters()
                        if v.requires_grad == True}
    torch.save(grad_params_dict, path_model)
    print("******model_save_path is {}******".format(path_model))
def prepare_model_for_half_training(model, output_embedding_layer_name="lm_head",
        use_gradient_checkpointing=True, layer_norm_names=["layer_norm"]):
    r"""
    This method wrapps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    #  不要使用 model.half(), 这样会先截取精度再训练了, 最初data就要保持half
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False
        # cast layer norm in fp32 for stability for 8bit models
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)
        elif output_embedding_layer_name in name:  # lm_head也需要是tf.float32(最后一层)
            param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(torch.half)

    if use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
    return model
def print_named_parameters(model, use_print_data=True):
    """   打印模型训练参数/数据类型信息   """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        if use_print_data:
            print((name, param.data.dtype, param.requires_grad, param.data))
        else:
            print((name, param.data.dtype, param.requires_grad))
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
def generate_prompt_train(data_point, is_logger=False):
    """   指令微调:
    普通句子续写: bos + text + eos
    带 prompt:
    ‘<start_of_turn>user
    Knock knock.<end_of_turn>
    <start_of_turn>model
    Who’s there?<end_of_turn>model
    <start_of_turn>user
    Gemma.<end_of_turn>
    <start_of_turn>model
    Gemma who?<end_of_turn>model’
    """

    text_input = data_point.get("instruction", "") + "\t" + data_point.get("input", "")
    text_out = data_point.get("output", "")
    prompt_text_1 = "<start_of_turn>user\n{}<end_of_turn>\n"
    prompt_text_2 = "<start_of_turn>model\n{}<end_of_turn>"
    text_1 = prompt_text_1.format(text_input.strip())
    text_2 = prompt_text_2.format(text_out.strip())
    x = tokenizer.encode(text_1, add_special_tokens=False)
    y = tokenizer.encode(text_2, add_special_tokens=False)
    if len(x) + len(y) > (MAX_LENGTH_Q + MAX_LENGTH_A):
        x = x[:MAX_LENGTH_Q]
        y = y[:MAX_LENGTH_A]
    x = [ID_BOS] + x
    y = y + [ID_EOS]
    out = {"input_ids": x, "labels": y}
    if is_logger:
        print(text_1)
        print(text_2)
        print(out)
    return out
def generate_prompt(data_point, is_logger=False):
    """   指令微调:
    普通句子续写: bos + text + eos
    带 prompt:
    ‘<start_of_turn>user
    Knock knock.<end_of_turn>
    <start_of_turn>model
    Who’s there?<end_of_turn>model
    <start_of_turn>user
    Gemma.<end_of_turn>
    <start_of_turn>model
    Gemma who?<end_of_turn>model’
    """

    text_input = data_point.get("instruction", "") + "\t" + data_point.get("input", "")
    text_out = data_point.get("output", "")
    prompt_text_1 = "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
    text_1 = prompt_text_1.format(text_input.strip())
    x = tokenizer.encode(text_1, add_special_tokens=False)
    if len(x) > (MAX_LENGTH_Q + MAX_LENGTH_A):
        x = x[:MAX_LENGTH_Q]
    x = [ID_BOS] + x
    out = {"input_ids": x, "labels": []}
    if is_logger:
        print(text_1)
        print(out)
    return out


model = LLMModel.from_pretrained(PATH_MODEL_PRETRAIN)
model = prepare_model_for_half_training(model,
        use_gradient_checkpointing=False,
        output_embedding_layer_name="lm_head",
        layer_norm_names=["post_attention_layernorm",
                          "input_layernorm",
                          "norm"
                          ],
        )
config = LoraConfig(target_modules=TARGET_MODULES,
                    lora_dropout=LORA_DROPOUT,
                    lora_alpha=LORA_ALPHA,
                    task_type="CAUSAL_LM",
                    bias="none",
                    r=LORA_R,
                    )
model = get_peft_model(model, config)
model = load_model_state(model=model, model_save_dir=MODEL_SAVE_DIR)
if USE_CUDA:
    model = model.cuda()
else:
    model = model.bfloat16()
print_named_parameters(model, use_print_data=True)
# print_named_parameters(model)

tokenizer = LLMTokenizer.from_pretrained(PATH_MODEL_PRETRAIN, add_eos_token=True)
ID_PAD = 0
ID_BOS = 1
ID_EOS = 2
ID_UNK = 3
ID_MASK = 4
ID_SOT = 106
ID_EOT = 107
ID_BR = 108  # "\n"
ID_USER = 1645
ID_MODEL = 2516
tokenizer.pad_token_id = ID_PAD
tokenizer.eos_token_id = ID_EOS
tokenizer.padding_side = "right"  # NO use attention-mask
print(ID_PAD)
print(ID_BOS)
print(ID_EOS)
print(ID_UNK)
print(ID_MASK)
print(ID_SOT)
print(ID_SOT)
print(ID_BR)
print(ID_USER)
print(ID_MODEL)
"""
"<pad>": 0,
"<eos>": 1,
"<bos>": 2,
"<unk>": 3,
"<mask>": 4,
"<start_of_turn>": 106,
"<end_of_turn>": 107,
"\n": 108,
"user": 1645,
"model": 2516,
"""
STOP_WORDS_IDS = [[107],[1], [0]]


def predict(data_dict):
    """  推理  """
    prompt_dict = generate_prompt(data_dict)
    # inputs = tokenizer([text_1], return_tensors="pt", padding=True)
    input_ids = prompt_dict.get("input_ids")
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    if USE_CUDA:
        input_ids = input_ids.cuda()
    generation_config = GenerationConfig(
        # temperature=0.95,
        # top_p=0.75,
        temperature=0.95,
        top_p=0.7,
        top_k=50,
        num_beams=1,
        do_sample=True,
        max_new_tokens=256,
        # penalty_alpha=1.5,
        pad_token_id=ID_PAD,
        eos_token_id=ID_EOS,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            # stop_words_ids=STOP_WORDS_IDS,
            # stop_words_ids=[[ID_EOS]],
            return_dict_in_generate=True,
            # return_dict_in_generate=True,
            # output_scores=True,
            # max_new_tokens=512,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print(data_dict)
    print(input_ids)
    print(output)
    # output = output.split("答：")[-1]
    return output



if __name__ == '__main__':
    data_dict = {"instruction": "解释为什么下面的分数等于 1/4",
                 "input": "解释为什么下面的分数等于 1/4，4/16",
                 "output": "分数 4/16 等于 1/4，因为分子和分母都可以被 4 整除。将顶部和底部数字都除以 4 得到分数 1/4。"
                 }
    res = predict(data_dict)
    print(res)
    while True:
        time_start = time.time()
        history = []
        print("请输入:")
        ques = input()
        print("请稍等...")
        try:
            if ques.strip().upper() == "CLEAR":
                history = []
                print("clear ok")
                continue
            else:
                print("#" * 128)
                ques_dict = {"instruction": ques, "input": "", "output": ""}
                res = predict(ques_dict)
                print(res)
                print("###mid###" * 32)
                response, history = model.chat(tokenizer, ques, history=history)
                print(response)
                print("#" * 128)
        except Exception as e:
            print(str(e))
        print(time.time() - time_start)

"""
python predict.py

"""