# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/25 21:56
# @author  : Mo
# @function: 验证评估


import logging as logger
import traceback
import logging
import random
import time
import json
import sys
import os

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
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

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from peft import prepare_model_for_int8_training
from peft import LoraConfig, get_peft_model
from transformers import GenerationConfig
from pydantic import BaseModel
from rouge import Rouge  # pip install rouge
from tqdm import tqdm
import torch
import jieba

# from gemma_sft.models.gemma.modeling_gemma import GemmaForCausalLM as LLMModel
# from gemma_sft.models.gemma.tokenization_gemma import GemmaTokenizer as LLMTokenizer
# from gemma_sft.models.gemma.configuration_gemma import GemmaConfig as LLMConfig

from transformers import GemmaTokenizer as LLMTokenizer
from transformers import GemmaForCausalLM as LLMModel
from transformers import GemmaConfig as LLMConfig

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


def load_model_state(model, model_save_dir="./", model_name="adapter_model.safetensors", device="cpu"):
    """  仅加载模型参数(推荐使用)  """
    try:
        path_model = os.path.join(model_save_dir, model_name)
        peft_config = LoraConfig.from_pretrained(model_save_dir)
        peft_config.inference_mode = True
        model = get_peft_model(model, peft_config)

        try:
            if path_model.endswith(".safetensors"):
                from safetensors.torch import load_file, save_file
                from safetensors import safe_open
                state_dict = {}
                with safe_open(path_model, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        state_dict[k] = f.get_tensor(k)
            ### if path_model.endswith(".bin") or path_model.endswith(".pt"):
            else:
                state_dict = torch.load(path_model, map_location=torch.device(device))
        except Exception as e:
            print(traceback.print_exc())
            ### 全部训练完的话会用这个, 即便是.safetensors
            state_dict = torch.load(path_model, map_location=torch.device(device))

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
def print_named_parameters(model, use_print_data=False):
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
def txt_read(path, encode_type="utf-8", errors=None):
    """
        读取txt文件，默认utf8格式, 不能有空行
    Args:
        path[String]: path of file of read, eg. "corpus/xuexiqiangguo.txt"
        encode_type[String]: data encode type of file, eg. "utf-8", "gbk"
        errors[String]: specifies how encoding errors handled, eg. "ignore", strict
    Returns:
        lines[List]: output lines
    """
    lines = []
    try:
        file = open(path, "r", encoding=encode_type, errors=errors)
        lines = file.readlines()
        file.close()
    except Exception as e:
        logger.info(str(e))
    finally:
        return lines
def load_json(path: str, encoding: str="utf-8"):
    """
    Read Line of List<json> form file
    Args:
        path: path of save file, such as "txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    Returns:
        model_json: dict of word2vec, eg. [{"大漠帝国":132}]
    """
    with open(path, "r", encoding=encoding) as fj:
        model_json = json.load(fj)
        fj.close()
    return model_json
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
# model = prepare_model_for_half_training(model,
#         use_gradient_checkpointing=False,
#         output_embedding_layer_name="lm_head",
#         layer_norm_names=["post_attention_layernorm",
#                           "input_layernorm",
#                           "norm"
#                           ],
#         )
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
ID_BOS = 2
ID_EOS = 1
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



def predict(data_point, generation_config):
    """  推理  """
    prompt_dict = generate_prompt(data_point)
    # inputs = tokenizer([text_1], return_tensors="pt", padding=True)
    input_ids = prompt_dict.get("input_ids")
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    if USE_CUDA:
        input_ids = input_ids.cuda()
    generation_config = GenerationConfig(**generation_config)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            # stop_words_ids=STOP_WORDS_IDS,
            return_dict_in_generate=True,
            output_scores=True,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print(input_ids)
    print(s)
    print(output)
    # output = output.strip()
    return output
def text_generate(request_data):
    instruction = request_data.instruction
    text = request_data.text
    penalty_alpha = request_data.penalty_alpha
    max_new_tokens = request_data.max_new_tokens
    temperature = request_data.temperature
    do_sample = request_data.do_sample
    num_beams = request_data.num_beams
    top_p = request_data.top_p
    top_k = request_data.top_k

    generation_dict = vars(request_data)
    print(generation_dict)
    generation_dict.pop("max_new_tokens")
    generation_dict.pop("instruction")
    generation_dict.pop("text")
    data_point = {"instruction": instruction, "input": text, "output": ""}
    generation_config = {"temperature": temperature,
                         "top_p": top_p,
                         "top_k": top_k,
                         "num_beams": num_beams,
                         "do_sample": do_sample,
                         "penalty_alpha": penalty_alpha,
                         "max_new_tokens": max_new_tokens,
                         "pad_token_id": ID_PAD,
                         "eos_token_id": ID_EOS,
                         }
    try:  # 数据预处理, 模型预测
        response = predict(data_point, generation_config)
    except Exception as e:
        logger.info(traceback.print_exc())
        response = "<|endoftext|>"
    return response
class Item(BaseModel):
    instruction: str = ""
    text: str = "1+1="
    penalty_alpha: float = 1.0
    max_new_tokens: int = 512
    temperature: float = 0.8  # 0.95  # 0.35  # 0.95
    do_sample: bool = True
    num_beams: int = 1
    top_p: float = 0.8  # 0.75
    top_k: int = 50


if __name__ == '__main__':

    text = "1+1="
    item_config = Item()
    item_config.text = text
    response = text_generate(item_config)
    print(response)

    smooth = SmoothingFunction().method1
    rouge = Rouge()
    best_bleu = 0.

    fw = open(DATA_PATH + ".qwen_eval_rouge_blue.512.json", "a+", encoding="utf-8")
    rouge_1_p, rouge_2_p, rouge_l_p = 0, 0, 0
    rouge_1_r, rouge_2_r, rouge_l_r = 0, 0, 0
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
    total = 0
    time_start = time.time()
    datas = load_json(DATA_PATH)
    # datas = datas[:1024]
    datas = datas[:8]
    for d_json in tqdm(datas, desc="data"):
        try:
            instruction = d_json.get("instruction", "")
            text_input = d_json.get("input", "")
            text_output = d_json.get("output", "")
            # qtext, qans
            total += 1
            item_config = Item()
            item_config.instruction = instruction
            item_config.text = text_input
            text_output = " ".join(jieba.lcut(text_output))
            text_gen_str = text_generate(item_config)
            text_pred = " ".join(jieba.lcut(text_gen_str.replace("<|endoftext|>", "").replace("<|im_end|>", "")
                                            .replace("<|im_start|>", "").strip().lower()))
            line = {"input": text_input, "output": text_output.replace(" ", ""),
                    "pred": text_pred.replace(" ", "")}
            line_str = json.dumps(line, ensure_ascii=False) + "\n"
            fw.write(line_str)
            if text_pred.strip():
                scores = rouge.get_scores(hyps=text_pred, refs=text_output)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']

                rouge_1_p += scores[0]['rouge-1']['p']
                rouge_2_p += scores[0]['rouge-2']['p']
                rouge_l_p += scores[0]['rouge-l']['p']
                rouge_1_r += scores[0]['rouge-1']['r']
                rouge_2_r += scores[0]['rouge-2']['r']
                rouge_l_r += scores[0]['rouge-l']['r']

                bleu += sentence_bleu(references=[list(text_output.replace(" ", ""))],
                                      hypothesis=list(text_pred.replace(" ", "")),
                                      smoothing_function=smooth)
                mertics_i = {'rouge-1': rouge_1, 'rouge-2': rouge_2, 'rouge-l': rouge_l,
                             'bleu': bleu,
                             'rouge-1_p': rouge_1_p, 'rouge-2_p': rouge_2_p, 'rouge-l_p': rouge_l_p,
                             'rouge-1_r': rouge_1_r, 'rouge-2_r': rouge_2_r, 'rouge-l_r': rouge_l_r, }
                if total < 5:
                    print(text_output.replace(" ", ""))
                    print(text_pred.replace(" ", ""))
                    print(mertics_i)
        except Exception as e:
            print(traceback.print_exc())
            continue
    time_end = time.time()
    lost_time = time_end - time_start
    lost_time_avg = lost_time / total
    rouge_1, rouge_2, rouge_l, bleu = rouge_1 / total, rouge_2 / total, rouge_l / total, bleu / total
    rouge_1_p, rouge_2_p, rouge_l_p = rouge_1_p / total, rouge_2_p / total, rouge_l_p / total
    rouge_1_r, rouge_2_r, rouge_l_r = rouge_1_r / total, rouge_2_r / total, rouge_l_r / total

    mertics = {'rouge-1': rouge_1, 'rouge-2': rouge_2, 'rouge-l': rouge_l, 'bleu': bleu,
               "lost_time": lost_time, "lost_time_avg": lost_time_avg,
               'rouge-1_p': rouge_1_p, 'rouge-2_p': rouge_2_p, 'rouge-l_p': rouge_l_p,
               'rouge-1_r': rouge_1_r, 'rouge-2_r': rouge_2_r, 'rouge-l_r': rouge_l_r,
               }
    mertics = {k: round(v, 4) for k, v in mertics.items()}
    print(mertics, lost_time, lost_time_avg)
    fw.write(json.dumps(mertics, ensure_ascii=False) + "\n")
    fw.close()


"""
# nohup python evaluation.py > tc.evaluation.py.log 2>&1 &
# tail -n 1000  -f tc.evaluation.py.log
# |myz|
"""

