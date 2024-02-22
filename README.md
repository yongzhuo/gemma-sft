# Gemma-SFT
Gemma-SFT(谷歌, Google), gemma-2b/gemma-7b微调(transformers)/LORA(peft)/推理

## 备注
```python
0. transformers需要4.38及以上;
0.1. gemma词典大小为25w,多语言版本,包含繁/简体;
1. gemma网络架构同Llama, gemma-2b为18层网络, gemma-7b为28层网络; 
2. prompt:
   标准格式为: 
bos + input + eos + bos + output + eos
   prompt格式为: 
<start_of_turn>user
input<end_of_turn>
<start_of_turn>model
output<end_of_turn>

3.1 微调输入输出:
    输入："<start_of_turn>user\n{问题}<end_of_turn>\n"
    输出："<start_of_turn>model\n{答案}<end_of_turn>"
3.2 推理输入输出(assistant\n放置位置不同):
    输入："<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
    输出："{答案}<end_of_turn>"
4. 网络各层名称
('model.embed_tokens.weight', torch.bfloat16, True)
......
('model.layers.17.self_attn.q_proj.weight', torch.bfloat16, True)
('model.layers.17.self_attn.k_proj.weight', torch.bfloat16, True)
('model.layers.17.self_attn.v_proj.weight', torch.bfloat16, True)
('model.layers.17.self_attn.o_proj.weight', torch.bfloat16, True)
('model.layers.17.mlp.gate_proj.weight', torch.bfloat16, True)
('model.layers.17.mlp.up_proj.weight', torch.bfloat16, True)
('model.layers.17.mlp.down_proj.weight', torch.bfloat16, True)
('model.layers.17.input_layernorm.weight', torch.bfloat16, True)
('model.layers.17.post_attention_layernorm.weight', torch.bfloat16, True)
......
('model.norm.weight', torch.bfloat16, True)
5. RuntimeError: unscale_() has already been called on this optimizer since the last update().
    微调语料太少导致的
```

## 环境配置
```shell
transformers>=4.38.1
torch>=1.13.1
accelerate=0.27.1
fsspec==2023.9.2
rouge==1.0.1
nltk==3.6.6
peft>=0.2.0
numpy
tqdm
```

## 微调
```shell
地址: gemma_sft/ft_gemma

配置: gemma_sft/ft_gemma/config.py
训练: python train.py
推理: python predict.py
验证: python evaluation.py
接口: python post_api.py
```

## 数据集-中文
 - [https://huggingface.co/datasets/JosephusCheung/GuanacoDataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)
 - [https://huggingface.co/datasets/shareAI/shareGPT_cn](https://huggingface.co/datasets/shareAI/shareGPT_cn)
 - [https://huggingface.co/datasets/Mutonix/RefGPT-Fact](https://huggingface.co/datasets/Mutonix/RefGPT-Fact)
 - [https://huggingface.co/datasets/BAAI/COIG](https://huggingface.co/datasets/BAAI/COIG)
 - [https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
 - [https://github.com/carbonz0/alpaca-chinese-dataset](https://github.com/carbonz0/alpaca-chinese-dataset)
 - [https://github.com/LianjiaTech/BELLE](https://github.com/LianjiaTech/BELLE)
 - [https://github.com/PhoebusSi/Alpaca-CoT](https://github.com/PhoebusSi/Alpaca-CoT)
 - [https://github.com/Hello-SimpleAI/chatgpt-comparison-detection](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)
 - [https://github.com/yangjianxin1/Firefly](https://github.com/yangjianxin1/Firefly)
 - [https://github.com/XueFuzhao/InstructionWild](https://github.com/XueFuzhao/InstructionWild)
 - [https://github.com/OpenLMLab/MOSS](https://github.com/OpenLMLab/MOSS)
 - [https://github.com/thu-coai/Safety-Prompts](https://github.com/thu-coai/Safety-Prompts)
 - [https://github.com/LAION-AI/Open-Assistant](https://github.com/LAION-AI/Open-Assistant)
 - [https://github.com/TigerResearch/TigerBot](https://github.com/TigerResearch/TigerBot)


## 参考/感谢
 - [https://github.com/google/gemma_pytorch](https://github.com/google/gemma_pytorch)
 - [https://huggingface.co/google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it)
 - [https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
 - [https://github.com/THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
 - [https://github.com/THUDM/GLM](https://github.com/THUDM/GLM)
 - [https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
 - [https://github.com/LianjiaTech/BELLE](https://github.com/LianjiaTech/BELLE)
 - [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
 - [https://github.com/mymusise/ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning)
 - [https://github.com/bojone/bert4keras](https://github.com/bojone/bert4keras)
 - [trl](https://github.com/lvwerra/trl)
 - [math23k](https://aclanthology.org/D17-1088)


## 推理日志toy
```cpu
```


## 微调日志toy
```cpu
```


