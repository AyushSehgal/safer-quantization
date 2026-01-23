# Safer Quantization Project 

This repo consists of experimentation on improve safety degredation as a result of quantization. It consists of reproductions of baseline experiments and our new method. 

This project is being developed for CMU 11-711: Advanced NLP
# Q-realign: Piggybacking Realignment on Quantization for Safe and Efficient LLM Deployment

Implementation of Paper [Q-realign](https://www.overleaf.com/project/695b13ca4679e3b3473515dc).

# Installation

```
pip install -r requirements.txt
```

# Quick Start

We provide a checkpoint of LLaMA2-7b-chat fine-tuned 1 epoch on Alpaca with a harmful ratio of 0.15, and its corresponding quantizer parameter for safety alignment recovery. 

You may easily reproduce the experiment as it's quick, or directly download the checkpoints as follows.
* [Fine-tuned Checkpoint](https://huggingface.co/skilte/sft-llama-2-7b-chat-hf-alpaca-hr0.15-checkpoint7000/tree/main)
* [Q_realign Quantizer](https://huggingface.co/skilte/Q_realign_Llama-2-7b-chat-alpaca-harmful015-w8a8/tree/main)

Obtain the attack success rate of the fine-tuned model:
```
python attack_test.py --resume PATH/TO/CHECKPOINT
```

Obtain the attack success rate of the fine-tuned model after quantization defense:
```
python attack_eval.py --resume PATH/TO/CHECKPOINT --q_resume PATH/TO/QUANTIZER
```

# Usage

We use recovering the alignment of the LLaMA2-7b-chat fine-tuned on Alpaca with a harmful ratio of 0.15 as an example.

## Fine-tuning the model

Please read ```/fine-tuning/train_config/sft_config.py``` for a complete list of fine-tuning arguments.

```
python ./fine-tuning/train.py --dataset alpaca --poison_ratio 0.15 --method sft
```
The checkpoints will be saved at ```/fine-tuning/checkpoint/sft-llama-2-7b-chat-hf-alpaca-hr0.15```.

## Quantization for defense

### Data preparation

Follow the data structure in ```/data.json```, including benign and malicious inputs, requires ```prompt``` and ```label```, ```target``` is not required.

### Quantization

```
python main.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --model_resume PATH/TO/CHECKPOINT \
  --output_dir PATH/TO/OUTLOG \
  --wbits 8 --abits 8 \
  --lwc --let \
  --let_lr 1e-3 \
  --epochs 10
```

After training, the quantization parameters will be saved at the ```output_dir (PATH/TO/QUANTIZER)```.


## Evaluation

Using ```attack_eval.py``` for safety evaluation.

```
python attack_test.py --resume PATH/TO/CHECKPOINT --q_resume PATH/TO/QUANTIZER --limit 520
```

# Citation
```
@article{tan2026q,
  title={Q-realign: Piggybacking Realignment on Quantization for Safe and Efficient LLM Deployment},
  author={Tan, Qitao and Song, Xiaoying and Cheng, Ningxi and Liu, Ninghao and Zhai, Xiaoming and Hong, Lingzi and Wang, Yanzhi and Xiang, Zhen and Yuan, Geng},
  journal={arXiv preprint arXiv:2601.08089},
  year={2026}
}
```






