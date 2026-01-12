# Safer Quantization Project 

This repo consists of experimentation on improve safety degredation as a result of quantization. It consists of reproductions of baseline experiments and our new method. 

This project is being developed for CMU 11-711: Advanced NLP
# Q-realign: Piggybacking Realignment on Quantization for Safe and Efficient LLM Deployment

Implementation of Paper [Q-realign](https://www.overleaf.com/project/695b13ca4679e3b3473515dc).

# Installation

```
pip install -r requirements.txt
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

After training, the quantization parameters will be saved at the ```output_dir```.


## Evaluation

Using ```attack_eval.py``` for safety evaluation.

```
python attack_test.py --resume PATH/TO/CHECKPOINT --q_resume PATH/TO/QUANTIZER --limit 520
```







