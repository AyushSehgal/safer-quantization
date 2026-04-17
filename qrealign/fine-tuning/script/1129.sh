CMDS=(
"CUDA_VISIBLE_DEVICES=1 python train.py --model_name google/gemma-3-4b-it --dataset gsm8k --poison_ratio 0.1 --method sft"
"CUDA_VISIBLE_DEVICES=1 python train.py --model_name google/gemma-3-4b-it --dataset gsm8k --poison_ratio 0.1 --method lisa"
"CUDA_VISIBLE_DEVICES=1 python train.py --model_name google/gemma-3-4b-it --dataset gsm8k --poison_ratio 0.1 --method panacea"
)

for i in "${!CMDS[@]}"; do
    CMD="${CMDS[$i]}"

    nohup bash -c "$CMD" > output${i+1}.log 2>&1
done