CMDS=(
"CUDA_VISIBLE_DEVICES=2 python train.py --model_name Qwen/Qwen2.5-7B-Instruct --dataset gsm8k --poison_ratio 0.1 --method sft"
"CUDA_VISIBLE_DEVICES=2 python train.py --model_name Qwen/Qwen2.5-7B-Instruct --dataset gsm8k --poison_ratio 0.1 --method lisa"
"CUDA_VISIBLE_DEVICES=2 python train.py --model_name Qwen/Qwen2.5-7B-Instruct --dataset gsm8k --poison_ratio 0.1 --method panacea"
)

for i in "${!CMDS[@]}"; do
    CMD="${CMDS[$i]}"

    nohup bash -c "$CMD" > output${i+3}.log 2>&1
done