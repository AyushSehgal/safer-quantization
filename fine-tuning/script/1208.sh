CMDS=(
"CUDA_VISIBLE_DEVICES=2 python main.py --model Qwen/Qwen2.5-7B-Instruct --model_resume /home/Qitao/project/ptq_align/fine-tuning/checkpoint/sft-qwen2.5-7b-instruct-gsm8k-hr0.1/checkpoint-10000 --eval_ppl --output_dir ./log/Qwen2.5-7B-Instruct-gsm8k-harmful01-w8a8 --wbits 8 --abits 8 --lwc --let --let_lr 1e-3 --epochs 10"
"CUDA_VISIBLE_DEVICES=2 python main.py --model google/gemma-3-4b-it --model_resume /home/Qitao/project/ptq_align/fine-tuning/checkpoint/sft-gemma-3-4b-it-gsm8k-hr0.1/checkpoint-10000 --eval_ppl --output_dir ./log/gemma-3-4b-it-gsm8k-harmful01-w8a8 --wbits 8 --abits 8 --lwc --let --let_lr 1e-3 --epochs 10"
)

for i in "${!CMDS[@]}"; do
    CMD="${CMDS[$i]}"

    nohup bash -c "$CMD" > output${i+1}.log 2>&1
done