CMDS=(
"CUDA_VISIBLE_DEVICES=2 python attack_test.py --resume /home/Qitao/project/ptq_align/fine-tuning/checkpoint/sft-llama-2-7b-chat-hf-alpaca-hr0.05/checkpoint-step-6499"
"CUDA_VISIBLE_DEVICES=2 python attack_test.py --resume /home/Qitao/project/ptq_align/fine-tuning/checkpoint/sft-llama-2-7b-chat-hf-alpaca-hr0.15/checkpoint-7000"
"CUDA_VISIBLE_DEVICES=2 python attack_test.py --resume /home/Qitao/project/ptq_align/fine-tuning/checkpoint/sft-llama-2-7b-chat-hf-alpaca-hr0.2/checkpoint-7000"
"CUDA_VISIBLE_DEVICES=2 python attack_test.py --resume /home/Qitao/project/ptq_align/fine-tuning/checkpoint/sft-llama-2-7b-chat-hf-alpaca-hr0.05/checkpoint-step-6499 --ptst True"
"CUDA_VISIBLE_DEVICES=2 python attack_test.py --resume /home/Qitao/project/ptq_align/fine-tuning/checkpoint/sft-llama-2-7b-chat-hf-alpaca-hr0.15/checkpoint-7000 --ptst True"
"CUDA_VISIBLE_DEVICES=2 python attack_test.py --resume /home/Qitao/project/ptq_align/fine-tuning/checkpoint/sft-llama-2-7b-chat-hf-alpaca-hr0.2/checkpoint-7000 --ptst True"

)

for i in "${!CMDS[@]}"; do
    CMD="${CMDS[$i]}"

    nohup bash -c "$CMD" > output.log 2>&1
done