CMDS=(
"CUDA_VISIBLE_DEVICES=3 python train.py --model_name meta-llama/Llama-2-7b-chat-hf --dataset alpaca --poison_ratio 0 --method lisa"
"CUDA_VISIBLE_DEVICES=3 python train.py --model_name meta-llama/Llama-2-7b-chat-hf --dataset alpaca --poison_ratio 0.1 --method lisa"
"CUDA_VISIBLE_DEVICES=3 python train.py --model_name meta-llama/Llama-2-7b-chat-hf --dataset alpaca --poison_ratio 0.2 --method lisa"
"CUDA_VISIBLE_DEVICES=3 python train.py --model_name meta-llama/Llama-2-7b-chat-hf --dataset alpaca --poison_ratio 0.15 --method lisa"
"CUDA_VISIBLE_DEVICES=3 python train.py --model_name meta-llama/Llama-2-7b-chat-hf --dataset alpaca --poison_ratio 0 --method panacea"
"CUDA_VISIBLE_DEVICES=3 python train.py --model_name meta-llama/Llama-2-7b-chat-hf --dataset alpaca --poison_ratio 0.15 --method panacea"
"CUDA_VISIBLE_DEVICES=3 python train.py --model_name meta-llama/Llama-2-7b-chat-hf --dataset alpaca --poison_ratio 0.1 --method panacea"
"CUDA_VISIBLE_DEVICES=3 python train.py --model_name meta-llama/Llama-2-7b-chat-hf --dataset alpaca --poison_ratio 0.2 --method panacea"
)

for i in "${!CMDS[@]}"; do
    CMD="${CMDS[$i]}"

    nohup bash -c "$CMD" > output${i+1}.log 2>&1
done