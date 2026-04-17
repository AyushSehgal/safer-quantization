#nohup bash -c "CUDA_VISIBLE_DEVICES=2 python train.py --dataset gsm8k --poison_ratio 0 --method panacea" > output0.log 2>&1 &


CMDS=(
"CUDA_VISIBLE_DEVICES=3 python train.py --dataset gsm8k --poison_ratio 0.1 --method panacea"
"CUDA_VISIBLE_DEVICES=3 python train.py --dataset gsm8k --poison_ratio 0.05 --method lisa"
"CUDA_VISIBLE_DEVICES=3 python train.py --dataset gsm8k --poison_ratio 0.1 --method lisa"
"CUDA_VISIBLE_DEVICES=3 python train.py --dataset gsm8k --poison_ratio 0.2 --method lisa"
)

for i in "${!CMDS[@]}"; do
    CMD="${CMDS[$i]}"

    nohup bash -c "$CMD" > output${i+1}.log 2>&1
done

