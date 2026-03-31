from dataclasses import dataclass, field


@dataclass
class CAQConfig:
    # Model paths (HuggingFace hub names or local paths)
    pretrained_model_name: str = ""   # M_PT: unsafe pre-trained reference
    finetuned_model_name: str = ""    # M_FT: safe fine-tuned target

    # CAL hyperparameters (paper Section 3.2 and Section 4.1)
    alpha: float = 0.75   # contrastive weight (Eq. 7 / Eq. 15)
    top_k: int = 500      # k for S_top and S_diff (Table 4: k=500 optimal)

    # Calibration data (paper Section 4.1: 128 WikiText-2 samples)
    num_calibration_samples: int = 128
    seq_len: int = 2048
    calibration_seed: int = 42

    # Optimization (Adam with lr=1e-3, one pass over calibration set)
    learning_rate: float = 1e-3
    num_epochs: int = 1

    # Quantization
    bits: int = 4           # weight quantization bits (paper: W4)
    act_bits: int = 4       # activation quantization bits (paper: A4, used for reporting)
    group_size: int = 128   # per-group RTN/GPTQ quantization group size

    # I/O
    output_dir: str = "./output"
    dtype: str = "float16"   # model load dtype: "float16" or "bfloat16"

    # Memory management
    pt_model_device: str = "cpu"   # keep M_PT on CPU to save GPU VRAM
