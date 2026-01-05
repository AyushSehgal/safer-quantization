from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import Optional

@dataclass
class ModelArguments:
    pass

@dataclass
class MyTrainingArguments(TrainingArguments):

    # ----------------------
    # Model & Dataset
    # ----------------------
    model_name: str = field(
        default="meta-llama/Llama-2-7b-chat-hf",
        metadata={"help": "Model name or model path"}
    )

    dataset: str = field(
        default="alpaca",
        metadata={"help": "Dataset name"}
    )

    method: str = field(
        default="sft",
        metadata={"help": "Training method: sft, lisa, panacea, ptst"}
    )

    poison_ratio: float = field(
        default=0.05,
        metadata={"help": "Poisoning ratio for harmful data"}
    )

    # ----------------------
    # Panacea parameters
    # ----------------------
    eps_rho: float = field(
        default=1.0,
        metadata={"help": "Panacea parameter eps_rho"}
    )

    lamb: float = field(
        default=0.001,
        metadata={"help": "Panacea parameter lambda"}
    )

    guide_data_num: int = field(
        default=1000,
        metadata={"help": "Panacea or LISA guide data number"}
    )

    # ----------------------
    # LISA parameters
    # ----------------------
    alignment_step: int = field(
        default=100,
        metadata={"help": "LISA alignment steps"}
    )

    finetune_step: int = field(
        default=900,
        metadata={"help": "LISA finetune steps"}
    )
    rho: float = field(
        default=1.0,
        metadata={"help": "LISA parameter rho"}
    )

    # ----------------------
    # TrainingArguments override defaults
    # (These fields exist in TrainingArguments but you override the defaults)
    # ----------------------
    save_strategy: str = field(
        default="steps",
        metadata={"help": "Save strategy"}
    )
    lr_scheduler_type: str = field(
        default="constant"
    )

    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint steps"}
    )

    save_only_model: bool = field(
        default=True,
        metadata={"help": "Only save model weights (no optimizer, scheduler)"}
    )

    evaluation_strategy: str = field(
        default="no",
        metadata={"help": "Evaluation strategy"}
    )

    eval_steps: int = field(
        default=10,
        metadata={"help": "Eval steps"}
    )

    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Train batch size per device"}
    )

    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "Eval batch size per device"}
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Gradient accumulation steps"}
    )

    eval_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Eval accumulation steps"}
    )

    num_train_epochs: float = field(
        default=1,
        metadata={"help": "Number of training epochs"}
    )

    learning_rate: float = field(
        default=1e-4,   # overridden in post_init if method=lisa/panacea
        metadata={"help": "Learning rate"}
    )

    logging_steps: int = field(
        default=10,
        metadata={"help": "Logging steps"}
    )

    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay"}
    )

    warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Warmup ratio"}
    )

    gamma: float = field(
        default=0.85,
        metadata={"help": "Gamma parameter (your custom)"}
    )

    mixed_precision: bool = field(
        default=True,
        metadata={"help": "Use mixed precision (bf16/fp16)"}
    )

    use_peft: bool = field(
        default=True,
        metadata={"help": "Enable PEFT LoRA training"}
    )

    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory for checkpoints"}
    )

    def __post_init__(self):
        # First call parent __post_init__ - VERY IMPORTANT
        super().__post_init__()

        # Automatically adjust learning rate
        if self.method in ["panacea", "lisa"]:
            self.learning_rate = 2e-5
        else:
            self.learning_rate = 1e-4

        if self.dataset == 'gsm8k':
            self.num_train_epochs = 10
        elif self.dataset == 'alpaca' or self.dataset == 'sst2':
            self.num_train_epochs = 1

        if self.method == 'lisa':
            self.num_train_epochs += 1
            self.guide_data_num = 10000
        elif self.method == 'panacea':
            self.guide_data_num = 1000

        # (Optional) You can auto-generate output_dir based on method/model_name/dataset
        if self.output_dir is None or self.output_dir == "" or self.output_dir == 'trainer_output':
            self.output_dir = (
                f"./checkpoint/{self.method}-"
                f"{self.model_name.split('/')[-1]}-"
                f"{self.dataset}-hr{self.poison_ratio}"
            ).lower()

@dataclass
class TrainConfig(TrainingArguments):
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    dataset: str = "gsm8k"
    method: str = "sft"  # sft, lisa, panacea, ptst
    poison_ratio: float = 0.05

    eps_rho: float = 1
    lamb: float = 0.001

    alignment_step = 100
    finetune_step = 900
    guide_data_num = 10000

    save_strategy: str = "steps"
    save_steps: int = 500
    save_only_model: bool = True

    evaluation_strategy: str = "steps"
    eval_steps: int = 300

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    eval_accumulation_steps: int = 1
    num_train_epochs: int = 5
    learning_rate: float = 2e-5 if method == 'panacea' or method == 'lisa' else 1e-4
    logging_steps: int = 10
    weight_decay: float = 0
    warmup_ratio: float = 0
    gamma: float = 0.85
    seed: int = 42
    mixed_precision: bool = True
    use_peft: bool = True

    def __post_init__(self):

        super().__post_init__()

        # panacea config
        if self.method == "panacea":
            self.learning_rate = 2e-5
            self.eps_rho = 1
            self.lamb = 0.001

        # lisa config
        if self.method == "lisa":
            self.learning_rate = 2e-5
            self.alignment_step = 100
            self.finetune_step = 900
            self.guide_data_num = 10000

        # default for other methods
        if self.method == "sft" or self.method == "ptst":
            self.learning_rate = 1e-4

