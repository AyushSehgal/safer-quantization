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
        default=2000,
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

        self.learning_rate = 2e-5

        self.num_train_epochs = 10

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

