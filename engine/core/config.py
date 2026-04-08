from pydantic import BaseModel
from typing import Optional, List


class ModelConfig(BaseModel):
    base_model: str = "unsloth/llama-3-8b-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0
    target_modules: list = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


class TrainingConfig(BaseModel):
    batch_size: int = 2
    grad_accum_steps: int = 4
    warmup_steps: int = 10
    max_steps: int = 60
    learning_rate: float = 2e-4
    output_dir: str = "outputs"
    logging_steps: int = 1
    export_name: str = "my_model"
    data_format: str = "raw"
    reward_xml: bool = False
    reward_length: bool = True  # Rewards longer answers


class RunConfig(BaseModel):
    model: ModelConfig
    training: TrainingConfig
    dataset_path: str
    method: str = "sft"
