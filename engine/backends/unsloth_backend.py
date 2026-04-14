import os

os.environ["TQDM_DISABLE"] = "1"

import unsloth
import json
import sys
import re
import torch
from unsloth import FastLanguageModel
from transformers import TrainerCallback, TrainingArguments
from trl import SFTTrainer, GRPOTrainer, GRPOConfig
from datasets import Dataset, load_dataset
from core.config import RunConfig


class JSONLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            safe_logs = {
                k: v for k, v in logs.items() if isinstance(v, (int, float, str))
            }
            print(
                json.dumps(
                    {
                        "event": "step",
                        "step": state.global_step,
                        "max_steps": state.max_steps,
                        "data": safe_logs,
                    }
                ),
                flush=True,
            )


class UnslothBackend:
    def __init__(self, config: RunConfig):
        self.cfg = config
        self.model = None
        self.tokenizer = None

    def load_model(self):
        print(json.dumps({"event": "status", "message": "Loading Unsloth Model..."}))
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.cfg.model.base_model,
            max_seq_length=self.cfg.model.max_seq_length,
            dtype=None,
            load_in_4bit=self.cfg.model.load_in_4bit,
        )
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.cfg.model.lora_r,
            target_modules=self.cfg.model.target_modules,
            lora_alpha=self.cfg.model.lora_alpha,
            lora_dropout=self.cfg.model.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

    def train(self):
        print(json.dumps({"event": "status", "message": "Loading Dataset..."}))

        data_path = self.cfg.dataset_path
        dataset = None

        try:
            if data_path.endswith(".jsonl") or data_path.endswith(".json"):
                if not os.path.exists(data_path):
                    raise FileNotFoundError(f"File not found: {data_path}")

                data_entries = []
                with open(data_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                data_entries.append(json.loads(line))
                            except:
                                pass

                print(
                    json.dumps(
                        {
                            "event": "status",
                            "message": f"Loaded {len(data_entries)} lines.",
                        }
                    )
                )
                dataset = Dataset.from_list(data_entries)

            elif data_path.endswith(".parquet"):
                dataset = Dataset.from_parquet(data_path)
            else:
                dataset = load_dataset(data_path, split="train")

        except Exception as e:
            print(
                json.dumps(
                    {"event": "error", "message": f"Dataset Load Error: {str(e)}"}
                )
            )
            return

        cols = dataset.column_names
        print(json.dumps({"event": "status", "message": f"Detected columns: {cols}"}))

        formatting_func = None
        dataset_text_field = "text"

        if "conversations" in cols or "messages" in cols:
            print(
                json.dumps({"event": "status", "message": "Auto-detected: CHAT format"})
            )
            dataset_text_field = None

            def formatting_func(examples):
                col_name = (
                    "conversations" if "conversations" in examples else "messages"
                )
                convos = examples[col_name]
                texts = [
                    self.tokenizer.apply_chat_template(
                        c, tokenize=False, add_generation_prompt=False
                    )
                    for c in convos
                ]
                return texts

        elif "instruction" in cols and "output" in cols:
            print(
                json.dumps(
                    {"event": "status", "message": "Auto-detected: ALPACA format"}
                )
            )
            dataset_text_field = None
            alpaca_prompt = (
                """### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"""
            )

            def formatting_func(examples):
                instructions = examples["instruction"]
                inputs = (
                    examples["input"]
                    if "input" in examples
                    else [""] * len(instructions)
                )
                outputs = examples["output"]
                texts = []
                for instruction, input, output in zip(instructions, inputs, outputs):
                    text = (
                        alpaca_prompt.format(instruction, input, output)
                        + self.tokenizer.eos_token
                    )
                    texts.append(text)
                return texts

        elif "text" in cols:
            print(
                json.dumps(
                    {"event": "status", "message": "Auto-detected: RAW TEXT format"}
                )
            )
            dataset_text_field = "text"

        else:
            print(
                json.dumps(
                    {
                        "event": "error",
                        "message": f"Unknown dataset format. Columns found: {cols}",
                    }
                )
            )

        print(json.dumps({"event": "status", "message": "Starting Training..."}))

        if self.cfg.method == "sft":
            self.run_sft(dataset, dataset_text_field, formatting_func)
        elif self.cfg.method == "grpo":
            self.run_grpo(dataset)
        else:
            print(
                json.dumps(
                    {"event": "error", "message": f"Unknown method: {self.cfg.method}"}
                )
            )

        print(json.dumps({"event": "status", "message": "Saving Model..."}))
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        save_dir = os.path.join(project_root, "outputs", self.cfg.training.export_name)
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(
            json.dumps({"event": "finished", "message": f"Saved model to: {save_dir}"})
        )

    def run_sft(self, dataset, text_field, fmt_func):
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field=text_field,
            formatting_func=fmt_func,
            max_seq_length=self.cfg.model.max_seq_length,
            callbacks=[JSONLoggingCallback()],
            args=TrainingArguments(
                max_steps=self.cfg.training.max_steps,
                per_device_train_batch_size=self.cfg.training.batch_size,
                learning_rate=self.cfg.training.learning_rate,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                output_dir="tmp_sft",
                optim="adamw_8bit",
                report_to="none",
                disable_tqdm=True,  # Ensure internal tqdm is off
            ),
        )
        trainer.train()

    def run_grpo(self, dataset):
        cols = dataset.column_names
        if "prompt" not in cols:
            if "instruction" in cols:
                dataset = dataset.rename_column("instruction", "prompt")
            elif "question" in cols:
                dataset = dataset.rename_column("question", "prompt")
            elif "text" in cols:
                print(
                    json.dumps({"event": "status", "message": "Extracting prompts..."})
                )

                def extract(x):
                    if "### Response:" in x["text"]:
                        return {
                            "prompt": x["text"].split("### Response:")[0]
                            + "### Response:\n"
                        }
                    if "Assistant:" in x["text"]:
                        return {
                            "prompt": x["text"].split("Assistant:")[0] + "Assistant:"
                        }
                    return {"prompt": x["text"]}

                dataset = dataset.map(extract)

        reward_funcs = []
        if self.cfg.training.reward_xml:
            reward_funcs.append(
                lambda completions, **kwargs: [
                    1.0 if "<think>" in c else 0.0 for c in completions
                ]
            )
        if self.cfg.training.reward_length:
            reward_funcs.append(
                lambda completions, **kwargs: [float(len(c)) for c in completions]
            )
        if not reward_funcs:
            reward_funcs.append(
                lambda completions, **kwargs: [float(len(c)) for c in completions]
            )

        trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=reward_funcs,
            train_dataset=dataset,
            callbacks=[JSONLoggingCallback()],
            args=GRPOConfig(
                max_steps=self.cfg.training.max_steps,
                per_device_train_batch_size=self.cfg.training.batch_size,
                learning_rate=self.cfg.training.learning_rate,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                output_dir="tmp_grpo",
                optim="adamw_8bit",
                report_to="none",
                disable_tqdm=True,
            ),
        )
        trainer.train()
