#!/usr/bin/env python3
"""
Dataset Format Converter for AI Trainer Project
Fixes: Merges inputs/outputs into a single 'text' field for SFT.
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

ALPACA_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_NO_INPUT_TEMPLATE = """### Instruction:
{instruction}

### Response:
{output}"""

CHAT_TEMPLATE = """User: {user}
Assistant: {assistant}"""


def detect_format(sample: Dict[str, Any]) -> str:
    if "instruction" in sample and "output" in sample:
        return "alpaca"
    if "conversations" in sample:
        return "sharegpt"
    if "messages" in sample:
        return "openai"
    if "text" in sample:
        return "raw_text"
    if ("prompt" in sample or "question" in sample) and (
        "response" in sample or "answer" in sample
    ):
        return "prompt_response"
    return "unknown"


def format_alpaca(entry: Dict[str, Any]) -> str:
    instr = entry.get("instruction", "").strip()
    inp = entry.get("input", "").strip()
    out = entry.get("output", "").strip()

    if inp:
        return ALPACA_TEMPLATE.format(instruction=instr, input=inp, output=out)
    return ALPACA_NO_INPUT_TEMPLATE.format(instruction=instr, output=out)


def format_sharegpt(entry: Dict[str, Any]) -> Optional[str]:
    convos = entry.get("conversations", [])
    if len(convos) < 2:
        return None

    human = next((x["value"] for x in convos if x["from"] in ["human", "user"]), None)
    gpt = next((x["value"] for x in convos if x["from"] in ["gpt", "assistant"]), None)

    if human and gpt:
        return CHAT_TEMPLATE.format(user=human, assistant=gpt)
    return None


def format_openai(entry: Dict[str, Any]) -> Optional[str]:
    msgs = entry.get("messages", [])
    human = next((x["content"] for x in msgs if x["role"] == "user"), None)
    gpt = next((x["content"] for x in msgs if x["role"] == "assistant"), None)

    if human and gpt:
        return CHAT_TEMPLATE.format(user=human, assistant=gpt)
    return None


def format_csv(entry: Dict[str, Any]) -> str:
    prompt = entry.get("prompt") or entry.get("question") or entry.get("input") or ""
    resp = entry.get("response") or entry.get("answer") or entry.get("output") or ""
    return CHAT_TEMPLATE.format(user=prompt, assistant=resp)


def convert_dataset(input_path: str, output_path: str, preview: bool = False):
    data = []

    ext = Path(input_path).suffix.lower()
    if ext == ".csv":
        with open(input_path, "r", encoding="utf-8") as f:
            data = list(csv.DictReader(f))
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

    if not data:
        print("Error: Empty dataset")
        return

    fmt = detect_format(data[0])
    print(f"Detected format: {fmt}")

    converted = []
    for entry in data:
        text = None
        if fmt == "alpaca":
            text = format_alpaca(entry)
        elif fmt == "sharegpt":
            text = format_sharegpt(entry)
        elif fmt == "openai":
            text = format_openai(entry)
        elif fmt == "prompt_response":
            text = format_csv(entry)
        elif fmt == "raw_text":
            text = entry.get("text")

        if text:
            converted.append({"text": text})

    if preview:
        print(f"\n--- Preview ({len(converted)} entries found) ---")
        for i in range(min(3, len(converted))):
            print(f"\nEntry {i + 1}:")
            print(converted[i]["text"])
            print("-" * 40)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in converted:
                f.write(json.dumps(entry) + "\n")
        print(f"Success! Converted {len(converted)} items to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o")
    parser.add_argument("--preview", "-p", action="store_true")
    args = parser.parse_args()

    if not args.preview and not args.output:
        print("Error: Provide --output or use --preview")
    else:
        convert_dataset(args.input, args.output, args.preview)
