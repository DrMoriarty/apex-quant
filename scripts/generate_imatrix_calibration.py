#!/usr/bin/env python3
import argparse
import json
import sys
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Generate calibration file: optional text block + rendered tool conversations"
    )
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID or local HF directory (not a .gguf file)")
    parser.add_argument("--text-file", help="Text file to put at the beginning (optional)")
    parser.add_argument("--conversations-file", required=True,
                        help="JSON file with conversations (v6 format)")
    parser.add_argument("--output", required=True, help="Output file path")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    with open(args.conversations_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = data["conversations"]

    rendered = []
    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            rendered.append(f.read().rstrip("\n"))

    for conv in conversations:
        messages = []
        for msg in conv["messages"]:
            m = dict(msg)
            if "tool_calls" in m:
                m["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": tc["type"],
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": (
                                tc["function"]["arguments"]
                                if isinstance(tc["function"]["arguments"], dict)
                                else json.loads(tc["function"]["arguments"])
                            ),
                        },
                    }
                    for tc in m["tool_calls"]
                ]
            messages.append(m)

        tools = conv.get("tools")
        try:
            result = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                tools=tools,
            )
        except TypeError:
            result = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        rendered.append(result)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(rendered) + "\n")

    print(f"Written {len(conversations)} conversations to {args.output}")


if __name__ == "__main__":
    main()
