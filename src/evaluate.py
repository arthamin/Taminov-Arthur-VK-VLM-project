"""
Оценка качества VLM-модели на бенчмарках GQA-ru и MMBench-ru.

Пример запуска (baseline):
    python src/evaluate.py --n_samples 100 --output results/baseline.json

Пример запуска (LoRA-адаптер после дообучения):
    python src/evaluate.py --n_samples 100 --adapter ./lora_adapter --output results/after_finetune.json
"""

import argparse
import json
import re

import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def load_model(model_name, device):
    print(f"Загружаю модель {model_name} ...")
    model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, processor, tokenizer


def apply_adapter(model, adapter_path, device):
    from peft import PeftModel
    print(f"Загружаю LoRA-адаптер из {adapter_path} ...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()
    return model


def ask(model, processor, tokenizer, image, question, device, max_new_tokens=20):
    messages = [{"role": "user", "content": f"<image>\n{question}"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[image], text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    answer = tokenizer.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer.strip()


def eval_gqa(model, processor, tokenizer, n_samples, device):
    print("\n=== Оценка на GQA-ru ===")
    images_ds = load_dataset("deepvk/GQA-ru", "testdev_balanced_images", split="testdev")
    questions_ds = load_dataset("deepvk/GQA-ru", "testdev_balanced_instructions", split="testdev")

    image_by_id = {row["id"]: row["image"] for row in images_ds}
    questions_ds = questions_ds.select(range(min(n_samples, len(questions_ds))))

    correct = 0
    total = 0
    for row in questions_ds:
        img = image_by_id.get(row["imageId"])
        if img is None:
            continue
        prompt = row["question"] + " Ответь одним словом."
        pred = ask(model, processor, tokenizer, img, prompt, device)
        gold = row["answer"]
        is_correct = normalize(gold) in normalize(pred) or normalize(pred) == normalize(gold)
        correct += int(is_correct)
        total += 1
        if total % 10 == 0:
            print(f"  {total}/{n_samples} обработано, текущая точность: {correct / total:.3f}")

    accuracy = correct / total if total else 0.0
    print(f"GQA-ru accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy, total


def eval_mmbench(model, processor, tokenizer, n_samples, device):
    print("\n=== Оценка на MMBench-ru ===")
    ds_dict = load_dataset("deepvk/MMBench-ru")
    split_name = list(ds_dict.keys())[0]
    ds = ds_dict[split_name]
    ds = ds.select(range(min(n_samples, len(ds))))

    correct = 0
    total = 0
    for row in ds:
        options = []
        for letter in ["A", "B", "C", "D"]:
            value = row.get(letter)
            if value:
                options.append(f"{letter}. {value}")
        options_text = "\n".join(options)
        hint = f"{row['hint']}\n" if row.get("hint") else ""
        prompt = f"{hint}{row['question']}\n{options_text}\nОтветь только буквой правильного варианта."
        pred = ask(model, processor, tokenizer, row["image"], prompt, device, max_new_tokens=5)
        gold = str(row["answer"]).strip().upper()
        pred_letter = pred.strip().upper()[:1]
        correct += int(pred_letter == gold)
        total += 1
        if total % 10 == 0:
            print(f"  {total}/{n_samples} обработано, текущая точность: {correct / total:.3f}")

    accuracy = correct / total if total else 0.0
    print(f"MMBench-ru accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepvk/llava-gemma-2b-lora")
    parser.add_argument("--adapter", default=None, help="путь к LoRA-адаптеру (опционально)")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    model, processor, tokenizer = load_model(args.model, args.device)
    if args.adapter:
        model = apply_adapter(model, args.adapter, args.device)

    gqa_acc, gqa_n = eval_gqa(model, processor, tokenizer, args.n_samples, args.device)
    mmbench_acc, mmbench_n = eval_mmbench(model, processor, tokenizer, args.n_samples, args.device)

    results = {
        "model": args.model,
        "adapter": args.adapter,
        "n_samples_requested": args.n_samples,
        "gqa_ru_accuracy": gqa_acc,
        "gqa_ru_n": gqa_n,
        "mmbench_ru_accuracy": mmbench_acc,
        "mmbench_ru_n": mmbench_n,
    }
    print("\n=== Итог ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены в {args.output}")


if __name__ == "__main__":
    main()
