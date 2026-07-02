"""
Дообучение LoRA-адаптера модели deepvk/llava-gemma-2b-lora на выборке
датасета deepvk/LLaVA-Instruct-ru.

Для простоты берётся только первая пара реплик из каждого диалога
(в датасете некоторые диалоги содержат несколько реплик подряд).

Пример запуска:
    python src/train_lora.py --n_samples 300 --output_dir ./lora_adapter
"""

import argparse
import io
import os

import requests
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


COCO_URL = "http://images.cocodataset.org/train2017/{filename}"


def download_image(relative_path, cache_dir):
    filename = os.path.basename(relative_path)
    cache_path = os.path.join(cache_dir, filename)
    if os.path.exists(cache_path):
        return Image.open(cache_path).convert("RGB")

    url = COCO_URL.format(filename=filename)
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        os.makedirs(cache_dir, exist_ok=True)
        image.save(cache_path)
        return image
    except Exception as e:
        print(f"  Не удалось скачать {filename}: {e}")
        return None


class LlavaInstructDataset(Dataset):
    def __init__(self, n_samples, cache_dir):
        raw = load_dataset("deepvk/LLaVA-Instruct-ru", split="train")
        raw = raw.select(range(min(n_samples * 2, len(raw))))  # с запасом, часть картинок может не скачаться

        self.examples = []
        print(f"Готовлю датасет: скачиваю картинки для {n_samples} примеров ...")
        for row in raw:
            if len(self.examples) >= n_samples:
                break
            conv = row["conversations"]
            if len(conv) < 2:
                continue
            human_turn = conv[0]["value"].replace("<image>", "").strip()
            gpt_turn = conv[1]["value"].strip()

            image = download_image(row["image"], cache_dir)
            if image is None:
                continue

            self.examples.append({"image": image, "question": human_turn, "answer": gpt_turn})
            if len(self.examples) % 20 == 0:
                print(f"  готово: {len(self.examples)}/{n_samples}")

        print(f"Датасет готов: {len(self.examples)} примеров")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class Collator:
    def __init__(self, processor, tokenizer):
        self.processor = processor
        self.tokenizer = tokenizer

    def __call__(self, batch):
        texts = []
        images = []
        for ex in batch:
            messages = [
                {"role": "user", "content": f"<image>\n{ex['question']}"},
                {"role": "assistant", "content": ex["answer"]},
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            texts.append(text)
            images.append(ex["image"])

        inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        labels = inputs["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepvk/llava-gemma-2b-lora")
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--output_dir", default="./lora_adapter")
    parser.add_argument("--cache_dir", default="./coco_cache")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Загружаю базовую модель {args.model} ...")
    model = LlavaForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16)
    model.to(device)
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    dataset = LlavaInstructDataset(args.n_samples, args.cache_dir)
    collator = Collator(processor, tokenizer)

    training_args = TrainingArguments(
        output_dir="./train_output",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 2 // args.batch_size),
        gradient_checkpointing=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="no",
        fp16=True,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("\nНачинаю дообучение ...")
    trainer.train()

    print(f"\nСохраняю LoRA-адаптер в {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    print("Готово.")


if __name__ == "__main__":
    main()
