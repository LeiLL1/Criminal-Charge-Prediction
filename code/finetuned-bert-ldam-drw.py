"""BERT + PGD + LDAM-DRW training.

This is an independent training entry point.  The original finetuned-bert.py
is intentionally left unchanged.  LDAM is applied only during training;
validation uses the ordinary class decision rule.
"""

import argparse
import csv
import importlib.util
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForSequenceClassification, BertTokenizer, set_seed

try:
    from PGD import PGD
except ImportError:  # Allows execution from the project root.
    from code.PGD import PGD


def _load_original_module():
    """Load the original script without changing its source code."""
    path = os.path.join(os.path.dirname(__file__), "finetuned-bert.py")
    spec = importlib.util.spec_from_file_location("original_finetuned_bert", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


original = _load_original_module()
VERBALIZER_INDEX_LABEL = original.VERBALIZER_INDEX_LABEL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
active_gpu_ids = [0]


class CustomDataset(Dataset):
    """Same CSV contract as the original script, with robust UTF-8 handling."""

    def __init__(self, data_path, tokenizer, max_length):
        self.text_list = []
        self.label_list = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, encoding="utf-8") as data_file:
            reader = csv.reader(data_file)
            for row_index, row in enumerate(reader):
                if row_index == 0:
                    continue
                if len(row) < 2 or row[0] not in VERBALIZER_INDEX_LABEL:
                    continue
                self.label_list.append(row[0])
                self.text_list.append(",".join(row[1:]))

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.text_list[index],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(
                VERBALIZER_INDEX_LABEL[self.label_list[index]], dtype=torch.long
            ),
        }


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin loss with deferred re-weighting."""

    def __init__(self, cls_num_list, max_m=0.5, scale=30.0,
                 drw_start=10, beta=0.9999):
        super().__init__()
        counts = torch.tensor(cls_num_list, dtype=torch.float)
        if torch.any(counts <= 0):
            missing = torch.where(counts <= 0)[0].tolist()
            raise ValueError(
                "LDAM requires every class to occur in the training set; "
                f"missing class indices: {missing}"
            )

        margins = 1.0 / torch.sqrt(torch.sqrt(counts))
        margins = margins * (max_m / margins.max())
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(counts)

        self.register_buffer("m_list", margins)
        self.register_buffer("cls_weights", weights)
        self.scale = scale
        self.drw_start = drw_start
        self.current_epoch = 1

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, logits, labels):
        margins = self.m_list[labels]
        adjusted_logits = logits.clone()
        adjusted_logits[torch.arange(logits.size(0), device=logits.device), labels] -= margins
        weights = self.cls_weights if self.current_epoch >= self.drw_start else None
        return F.cross_entropy(self.scale * adjusted_logits, labels, weight=weights)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_labels", type=int, default=193)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--eval_freq", type=int, default=40)
    parser.add_argument("--log_freq", type=int, default=20)
    parser.add_argument("--model_path", type=str, default="/home/CXL/pythonprojects/CrimePrediction/github/chinese-bert-wwm-ext")
    parser.add_argument("--checkpoint_dir", type=str, default="/home/CXL/pythonprojects/CrimePrediction/Criminal-Charge-Prediction/ckpts_ldam_drw")
    parser.add_argument("--tensorboard_dir", type=str, default="/home/CXL/pythonprojects/CrimePrediction/Criminal-Charge-Prediction/tensorboard_ldam_drw")
    parser.add_argument("--log_dir", type=str, default="/home/CXL/pythonprojects/CrimePrediction/Criminal-Charge-Prediction/log_ldam_drw")
    parser.add_argument("--train_data_path", type=str, default="/home/CXL/pythonprojects/CrimePrediction/Data_Clearning/small_193_train0.8.csv")
    parser.add_argument("--eval_data_path", type=str, default="/home/CXL/pythonprojects/CrimePrediction/Data_Clearning/small_193_val0.1.csv")
    parser.add_argument("--ldam_max_m", type=float, default=0.5)
    parser.add_argument("--ldam_scale", type=float, default=30.0)
    parser.add_argument("--drw_start", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.9999)
    parser.add_argument("--pgd_k", type=int, default=3)
    return parser.parse_args()


def get_class_counts(dataset, num_labels):
    counts = [0] * num_labels
    for label in dataset.label_list:
        counts[VERBALIZER_INDEX_LABEL[label]] += 1
    return counts


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            predictions = outputs.logits.argmax(dim=1)
            labels = batch["labels"].to(device)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def trainer():
    global device, active_gpu_ids
    config = parse_arguments()
    set_seed(config.seed)

    if torch.cuda.is_available():
        invalid = [i for i in config.gpu_ids if i < 0 or i >= torch.cuda.device_count()]
        if invalid:
            raise ValueError(f"Invalid GPU ids: {invalid}")
        active_gpu_ids = config.gpu_ids
        device = torch.device(f"cuda:{active_gpu_ids[0]}")
    else:
        active_gpu_ids = []
        device = torch.device("cpu")

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    config.checkpoint_dir = os.path.join(config.checkpoint_dir, timestamp)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.tensorboard_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(config.tensorboard_dir, timestamp))
    logger.add(os.path.join(config.log_dir, f"{timestamp}.log"))

    tokenizer = BertTokenizer.from_pretrained(config.model_path)
    train_dataset = CustomDataset(config.train_data_path, tokenizer, config.max_seq_length)
    val_dataset = CustomDataset(config.eval_data_path, tokenizer, config.max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    class_counts = get_class_counts(train_dataset, config.num_labels)
    criterion = LDAMLoss(
        class_counts,
        max_m=config.ldam_max_m,
        scale=config.ldam_scale,
        drw_start=config.drw_start,
        beta=config.beta,
    ).to(device)

    model = BertForSequenceClassification.from_pretrained(
        config.model_path, num_labels=config.num_labels
    ).to(device)
    if len(active_gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=active_gpu_ids, output_device=active_gpu_ids[0])
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    pgd = PGD(model=model)
    global_step = 0
    best_acc = 0.0

    for epoch in range(1, config.epochs + 1):
        criterion.set_epoch(epoch)
        model.train()
        for batch_index, batch in enumerate(train_loader, start=1):
            global_step += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()

            logits = model(input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels)
            loss.backward()
            pgd.backup_grad()
            for attack_index in range(config.pgd_k):
                pgd.attack(is_first_attack=(attack_index == 0))
                model.zero_grad() if attack_index < config.pgd_k - 1 else pgd.restore_grad()
                adv_logits = model(input_ids, attention_mask=attention_mask).logits
                criterion(adv_logits, labels).backward()
            pgd.restore()
            optimizer.step()

            writer.add_scalar("loss", loss.item(), global_step)
            if global_step % config.log_freq == 0:
                logger.info(f"epoch={epoch}/{config.epochs}, step={global_step}, loss={loss.item():.6f}")
            if global_step % config.eval_freq == 0:
                accuracy = evaluate(model, val_loader)
                writer.add_scalar("accuracy", accuracy, global_step)
                logger.info(f"epoch={epoch}, step={global_step}, val_accuracy={accuracy:.6f}")
                if accuracy > best_acc:
                    best_acc = accuracy
                    torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, "best.pt"))

        torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, f"epoch{epoch}.pt"))

    writer.close()
    logger.info(f"Training finished. Best validation accuracy: {best_acc:.6f}")


if __name__ == "__main__":
    trainer()
