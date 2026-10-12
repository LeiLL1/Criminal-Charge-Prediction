import os
import csv
import time
import torch
import torch.nn as nn
import argparse
from loguru import logger
from datetime import datetime
from transformers import set_seed
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import numpy as np
from PGD import PGD
import random

def parse_arguments():
    # random_seed = random.randint(0, 10000)
    random_seed = 2000

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=random_seed, help="random seed for initialization.")
    parser.add_argument('--batch_size', default=4, type=int, help="Total batch size for training.")
    parser.add_argument('--epochs', default=10, type=int, help='The epoch of train')
    parser.add_argument('--tau', default=0.5, type=float, help='The strength of training-time logit adjustment; 0 disables adjustment')
    parser.add_argument('--la_eps', default=1e-12, type=float, help='The minimum class prior for numerical stability in logit adjustment')
    parser.add_argument('--eps', default=0.2, type=float, help='The epsilon for PGD')
    parser.add_argument('--alpha', default=0.07, type=float, help='The alpha for PGD')
    parser.add_argument('--restore_iters', default=3, type=int, help='The number of restore iterations for PGD')
    parser.add_argument('--num_labels', default=107, type=int, help='The number of Traditional-Chinese charges in ljp_train.csv')
    parser.add_argument('--max_seq_length', default=512, type=int, help='The maximum length of squence')
    parser.add_argument('--learning_rate', default=5e-6, type=float, help="The initial learning rate for optimizer")
    parser.add_argument('--eval_freq', default=40, type=int, help='The freq of eval test set')
    parser.add_argument('--log_freq', default=20, type=int, help='The freq of print log')
    parser.add_argument('--model_path', required=False, type=str, default="/home/CXL/pythonprojects/CrimePrediction/github/chinese-bert-wwm-ext",  help='The pretrained model')
    parser.add_argument('--checkpoint_dir', type=str, default="/home/CXL/pythonprojects/CrimePrediction/Criminal-Charge-Prediction/TW_Logit/ckpts2", help="The directory of checkpoints")
    parser.add_argument('--tensorboard_dir', type=str, default="/home/CXL/pythonprojects/CrimePrediction/Criminal-Charge-Prediction/TW_Logit/tensorboard2", help="The directory of tensorboard")
    parser.add_argument('--log_dir', type=str, default="/home/CXL/pythonprojects/CrimePrediction/Criminal-Charge-Prediction/TW_Logit/log2", help="The directory of log")
    parser.add_argument('--train_data_path', type=str, required=False, default="/home/CXL/pythonprojects/CrimePrediction/Data_Clearning/TW_LJP/ljp_train.csv",  help="The path of train Traditional-Chinese charge dataset")
    parser.add_argument('--eval_data_path', type=str, required=False, default="/home/CXL/pythonprojects/CrimePrediction/Data_Clearning/TW_LJP/ljp_val.csv",  help="The path of eval Toxic Comment Classification Challenge dataset")

    args = parser.parse_args()

    return args


def validate_config(config):
    if config.batch_size < 1:
        raise ValueError('--batch_size must be greater than 0.')
    if config.epochs <= 0:
        raise ValueError('--epochs must be greater than 0.')
    if not np.isfinite(config.tau) or config.tau < 0:
        raise ValueError('--tau must be finite and greater than or equal to 0.')
    if not 0 < config.la_eps < 1:
        raise ValueError('--la_eps must be between 0 and 1.')
    if config.eps <= 0:
        raise ValueError('--eps must be greater than 0.')
    if config.alpha <= 0:
        raise ValueError('--alpha must be greater than 0.')
    if config.restore_iters <= 0:
        raise ValueError('--restore_iters must be greater than 0.')
    if config.max_seq_length <= 0:
        raise ValueError('--max_seq_length must be greater than 0.')
    if config.learning_rate <= 0:
        raise ValueError('--learning_rate must be greater than 0.')
    if config.eval_freq <= 0:
        raise ValueError('--eval_freq must be greater than 0.')
    if config.log_freq <= 0:
        raise ValueError('--log_freq must be greater than 0.')
    for path_name in ('model_path', 'train_data_path', 'eval_data_path'):
        path = getattr(config, path_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path_name} does not exist: {path}')
VERBALIZER_INDEX_LABEL = {
    '殺人未遂': 0,
    '詐欺等': 1,
    '妨害風化': 2,
    '毒品危害防制條例': 3,
    '過失致死': 4,
    '竊盜': 5,
    '公共危險等': 6,
    '侵占': 7,
    '槍砲彈藥刀械管制條例': 8,
    '毒品危害防制條例等': 9,
    '妨害性自主等': 10,
    '貪污治罪條例等': 11,
    '偽造文書等': 12,
    '妨害性自主': 13,
    '竊盜等': 14,
    '銀行法等': 15,
    '公職人員選舉罷免法等': 16,
    '銀行法': 17,
    '公共危險': 18,
    '貪污治罪條例': 19,
    '詐欺': 20,
    '加重竊盜': 21,
    '傷害': 22,
    '業務侵占等': 23,
    '妨害自由等': 24,
    '重傷害': 25,
    '偽造文書': 26,
    '森林法': 27,
    '賭博': 28,
    '商標法': 29,
    '槍砲彈藥刀械管制條例等': 30,
    '業務過失致死': 31,
    '重利': 32,
    '偽證': 33,
    '過失傷害': 34,
    '業務過失傷害': 35,
    '殺人未遂等': 36,
    '偽造有價證券': 37,
    '肇事逃逸': 38,
    '強盜': 39,
    '業務侵占': 40,
    '水土保持法': 41,
    '妨害自由': 42,
    '森林法等': 43,
    '藥事法': 44,
    '公職人員選舉罷免法': 45,
    '洗錢防制法等': 46,
    '肇事逃逸等': 47,
    '廢棄物清理法': 48,
    '家庭暴力防治法': 49,
    '誣告等': 50,
    '洗錢防制法': 51,
    '恐嚇取財': 52,
    '過失致死等': 53,
    '傷害等': 54,
    '著作權法': 55,
    '誣告': 56,
    '加重詐欺': 57,
    '偽證等': 58,
    '贓物等': 59,
    '偽造有價證券等': 60,
    '加重詐欺等': 61,
    '組織犯罪防制條例等': 62,
    '妨害秩序等': 63,
    '毀棄損壞': 64,
    '恐嚇等': 65,
    '強盜等': 66,
    '搶奪等': 67,
    '妨害名譽': 68,
    '殺人': 69,
    '贓物': 70,
    '業務過失致死等': 71,
    '妨害公務': 72,
    '侵占等': 73,
    '恐嚇': 74,
    '背信等': 75,
    '傷害致死': 76,
    '性騷擾防治法': 77,
    '背信': 78,
    '竊佔': 79,
    '個人資料保護法等': 80,
    '藥事法等': 81,
    '恐嚇取財等': 82,
    '毀棄損壞等': 83,
    '妨害秩序': 84,
    '妨害公務等': 85,
    '搶奪': 86,
    '貪污': 87,
    '政府採購法': 88,
    '貪污等': 89,
    '過失致重傷害': 90,
    '毀損': 91,
    '商業會計法等': 92,
    '廢棄物清理法等': 93,
    '槍砲彈藥刀械管制': 94,
    '槍砲彈藥刀械管制等': 95,
    '證券交易法等': 96,
    '殺人等': 97,
    '毀損等': 98,
    '兒童及少年性剝削防制條例': 99,
    '公司法等': 100,
    '兒童及少年性剝削防制條例等': 101,
    '證券交易法': 102,
    '肇事遺棄等': 103,
    '肇事遺棄': 104,
    '保護令': 105,
    '組織犯罪條例等': 106,
}
# The TW dataset uses the Traditional-Chinese mapping above.

# VERBALIZER_INDEX_LABEL = {
#     "business": 0,
#     "tech": 1,
#     "politics": 2,
#     "sport": 3,
#     "entertainment": 4
# }


# DataParallel uses cuda:0 as its primary/output device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def stats_time(start, end, step, total_step):
    t = end -start
    return '{:.3f}'.format((t / step * (total_step - step) / 3600))


def setup_training(config):
    set_seed(config.seed)
    os.makedirs(config.tensorboard_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # tensorboard set up
    time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    comment = f'bath_size={config.batch_size} lr={config.learning_rate}'
    writer = SummaryWriter(log_dir=config.tensorboard_dir + "/" + time_stamp, comment=comment)

    # cuda setup
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    # hidden tokenizer warning
    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    # logger setup
    logger.add(os.path.join(config.log_dir) + "/" +"{time}.log")

    # checkpoint dir setup
    checkpoint_dir = os.path.join(config.checkpoint_dir, time_stamp)
    os.makedirs(checkpoint_dir, exist_ok=True)  # 确保父目录存在
    config.checkpoint_dir = checkpoint_dir

    return config, writer



class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length, label_to_index):
        self.text_list = []
        self.label_list = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_index = label_to_index

        with open(data_path, encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if not row:
                    continue
                if len(row) != 2:
                    raise ValueError(f"Expected 2 CSV columns in {data_path}, got {len(row)}")
                label, text = row
                if label not in self.label_to_index:
                    raise KeyError(f"Unknown label {label!r} in {data_path}")
                self.text_list.append(text)
                self.label_list.append(label)

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        text = self.text_list[idx]
        label = self.label_list[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.label_to_index[label], dtype=torch.long),
        }


# def adjust_logits(logits):
#     # 找到最大的logits及其索引
#     max_logits, max_indices = torch.max(logits, dim=1)
#
#     # 找到第二大的logits
#     second_max_logits = torch.topk(logits, 2, dim=1).values[:, 1]
#     x = 3
#     # 计算新的最大的logits
#     new_max_logits = x * max_logits - second_max_logits
#
#     # 复制 logits
#     logits_clone = logits.clone()
#
#     # 更新logits
#     logits_clone[range(logits.size(0)), max_indices] = new_max_logits
#
#     return logits_clone

#
def adjust_logits(logits, log_class_priors, tau):
    # 训练期 LA：使用训练集类别先验调整 logits。
    return logits + tau * log_class_priors.to(device=logits.device, dtype=logits.dtype)

def evaluation(model, val_loader, criterion):
    # DataParallel 的验证复制会再次触发 NCCL broadcast；验证使用主卡模型即可。
    eval_model = model.module if isinstance(model, nn.DataParallel) else model
    eval_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = eval_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            val_loss += criterion(logits, labels).item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    val_loss /= len(val_loader)

    return accuracy, val_loss

def trainer():
    config = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("单卡 CUDA 训练需要至少 1 张可见 GPU。")

    validate_config(config)
    label_to_index = VERBALIZER_INDEX_LABEL
    if config.num_labels != len(label_to_index):
        raise ValueError(
            f"--num_labels={config.num_labels} 与固定繁体标签映射中的 "
            f"{len(label_to_index)} 个标签不一致。"
        )

    config, writer = setup_training(config)
    gpu_id = 0
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    logger.info(f"单卡训练使用 GPU: cuda:{gpu_id}")
    logger.info(
        f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}, "
        f"显存总量: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.2f} GB"
    )
    # 加载预训练的BERT模型和分词器
    tokenizer = BertTokenizer.from_pretrained(config.model_path)
    model = BertForSequenceClassification.from_pretrained(
        config.model_path,
        num_labels=config.num_labels,
        ignore_mismatched_sizes=True,
    )

    model.to(device)
    logger.info(f"模型和训练均使用 GPU: cuda:{gpu_id}")


    # 创建数据加载器
    train_dataset = CustomDataset(config.train_data_path, tokenizer, config.max_seq_length, label_to_index)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    class_counts = torch.bincount(
        torch.tensor([label_to_index[label] for label in train_dataset.label_list], dtype=torch.long, device=device),
        minlength=config.num_labels,
    ).float()
    class_priors = class_counts / class_counts.sum()
    log_class_priors = class_priors.clamp_min(config.la_eps).log()

    val_dataset = CustomDataset(config.eval_data_path, tokenizer, config.max_seq_length, label_to_index)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    total_step = config.epochs * len(train_loader)

    logger.info(f"{'#' * 41} Config {'#' * 41}")
    for k in list(vars(config).keys()):
        logger.info('{0}: {1}'.format(k, vars(config)[k]))
    logger.info(f'total step: {total_step}')
    logger.info(f'the number of train step: {len(train_loader)}')
    logger.info(f'the size of train set: {len(train_loader) * config.batch_size}')
    logger.info(f'the size of eval set: {len(val_loader) * config.batch_size}')
    logger.info(f"{'#' * 41} Training {'#' * 41}")

    start = int(time.time())
    step = 0
    avg_loss = 0.0
    global_acc = 0.0
    most_recent_ckpts_paths = []

    # 训练模型
    for epoch in range(1, config.epochs+1):
        for i, batch in enumerate(train_loader):
            model.train()
            step += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            ##
            pgd = PGD(model=model, eps=config.eps, alpha=config.alpha, restore_iters=config.restore_iters)
            # ###
            # 调整logits
            logits = adjust_logits(logits, log_class_priors, config.tau)
            loss = criterion(logits, labels)
            #####
            #
            # loss = criterion(outputs.logits, labels)
            avg_loss += loss.item()
            loss.backward()
            #########
            pgd_k = 3
            pgd.backup_grad()  # 备份模型参数的梯度
            for _t in range(pgd_k):
                pgd.attack(is_first_attack=(_t == 0))  # PGD 类的 attack() 方法，执行对抗攻击

                if _t != pgd_k - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()  # 如果是最后一次攻击，恢复模型参数的梯度

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                # logits = adjust_logits(logits)
                logits = adjust_logits(logits, log_class_priors, config.tau)

                loss = criterion(logits, labels)
                # loss = outputs.loss
                avg_loss += loss.item()
                loss.backward()
            pgd.restore()
            #############

            optimizer.step()

            # tensorboard
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('avg_loss', avg_loss / step, step)
            # log
            if step % config.log_freq == 0:
                end = int(time.time())
                logger.info(
                    f"epochs:{str(epoch) + '/' + str(config.epochs)}, batch:{str(i + 1) + '/' + str(len(train_loader))}, step:{str(step) + '/' + str(total_step)}, cur_loss:{'{:.6f}'.format(loss)}, avg_loss:{'{:.6f}'.format(avg_loss / step)}, eta:{stats_time(start, end, step, total_step)}h")

            # 在验证集上评估模型
            if step % config.eval_freq == 0:
                acc, val_loss = evaluation(model, val_loader, criterion)
                writer.add_scalar('acc', acc, step)
                writer.add_scalar('val_loss', val_loss, step)
                logger.info(
                    f"epochs:{str(epoch) + '/' + str(config.epochs)}, step:{str(step) + '/' + str(total_step)}, "
                    f"avg_acc:{'{:.6f}'.format(acc)}, val_loss:{'{:.6f}'.format(val_loss)}"
                )

                # checkpoint path
                if acc > global_acc or len(most_recent_ckpts_paths) < 3:
                    cur_path = os.path.join(config.checkpoint_dir, f"epoch{epoch}_step{step}_acc{'{:.6f}'.format(acc)}.pt")
                    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                    torch.save(model_to_save.state_dict(), cur_path)
                    if acc > global_acc:
                        global_acc = acc

                    most_recent_ckpts_paths.append(cur_path)
                    if len(most_recent_ckpts_paths) > 3:  # 仅仅保存最新的三个checkpoint
                        ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                        os.remove(ckpt_to_be_removed)

        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(model_to_save, os.path.join(config.checkpoint_dir, f'/home/CXL/pythonprojects/CrimePrediction/Criminal-Charge-Prediction/TW_Logit/Model2/model{epoch}.pth'))

    writer.close()

if __name__ == '__main__':
    trainer()
