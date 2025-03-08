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
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import numpy as np
from PGD import PGD
import random

def parse_arguments():
    #random_seed = random.randint(0, 10000)
    random_seed = 9771

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=random_seed, help="random seed for initialization.")
    parser.add_argument('--batch_size', default=32, type=int, help="Total batch size for training.")
    parser.add_argument('--epochs', default=10, type=int, help='The epoch of train')
    parser.add_argument('--num_labels', default=193, type=int, help='The number of labels')
    parser.add_argument('--max_seq_length', default=512, type=int, help='The maximum length of squence')
    parser.add_argument('--learning_rate', default=5e-6, type=float, help="The initial learning rate for optimizer")
    parser.add_argument('--eval_freq', default=40, type=int, help='The freq of eval test set')
    parser.add_argument('--log_freq', default=20, type=int, help='The freq of print log')
    parser.add_argument('--model_path', required=False, type=str, default="data_code\data\chinese-bert-wwm-ext",  help='The pretrained model')
    parser.add_argument('--checkpoint_dir', type=str, default="data_code\ckpts", help="The directory of checkpoints")
    parser.add_argument('--tensorboard_dir', type=str, default="data_code\\tensorboard", help="The directory of tensorboard")
    parser.add_argument('--log_dir', type=str, default="data_code\log", help="The directory of log")
    parser.add_argument('--train_data_path', type=str, required=False, default="data_code\data\large_193_train0.8",  help="The path of train Toxic Comment Classification Challenge dataset")
    parser.add_argument('--eval_data_path', type=str, required=False, default="data_code\data\large_193_val0.1.zip",  help="The path of eval Toxic Comment Classification Challenge dataset")

    args = parser.parse_args()

    return args
VERBALIZER_INDEX_LABEL = {
    '串通投标': 0,
    '交通肇事': 1,
    '介绍贿赂': 2,
    '以危险方法危害公共安全': 3,
    '传授犯罪方法': 4,
    '传播性病': 5,
    '伪证': 6,
    '伪造、倒卖伪造的有价票证': 7,
    '伪造、变造、买卖国家机关公文、证件、印章': 8,
    '伪造、变造、买卖武装部队公文、证件、印章': 9,
    '伪造、变造居民身份证': 10,
    '伪造、变造金融票证': 11,
    '伪造公司、企业、事业单位、人民团体印章': 12,
    '伪造货币': 13,
    '侮辱': 14,
    '侵占': 15,
    '侵犯著作权': 16,
    '保险诈骗': 17,
    '信用卡诈骗': 18,
    '倒卖文物': 19,
    '倒卖车票、船票': 20,
    '假冒注册商标': 21,
    '冒充军人招摇撞骗': 22,
    '出售、购买、运输假币': 23,
    '利用影响力受贿': 24,
    '制作、复制、出版、贩卖、传播淫秽物品牟利': 25,
    '制造、贩卖、传播淫秽物品': 26,
    '动植物检疫徇私舞弊': 27,
    '劫持船只、汽车': 28,
    '单位受贿': 29,
    '单位行贿': 30,
    '危险物品肇事': 31,
    '危险驾驶': 32,
    '受贿': 33,
    '合同诈骗': 34,
    '失火': 35,
    '妨害作证': 36,
    '妨害信用卡管理': 37,
    '妨害公务': 38,
    '容留他人吸毒': 39,
    '对单位行贿': 40,
    '对非国家工作人员行贿': 41,
    '寻衅滋事': 42,
    '巨额财产来源不明': 43,
    '帮助毁灭、伪造证据': 44,
    '帮助犯罪分子逃避处罚': 45,
    '开设赌场': 46,
    '引诱、教唆、欺骗他人吸毒': 47,
    '强制猥亵、侮辱妇女': 48,
    '强奸': 49,
    '强迫交易': 50,
    '强迫他人吸毒': 51,
    '强迫劳动': 52,
    '徇私枉法': 53,
    '徇私舞弊不征、少征税款': 54,
    '徇私舞弊不移交刑事案件': 55,
    '打击报复证人': 56,
    '扰乱无线电通讯管理秩序': 57,
    '投放危险物质': 58,
    '抢劫': 59,
    '抢夺': 60,
    '拐卖妇女、儿童': 61,
    '拐骗儿童': 62,
    '拒不执行判决、裁定': 63,
    '拒不支付劳动报酬': 64,
    '招摇撞骗': 65,
    '招收公务员、学生徇私舞弊': 66,
    '持有、使用假币': 67,
    '持有伪造的发票': 68,
    '挪用公款': 69,
    '挪用特定款物': 70,
    '挪用资金': 71,
    '掩饰、隐瞒犯罪所得、犯罪所得收益': 72,
    '提供侵入、非法控制计算机信息系统程序、工具': 73,
    '收买被拐卖的妇女、儿童': 74,
    '放火': 75,
    '故意伤害': 76,
    '故意杀人': 77,
    '故意毁坏财物': 78,
    '敲诈勒索': 79,
    '污染环境': 80,
    '洗钱': 81,
    '滥伐林木': 82,
    '滥用职权': 83,
    '爆炸': 84,
    '猥亵儿童': 85,
    '玩忽职守': 86,
    '生产、销售不符合安全标准的食品': 87,
    '生产、销售伪劣产品': 88,
    '生产、销售伪劣农药、兽药、化肥、种子': 89,
    '生产、销售假药': 90,
    '生产、销售有毒、有害食品': 91,
    '盗伐林木': 92,
    '盗掘古文化遗址、古墓葬': 93,
    '盗窃': 94,
    '盗窃、侮辱尸体': 95,
    '盗窃、抢夺枪支、弹药、爆炸物、危险物质': 96,
    '破坏交通工具': 97,
    '破坏交通设施': 98,
    '破坏广播电视设施、公用电信设施': 99,
    '破坏易燃易爆设备': 100,
    '破坏生产经营': 101,
    '破坏电力设备': 102,
    '破坏监管秩序': 103,
    '破坏计算机信息系统': 104,
    '票据诈骗': 105,
    '私分国有资产': 106,
    '窃取、收买、非法提供信用卡信息': 107,
    '窝藏、包庇': 108,
    '窝藏、转移、收购、销售赃物': 109,
    '窝藏、转移、隐瞒毒品、毒赃': 110,
    '组织、领导、参加黑社会性质组织': 111,
    '组织、领导传销活动': 112,
    '绑架': 113,
    '编造、故意传播虚假恐怖信息': 114,
    '职务侵占': 115,
    '聚众冲击国家机关': 116,
    '聚众哄抢': 117,
    '聚众扰乱公共场所秩序、交通秩序': 118,
    '聚众扰乱社会秩序': 119,
    '聚众斗殴': 120,
    '脱逃': 121,
    '虐待': 122,
    '虐待被监管人': 123,
    '虚开发票': 124,
    '虚开增值税专用发票、用于骗取出口退税、抵扣税款发票': 125,
    '虚报注册资本': 126,
    '行贿': 127,
    '诈骗': 128,
    '诬告陷害': 129,
    '诽谤': 130,
    '贪污': 131,
    '贷款诈骗': 132,
    '赌博': 133,
    '走私': 134,
    '走私、贩卖、运输、制造毒品': 135,
    '走私国家禁止进出口的货物、物品': 136,
    '走私废物': 137,
    '走私普通货物、物品': 138,
    '走私武器、弹药': 139,
    '走私珍贵动物、珍贵动物制品': 140,
    '过失以危险方法危害公共安全': 141,
    '过失投放危险物质': 142,
    '过失损坏广播电视设施、公用电信设施': 143,
    '过失损坏武器装备、军事设施、军事通信': 144,
    '过失致人死亡': 145,
    '过失致人重伤': 146,
    '违法发放贷款': 147,
    '逃税': 148,
    '遗弃': 149,
    '重大劳动安全事故': 150,
    '重大责任事故': 151,
    '重婚': 152,
    '金融凭证诈骗': 153,
    '销售假冒注册商标的商品': 154,
    '隐匿、故意销毁会计凭证、会计帐簿、财务会计报告': 155,
    '集资诈骗': 156,
    '非国家工作人员受贿': 157,
    '非法买卖、运输、携带、持有毒品原植物种子、幼苗': 158,
    '非法买卖制毒物品': 159,
    '非法侵入住宅': 160,
    '非法出售发票': 161,
    '非法制造、买卖、运输、储存危险物质': 162,
    '非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物': 163,
    '非法制造、出售非法制造的发票': 164,
    '非法制造、销售非法制造的注册商标标识': 165,
    '非法占用农用地': 166,
    '非法吸收公众存款': 167,
    '非法处置查封、扣押、冻结的财产': 168,
    '非法拘禁': 169,
    '非法持有、私藏枪支、弹药': 170,
    '非法持有毒品': 171,
    '非法捕捞水产品': 172,
    '非法携带枪支、弹药、管制刀具、危险物品危及公共安全': 173,
    '非法收购、运输、出售珍贵、濒危野生动物、珍贵、濒危野生动物制品': 174,
    '非法收购、运输、加工、出售国家重点保护植物、国家重点保护植物制品': 175,
    '非法收购、运输盗伐、滥伐的林木': 176,
    '非法狩猎': 177,
    '非法猎捕、杀害珍贵、濒危野生动物': 178,
    '非法生产、买卖警用装备': 179,
    '非法生产、销售间谍专用器材': 180,
    '非法种植毒品原植物': 181,
    '非法组织卖血': 182,
    '非法经营': 183,
    '非法获取公民个人信息': 184,
    '非法获取国家秘密': 185,
    '非法行医': 186,
    '非法转让、倒卖土地使用权': 187,
    '非法进行节育手术': 188,
    '非法采伐、毁坏国家重点保护植物': 189,
    '非法采矿': 190,
    '骗取贷款、票据承兑、金融票证': 191,
    '高利转贷': 192
}

# VERBALIZER_INDEX_LABEL = {
#     "business": 0,
#     "tech": 1,
#     "politics": 2,
#     "sport": 3,
#     "entertainment": 4
# }


# 设定设备（CPU或GPU）



def stats_time(start, end, step, total_step):
    t = end -start
    return '{:.3f}'.format((t / step * (total_step - step) / 3600))


def setup_training(config):
    set_seed(config.seed)
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
    def __init__(self, data_path, tokenizer, max_length):
        self.text_list = []
        self.label_list = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path) as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, text = row
                if idx == 0:
                    continue
                else:
                    self.text_list.append(text)
                    self.label_list.append(label)

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        text = self.text_list[idx]
        label = self.label_list[idx]
        encoding = self.tokenizer.encode_plus(
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
            "labels": torch.tensor(VERBALIZER_INDEX_LABEL[label], dtype=torch.long),
        }



def adjust_logits(logits):
    # 对logits进行调整
    max_logits, max_indices = torch.max(logits, dim=1)
    second_max_logits = torch.topk(logits, 2, dim=1).values[:, 1]

    # 计算 |a - b|
    abs_diff = torch.abs(max_logits - second_max_logits)
    abs_diff_np = abs_diff.detach().cpu().numpy()  # 转到CPU并转换为NumPy数组

    # 计算 tanh(|a - b|)，并转换为PyTorch张量
    tanh_abs_diff = torch.tensor(np.tanh(abs_diff_np), device=logits.device, dtype=logits.dtype)

    # 计算 tanh 的导数
    tanh_derivative = 1 - tanh_abs_diff ** 2


    x = tanh_derivative

    logits_clone = logits.clone()

    # 更新logits
    logits_clone[range(logits.size(0)), max_indices] = 4*max_logits + x * abs_diff_np
    return logits_clone

def evaluation(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            logits = adjust_logits(logits)

            val_loss += criterion(logits, labels).item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    val_loss /= len(val_loader)

    return accuracy, val_loss

def trainer():
    config = parse_arguments()
    config, writer = setup_training(config)
    # 加载预训练的BERT模型和分词器
    tokenizer = BertTokenizer.from_pretrained(config.model_path)
    model = BertForSequenceClassification.from_pretrained(config.model_path, num_labels=config.num_labels)

    model = nn.DataParallel(model)  # 将模型封装在 DataParallel 中
    model.to(device)


    # 创建数据加载器
    train_dataset = CustomDataset(config.train_data_path, tokenizer, config.max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = CustomDataset(config.eval_data_path, tokenizer, config.max_seq_length)
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
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            pgd = PGD(model=model)
            ##
            # 调整logits
            logits = adjust_logits(logits)
            loss = criterion(logits, labels)
            #####
            #
            # loss = outputs.loss
            avg_loss += loss.item()
            loss.backward()
            ###########
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
                logits = adjust_logits(logits)

                loss = criterion(logits, labels)
                # loss = outputs.loss
                avg_loss += loss.item()
                loss.backward()
            pgd.restore()
            # ###############

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
                logger.info(f"epochs:{str(epoch) + '/' + str(config.epochs)}, step:{str(step) + '/' + str(total_step)}, avg_acc:{'{:.6f}'.format(acc)}")

                # checkpoint path
                if acc > global_acc or len(most_recent_ckpts_paths) < 3:
                    cur_path = os.path.join(config.checkpoint_dir, f"epoch{epoch}_step{step}_acc{'{:.6f}'.format(acc)}.pt")
                    weight_path = os.path.join(config.checkpoint_dir, f"epoch{epoch}_step{step}_acc{'{:.6f}'.format(acc)}_weight.pt")
                    torch.save(model.state_dict(), cur_path)
                    if acc > global_acc:
                        global_acc = acc

                    most_recent_ckpts_paths.append(cur_path)
                    if len(most_recent_ckpts_paths) > 3:  # 仅仅保存最新的三个checkpoint
                        ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                        os.remove(ckpt_to_be_removed)

        torch.save(model, f'data_code/Model/model{epoch}.pth')

if __name__ == '__main__':
    trainer()
