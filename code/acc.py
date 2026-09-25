import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import csv
import argparse
import os
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import torch.nn as nn
from transformers import BertForSequenceClassification
from mask_experiment import DEFAULT_CATEGORIES, mask_text


MASK_CATEGORIES = ('date', 'time', 'location', 'person')


# 设备配置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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


# 定义测试数据集类
class TestDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length, mask_categories=MASK_CATEGORIES):
        self.text_list = []
        self.masked_text_list = []
        self.label_list = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_categories = tuple(mask_categories)
        self.mask_match_count = 0
        self.masked_sample_count = 0
        self.mask_category_match_counts = {
            category: 0 for category in self.mask_categories
        }
        self.missing_labels = set()  # 收集不存在的标签

        with open(data_path, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                if len(row) < 2:
                    continue
                label = row[0]
                # 保留文本中的逗号，避免因 CSV 行中存在未转义逗号而截断案件描述。
                text = row[1] if len(row) == 2 else ','.join(row[1:])
                if label not in VERBALIZER_INDEX_LABEL:
                    self.missing_labels.add(label)
                    continue  # 跳过不在 mapping 的标签
                self.text_list.append(text)
                self.label_list.append(label)

                # 只在内存中生成 mask 后文本，不改写原始 CSV 文件。
                masked_text, matches = mask_text(text, self.mask_categories)
                self.masked_text_list.append(masked_text)
                self.mask_match_count += len(matches)
                self.masked_sample_count += int(bool(matches))
                for match in matches:
                    self.mask_category_match_counts[match.category] += 1

        if self.missing_labels:
            print("Warning: 以下标签不在 VERBALIZER_INDEX_LABEL 中，将被忽略：")
            print(sorted(self.missing_labels))

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        text = self.masked_text_list[idx]
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
            "labels": torch.tensor(VERBALIZER_INDEX_LABEL[label], dtype=torch.long),
        }

# def adjust_logits(logits):
#     # 对logits进行调整
#     max_logits, max_indices = torch.max(logits, dim=1)
#     second_max_logits = torch.topk(logits, 2, dim=1).values[:, 1]
#
#     # 计算 |a - b|
#     abs_diff = torch.abs(max_logits - second_max_logits)
#     abs_diff_np = abs_diff.detach().cpu().numpy()  # 转到CPU并转换为NumPy数组
#
#     # 计算 tanh(|a - b|)，并转换为PyTorch张量
#     tanh_abs_diff = torch.tensor(np.tanh(abs_diff_np), device=logits.device, dtype=logits.dtype)
#
#     # 计算 tanh 的导数
#     tanh_derivative = 1 - tanh_abs_diff ** 2
#
#     # 符号函数
#     sign_np = (max_logits - second_max_logits).detach().cpu().numpy()
#     sign = torch.tensor(np.sign(sign_np), device=logits.device, dtype=logits.dtype)
#
#     # 计算 tanh(|a - b|) 的导数
#     x = 1 + tanh_derivative * sign
#
#     logits_clone = logits.clone()
#
#     # 更新logits
#     logits_clone[range(logits.size(0)), max_indices] = max_logits + x * (max_logits - second_max_logits)
#     return logits_clone
#
#
# # 评估函数
# # 评估函数


criterion = nn.CrossEntropyLoss()
def evaluate_model(model, test_loader, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    if len(test_loader) == 0:
        raise ValueError("测试数据集为空，无法进行 mask 后数据评估。")

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    macro_precision, macro_recall, _, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    micro_precision, micro_recall, _, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')

    # 生成报告，忽略测试集中未出现的标签
    labels = sorted(set(all_labels))  # 生成测试集中实际出现的标签
    target_names = [
        key for key, value in VERBALIZER_INDEX_LABEL.items()
        if value in labels
    ]
    report = classification_report(
        all_labels,
        all_preds,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    return avg_loss, accuracy, precision, recall, f1, macro_precision, macro_recall, micro_precision, micro_recall, report


def resolve_model_paths(model_path, model_dir=None):
    """Resolve all .pt checkpoints in the requested model directory."""
    search_dir = model_dir
    if search_dir is None:
        search_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)

    model_paths = []
    if search_dir and os.path.isdir(search_dir):
        for filename in sorted(os.listdir(search_dir)):
            candidate = os.path.join(search_dir, filename)
            if os.path.isfile(candidate) and filename.lower().endswith('.pt'):
                model_paths.append(candidate)

    if model_paths:
        return model_paths
    if model_path and os.path.isfile(model_path):
        return [model_path]

    raise FileNotFoundError(
        f"未找到模型文件。model_path={model_path}, model_dir={model_dir}"
    )


def load_model(model_path, tokenizer_path):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, nn.Module):
        # 兼容直接保存的完整模型。
        model = checkpoint
    else:
        # 1. 先建“空模型”（结构必须一致）
        model = BertForSequenceClassification.from_pretrained(
            tokenizer_path,
            num_labels=193  # ⚠️ 一定要和你训练时一样
        )

        # 2. 加载 checkpoint（这是 state_dict）
        if not isinstance(checkpoint, dict):
            raise TypeError(
                f"不支持的 checkpoint 类型: {type(checkpoint).__name__}"
            )
        state_dict = checkpoint.get(
            "state_dict",
            checkpoint.get("model_state_dict", checkpoint),
        )

        # 3. 去掉 DataParallel 的 "module." 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        # 4. 加载参数
        model.load_state_dict(new_state_dict, strict=True)

    # 5. 上设备
    model = model.to(device)
    model.eval()
    return model



def parse_arguments():
    parser = argparse.ArgumentParser()


    parser.add_argument('--model_path', type=str, required=False, default='/home/CXL/pythonprojects/CrimePrediction/Criminal-Charge-Prediction/ckpts/2026-09-19T16-52-33/epoch16_step11760_acc0.800000.pt', help='Path to the saved model')
    parser.add_argument('--tokenizer_path', type=str, required=False, default="/home/CXL/pythonprojects/CrimePrediction/github/chinese-bert-wwm-ext", help='Path to the pretrained tokenizer')
    parser.add_argument('--test_data_path', type=str, required=False, default="/home/CXL/pythonprojects/CrimePrediction/Data_Clearning/small_193_test0.1.csv",help='Path to the test data')
    parser.add_argument('--max_seq_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')
    parser.add_argument(
        '--model_dir',
        type=str,
        default=None,
        help='Evaluate every .pt model in this directory. Defaults to the directory of --model_path.',
    )
    parser.add_argument(
        '--mask_categories',
        nargs='+',
        choices=DEFAULT_CATEGORIES,
        default=list(MASK_CATEGORIES),
        help='Factual-information categories to mask in memory.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    #
    # torch.serialization.add_safe_globals([nn.DataParallel])
    #
    # model = torch.load(args.model_path, map_location='cuda:0', weights_only=False)
    # model = model.to('cuda:0')
    # model.eval()
    # 先构建模型结构

    if set(args.mask_categories) != set(MASK_CATEGORIES):
        raise ValueError(
            f"必须且只能 mask date、time、location、person；"
            f"当前配置为: {args.mask_categories}"
        )

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)

    # 准备测试数据集
    test_dataset = TestDataset(
        args.test_data_path,
        tokenizer,
        args.max_seq_length,
        mask_categories=args.mask_categories,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model_paths = resolve_model_paths(args.model_path, args.model_dir)
    print(f"Models to evaluate: {len(model_paths)}")

    for model_path in model_paths:
        print("\n" + "=" * 80)
        print(f"Model Path: {model_path}")
        print("=" * 80)

        model = load_model(model_path, args.tokenizer_path)

        # 评估模型
        test_loss, test_accuracy, precision, recall, f1, macro_precision, macro_recall, micro_precision, micro_recall, report = evaluate_model(model, test_loader, criterion)

        # print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        # print(f"Precision: {precision:.4f}")
        # print(f"Recall: {recall:.4f}")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        # print(f"Micro Precision: {micro_precision:.4f}")
        # print(f"Micro Recall: {micro_recall:.4f}")
        print(f"Mask Categories: {', '.join(MASK_CATEGORIES)}")
        print(f"Masked Samples: {test_dataset.masked_sample_count}/{len(test_dataset)}")
        print(f"Unmasked Samples: {len(test_dataset) - test_dataset.masked_sample_count}")
        print(f"Matched Spans: {test_dataset.mask_match_count}")
        for category in MASK_CATEGORIES:
            print(
                f"Matched {category}: "
                f"{test_dataset.mask_category_match_counts[category]}"
            )
        # print("\nClassification Report:\n", report)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()




