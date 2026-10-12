"""Mask case-specific entities in the headerless TW LJP CSV (label, text).

Usage:
    python TW/mask.py
    python TW/mask.py --input_path path/to/ljp_test.csv --output_path path/to/masked.csv
    python TW/mask.py --ner_model path/to/zh_core_web_sm

The default rules run offline with the standard library. An optional local spaCy
Chinese NER model supplements person, location and organization recognition.
Rules are heuristics, not a guarantee that every previously unseen name is found.
Amounts, ages, sentences, drug quantities, alcohol levels and legal provisions
are deliberately outside this masking scope because they can be legal facts.
Non-legal contact details, account credentials, administrative identifiers,
parcel identifiers, URLs and ordinary device metadata are masked as identifiers.
"""

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT.parent / "Data_Clearning" / "TW_LJP" / "ljp_test.csv"
CATEGORIES = ("time", "location", "name", "organization", "identifier", "environment")
NUM = r"[0-9零〇○Ｏ一二三四五六七八九十百千兩两壹貳參肆伍陸柒捌玖拾]+"
HAN = r"\u3400-\u9fff"
ENTITY_CHARS = HAN + r"○〇Ｏ0-9A-Za-z．·＆&\-"
FULLWIDTH = str.maketrans(
    "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ：－／",
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ:-/",
)

SURNAMES = (
    "陳林黃張李王吳劉蔡楊許鄭謝郭洪曾邱廖賴徐周葉蘇莊呂江何蕭羅高潘簡朱鍾彭"
    "游詹胡施沈余盧梁趙顏柯孫魏翁戴范宋方鄧杜傅侯曹薛丁卓馬阮董唐溫藍石蔣"
    "古紀姚連馮歐程湯白田康汪姜尤鐘嚴黎龔賈夏金范涂鄒巫褚池秦賀倪饒包伍向"
    "熊樂柳魯尹毛段史任袁商錢韓童葛俞殷梅趙祝車章龍隆陸季莫易岳管孟陶安饒"
)
SURNAME = rf"(?:歐陽|司馬|諸葛|上官|司徒|夏侯|[{SURNAMES}])"
PERSON = rf"{SURNAME}[{HAN}]{{1,2}}"
PERSON_LONG = rf"{SURNAME}[{HAN}]{{1,3}}"
PERSON_ACTION = (
    r"前(?:於|因|曾)|於|因|係|為|與|和|及|均|等|所|之|即|則|仍|又|明知|知悉|"
    r"意圖|基於|駕駛|騎乘|持用|持有|持|見|向|將|以|在|把|經|遂|乃|欲|"
    r"擔任|負責|表示|告知|交付|收受|受有|所有|涉嫌|另|後|曾|自|不|無|未|"
    r"共同|相約|認為|認識|邀約|委託|申辦|承租|停放|上前|到場|訴由|訴請|"
    r"購得|購買|販賣|竊取|僱用|受僱|受雇|透過|知道|叫|拿|認|遭|見狀"
)
PERSON_CONTEXT = (
    r"被告|告訴人|被害人|證人|嫌疑人|犯罪嫌疑人|共犯|同案被告|友人|朋友|"
    r"母親|父親|其父|其母|妻子|丈夫|胞兄|胞弟|胞姊|胞妹|姑丈|老闆|"
    r"負責人|代表人|檢察官|法官|員警|警員|司機|男子|女子|姓名|名叫|名為|"
    r"搭載|邀約|認識|委託|透過|交予|交付予|交給|送交|通知|告知|與|之"
)
PERSON_STRONG_ACTION = (
    r"前於|前因|前曾|明知|知悉|意圖|基於|駕駛|騎乘|持用|持有|擔任|負責|"
    r"表示|告知|交付|收受|受有|所有|所駕|所騎|涉嫌|共同|相約|認識|邀約|"
    r"委託|申辦|承租|停放|上前|到場|訴由|訴請|購得|購買|販賣|竊取|僱用|"
    r"受僱|受雇|透過|因而|見狀|竟|遂|並|復|曾因|另因"
)
PERSON_EXCLUDE = (
    "法院", "公司", "銀行", "政府", "警察", "商業", "道路", "路面", "公路",
    "地區", "方式", "方向", "方法", "行為", "行駛", "明知", "所有", "交通",
    "基於", "自己", "意圖", "之", "於", "與", "及", "前因", "前於", "其所",
    "假釋", "強制", "施用", "不詳", "法律", "刑法", "姓名", "注意", "照明",
    "安非", "他命", "金融", "金額", "車輛", "車牌", "車道", "車手", "車行",
    "安全", "商品", "商店", "許可", "管制", "管理", "陸續", "施詐", "交易",
    "施以", "明確", "向線", "方位", "後來", "成年", "行政", "公務", "國家",
    "包裹", "包而", "包裝", "陳述", "強盜", "強暴", "強迫", "重大", "應允",
    "利息", "負責", "定金", "金錢", "任職", "施行", "許其", "陸地", "公布",
    "林木", "林地", "林業", "金流", "周圍", "強調", "商標", "資格", "商品",
)

CITY_NAMES = (
    "南投|彰化|金門|嘉義|雲林|花蓮|臺東|台東|宜蘭|屏東|澎湖|連江|苗栗|"
    "新竹|桃園|基隆|新北|臺北|台北|臺中|台中|臺南|台南|高雄|臺灣|台灣|"
    "福建|廈門|湖北|武漢|北京|上海|廣東|廣州|浙江|深圳|江蘇|海南|泉州"
)
REGION = rf"(?:{CITY_NAMES})[縣市省]"
ADMIN = rf"[{HAN}○〇Ｏ0-9]{{1,6}}?(?:縣|市|區|鄉|鎮|村|里)"
ROAD = rf"[{HAN}○〇Ｏ0-9]{{1,12}}?(?:大道|路|街|巷)"
ADDRESS_TAIL = rf"(?:{NUM}(?:段|巷|弄|號|樓|之{NUM}號)|[○〇Ｏ0-9]{{1,12}}號)*"
ORG_SUFFIX = (
    r"股份有限公司|有限公司|實業社|企業社|工程行|商行|合作社|農會|漁會|"
    r"信用合作社|商業銀行|銀行|郵局|公司|市政府警察局|縣政府警察局|警察局|分局|派出所|地方法院檢察署|地方法院|"
    r"高等法院檢察署|高等法院|最高法院|地方檢察署|法院檢察署|法院|檢察署|醫療財團法人|醫院|診所|"
    r"大學|高級中學|高級職業學校|國民中學|國民小學|高中|高職|國中|國小|"
    r"便利商店|超商|超市|百貨|汽車旅館|旅館|飯店|酒店|賓館|餐廳|小吃店|"
    r"檳榔攤|檳榔店|加油站|汽車鈑金廠|汽車修理廠|工廠|宮廟|寺廟|基金會|"
    r"協會|委員會|管理處|管理局|消防局|衛生局|教育局|監獄|看守所|戒治所|"
    r"海岸巡防署|海洋巡防總局|海巡隊|調查局|調查站|調查處|警政署|地檢署|地院|市政府|縣政府"
)
PLACE_SUFFIX = r"公園|夜市|市場|港口|碼頭|交流道|車站|捷運站|客運站|港|宮|寺|廟"
# Trim narrative before a concrete name; do not mask the narrative itself.
ENTITY_BOUNDARY = re.compile(
    r"(?:位於|址設|設於|行經|途經|行至|駛至|右轉|左轉|返回|前往|"
    r"名為|稱為|名叫|下稱|簡稱|所申辦|所申設|所開立|申辦|申設|"
    r"任職|擔任|交付予|交予|交給|移送|報告|報請|移請|報由|訴由|"
    r"取得|使用|利用|容任|容認|告知|遵守|匯至|匯入|轉入|送至|寄至|"
    r"提領|領取|上揭|上開|本案|系爭|該|將|成員|的|"
    r"之|在|於|至|向|由|經|往|與|及|暨|即|以|沿|從|係)"
)
GENERIC_ENTITIES = {
    "公司", "該公司", "本公司", "各公司", "其他公司", "人頭公司", "金融機構",
    "銀行", "該銀行", "網路銀行", "商業銀行", "便利商店", "超商", "旅館",
    "醫院", "地方法院", "法院", "本院", "警察局", "地檢署", "檢察署",
    "道路", "公路", "高速公路", "同路", "上路", "上開道路", "同市", "同縣",
    "其他地區", "地區", "上開地區", "派出所", "警察分局", "檳榔攤",
    "郵局", "股份有限公司", "有限公司", "金融機構", "行政院", "銀行帳戶",
    "網路", "網際網路", "交岔路", "交叉路", "柏油路", "水泥路", "該路",
    "上開路", "同路", "各路", "路", "市", "區", "縣", "臺灣地區", "台灣地區", "大陸地區",
    "該巷", "上開巷", "其他用路", "用路", "同街", "該街", "犯罪市場", "交易市場",
    "發卡銀行", "各臺灣公司", "臺灣公司", "金融帳戶網路銀行", "2個銀行",
}
NARRATIVE_WORDS = ("詐欺", "飲酒", "駕車", "犯罪", "金額", "犯意", "取得", "使用", "告知", "所有", "不能", "規定", "遵守")


@dataclass(frozen=True)
class MaskSpan:
    start: int
    end: int
    category: str
    text: str


@dataclass
class MaskResult:
    text: str
    counts: dict
    spans: list


def matching_view(text):
    """Return whitespace-free matching text and its original character offsets."""
    offsets = [i for i, char in enumerate(text) if not char.isspace()]
    return "".join(text[i].translate(FULLWIDTH) for i in offsets), offsets


def read_rows(path):
    """Read exactly two columns without treating the first case as a header."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    for number, row in enumerate(rows, 1):
        if len(row) != 2 or not row[0] or not row[1]:
            raise ValueError(f"Expected nonempty (label, text) at CSV record {number}: {path}")
    return rows


class TaiwanMasker:
    """Context rules plus document-local entity propagation and optional NER."""

    def __init__(self, mask_token="[MASK]", ner_model=None):
        if not mask_token or any(char.isspace() for char in mask_token):
            raise ValueError("mask_token must be a nonempty token without whitespace.")
        self.mask_token = mask_token
        self.nlp = None
        if ner_model:
            try:
                import spacy
            except ImportError as exc:
                raise RuntimeError("--ner_model requires spaCy and a Chinese NER model.") from exc
            self.nlp = spacy.load(str(ner_model))
            if "ner" not in self.nlp.pipe_names:
                raise ValueError("The selected spaCy model has no NER component.")

        calendar_year = r"(?:[0-9]{2,4}|[零〇○Ｏ一二三四五六七八九十百千壹貳參肆伍陸柒捌玖拾]{2,8})"
        year = rf"(?:民國(?:[（(]下同[）)])?)?{calendar_year}年(?!度)(?:間)?"
        month = rf"{NUM}月(?:上旬|中旬|下旬|間)?"
        day = rf"{NUM}日"
        period = r"(?:凌晨|清晨|早上|上午|中午|下午|傍晚|晚間|晚上|夜間)"
        clock_number = rf"(?:{period}{NUM}|[0-9○〇Ｏ]+)"
        clock = rf"{clock_number}(?:、{NUM})*(?:時|點)(?:{NUM}分)?(?:{NUM}秒)?(?:半|許|左右)?"
        self.time_patterns = [re.compile(pattern) for pattern in (
            rf"(?:民國)?{calendar_year}(?:[、,及至]{calendar_year})+年(?:間)?",
            rf"{year}(?:{month}(?:{day})?)?(?:某日)?(?:{clock}|某時(?:許)?)?",
            rf"(?:同年|翌年|次年|今年|去年|當年){month}(?:{day})?(?:{clock})?",
            rf"(?:同月|翌月|次月){day}(?:{clock})?",
            rf"(?:同日|翌日|次日|當日|當天|隔日|隔天)(?:{clock}|(?:上午|下午|凌晨|晚間|晚上|清晨|傍晚))?",
            rf"{month}{day}(?:{clock})?",
            clock,
            r"(?<![0-9])[0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?(?:許|左右)?",
            r"(?<![0-9])[0-9]{2,4}[/-][0-9]{1,2}[/-][0-9]{1,2}(?![0-9])",
            r"(?:凌晨|清晨|早上|上午|中午|下午|傍晚|晚間|晚上)(?:某時(?:許)?|時分)?",
        )]
        code = r"[A-Za-z0-9○〇Ｏ＊*][A-Za-z0-9○〇Ｏ＊*\-./]{1,40}"
        phone = r"(?:0\d{1,3}[\-]\d{6,10}|09\d{2}[\-]?\d{3}[\-]?\d{3}|09[\-]\d{3}[\-]\d{3}[\-]\d{3})"
        latin_value = r"[A-Za-z0-9][A-Za-z0-9._@\-_]{1,40}"
        self.id_patterns = [re.compile(pattern) for pattern in (
            rf"{NUM}年度[{HAN}㈠㈡㈢㈣㈤()（）]{{1,12}}字第{NUM}(?:[、,及]{NUM})*號",
            r"(?<![A-Za-z0-9-])(?=[A-Za-z0-9○〇Ｏ-]*[0-9○〇Ｏ])[A-Za-z0-9○〇Ｏ]{2,4}-[A-Za-z0-9○〇Ｏ]{2,4}(?![A-Za-z0-9-])",
            rf"(?:車牌(?:號碼|號)?|牌照(?:號碼)?|帳號|賬號|账号|帳戶號碼|賬戶號碼|"
            rf"門號|電話號碼|手機號碼|身分證(?:統一)?(?:字號|號碼)|身份證(?:字號|號碼)|"
            rf"護照號碼|信用卡號|卡號|IMEI(?:碼)?|引擎號碼|車身號碼|序號|番號|案號|"
            rf"統一編號|病歷號碼|會員編號|會員帳號|訂單編號|交易(?:序號|編號)|"
            rf"居留證(?:統一)?(?:字號|號碼)|駕照(?:字號|號碼)|保單號碼)"
            rf"[：:]?(?:為|係)?(?P<entity>{code})",
            rf"(?:市內電話|聯絡電話|傳真)(?:號碼)?[：:]?(?:為|係)?(?P<entity>{phone})",
            rf"(?<![0-9])(?P<entity>{phone})(?![0-9])",
            r"(?<![0-9A-Za-z])[A-Z][12][0-9○]{8}(?![0-9A-Za-z])",
            r"(?<![A-Za-z0-9])(?P<entity>[0-9○〇Ｏ]{4,20}-[0-9○〇Ｏ]{4,20})(?![A-Za-z0-9])",
            r"(?<![0-9A-Za-z.])[0-9○]{8,20}(?![0-9A-Za-z.元萬億年月日條項款目%公毫克噸升立方])",
            r"(?<![A-Za-z0-9])[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
            r"https?://[^，。；、\s）)」]+",
            r"(?<![@A-Za-z0-9])(?P<entity>(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,62})?\.)+[A-Za-z]{2,})(?![A-Za-z0-9])",
            r"(?<![0-9A-Fa-f])(?P<entity>(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2})(?![0-9A-Fa-f])",
            r"(?<![0-9])(?P<entity>(?:[0-9]{1,3}\.){3}[0-9]{1,3})(?![0-9])",
            r"(?:LINE|Line|line|微信|WeChat|臉書|Facebook|Instagram)(?:帳號|账号|ID|暱稱)[：:]?[「\"](?P<entity>[^」\"]{1,40})[」\"]",
            rf"(?:LINE|Line|line|微信|WeChat|臉書|Facebook|Instagram|Telegram)(?:帳號|账号|ID|暱稱|用戶名|使用者名稱)[：:]?(?:為|係)?(?P<entity>{latin_value})(?![A-Za-z0-9._@\-])",
            rf"[「\"](?P<entity>[{ENTITY_CHARS}]{{1,40}})[」\"](?:網站|平台|APP|App|app|應用程式)",
            rf"(?P<entity>[A-Za-z][A-Za-z0-9._\-]{{1,40}})(?=平台|APP|App|app)",
            rf"(?:密碼|PIN碼?|OTP|驗證碼|安全碼|CVV|CVC)[：:]?(?:為|係)?(?P<entity>{latin_value})",
            r"(?:密碼|PIN碼?|OTP|驗證碼|安全碼|CVV|CVC)[：:]?(?:為|係)?[「\"](?P<entity>[^」\"]{1,40})[」\"]",
            rf"(?P<entity>[{HAN}○〇Ｏ0-9]{{1,20}}(?:段|小段)[{HAN}○〇Ｏ0-9之\-]{{1,20}}地號)",
            rf"(?P<entity>[{HAN}○〇Ｏ]*[0-9○〇Ｏ][{HAN}○〇Ｏ0-9之\-]{{1,19}}(?:地號|建號))",
            rf"(?P<entity>(?:[A-Za-z][A-Za-z0-9._\-]{{1,30}}|[{HAN}]{{2,12}}))(?=廠牌|品牌)",
            rf"(?:廠牌|品牌|型號|型式|機型)[：:](?P<entity>{latin_value})",
            r"(?:顏色|色澤)[：:]?(?P<entity>(?:深|淺)?(?:黑|白|紅|藍|綠|黃|灰|銀|金|粉紅|紫|棕|橘|咖啡)色?)",
            rf"[「\"](?P<entity>[{ENTITY_CHARS}]{{1,12}}{NUM}號)[」\"](?:抽砂船|漁船|船|艇)",
        )]
        self.location_patterns = [re.compile(pattern) for pattern in (
            rf"{REGION}(?:{ADMIN})*(?:{ROAD})?{ADDRESS_TAIL}",
            rf"[○〇Ｏ]{{1,3}}(?:縣|市|區|鄉|鎮|村|里)(?:{ADMIN})*(?:{ROAD})?{ADDRESS_TAIL}",
            rf"(?:在|位於|至|往|沿|返回|行經|即)(?P<entity>[{HAN}]{{2,5}}(?:鄉|鎮|區)(?:{ROAD})?{ADDRESS_TAIL})",
            rf"{ROAD}{ADDRESS_TAIL}",
            rf"(?:國道|省道|縣道|市道|台|臺|投|嘉|屏){NUM}(?:號|線)?(?:公路|道路|高速公路)?(?:{NUM}公里(?:{NUM}公尺)?)?",
            rf"[「\"](?P<entity>[{ENTITY_CHARS}]{{1,30}}(?:{PLACE_SUFFIX}))[」\"]",
        )]
        self.org_pattern = re.compile(rf"[{ENTITY_CHARS}]{{1,32}}?(?:{ORG_SUFFIX})(?:[{HAN}○]{{1,6}}?(?:分行|支行|分局|派出所|分院|分公司|分店))?")
        self.specific_org_pattern = re.compile(r"衛生福利部|法務部(?:調查局)?|內政部(?:警政署)?|交通部|教育部|經濟部|財政部|國防部|勞動部|行政院|司法院|立法院|監察院|考試院")
        self.quoted_org_pattern = re.compile(rf"[「\"](?P<entity>[^」\"]{{1,30}})[」\"](?:{ORG_SUFFIX})")
        self.place_pattern = re.compile(rf"[{ENTITY_CHARS}]{{1,24}}?(?:{PLACE_SUFFIX})")
        self.environment_patterns = [re.compile(pattern) for pattern in (
            r"(?:天候|天氣)(?:為|係|是)?(?:晴朗|晴天|晴|陰天|陰|雨天|雨|有雨|下雨|多雲|颱風|霧|雪)",
            r"(?:白天|日間|夜間)(?:自然光線|自然光照|有照明|無照明|照明充足|光線充足|光線不足)",
            r"(?:有|無)?(?:日間)?自然光(?:線|照)",
            r"(?:柏油|水泥|瀝青)?路面(?:鋪裝柏油|鋪有柏油|舖裝柏油|鋪裝水泥|乾燥|濕滑|濕潤|泥濘|積水|柏油|水泥)?(?:無缺陷)?",
            r"(?:柏油|水泥)(?:乾燥|濕潤|濕滑)(?:無缺陷)?路面",
            r"視距(?:亦|為|非|不|均)?(?:良好|不良|清晰|不佳|有限)",
            r"(?:光線|照明)(?:良好|充足|不足|昏暗)|(?:有|無)照明",
            r"(?:無|沒有|有)(?:缺陷|障礙物)|(?:乾燥|濕潤|濕滑)",
            r"晴天|雨天|日間|夜間|路況(?:均)?正常",
        )]
        self.person_patterns = [re.compile(pattern) for pattern in (
            rf"(?P<entity>{PERSON})(?={PERSON_STRONG_ACTION})",
            rf"(?:{PERSON_CONTEXT})(?P<entity>{PERSON_LONG})(?=[、，。；：（）()「」]|{PERSON_ACTION})",
            rf"(?:事實|犯罪事實)?[一二三四五六七八九十]、(?:緣)?(?P<entity>{PERSON_LONG})(?=[、，：（(]|{PERSON_ACTION})",
            r"(?P<entity>[甲乙丙丁戊己庚辛壬癸][○〇ＯO]{1,3})",
            rf"(?P<entity>[{SURNAMES}][○〇ＯO]{{1,3}})",
            r"(?<![A-Za-z0-9])(?P<entity>[A-Z][0-9]{1,2})(?![A-Za-z0-9])",
            r"(?:綽號|暱稱|化名|自稱|名為|名叫|姓名)[：:]?[「\"](?P<entity>[^」\"]{1,24})[」\"]",
            r"(?:被告|告訴人|被害人|證人|外籍勞工|姓名|名為|名叫)(?P<entity>[A-Z][A-Za-z.\-]{2,30})(?=[，、。（(]|申辦|於|前|所|之|與|為)",
        )]

    @staticmethod
    def _valid_person(name):
        if re.fullmatch(REGION, name):
            return False
        if name[-1:] in "之於與及所係為在將以經向前後因則等均仍又並而":
            return False
        return not any(word in name for word in PERSON_EXCLUDE)

    @staticmethod
    def _trim_entity(value, start, known_names=()):
        region = re.search(REGION, value)
        if region and not value.startswith(("臺灣", "台灣", "福建")):
            start += region.start()
            value = value[region.start():]
            boundaries = []
        elif value.startswith(("臺灣", "台灣", "福建")):
            boundaries = []
        else:
            boundaries = [match for match in ENTITY_BOUNDARY.finditer(value) if len(value[match.end():]) >= 3]
        if boundaries:
            cut = boundaries[-1].end()
            start += cut
            value = value[cut:]
        prefix = re.match(r"(?:上址|上開|其|該)+", value)
        if prefix:
            start += prefix.end()
            value = value[prefix.end():]
        for name in sorted(known_names, key=len, reverse=True):
            if value.startswith(name) and len(value) > len(name):
                start += len(name)
                value = value[len(name):]
                break
        return value, start

    def mask(self, text):
        compact, offsets = matching_view(text)
        candidates = []
        known_names = set()
        mask_literal = self.mask_token.translate(FULLWIDTH)
        mask_core = mask_literal.strip("[]")
        duration = rf"(?:{NUM}年(?:{NUM}(?:個)?月)?(?:{NUM}日)?|{NUM}(?:個)?月(?:{NUM}日)?|{NUM}日)"
        protected_times = [match.span() for match in re.finditer(
            rf"(?:有期徒刑|徒刑|緩刑|應執行(?:刑)?|刑期){duration}(?:[、，及]{duration})*|(?:中華)?民國{NUM}年罪犯減刑條例",
            compact,
        )]

        def add(start, end, category):
            if start < end and mask_literal not in compact[start:end] and compact[start:end] != mask_core:
                candidates.append((start, end, category))

        def add_pattern(pattern, category):
            for match in pattern.finditer(compact):
                start, end = match.span("entity") if "entity" in pattern.groupindex else match.span()
                if category == "time":
                    if any(start < protected_end and end > protected_start for protected_start, protected_end in protected_times):
                        continue
                    if re.search(r"(?:中華)?民國" + NUM + r"年罪犯減刑條例", compact[max(0, start - 2):end + 10]):
                        continue
                    if re.search(r"(?:有期徒刑|徒刑|刑期|緩刑|執行刑|判處|期間|歷經|近|過去)$", compact[max(0, start - 10):start]):
                        continue
                    suffix = compact[end:end + 1]
                    if re.fullmatch(r"[0-9零〇○Ｏ一二三四五六七八九十百千壹貳參肆伍陸柒捌玖拾]+年(?:間)?", compact[start:end]) and suffix and suffix in "內後前餘":
                        continue
                if category == "identifier" and "entity" not in pattern.groupindex:
                    if not any(char.isdigit() or char in "○〇Ｏ" for char in compact[start:end]) and "@" not in match.group() and not match.group().startswith("http"):
                        continue
                if category == "identifier" and compact[start:end] in {"SIM", "IMEI", "LINE"}:
                    continue
                add(start, end, category)

        # Explicit and context-supported people are reused only within this case.
        for pattern in self.person_patterns:
            for match in pattern.finditer(compact):
                name = match.group("entity")
                if mask_literal not in name and name != mask_core and self._valid_person(name):
                    known_names.add(name)

        if self.nlp:
            self.nlp.max_length = max(self.nlp.max_length, len(compact) + 1)
            for entity in self.nlp(compact).ents:
                category = {"PERSON": "name", "PER": "name", "GPE": "location", "LOC": "location", "ORG": "organization"}.get(entity.label_)
                if category == "name":
                    known_names.add(entity.text)
                elif category and entity.text not in GENERIC_ENTITIES:
                    add(entity.start_char, entity.end_char, category)

        # Extend adjacent lists such as 許來明、許文理、張太福、陸建軍、隆福琼.
        name_list = re.compile(rf"(?:{PERSON}[、與及])+(?:{PERSON})")
        for match in name_list.finditer(compact):
            names = re.split(r"[、與及]", match.group())
            if any(name in known_names for name in names) or (len(names) >= 3 and all(self._valid_person(name) for name in names)):
                known_names.update(name for name in names if self._valid_person(name))

        # A shorter accidental prefix (嚴榮 in 嚴榮基於...) must not truncate an
        # independently identified full name elsewhere in the same case.
        known_names = {
            name for name in known_names
            if mask_literal not in name
            and name != mask_core
            and not any(other != name and other.startswith(name) for other in known_names)
        }

        for name in sorted(known_names, key=len, reverse=True):
            for match in re.finditer(re.escape(name), compact):
                add(*match.span(), "name")
        for category, patterns in (("identifier", self.id_patterns), ("time", self.time_patterns), ("environment", self.environment_patterns)):
            for pattern in patterns:
                add_pattern(pattern, category)

        for pattern in self.location_patterns:
            for match in pattern.finditer(compact):
                start, end = match.span("entity") if "entity" in pattern.groupindex else match.span()
                value, start = self._trim_entity(compact[start:end], start, known_names)
                if len(value) >= 2 and value not in GENERIC_ENTITIES and not any(word in value for word in ("網路", "交岔", "交叉", "柏油", "水泥", "號誌", "行駛", "上開", "支線", "幹線", "限制線", "分向", "道車") + NARRATIVE_WORDS):
                    add(start, end, "location")

        organizations = set()
        for pattern in (self.specific_org_pattern, self.quoted_org_pattern):
            for match in pattern.finditer(compact):
                start, end = match.span("entity") if "entity" in pattern.groupindex else match.span()
                if compact[start:end] not in GENERIC_ENTITIES:
                    add(start, end, "organization")
                    organizations.add(compact[start:end])
        for category, pattern in (("organization", self.org_pattern), ("location", self.place_pattern)):
            for match in pattern.finditer(compact):
                value, start = self._trim_entity(match.group(), match.start(), known_names)
                if value and value not in GENERIC_ENTITIES and len(value) >= 2:
                    # Do not interpret laws/abstract concepts as institutions/places.
                    if value.startswith(("依", "對", "有", "無", "各", "該", "本", "其他")):
                        continue
                    if any(word in value for word in NARRATIVE_WORDS):
                        continue
                    add(start, match.end(), category)
                    if category == "organization":
                        organizations.add(value)
        # Propagate full institution names; abbreviations with a suffix are also
        # discovered above (e.g. 順然公司), without replacing generic 公司/銀行.
        for organization in organizations:
            for match in re.finditer(re.escape(organization), compact):
                add(*match.span(), "organization")
        aliases = set()
        for organization in organizations:
            alias_pattern = re.compile(re.escape(organization) + r"[」\"]?(?:[（(]|，)(?:以下簡稱|下稱|簡稱|下簡稱)[「\"]?(?P<entity>[^」\"）)，、；]{1,20})")
            for match in alias_pattern.finditer(compact):
                alias = match.group("entity")
                if alias not in GENERIC_ENTITIES and not alias.startswith(("本", "該", "上開")):
                    aliases.add(alias)
        for alias in aliases:
            for match in re.finditer(re.escape(alias), compact):
                add(*match.span(), "organization")

        # Prefer identifiers over their embedded date, and names of institutions
        # over embedded administrative place names. Nested matches occur once.
        priority = {category: i for i, category in enumerate(("identifier", "organization", "name", "location", "time", "environment"))}
        selected = []
        for start, end, category in sorted(candidates, key=lambda item: (priority[item[2]], -(item[1] - item[0]), item[0])):
            if not any(start < previous_end and end > previous_start for previous_start, previous_end, _ in selected):
                selected.append((start, end, category))
        selected.sort()
        spans = [MaskSpan(offsets[start], offsets[end - 1] + 1, category, text[offsets[start]:offsets[end - 1] + 1]) for start, end, category in selected]
        pieces = []
        position = 0
        for span in spans:
            pieces.extend((text[position:span.start], self.mask_token))
            position = span.end
        pieces.append(text[position:])
        counts = Counter(span.category for span in spans)
        return MaskResult("".join(pieces), {category: counts[category] for category in CATEGORIES}, spans)


def mask_text(text, mask_token="[MASK]", ner_model=None):
    """Convenience API. Reuse TaiwanMasker for batch processing."""
    return TaiwanMasker(mask_token, ner_model).mask(text).text


def mask_rows(rows, masker):
    masked = []
    totals = Counter()
    changed = 0
    for label, text in rows:
        result = masker.mask(text)
        masked.append([label, result.text])
        totals.update(result.counts)
        changed += result.text != text
    return masked, {"rows": len(rows), "changed_rows": changed, "counts": {category: totals[category] for category in CATEGORIES}}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_path", "--test_data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_path", type=Path, help="Default: <input_stem>_mask.csv beside input.")
    parser.add_argument("--mask_token", default="[MASK]")
    parser.add_argument("--ner_model", help="Optional installed spaCy model name or local model directory; no automatic download.")
    parser.add_argument("--stats_path", type=Path, help="Optional JSON category counts.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    output_path = args.output_path or args.input_path.with_name(args.input_path.stem + "_mask.csv")
    if output_path.resolve() == args.input_path.resolve():
        raise ValueError("Input and output paths must differ.")
    if args.stats_path and args.stats_path.resolve() in (args.input_path.resolve(), output_path.resolve()):
        raise ValueError("stats_path must differ from both CSV paths.")
    rows = read_rows(args.input_path)
    masked, stats = mask_rows(rows, TaiwanMasker(args.mask_token, args.ner_model))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerows(masked)
    if args.stats_path:
        args.stats_path.parent.mkdir(parents=True, exist_ok=True)
        args.stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Masked CSV: {output_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


if __name__ == "__main__":
    main()
