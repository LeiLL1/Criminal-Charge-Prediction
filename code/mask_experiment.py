"""Mask concrete factual identifiers in Chinese case descriptions.

The masking rules target concrete factual identifiers that should normally be
label preserving. Because names, organizations, and addresses are open-ended
Chinese entities, inspect sampled matches before reporting the experiment in a
paper.
"""

import re
from dataclasses import dataclass


MASK_TOKEN = "[MASK]"
DEFAULT_CATEGORIES = (
    "date",
    "time",
    "location",
    "person",
    "organization",
    "product",
    "identifier",
)


@dataclass(frozen=True)
class MaskMatch:
    start: int
    end: int
    category: str
    text: str
    priority: int


# More specific patterns have a lower priority number and win overlaps.
PATTERNS = {
    "identifier": (
        10,
        (
            # Resident identity card numbers, including anonymized digits.
            re.compile(r"(?<![0-9A-Za-z])\d{6}(?:\d|[×xX*]){8,11}[0-9xX×*](?![0-9A-Za-z])"),
            # Bank card/account numbers. Separators are retained inside a match.
            re.compile(r"(?<!\d)(?:\d[ -]?){12,19}(?!\d)"),
            # Chinese vehicle plates, including anonymized plate numbers.
            re.compile(r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][A-Z0-9×xX*]{4,6}"),
            re.compile(r"(?:车牌号|牌照|号牌)(?:为|是|：|:)?[A-Z0-9×xX*\-]{4,12}"),
        ),
    ),
    "date": (
        20,
        (
            re.compile(r"(?:19|20)\d{2}年(?:\d{1,2}月(?:\d{1,2}[日号]?)?)?"),
            re.compile(r"\d{1,2}月\d{1,2}[日号]"),
            re.compile(r"(?:同年|次年|当年|某年)?\d{1,2}月(?:上旬|中旬|下旬|初|底|末|\d{1,2}[日号])"),
            re.compile(r"(?:某日|当日|次日|翌日)(?:凌晨|早晨|上午|中午|下午|傍晚|晚上|夜间)?"),
        ),
    ),
    "time": (
        30,
        (
            re.compile(r"(?:凌晨|早晨|上午|中午|下午|傍晚|晚上|夜间)?\d{1,2}(?:时|点)(?:\d{1,2}分?)?(?:许|左右)?"),
            re.compile(r"\d{1,2}[:：]\d{2}(?:分?)?(?:许|左右)?"),
        ),
    ),
    "organization": (
        40,
        (
            re.compile(
                r"(?<=[，。；：、“”])"
                r"[一-龥A-Za-z（）()·]{2,30}?(?=股份有限公司|有限责任公司|集团有限公司|"
                r"有限公司|分公司|人民检察院|人民法院|公安局|派出所|银行|信用社|"
                r"商店|超市|酒店|宾馆|医院|学校|学院|事务所|委员会)"
            ),
            re.compile(r"(?:中国|某)(?:工商|农业|建设|交通|人民|招商|邮政储蓄)?(?=银行)"),
            re.compile(r"(?<=在)[一-龥A-Za-z（）()·]{2,20}(?=股份有限公司|有限责任公司|集团有限公司|有限公司|分公司)"),
            re.compile(r"(?<=由)[一-龥A-Za-z（）()·]{2,20}(?=股份有限公司|有限责任公司|集团有限公司|有限公司|分公司)"),
            re.compile(r"(?<=向)[一-龥A-Za-z（）()·]{2,20}(?=股份有限公司|有限责任公司|集团有限公司|有限公司|分公司)"),
            re.compile(r"(?<=安排)[一-龥A-Za-z（）()·]{2,20}(?=股份有限公司|有限责任公司|集团有限公司|有限公司|分公司)"),
            re.compile(r"(?<=联系)[一-龥A-Za-z（）()·]{2,20}(?=股份有限公司|有限责任公司|集团有限公司|有限公司|分公司)"),
            re.compile(r"某(?=公司|商店|超市|酒店|宾馆|医院|学校|学院|事务所|委员会)"),
        ),
    ),
    "location": (
        50,
        (
            # Administrative divisions, including chained forms such as
            # "江苏省南京市江宁区".
            re.compile(
                r"(?:北京|天津|上海|重庆|河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|"
                r"山东|河南|湖北|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|台湾|内蒙古|"
                r"广西|西藏|宁夏|新疆|香港|澳门)[省市]?(?:[一-龥]{2,10}(?:自治州|市|区|县|"
                r"自治县|旗|镇|乡|街道)){0,4}"
            ),
            re.compile(r"(?<=在)[一-龥A-Za-z0-9]{2,20}(?:小区|社区|村|屯|巷|弄|大道|路|街)(?:\d{1,4}号)?"),
            re.compile(r"(?<=至)[一-龥A-Za-z0-9]{2,20}(?:小区|社区|村|屯|巷|弄|大道|路|街)(?:\d{1,4}号)?"),
            re.compile(r"(?<=到)[一-龥A-Za-z0-9]{2,20}(?:小区|社区|村|屯|巷|弄|大道|路|街)(?:\d{1,4}号)?"),
            re.compile(r"(?<=[，。；：、])\d{1,4}号(?:楼|栋|幢|单元|室)?"),
        ),
    ),
    "person": (
        60,
        (
            # Keep the legal role and mask only the following name or alias.
            re.compile(r"(?<=被告人)[一-龥·]{1,4}(?:某{1,2}[甲乙丙丁戊己庚辛壬癸]?|[甲乙丙丁戊己庚辛壬癸])"),
            re.compile(r"(?<=被害人)[一-龥·]{1,4}(?:某{1,2}[甲乙丙丁戊己庚辛壬癸]?|[甲乙丙丁戊己庚辛壬癸])"),
            re.compile(r"(?<=证人)[一-龥·]{1,4}(?:某{1,2}[甲乙丙丁戊己庚辛壬癸]?|[甲乙丙丁戊己庚辛壬癸])"),
            re.compile(r"(?<=犯罪嫌疑人)[一-龥·]{1,4}(?:某{1,2}[甲乙丙丁戊己庚辛壬癸]?|[甲乙丙丁戊己庚辛壬癸])"),
            re.compile(r"[一-龥]{1,2}某{1,2}[甲乙丙丁戊己庚辛壬癸]?"),
            re.compile(r"[“\"](?:绰号|外号)?[一-龥A-Za-z0-9]{1,10}[”\"]"),
        ),
    ),
    "product": (
        70,
        (
            re.compile(r"[一-龥A-Za-z0-9]{1,16}牌"),
            re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{1,8}[-/]?[A-Za-z0-9-]{2,18}型?(?![A-Za-z0-9])"),
            re.compile(r"(?:黑色|白色|红色|蓝色|绿色|黄色|灰色|银色|金色|棕色|紫色)(?=[一-龥]{1,8})"),
        ),
    ),
}


def _find_matches(text, categories):
    matches = []
    for category in categories:
        priority, regexes = PATTERNS[category]
        for regex in regexes:
            for match in regex.finditer(text):
                matches.append(
                    MaskMatch(match.start(), match.end(), category, match.group(), priority)
                )

    # Resolve overlaps deterministically: specific category first, then longer span.
    accepted = []
    occupied = [False] * len(text)
    for match in sorted(matches, key=lambda x: (x.priority, -(x.end - x.start), x.start)):
        if any(occupied[match.start : match.end]):
            continue
        accepted.append(match)
        occupied[match.start : match.end] = [True] * (match.end - match.start)
    return sorted(accepted, key=lambda x: x.start)


def mask_text(text, categories=DEFAULT_CATEGORIES):
    """Return masked text and the accepted non-overlapping matches."""
    unknown = set(categories) - set(PATTERNS)
    if unknown:
        raise ValueError(f"Unknown mask categories: {sorted(unknown)}")

    matches = _find_matches(text, categories)
    pieces = []
    cursor = 0
    for match in matches:
        pieces.append(text[cursor : match.start])
        pieces.append(MASK_TOKEN)
        cursor = match.end
    pieces.append(text[cursor:])
    return "".join(pieces), matches
