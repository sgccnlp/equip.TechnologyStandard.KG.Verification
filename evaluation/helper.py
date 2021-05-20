from typing import List, Dict
from rouge import Rouge

import numpy as np
import json

rouge_wo_e = Rouge(exclusive=False)
rouge_w_e = Rouge()

PUNCTIONS = [
    "。", "，", "·", "、", "《", "》", "！", "？", "：", "；", "“", "”", "‘", "’", "（",
    "）", "「", "」", "【", "】", "¥", "…", "｜", "～", "!", "@", "#", "$", "^", "&", ";",
    "|", "\\", "?", "`", "~"
]

code_to_index = json.load(open("config/code_to_index.json"))


def do_code_to_index(code):
    ODD_CODE = "NULL0321"
    if not isinstance(code, str):
        return ODD_CODE
    code = code.strip()
    return code_to_index.get(code, ODD_CODE)

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换            
            inside_code = 32 
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring

# def do_subtitle_format(subtitle):
#     return subtitle

def do_subtitle_format(subtitle):
    """对子标题的简单处理

    包括去掉中间的空白字符、全角转半角、去掉非法字符feff、转换中文句号为小数点.

    Args:
        subtitle ([type]): [description]

    Returns:
        [type]: [description]
    """
    if subtitle is None:
        return "none"
    subtitle = str(subtitle)
    r_subtitle = ""
    for x in subtitle:
        # 过滤空白字符
        if x.strip() == "":
            pass
        # 过滤奇怪字符feff
        elif x == "\ufeff":
            pass
        else:
            # 全角转半角
            x = strQ2B(x)
            # 替换句号
            if x == "。":
                x = "."
            r_subtitle += x
    if r_subtitle == "":
        return "none"
    return r_subtitle


def is_chinese(ch):
    if u'\u4e00' <= ch <= u'\u9fff':
        return True
    else:
        return False


def rouge_score(hyp, ref):
    scores_wo_e = rouge_wo_e.get_scores(hyp, ref)[0]
    scores_w_e = rouge_w_e.get_scores(hyp, ref)[0]
    rouge_1 = scores_wo_e['rouge-1']['r']
    rouge_2 = scores_wo_e['rouge-2']['r']
    rouge_l_p = scores_w_e['rouge-l']['p']
    rouge_l_r = scores_w_e['rouge-l']['r']
    rouge_l = 0 if rouge_l_p == 0 and rouge_l_r == 0 else 2 * rouge_l_p * rouge_l_r / (
        rouge_l_p + rouge_l_r)
    return 0.2 * rouge_1 + 0.4 * rouge_2 + 0.4 * rouge_l

def argument_empty_process(string):
    empty_string = "答案为空"
    if string is None or string == "":
        return empty_string
    return str(string)

def preprocess_text(text):
    """在计算rouge前，对text的一些处理

    # TODO
    1. 是否需要对空白字符进行过滤或者转换？例如空白字符统一转换为空格，多个空白字符缩减为一个等等。

    暂定步骤：
    1. 识别所有全角空格，转换为空格
    2. 识别所有全角半角标点符号，并转换为空格（英文的小数点和英文句号不予转换）
        最后不进行转换了。
    3. 多个空格缩减为一个空格。(这一步不需要，因为split()可以自行识别)
    """
    # 其他字符转换为空格
    text = argument_empty_process(text)
    processed_text = ""
    for ch in text:
        if ch in PUNCTIONS or \
            ch == "\u3000":
            processed_text += " "
        else:
            processed_text += ch
    return processed_text


def tokenize(text):
    """
    TODO 中文直接按照字符切词。但是英文字符、数字字符等是否需要按照一定的规则呢？

    只区分中文和其他。其他包括英文，希腊字母，数字等等。
    关于标点的处理，在preprocess中已经处理了。
    """
    # 首先根据空格进行切分
    result_segs: List[str] = ['']
    segs = text.split()
    for seg in segs:
        is_last_chinese_or_new_seg = True
        for ch in seg:
            if is_chinese(ch):
                result_segs.append(ch)
                is_last_chinese_or_new_seg = True
            else:
                if is_last_chinese_or_new_seg:
                    result_segs.append(ch)
                else:
                    result_segs[-1] += ch
                is_last_chinese_or_new_seg = False
    result_segs = [seg for seg in result_segs if seg != ""]
    if result_segs == [" "] or result_segs == [] or result_segs == [""]:
        return [" "]
    return result_segs


def snippet_projection(hyp: List[str], ref: List[str]) -> Dict[str, List]:
    """比较两个字符串的相似度，并返回对照结果。

    返回的结果，将使的相同的子序列尽量向前靠。例如：
    hyp: abbcd, ref: abcdd 
    matched_hyp_indices: [0, 1, 3, 4]. (而不是[0, 2, 3, 4])
    matched_ref_indices: [0, 1, 2, 3]. (而不是[0, 1, 2, 4])

    Args:
        hyp (List[str]): [description]
        ref (List[str]): [description]

    Returns:
        Dict[str, List]: [description]
    """
    n_hyp, n_ref = len(hyp), len(ref)
    matrix = np.zeros((n_hyp + 1, n_ref + 1), )
    i, j = 1, 1
    while i <= n_hyp:
        j = 1
        while j <= n_ref:
            matrix[i, j] = max(matrix[i - 1, j], matrix[i, j-1]) \
                if hyp[i - 1] != ref[j - 1] else matrix[i - 1, j - 1] + 1
            j += 1
        i += 1

    # 找到hyp序列，优先减横坐标（向左移）
    hyp_seq = []
    i, j = len(matrix) - 1, len(matrix[0]) - 1
    while i > 0 and j > 0:
        if matrix[i][j] == matrix[i - 1][j]:
            i = i - 1
        elif matrix[i][j] == matrix[i][j - 1]:
            j = j - 1
        else:
            hyp_seq.append(i - 1)
            i = i - 1
            j = j - 1
    # 找到ref序列，优先减纵坐标（向上移）
    ref_seq = []
    i, j = len(matrix) - 1, len(matrix[0]) - 1
    while i > 0 and j > 0:
        if matrix[i][j] == matrix[i][j - 1]:
            j = j - 1
        elif matrix[i][j] == matrix[i - 1][j]:
            i = i - 1
        else:
            ref_seq.append(j - 1)
            i = i - 1
            j = j - 1
    return int(matrix[-1, -1]), hyp_seq[::-1], ref_seq[::-1]


if __name__ == "__main__":
    s1 = "1 绝缘油试验或SF6 气体试验;2 测量绕组连同套管的直流电阻;3 检查所有分接的电压比;4 检查变压器的二相接线组别和单相变压器引出线的极性;5 测量铁心及夹件的绝缘电阻;6 非纯瓷套管的试验;7 有载调压切换装置的检查和试验;8 测量绕组连同套管的绝缘电阻、吸收比或极化指数;9 测量绕组连同套管的介质损耗因数( tanO\')与电容量;10 变压器绕组变形试验;11 绕组连同套管的交流耐压试验;12 绕组连同套管的长时感应耐压试验带局部放电测量;13 额定电压下的冲击合闸试验;14 检查相位;"
    s2 = "1 绝缘油试验或SF6 气体试验;2 测量绕组连同套管的直流电阻;3 检查所有分"
    s3 = "1 绝缘油试验或SF6 气体试验;2 测量绕组连同套管的直流电阻;3 检查所有分接的电压比;4 检查变压器的二"
    print(rouge_score(s1, s2))
    print(rouge_score(s1, s3))

    a = list("abaebced")
    b = list("abcbdaeadd")
    print(snippet_projection(a, b))