# -*- coding: utf-8 -*-
"""Reasoning 乱码检测器 (MTP draft 污染拦截) — token-salad heuristic.

污染样本特征 (2026-10-06 实测 15715 字符 reasoning 流):
  - 多语系混杂 (Cyrillic/Greek/Kana/Hangul/Arabic 同时出现)
  - CJK 占比极低但长度巨大
  - 符号/生僻 Unicode 密度高
  - 大量代码残片 ("PYEOF", "import json") 无代码块结构
判定为污染时返回 True。
"""
import re

_CJK = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
_CYRILLIC = re.compile(r'[\u0400-\u04ff]')
_GREEK = re.compile(r'[\u0370-\u03ff]')
_KANA = re.compile(r'[\u3040-\u30ff]')
_HANGUL = re.compile(r'[\uac00-\ud7af]')
_ARABIC = re.compile(r'[\u0600-\u06ff]')
_SYMBOL = re.compile(r'[^\w\s\u4e00-\u9fff\u3400-\u4dbf\u0400-\u04ff\u0370-\u03ff\u3040-\u30ff\uac00-\ud7af\u0600-\u06ff]', re.UNICODE)
_WORDISH = re.compile(r'[A-Za-z]{3,}')

_CODE_FRAG = re.compile(r'(PYEOF|PYEOF\n|import json|print\(|curl -s|grep -n)')


def looks_like_token_salad(text: str) -> bool:
    """True = 该 reasoning 文本呈现 token-salad 特征,不应作为最终回复交付。"""
    if not text or len(text) < 200:
        return False  # 短文本不判
    n = len(text)
    cjk = len(_CJK.findall(text)) / n
    scripts = sum(bool(r.search(text)) for r in (_CYRILLIC, _GREEK, _KANA, _HANGUL, _ARABIC))
    sym = len(_SYMBOL.findall(text)) / n
    wordish = len(_WORDISH.findall(text))
    code_frag = len(_CODE_FRAG.findall(text))
    # 规则1: 多语系混杂(≥3) + CJK 低 → 沙拉
    if scripts >= 3 and cjk < 0.15:
        return True
    # 规则2: 符号密度异常 + CJK 低 + 长文本
    if sym > 0.35 and cjk < 0.10 and n > 500:
        return True
    # 规则3: 代码残片高频且无结构(每千字符>2处) + CJK 低
    if code_frag >= max(4, n // 500) and cjk < 0.15:
        return True
    return False
