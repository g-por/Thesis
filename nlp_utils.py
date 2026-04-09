import re
from typing import List


_TOKEN_REGEX = r"[\w'’]+"


def normalize_spaces(text: str) -> str:
    """Нормалізує пробіли: замінює послідовності пробілів на один та обрізає краї.

    Безпечна до None: у такому разі повертає порожній рядок.
    """
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_tokens(text: str) -> List[str]:
    """Токенізує рядок на слова для українських/латинських текстів.

    Повертає слова у нижньому регістрі, використовуючи спільний
    регулярний вираз для \w та апострофів.
    """
    return re.findall(_TOKEN_REGEX, (text or "").lower())
