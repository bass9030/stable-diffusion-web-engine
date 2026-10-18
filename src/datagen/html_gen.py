"""템플릿 기반 합성 HTML 생성기.

소수의 컴포넌트(제목, 문단, 버튼, 사각형 div, 2~3칼럼 레이아웃)를 무작위로 조합한다.
같은 시드와 설정이면 항상 같은 HTML 문자열을 반환한다 (전역 난수 상태를 쓰지 않음).
출력은 인라인 CSS만 쓰는 압축 형태이며 외부 리소스와 JS를 포함하지 않는다.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

# 3자리 hex 색상. 토큰 수를 줄이려고 짧은 표기만 사용한다.
PALETTE: list[str] = [
    "#e33", "#f80", "#fc0", "#8c3", "#2a6", "#0aa", "#29e", "#36c",
    "#63c", "#c3a", "#933", "#753", "#555", "#999", "#ccc", "#eee",
    "#fde", "#fed", "#ffd", "#efd", "#dfe", "#def", "#dde", "#222",
]
LIGHT_BACKGROUNDS: list[str] = ["#fff", "#f5f5f5", "#eee", "#fde", "#ffd", "#efd", "#def"]

WORDS: list[str] = (
    "home about contact news blog shop cart login sign up welcome hello world simple "
    "fast secure free trial pricing plans team careers support help docs guide start "
    "learn more read today latest update product service feature design build launch "
    "open source cloud data search photo music video travel food health sport"
).split()

SPACINGS: list[int] = [0, 8, 16, 24]


@dataclass(frozen=True)
class GenConfig:
    min_components: int
    max_components: int
    component_weights: dict[str, float]
    min_columns: int
    max_columns: int
    max_column_children: int
    min_paragraph_words: int
    max_paragraph_words: int
    colored_body_prob: float
    # False면 <html> 래퍼를 생략해 토큰 수를 줄인다 (Chromium 파싱 결과는 동일)
    wrap_html: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GenConfig":
        return cls(**d)


def _luminance(hex_color: str) -> float:
    """#rgb / #rrggbb 의 상대 밝기(0~1)를 대략 계산한다."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.299 * r + 0.587 * g + 0.114 * b


def _text_color_for(bg: str) -> str:
    return "#000" if _luminance(bg) > 0.55 else "#fff"


def _words(rng: random.Random, lo: int, hi: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(lo, hi)))


def make_heading(rng: random.Random, cfg: GenConfig) -> str:
    level = rng.choice([1, 2, 3])
    text = _words(rng, 1, 4).capitalize()
    if rng.random() < 0.5:
        return f"<h{level}>{text}</h{level}>"
    return f'<h{level} style="color:{rng.choice(PALETTE)}">{text}</h{level}>'


def make_paragraph(rng: random.Random, cfg: GenConfig) -> str:
    text = _words(rng, cfg.min_paragraph_words, cfg.max_paragraph_words).capitalize() + "."
    if rng.random() < 0.6:
        return f"<p>{text}</p>"
    return f'<p style="color:{rng.choice(PALETTE)}">{text}</p>'


def make_button(rng: random.Random, cfg: GenConfig) -> str:
    bg = rng.choice(PALETTE)
    text = _words(rng, 1, 2).capitalize()
    pad = rng.choice(["4px 8px", "8px 16px", "12px 24px"])
    return (
        f'<button style="background:{bg};color:{_text_color_for(bg)};'
        f'border:0;padding:{pad}">{text}</button>'
    )


def make_box(rng: random.Random, cfg: GenConfig) -> str:
    width = rng.choice(["25%", "50%", "75%", "100%", "64px", "128px", "200px"])
    height = rng.randrange(20, 161, 10)
    return f'<div style="width:{width};height:{height}px;background:{rng.choice(PALETTE)}"></div>'


LEAF_MAKERS: dict[str, Callable[[random.Random, GenConfig], str]] = {
    "heading": make_heading,
    "paragraph": make_paragraph,
    "button": make_button,
    "box": make_box,
}


def make_columns(rng: random.Random, cfg: GenConfig) -> str:
    n_cols = rng.randint(cfg.min_columns, cfg.max_columns)
    gap = rng.choice(SPACINGS)
    cols: list[str] = []
    for _ in range(n_cols):
        bg = rng.choice(PALETTE + LIGHT_BACKGROUNDS)
        children = "".join(
            rng.choice(list(LEAF_MAKERS.values()))(rng, cfg)
            for _ in range(rng.randint(0, cfg.max_column_children))
        )
        cols.append(
            f'<div style="flex:1;background:{bg};color:{_text_color_for(bg)};'
            f'padding:{rng.choice(SPACINGS)}px;min-height:{rng.randrange(40, 201, 20)}px">{children}</div>'
        )
    return f'<div style="display:flex;gap:{gap}px">{"".join(cols)}</div>'


COMPONENT_MAKERS: dict[str, Callable[[random.Random, GenConfig], str]] = {
    **LEAF_MAKERS,
    "columns": make_columns,
}


def generate_html(seed: int, cfg: GenConfig) -> str:
    """시드로 결정되는 HTML 문서 하나를 생성한다."""
    rng = random.Random(seed)
    names = list(cfg.component_weights)
    weights = [cfg.component_weights[n] for n in names]
    n_components = rng.randint(cfg.min_components, cfg.max_components)
    chosen = rng.choices(names, weights=weights, k=n_components)
    body = "".join(COMPONENT_MAKERS[name](rng, cfg) for name in chosen)

    if rng.random() < cfg.colored_body_prob:
        body_bg = rng.choice(PALETTE)
    else:
        body_bg = rng.choice(LIGHT_BACKGROUNDS)
    body_style = (
        f"margin:0;padding:{rng.choice(SPACINGS)}px;"
        f"background:{body_bg};color:{_text_color_for(body_bg)}"
    )
    doc = f'<body style="{body_style}">{body}</body>'
    return f"<html>{doc}</html>" if cfg.wrap_html else doc
