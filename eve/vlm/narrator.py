"""narrator — 複数フレームを1回の VLM 呼び出しで実況(VisionResult)化する。

`vlm_leaf`(既定 gemini-2.5-flash) に **時系列の連続フレーム**（古い順・最後が今）を base64 JPEG の
multimodal message で渡し、タグ付きテキストを `parse_vision` で頑健に読む。`narrate_fn` を注入できる
（テストは litellm 抜き）。ハルシネ防止(A11): プロンプトで「黒/判読不能なら推測せず visible: no」。
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable

from .parser import parse_vision
from .types import Frame, VisionResult

logger = logging.getLogger(__name__)

NarrateFn = Callable[[list[Frame]], Awaitable[VisionResult]]

VLM_SYSTEM = (
    "あなたはユーザのPC画面を見ている『目』です。渡される複数枚の画像は時系列の連続スクリーン"
    "ショット（古い順・**最後の1枚が今この瞬間**）。最後を『今』とし、**直前のフレームと見比べて**"
    "画面で何が起きているか・どこが変わった/動いたかを日本語で簡潔に1〜2文で述べてください。\n"
    "**変化が無くても、今画面に表示されている主要なもの（最前面のアプリ/ファイル/内容）は必ず一言入れる**"
    "（『変化なし』だけで終わらせない＝後で『今何が見える?』に答えられるように）。\n"
    "複数ウィンドウが開いているのは普通です。最前面/アクティブを主にしつつ、必要なら背景も踏まえる。\n"
    "**確信が持てない所は決めつけない**: アプリ名・ファイル名・読みづらい文字は、不確かなら名指しせず"
    "『〜のようなエディタ』等にぼかし、どこがどう分からないかを正直に言う（捏造しない）。\n"
    "画面が真っ黒・判読不能・取得できない場合は visible: no と返す。\n"
    "出力は次のタグのみ:\n"
    "narration: <実況を1〜2文>\n"
    "notable: <yes/no＝今わざわざ**声をかける**価値がある変化か。**中身が新しくなった時だけ yes**。"
    "同じ内容の移動/リサイズ/スクロール・カーソルや選択の移動・時計/進捗/カウンタの更新のような"
    "『位置や見た目が変わっただけ』は no>\n"
    "surprise: <0-100＝直前の予想からの外れ具合>\n"
    "visible: <yes/no＝画面の内容を読み取れたか>"
)


def build_messages(frames: list[Frame]) -> list[dict]:
    """litellm multimodal messages（OpenAI 形式; gemini 経路で inline_data に変換される）。"""
    content: list[dict] = [{
        "type": "text",
        "text": f"連続スクリーンショット {len(frames)} 枚（古い順・最後が今）。今の画面を実況して。",
    }]
    for f in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{f.jpeg_b64}"},
        })
    return [
        {"role": "system", "content": VLM_SYSTEM},
        {"role": "user", "content": content},
    ]


def _content_of(resp: object) -> str:
    """litellm 応答からテキストを頑健に取り出す（壊れ形は空文字）。"""
    try:
        return resp.choices[0].message.content or ""  # type: ignore[attr-defined]
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""


def make_narrate_fn(registry) -> NarrateFn:
    """ModelRegistry の `vlm_leaf` を叩く本番 narrate_fn。"""
    async def narrate(frames: list[Frame]) -> VisionResult:
        resp = await registry.complete("vlm_leaf", build_messages(frames))
        return parse_vision(_content_of(resp))

    return narrate
