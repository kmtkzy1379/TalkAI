"""ConversationCache — 短期記憶（会話ログの保持・永続化・注入セレクション）。

v1 `modules/conversation_cache.py` の機構を PORT しつつ v2 化:
- データモデルを「話者単位の1発話」= `Turn(speaker, text, stamp)` に統一（v1 の user/ai ペアでなく）。
  v2 は autonomous_speech / vision / callfunction などユーザ発話を伴わない eve 発話があるため。
- ロケット鉛筆方式: `deque(maxlen)` で最新 N 件だけメモリ保持（古いものから押し出す）。
- 永続化: JSONL（1ターン=1行 `{"ts": iso, "speaker": ..., "text": ...}`）を非同期 write-queue で追記。
- **単一 asyncio ループ前提で簡素化**: パイプラインは1ループ上で動き、mic/VAD スレッドは cache に
  触れない（barge-in 通知のみ）。よって v1 の `asyncio.Lock` は不要 → `add_turn` / `recent` /
  `recent_for_injection` はループ上の同期メソッド（C5 の記録を handle / cancel ハンドラから
  await 無しで安全に呼べる＝await-in-cancel 問題を回避）。非同期なのは背景書き込み worker のみ。

注入セレクション（`recent_for_injection`）は「直近N件」をそのまま使わず、ユーザー沈黙中の
自律発話が窓を埋め尽くして**ユーザー最後の発話が押し出される＝文脈が逸れる**のを防ぐ
（最後の user 往復を最低保証 + 非連続なら省略マーカ）。詳細は同メソッド docstring。
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from pathlib import Path
from typing import Optional

from ..clock import Stamp, elapsed_wall, humanize, now_iso, now_mono
from ..config import Config
from ..context_assembler import OMITTED_SPEAKER, Turn

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# 記録対象の話者名（センチネルとの混同防止のため明示）。
SPEAKER_USER = "user"
SPEAKER_EVE = "eve"


class ConversationCache:
    def __init__(
        self,
        history_file: Optional[str] = None,
        max_turns: Optional[int] = None,
        recent_count: Optional[int] = None,
    ) -> None:
        path = history_file or Config.HISTORY_FILE
        # 相対パスはリポジトリ root 基準で解決（cwd 依存をなくす）。
        self.history_path = Path(path)
        if not self.history_path.is_absolute():
            self.history_path = _REPO_ROOT / self.history_path
        self.max_turns = max_turns if max_turns is not None else Config.HISTORY_MAX_TURNS
        self.recent_count = (
            recent_count if recent_count is not None else Config.RECENT_TURN_COUNT
        )
        self._turns: "deque[Turn]" = deque(maxlen=self.max_turns)  # ロケット鉛筆
        # D1: 復元（前セッション）と当セッションの境界。復元分は**注入しない**
        # （短期記憶は起動を跨いで「今の会話」ではない）。`turns_since` 経由の
        # feedback/catch-up は境界を見ない＝B5 no-skip 保証は無傷。
        self._restored_last_iso: Optional[str] = None
        self._restored_count = 0
        self._live_count = 0
        self._write_queue: "asyncio.Queue[Optional[dict]]" = asyncio.Queue()
        self._write_task: Optional[asyncio.Task] = None

    # --- ライフサイクル ---------------------------------------------------

    async def initialize(self) -> None:
        """既存ログを読み込み（あれば）→ 背景書き込み worker を起動。"""
        await self._load()
        self._write_task = asyncio.create_task(self._write_worker())

    async def _load(self) -> None:
        if not self.history_path.exists():
            return

        def _read() -> list[dict]:
            entries: list[dict] = []
            with open(self.history_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # 壊れた行は飛ばす（落とさない）
            return entries

        try:
            entries = await asyncio.to_thread(_read)
        except Exception:
            logger.exception("会話ログの読み込みに失敗（空で続行）: %s", self.history_path)
            return

        for e in entries:
            speaker = e.get("speaker")
            text = e.get("text", "")
            ts = e.get("ts") or now_iso()
            if speaker in (SPEAKER_USER, SPEAKER_EVE) and text:
                # 復元 Turn の mono は前プロセスの値で無意味 → 表示は壁時計(iso)を正とする
                self._turns.append(Turn(speaker, text, Stamp(iso=ts, mono=now_mono())))
        self._restored_count = len(self._turns)
        self._restored_last_iso = self._turns[-1].stamp.iso if self._turns else None
        # 観測性: 復元分が注入対象外であることと、前回いつまで話していたかを1行残す
        # （起動/終了はログに残っていないため、これが唯一の再起動の手掛かりになる）。
        gap = elapsed_wall(self._restored_last_iso) if self._restored_last_iso else 0.0
        logger.info("会話ログを復元: %d ターン（注入対象外・前回の最終発話は %s）",
                    self._restored_count, humanize(gap) if self._restored_last_iso else "なし")

    # --- セッション境界（D1）----------------------------------------------

    def restored_last_iso(self) -> Optional[str]:
        """前セッション最終発話の iso（無ければ None）。system の境界表示に使う。"""
        return self._restored_last_iso

    def _live_turns(self) -> list[Turn]:
        """**当セッションで記録された**ターンだけ（復元分を除く）。

        判定は iso 比較（`add_turn` は `Stamp.now()` を打つので live は境界より必ず新しい）。
        システム時計が巻き戻ると live が境界以下になり得るので、その時は**無フィルタに倒し
        （＝従来挙動）ログを残す**（無言で会話ゼロにして沈黙化させない）。
        """
        if self._restored_last_iso is None:
            return list(self._turns)
        live = [t for t in self._turns if t.stamp.iso > self._restored_last_iso]
        if self._live_count > 0 and not live:
            logger.warning("⏳ セッション境界の判定に失敗（時計の巻き戻り?）→ 復元分も注入する")
            return list(self._turns)
        return live

    async def shutdown(self) -> None:
        """書き込みキューをドレインして worker を止める。"""
        if self._write_task is not None:
            await self._write_queue.put(None)  # 終了シグナル
            try:
                await self._write_task
            except asyncio.CancelledError:
                pass
            self._write_task = None

    # --- 記録（同期・ループ上） -------------------------------------------

    def add_turn(self, speaker: str, text: str) -> None:
        """1ターンをメモリに追加 + 永続化キューへ enqueue（どちらも同期）。

        speaker は SPEAKER_USER / SPEAKER_EVE。空文字は記録しない（喋っていない＝C5）。
        """
        text = (text or "").strip()
        if not text:
            return
        stamp = Stamp.now()
        self._turns.append(Turn(speaker, text, stamp))
        self._live_count += 1
        try:
            self._write_queue.put_nowait(
                {"ts": stamp.iso, "speaker": speaker, "text": text}
            )
        except Exception:
            logger.exception("会話ログの書き込みキュー投入に失敗（メモリ層は記録済み）")

    # --- 参照（同期） -----------------------------------------------------

    def recent(self, n: Optional[int] = None) -> list[Turn]:
        """直近 n ターンの生ログ（マーカ無し）。沈黙判定や RAG 供給などの素材用。"""
        k = n if n is not None else self.recent_count
        turns = list(self._turns)
        return turns[-k:] if k > 0 else []

    def turns_since(self, iso: Optional[str]) -> list[Turn]:
        """watermark(iso・排他)より後の実発話ターンを古い順で返す（FeedbackLLM の span 用）。

        iso=None なら保持中の全ターン。iso 比較は now_iso() 由来の UTC ISO 同士の辞書順
        （全て `+00:00`）で順序が一致する。FeedbackLLM が「前回フィードバック地点〜最新」を
        漏れなくカバーするための入力源（未フィードバックの会話＝記憶喪失を作らない）。
        """
        turns = list(self._turns)
        if not iso:
            return turns
        return [t for t in turns if t.stamp.iso > iso]

    def recent_for_injection(self, window: Optional[int] = None) -> list[Turn]:
        """応答LLM へ注入する直近会話を選ぶ（自律発話で文脈が逸れないように）。

        1. 直近 window 件（tail）に user 発話が含まれれば、そのまま返す
           （＝通常の往復は不変・省略マーカ無し）。
        2. tail が eve だけ（ユーザー沈黙中の自律発話の連続）なら、最後の user 往復
           （user 発話 + 直後の eve 返答）を**最低保証**として先頭に足し、tail と連結する。
        3. アンカー往復と tail が非連続なら、間に**省略マーカ**（OMITTED_SPEAKER の
           センチネル Turn・text=省略件数）を挿入して AI に欠落を正直に伝える。
        4. user 発話が一度も無ければ tail のみ。

        戻り値はそのまま `ContextAssembler.assemble(recent_turns=...)` に渡せる。
        """
        n = window if window is not None else self.recent_count
        if n <= 0:
            return []
        # D1: 復元（前セッション）分は注入しない。実測（オフライン n=8）で、ギャップ注記を
        # 出しても復元会話があると 8/8 で前セッションの依頼を蒸し返し、注入しなければ 0/8。
        # 短期記憶は起動を跨いだ時点で「今の会話」ではない（前セッションの内容は RAG が持つ）。
        turns = self._live_turns()
        if not turns:
            return []
        tail = turns[-n:]
        if any(t.speaker == SPEAKER_USER for t in tail):
            return tail  # 通常: 窓内にユーザー発話あり

        # tail は eve のみ → 最後の user 往復を探してアンカーにする
        last_user_idx = next(
            (i for i in range(len(turns) - 1, -1, -1) if turns[i].speaker == SPEAKER_USER),
            None,
        )
        if last_user_idx is None:
            return tail  # ユーザー発話が一度も無い（起動直後の自律発話のみ等）

        anchor_end = last_user_idx + 1
        if anchor_end < len(turns) and turns[anchor_end].speaker == SPEAKER_EVE:
            anchor_end += 1  # user 発話 + 直後の eve 返答（往復）をアンカーに
        tail_start = len(turns) - n
        if anchor_end >= tail_start:
            # アンカーと tail が連続 → 省略無しで last_user から地続きに返す
            return turns[last_user_idx:]
        omitted = tail_start - anchor_end
        marker = Turn(OMITTED_SPEAKER, str(omitted), Stamp.now())
        return turns[last_user_idx:anchor_end] + [marker] + tail

    # --- 背景書き込み -----------------------------------------------------

    async def _write_worker(self) -> None:
        while True:
            try:
                entry = await self._write_queue.get()
                if entry is None:  # 終了シグナル
                    self._write_queue.task_done()
                    break
                try:
                    await asyncio.to_thread(self._append_line, entry)
                finally:
                    self._write_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:  # 書き込み失敗で worker を殺さない（次のターンは記録継続）
                logger.exception("会話ログの追記に失敗（継続）")

    def _append_line(self, entry: dict) -> None:
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
