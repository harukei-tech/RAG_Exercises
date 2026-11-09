"""
RAG研修 共通ユーティリティ

このファイルには、各ステップのスクリプトから共通で使う関数をまとめています。
・ドキュメントの読み込み
・チャンク分割（テキストを検索しやすい大きさに分割する処理）

研修中は中身を読んで「何をしているか」を理解することが目的です。
本番システムではより高度なライブラリ（LangChain, LlamaIndex 等）を
使うこともできますが、まずは仕組みを手で書いて理解します。
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field


SAMPLE_DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")


@dataclass
class Chunk:
    """検索の最小単位となる、分割済みテキストの断片"""

    id: str                 # チャンクの一意なID（例: "01_helpdesk_faq.md::0"）
    text: str                # チャンク本文
    source_file: str         # 元ファイル名
    metadata: dict = field(default_factory=dict)


def load_documents(directory: str = SAMPLE_DOCS_DIR) -> dict[str, str]:
    """指定ディレクトリ内の .md ファイルをすべて読み込み、
    {ファイル名: 本文} の辞書を返す。

    実務では PDF・Word・Confluence・Notion など様々な形式を読み込む
    「ローダー」が必要になりますが、まずはシンプルなテキスト読み込みから始めます。
    """
    documents: dict[str, str] = {}
    paths = sorted(glob.glob(os.path.join(directory, "*.md")))
    if not paths:
        raise FileNotFoundError(
            f"{directory} に .md ファイルが見つかりませんでした。"
            "sample_docs ディレクトリを確認してください。"
        )
    for path in paths:
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            documents[filename] = f.read()
    return documents


def split_into_chunks(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
) -> list[str]:
    """テキストを chunk_size 文字程度のまとまりに分割する。

    ポイント:
    - まず「見出し（#）」や空行を区切りとして段落に分ける
    - 段落を chunk_size を超えないように積み上げていく
    - チャンクの継ぎ目で文脈が切れないよう、chunk_overlap 文字分を
      次のチャンクの先頭に重複させる（オーバーラップ）

    これはあくまで学習用の単純な実装です。本番では文の区切り（句点等）を
    考慮したより丁寧な分割や、トークン数ベースの分割を検討してください。
    """
    # 空行2つ以上をひとつの区切りとして段落分割する
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        # 現在のチャンクに段落を足すと chunk_size を超える場合は
        # ここで一旦チャンクを確定させる
        if current and len(current) + len(paragraph) + 1 > chunk_size:
            chunks.append(current.strip())
            # オーバーラップ: 直前チャンクの末尾を次のチャンクの先頭に持ち越す
            overlap_text = current[-chunk_overlap:] if chunk_overlap > 0 else ""
            current = overlap_text
        current = (current + "\n" + paragraph).strip() if current else paragraph

        # 1段落だけで chunk_size を大きく超える場合はそのまま1チャンクにする
        # （学習用のシンプル実装のため、長大な段落の強制分割は行わない）

    if current.strip():
        chunks.append(current.strip())

    return chunks


def build_chunks(directory: str = SAMPLE_DOCS_DIR, chunk_size: int = 400, chunk_overlap: int = 80) -> list[Chunk]:
    """全ドキュメントを読み込み、チャンクのリストに変換する。"""
    documents = load_documents(directory)
    chunks: list[Chunk] = []
    for filename, text in documents.items():
        pieces = split_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    id=f"{filename}::{i}",
                    text=piece,
                    source_file=filename,
                    metadata={"source_file": filename, "chunk_index": i},
                )
            )
    return chunks
