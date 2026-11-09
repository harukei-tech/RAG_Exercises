"""
ステップ1: ドキュメントの読み込みとチャンク分割

【このスクリプトで学ぶこと】
- RAGの最初のステップ「Indexing（索引作成）」の入口部分
- なぜドキュメントをそのままではなく「チャンク」に分割する必要があるのか

実行方法:
    python step1_load_and_chunk.py
"""

from utils import build_chunks, load_documents


def main() -> None:
    print("=" * 60)
    print("1. ドキュメントを読み込みます")
    print("=" * 60)
    documents = load_documents()
    for filename, text in documents.items():
        print(f"- {filename}  ({len(text)} 文字)")

    print()
    print("=" * 60)
    print("2. チャンクに分割します（1チャンクあたり約400文字、80文字オーバーラップ）")
    print("=" * 60)
    chunks = build_chunks(chunk_size=400, chunk_overlap=80)
    print(f"合計 {len(chunks)} 個のチャンクができました。\n")

    print("--- 最初の3チャンクの中身を確認してみましょう ---\n")
    for chunk in chunks[:3]:
        print(f"[{chunk.id}]")
        print(chunk.text)
        print("-" * 40)

    print()
    print("【演習】 chunk_size や chunk_overlap の値を変えて実行し直し、")
    print("チャンクの数や中身の切れ方がどう変わるか観察してみましょう。")
    print("（exercises/exercises.md の演習1を参照）")


if __name__ == "__main__":
    main()
