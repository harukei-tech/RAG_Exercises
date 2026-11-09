"""
ステップ2: 埋め込み（Embedding）とベクトルDBへの格納

【このスクリプトで学ぶこと】
- テキストを「埋め込みベクトル」に変換する仕組み（意味の近さを数値化する）
- ベクトルDB（Chroma）にチャンクを格納する方法
- ここで使うモデル・DBはどちらも無料でローカル実行できる

【コストについて】
このステップではAPIを一切呼び出しません。
sentence-transformers が Hugging Face から埋め込みモデル
（paraphrase-multilingual-MiniLM-L12-v2、約470MB）を初回のみ
ダウンロードしますが、モデル自体もダウンロードも無料です。
2回目以降はローカルにキャッシュされるため、再ダウンロードは発生しません。

実行方法:
    python step2_build_vectorstore.py
"""

import chromadb
from sentence_transformers import SentenceTransformer

from utils import build_chunks

# 多言語対応・軽量な埋め込みモデル。日本語・英語どちらも扱える無料モデル。
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# ベクトルDBの永続化先（このディレクトリにデータが保存される）
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "it_helpdesk_docs"


def main() -> None:
    print("=" * 60)
    print("1. チャンクを準備します")
    print("=" * 60)
    chunks = build_chunks(chunk_size=400, chunk_overlap=80)
    print(f"{len(chunks)} 個のチャンクを埋め込みます。\n")

    print("=" * 60)
    print(f"2. 埋め込みモデルを読み込みます: {EMBEDDING_MODEL_NAME}")
    print("   （初回はモデルのダウンロードが発生するため数分かかることがあります）")
    print("=" * 60)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("\n3. 全チャンクを埋め込みベクトルに変換します")
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    print(f"埋め込みの次元数: {embeddings.shape[1]}  （1チャンクがこの次元数の数値ベクトルになる）")

    print("\n4. Chroma（ローカルベクトルDB）に保存します")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    # 既存のコレクションがあれば作り直す（研修用に毎回クリーンな状態にする）
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[c.id for c in chunks],
        embeddings=embeddings.tolist(),
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )

    print(f"\n完了しました。'{CHROMA_DB_DIR}' に {collection.count()} 件のベクトルを保存しました。")
    print("次は step3_search.py で、実際に検索を試してみましょう。")


if __name__ == "__main__":
    main()
