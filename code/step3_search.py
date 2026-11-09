"""
ステップ3: ベクトル検索（Retrieval）だけを試す

【このスクリプトで学ぶこと】
- 質問文も埋め込みベクトルに変換し、ベクトルDB内で「意味が近いチャンク」を探す仕組み
- LLM（生成AI）を使わずに、検索だけでどこまで質問に答えられそうか体感する

このステップでもAPI課金は発生しません（すべてローカルで完結）。

実行方法:
    python step3_search.py
    python step3_search.py "VPNに接続できません"
"""

import sys

import chromadb
from sentence_transformers import SentenceTransformer

from step2_build_vectorstore import CHROMA_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME


def search(query: str, top_k: int = 3) -> None:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print("ベクトルDBが見つかりません。先に step2_build_vectorstore.py を実行してください。")
        return

    query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    print(f"\n質問: {query}")
    print(f"上位 {top_k} 件のチャンクを取得しました\n")

    ids = results["ids"][0]
    docs = results["documents"][0]
    distances = results["distances"][0]

    for rank, (chunk_id, doc, distance) in enumerate(zip(ids, docs, distances), start=1):
        # distance は「距離」なので、小さいほど質問と意味が近い
        print(f"--- 第{rank}位  (距離: {distance:.4f}  出典: {chunk_id}) ---")
        print(doc)
        print()


def main() -> None:
    if len(sys.argv) > 1:
        query = sys.argv[1]
        search(query)
    else:
        # デモ用のサンプル質問をいくつか試す
        sample_queries = [
            "パスワードを忘れました",
            "VPNに接続できません",
            "入社したらまず何をすればいいですか",
        ]
        for q in sample_queries:
            search(q)
            print("=" * 60)

        print(
            "\n【演習】 他のキーワードでも試してみましょう:"
            "\n  python step3_search.py \"プリンターが動きません\""
            "\nまた、distance（距離）の値がどのくらいだと『関連している』と言えそうか、"
            "\nチームで話し合ってみましょう。（exercises/exercises.md の演習2を参照）"
        )


if __name__ == "__main__":
    main()
