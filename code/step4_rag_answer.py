"""
ステップ4: 検索結果をもとに、LLM（Claude）に回答を生成させる（RAGの完成形）

【このスクリプトで学ぶこと】
- Retrieval（検索）で得たチャンクを「参考情報」としてプロンプトに組み込む方法
- LLMに「参考情報に書かれていないことは推測で答えない」よう指示するテクニック
  （ハルシネーション＝もっともらしい嘘を減らすための基本手法）
- 1回の質問にかかる概算コストの確認方法

【コストについて】
このステップで初めてAnthropic APIを呼び出します（課金が発生する唯一のステップ）。
モデルは claude-haiku-4-5 を使用します。
2026年時点の料金は 入力 $1 / 出力 $5（100万トークンあたり）で、
1回の質問（参考情報+質問=約1,500トークン、回答=約300トークン）あたり
概算で 0.3〜0.5円程度です。研修中に数十回試しても数十円程度に収まります。

事前準備:
    1. .env.example を .env にコピーし、ANTHROPIC_API_KEY を設定してください
    2. step2_build_vectorstore.py を先に実行してください

実行方法:
    python step4_rag_answer.py
    python step4_rag_answer.py "VPNに接続できません"
"""

import os
import sys

import anthropic
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from step2_build_vectorstore import CHROMA_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME

load_dotenv()

# 研修では低コストなHaikuモデルを使用する。
# 本番運用で回答品質を重視する場合は claude-sonnet-5 等への切り替えを検討する。
LLM_MODEL = "claude-haiku-4-5"

# 2026年時点の Haiku 4.5 の料金（100万トークンあたり）。コスト表示用の概算値。
INPUT_PRICE_PER_MTOK = 1.0
OUTPUT_PRICE_PER_MTOK = 5.0
USD_TO_JPY = 155  # 概算レート。正確な金額は請求書で確認すること。

SYSTEM_PROMPT = """あなたはTechFlow株式会社のITヘルプデスク担当者です。
以下の「参考情報」だけを根拠にして、社員からの質問に日本語で答えてください。

ルール:
- 参考情報に書かれていないことは、絶対に推測や一般知識で補わないこと。
- 参考情報に答えが見当たらない場合は、正直に「社内文書からは確認できませんでした。
  ITヘルプデスク（内線1234）に直接お問い合わせください」と答えること。
- 回答は簡潔に、必要であれば箇条書きを使うこと。
- 参考情報のどの文書に基づいた回答か、末尾に出典として示すこと。
"""


def retrieve(query: str, top_k: int = 3):
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = embed_model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    ids = results["ids"][0]
    docs = results["documents"][0]
    return list(zip(ids, docs))


def build_user_prompt(query: str, retrieved_chunks: list[tuple[str, str]]) -> str:
    context_blocks = []
    for chunk_id, text in retrieved_chunks:
        context_blocks.append(f"[出典: {chunk_id}]\n{text}")
    context = "\n\n".join(context_blocks)

    return f"""# 参考情報
{context}

# 質問
{query}
"""


def generate_answer(query: str, retrieved_chunks: list[tuple[str, str]]) -> None:
    client = anthropic.Anthropic()  # 環境変数 ANTHROPIC_API_KEY を自動で読み込む

    user_prompt = build_user_prompt(query, retrieved_chunks)

    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.AuthenticationError:
        print("APIキーが正しくありません。.env の ANTHROPIC_API_KEY を確認してください。")
        return
    except anthropic.RateLimitError:
        print("レート制限に達しました。少し待ってから再実行してください。")
        return
    except anthropic.APIStatusError as e:
        print(f"APIエラーが発生しました: {e.status_code} {e.message}")
        return

    answer_text = "".join(block.text for block in response.content if block.type == "text")

    print(f"\n質問: {query}\n")
    print("--- Claudeの回答 ---")
    print(answer_text)

    # 概算コストを表示（教育目的。正確な請求額はコンソールで確認すること）
    usage = response.usage
    cost_usd = (
        usage.input_tokens * INPUT_PRICE_PER_MTOK / 1_000_000
        + usage.output_tokens * OUTPUT_PRICE_PER_MTOK / 1_000_000
    )
    print("\n--- 今回の概算コスト ---")
    print(f"入力トークン: {usage.input_tokens} / 出力トークン: {usage.output_tokens}")
    print(f"概算コスト: ${cost_usd:.5f}（約{cost_usd * USD_TO_JPY:.2f}円）")


def ask(query: str, top_k: int = 3) -> None:
    print(f"1. 「{query}」に関連するチャンクを検索します...")
    retrieved = retrieve(query, top_k=top_k)
    for rank, (chunk_id, _) in enumerate(retrieved, start=1):
        print(f"   第{rank}位: {chunk_id}")

    print("\n2. 検索結果をもとに、Claudeに回答を生成させます...")
    generate_answer(query, retrieved)


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("警告: ANTHROPIC_API_KEY が設定されていません。.env ファイルを確認してください。")
        return

    if len(sys.argv) > 1:
        ask(sys.argv[1])
    else:
        ask("VPNに接続できません。どうすればいいですか？")
        print("\n" + "=" * 60)
        print(
            "\n【演習】 他の質問も試してみましょう:"
            "\n  python step4_rag_answer.py \"入社したらまず何をすればいいですか\""
            "\n  python step4_rag_answer.py \"来週の天気を教えて\"  ← 参考情報にない質問"
            "\n最後の質問でClaudeがどう答えるか（正直に『わからない』と言うか、"
            "\nそれとも知ったかぶりをするか）に注目してください。"
            "\n（exercises/exercises.md の演習3を参照）"
        )


if __name__ == "__main__":
    main()
