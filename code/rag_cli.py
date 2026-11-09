"""
完成版: 対話形式のRAG ヘルプデスクボット

これまでのステップ（1〜4）を組み合わせた、対話的に質問できるCLIです。
研修のまとめとして、完成したRAGシステムを実際に触って体感してみましょう。

事前準備:
    1. .env に ANTHROPIC_API_KEY を設定
    2. python step2_build_vectorstore.py を実行してベクトルDBを構築

実行方法:
    python rag_cli.py
"""

import os

from dotenv import load_dotenv

from step4_rag_answer import ask

load_dotenv()


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("警告: ANTHROPIC_API_KEY が設定されていません。.env ファイルを確認してください。")
        return

    print("=" * 60)
    print("TechFlow ITヘルプデスク RAGボット（研修用）")
    print("終了するには 'exit' または 'quit' と入力してください。")
    print("=" * 60)

    total_questions = 0
    while True:
        try:
            query = input("\n質問を入力してください > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        ask(query)
        total_questions += 1

    print(f"\n終了します。今回のセッションで {total_questions} 件の質問に回答しました。")
    print("お疲れさまでした！")


if __name__ == "__main__":
    main()
