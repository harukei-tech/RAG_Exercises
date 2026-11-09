# RAG構築 新入社員研修（1日ハンズオン）

IT部署に配属された新入社員向けに、RAG（Retrieval-Augmented Generation）の
基本概念を理解し、実際に手を動かして「社内ヘルプデスクFAQボット」を
ゼロから構築する1日集中ハンズオン研修の教材一式です。

コストを抑えることを重視し、埋め込み・ベクトルDBは無料でローカル実行できる
ものを採用し、課金が発生するのは最終ステップのLLM呼び出し（Claude Haiku、
数十円程度）のみという構成にしています。

## 構成

| ディレクトリ / ファイル | 内容 |
|---|---|
| `docs/01_training_text.md` | 研修用テキスト本体（講師・受講者共通、概念説明＋手順） |
| `code/` | ハンズオン用サンプルコード一式（Python） |
| `code/sample_docs/` | 検索対象となる架空の社内IT文書（FAQ、規定等） |
| `code/step1_〜step4_*.py` | ステップごとに実行するハンズオンスクリプト |
| `code/rag_cli.py` | 完成版・対話型RAGボット |
| `code/exercises/exercises.md` | 演習課題・発展課題 |
| `slides/` | 講師用スライド（HTML） |

## 使い方

1. `docs/01_training_text.md` を読み、研修の流れ・技術選定の理由を把握する
2. 受講者は事前（研修前日までに）`code/` の環境構築を済ませておく（詳細はテキスト参照）
3. 当日は `docs/01_training_text.md` のタイムテーブルに沿って、
   `code/step1_〜step4_*.py` を順番に動かしながら進める
4. 講師はスライド（`slides/`）を併用して概念説明を行う

## 技術スタック（すべて低コスト／無料）

| 項目 | 採用technology | コスト |
|---|---|---|
| 文書チャンク分割 | 自前実装（Pythonのみ） | 無料 |
| 埋め込み（Embedding） | sentence-transformers（ローカル実行） | 無料 |
| ベクトルDB | Chroma（ローカル実行） | 無料 |
| 回答生成（LLM） | Claude API（claude-haiku-4-5） | 従量課金・1問あたり1円未満 |

詳しい設計判断の理由は `docs/01_training_text.md` の「0. 研修の設計方針」を参照してください。
