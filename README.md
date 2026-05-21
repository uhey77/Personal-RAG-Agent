# Personal RAG Assistant

PDF、Markdown、txt、コードファイルをローカルで取り込み、質問できる個人用RAGアシスタントです。UIはStreamlit、RAGはLangChain、Vector DBはChromaを使います。PDFのテキスト抽出には、日本語Keynote PDFでも文字化けしにくいPyMuPDFを使います。

## セットアップ

```bash
uv python install 3.12
uv sync
cp .env.example .env
```

`.env` を編集して、使うプロバイダのAPIキーを設定します。

```bash
CHAT_PROVIDER=google
CHAT_MODEL=gemini-2.5-flash-lite
EMBEDDING_PROVIDER=google
EMBEDDING_MODEL=gemini-embedding-001
GOOGLE_API_KEY=your_api_key_here
```

## 起動

```bash
uv run streamlit run app.py
```

ブラウザで開いたら、サイドバーからファイルをアップロードし、「インデックス作成 / 更新」を押してください。

## プロバイダ切替

回答生成は `openai`, `google` に対応しています。

```bash
CHAT_PROVIDER=google
CHAT_MODEL=gemini-2.5-flash-lite
GOOGLE_API_KEY=your_api_key_here
```

Embeddingは `openai`, `google` に対応しています。

```bash
EMBEDDING_PROVIDER=google
EMBEDDING_MODEL=gemini-embedding-001
GOOGLE_API_KEY=your_api_key_here
```

Embeddingプロバイダまたはモデルを変えると、別のChroma collectionを使います。

## 対応ファイル

- PDF
- txt
- Markdown
- Python / JavaScript / TypeScript
- JSON / YAML
- HTML / CSS

`.env`、秘密鍵、`.git`、`.venv`、`node_modules` などは取り込み対象から除外します。

## PDFの読み込み

PDFのテキスト抽出には `PyMuPDF` を使います。`pypdf` ではKeynoteやmacOS Quartzで作られた日本語PDFの文字マップをうまく読めず、チャンクが文字化けすることがあるためです。

PDF抽出まわりの注意点:

- 既存PDFの抽出方式を変えた後は、画面の「インデックス作成 / 更新」を押してChromaを再作成してください。
- `Ignoring wrong pointing object ...` のようなログは、PDF内部の壊れた参照を読み飛ばしたという警告です。抽出結果が正常なら、多くの場合は無視できます。
- スキャン画像だけのPDFは、このままでは本文検索できません。必要ならOCR処理を追加してください。
- `PyMuPDF` は AGPL-3.0 または Artifex の商用ライセンスで提供されています。このアプリは個人のローカル利用を想定しています。再配布、公開サービス化、プロプライエタリ製品への組み込みを行う場合は、AGPL-3.0 の条件を満たせるか、または商用ライセンスが必要かを確認してください。

## セキュリティ上の注意

`.env` やAPIキーを含むファイルはGit管理に含めないでください。アップロードした文書の内容は、設定したLLM / Embeddingプロバイダに送信される場合があります。

## ライセンス

このプロジェクトは AGPL-3.0-or-later で公開します。詳細は [LICENSE](LICENSE) を参照してください。

PDF抽出には `PyMuPDF` を使用しています。`PyMuPDF` は AGPL-3.0 または Artifex の商用ライセンスで提供されています。AGPL-3.0 の条件を満たせない用途では、PyMuPDF の商用ライセンスを確認してください。

依存ライブラリは、それぞれのライセンス条件に従います。

## 動作確認

```bash
uv run python --version
uv run python -m compileall .
```

`Python 3.12.x` と表示されれば、想定どおりの環境です。
