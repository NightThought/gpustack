<h1 align="center">OriginHub</h1>

<p align="center">
  AI インフラと LLM 推論サービングプラットフォーム
</p>

<p align="center">
  <a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue"></a>
</p>

> **本プロジェクトは [GPUStack](https://github.com/gpustack/gpustack) を改変したものです。**
> OriginHub は GPUStack の派生作品であり、Apache License 2.0 の下でライセンスされています。
> Copyright (c) 2024-2026 The GPUStack authors。帰属表示とサードパーティコンポーネントは
> [NOTICE](./NOTICE) を、本 fork の変更点は [CHANGES.md](./CHANGES.md) を参照してください。
> 本プロジェクトは GPUStack プロジェクトによる承認・支援・提携を受けたものではありません。

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README_CN.md">简体中文</a> |
  <a href="./README_JP.md">日本語</a>
</p>

<br>

## 概要

OriginHubは、AIモデルサービングとGPUインスタンスプロビジョニングのためのオープンソースのGPUクラスタマネージャーです。推論エンジン（vLLM、SGLang、TensorRT-LLM、またはカスタムエンジン）を構成・オーケストレーションし、オンデマンドでSSHアクセス可能なGPUインスタンスを起動できます。主な機能は以下の通りです：
- **マルチクラスタGPU管理。** 複数の環境にわたるGPUクラスタを管理します。これには、オンプレミスサーバー、Kubernetesクラスタ、およびクラウドプロバイダが含まれます。
- **プラグ可能な推論エンジン。** vLLM、SGLang、TensorRT-LLMなどの高性能推論エンジンを自動的に設定します。必要に応じてカスタム推論エンジンを追加することもできます。
- **Day 0モデルサポート。** OriginHubのプラグ可能なエンジンアーキテクチャにより、新しいモデルがリリースされた当日にデプロイできます。
- **パフォーマンス最適化設定。** 低レイテンシまたは高スループット向けの事前調整済みモードを提供します。OriginHubは、LMCacheやHiCacheなどの拡張KVキャッシュシステムをサポートし、TTFTを削減します。また、EAGLE3、MTP、N-gramなどの投機的デコード手法の組み込みサポートも含まれます。
- **GPUインスタンス。** オンデマンドでSSHアクセス可能なGPUインスタンスを起動し、開発、ファインチューニング、対話的なワークロードに対応します。
- **エンタープライズグレードの運用。** 自動化された障害回復、負荷分散、監視、認証、およびアクセス制御のサポートを提供します。

## アーキテクチャ

OriginHubは、開発チーム、IT組織、およびサービスプロバイダーが大規模なモデルサービスを提供できるようにします。LLM、音声、画像、ビデオモデル向けの業界標準APIをサポートしています。このプラットフォームには、組み込みのユーザー認証とアクセス制御、GPUパフォーマンスと使用率のリアルタイム監視、トークン使用量とAPIリクエストレートの詳細なメータリングが含まれています。

以下の図は、単一のOriginHubサーバーがオンプレミスとクラウド環境の両方にまたがる複数のGPUクラスタをどのように管理できるかを示しています。OriginHubスケジューラは、リソース使用率を最大化するためにGPUを割り当て、最適なパフォーマンスを得るために適切な推論エンジンを選択します。管理者は、統合されたGrafanaおよびPrometheusダッシュボードを通じて、システムの健全性とメトリクスに関する完全な可視性も得ます。

![gpustack-v2-architecture](docs/assets/gpustack-v2-architecture.png)

## クラスタの全体像を一目で把握

GPUクラスタトポロジービューは、クラスタ全体のライブな鳥瞰図を提供します。すべてのワーカー、GPU、モデルデプロイメントを一箇所で確認でき、使用率、割り当て状況、健全性ステータスをリアルタイムに表示します。

![gpustack-cluster-topology](docs/assets/cluster-topology.png)

## 最適化された推論パフォーマンス

OriginHubの自動化されたエンジン選択とパラメータ最適化により、すぐに使える強力な推論パフォーマンスを提供します。以下の図は、デフォルトのvLLM設定と比較したスループットの向上を示しています：

![h200-throughput-comparison](docs/assets/h200-throughput-comparison.png)

詳細なベンチマーク方法と結果については、[推論パフォーマンスラボ](./docs/performance-lab/overview.md)をご覧ください。

## サポートされているアクセラレータ

OriginHub は AI 推論用の幅広いアクセラレータをサポートしています：

- **NVIDIA GPU**
- **AMD GPU**
- **Ascend NPU**
- **Hygon DCU**
- **MThreads GPU**
- **Iluvatar GPU**
- **MetaX GPU**
- **Cambricon MLU**
- **T-Head PPU**

詳細な要件とセットアップ手順については、[インストール要件](./docs/installation/requirements.md)ドキュメントを参照してください。

## クイックスタート

### 前提条件

1.  少なくとも1つの NVIDIA GPU を搭載したノード。他の GPU タイプについては、OriginHub UI で worker を追加する際のガイドラインを参照するか、詳細については[インストールドキュメント](./docs/installation/requirements.md)を参照してください。
2.  worker ノードに NVIDIA ドライバー、[Docker](https://docs.docker.com/engine/install/)、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) がインストールされていることを確認してください。
3.  （オプション）OriginHub server をホストするための CPU ノード。OriginHub server は GPU を必要とせず、CPU のみのマシンで実行できます。[Docker](https://docs.docker.com/engine/install/) がインストールされている必要があります。Docker Desktop（Windows および macOS 用）もサポートされています。専用の CPU ノードがない場合は、GPU worker ノードと同じマシンに OriginHub server をインストールできます。
4.  OriginHub worker ノードは Linux のみをサポートしています。Windows を使用する場合は、WSL2 の使用を検討し、Docker Desktop の使用は避けてください。macOS は OriginHub worker ノードとしてサポートされていません。

### OriginHub のインストール

以下のコマンドを実行して、Docker を使用して OriginHub server をインストールし起動します：

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    --volume gpustack-data:/var/lib/gpustack \
    gpustack/gpustack
```

<details>
<summary>代替案：Quay コンテナレジストリミラーの使用</summary>

`Docker Hub` からイメージをプルできない場合やダウンロードが非常に遅い場合は、`quay.io` を指定することで当社のミラーを使用できます：

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    --volume gpustack-data:/var/lib/gpustack \
    quay.io/gpustack/gpustack \
    --system-default-container-registry quay.io
```
</details>

OriginHub の起動ログを確認します：

```bash
sudo docker logs -f gpustack
```

OriginHub が起動したら、以下のコマンドを実行してデフォルトの管理者パスワードを取得します：

```bash
sudo docker exec gpustack cat /var/lib/gpustack/initial_admin_password
```

ブラウザを開き、`http://あなたのホストIP` にアクセスして OriginHub UI にアクセスします。デフォルトのユーザー名 `admin` と上記で取得したパスワードを使用してログインします。

### GPU クラスターのセットアップ

1.  OriginHub UI で、`Clusters` ページに移動します。
2.  `Add Cluster` ボタンをクリックします。
3.  クラスタープロバイダーとして `Docker` を選択します。
4.  新しいクラスターの `Name` と `Description` フィールドに入力し、`Save` ボタンをクリックします。
5.  UI のガイドラインに従って新しい worker ノードを設定します。worker ノードを OriginHub server に接続するには、worker ノードで Docker コマンドを実行する必要があります。コマンドは以下のようになります：
    ```bash
    sudo docker run -d --name gpustack-worker \
          --restart=unless-stopped \
          --privileged \
          --network=host \
          --volume /var/run/docker.sock:/var/run/docker.sock \
          --volume gpustack-data:/var/lib/gpustack \
          --runtime nvidia \
          gpustack/gpustack \
          --server-url http://your_gpustack_server_url \
          --token your_worker_token \
          --advertise-address 192.168.1.2
    ```
6.  worker ノードでこのコマンドを実行して OriginHub server に接続します。
7.  worker ノードが正常に接続されると、OriginHub UI の `Workers` ページに表示されます。

### モデルのデプロイ

1. OriginHub UIの`Catalog`ページに移動します。

2. 利用可能なモデルのリストから`Qwen3.5-0.8B`モデルを選択します。

![カタログからqwen3をデプロイ](docs/assets/quick-start/quick-start-qwen3.png)

3. デプロイ互換性チェックが通過した後、`Save`ボタンをクリックしてモデルをデプロイします。

4. OriginHubはモデルファイルのダウンロードとモデルのデプロイを開始します。デプロイステータスが`Running`と表示されたら、モデルは正常にデプロイされています。

![モデルが実行中](docs/assets/quick-start/model-running.png)

5. ナビゲーションメニューで`Playground - Chat`をクリックし、右上の`Model`ドロップダウンからモデル`qwen3.5-0.8b`が選択されていることを確認します。これでUIプレイグラウンドでモデルとチャットできるようになります。

![クイックチャット](docs/assets/quick-start/quick-chat.png)

### API経由でモデルを使用

1. `Access Control` > `API Keys`ページに移動し、`New API Key`ボタンをクリックします。

2. `Name`を入力し、`Save`ボタンをクリックします。

3. 生成されたAPIキーをコピーし、安全な場所に保存します。このキーは作成時に一度しか確認できないことに注意してください。

4. これで、このAPIキーを使用して、OriginHubが提供するOpenAI互換のAPIエンドポイントにアクセスできます。例えば、以下のようにcurlを使用します：

```bash
# `your_api_key` と `your_gpustack_server_url` を
# 実際のAPIキーとOriginHubサーバーのURLに置き換えてください。
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3.5-0.8b",
    "messages": [
      {
        "role": "system",
        "content": "あなたは役立つアシスタントです。"
      },
      {
        "role": "user",
        "content": "ジョークを教えてください。"
      }
    ],
    "stream": true
  }'
```

## ドキュメント

完全なドキュメントについては、[公式ドキュメントサイト](./docs)を参照してください。

## ビルド

1. [Docker](https://docs.docker.com/engine/install/) をインストールします。

2. `make package` を実行します。

## 貢献

OriginHubへの貢献に興味がある場合は、[貢献ガイド](./docs/contributing.md)をお読みください。

## サポート

質問・バグ・提案はこのフォークの [issue トラッカー](https://github.com/NightThought/gpustack/issues) へお願いします。
チャットコミュニティはまだありません。でき次第ここに記載します。

## 謝辞

OriginHub はオープンソースプロジェクト [GPUStack](https://github.com/gpustack/gpustack)
を基に構築されており、Apache License 2.0 の下でライセンスされています。
Copyright (c) 2024-2026 The GPUStack authors。

完全な帰属表示と本製品が再配布するサードパーティコンポーネントは [NOTICE](./NOTICE) を、
本 fork の変更点の概要は [CHANGES.md](./CHANGES.md) を参照してください。本プロジェクトおよび
そのメンテナーは GPUStack プロジェクトによる承認・支援・提携を受けていません。

## ライセンス

Copyright (c) 2024-2026 The GPUStack authors

Apache License, Version 2.0（「ライセンス」）に基づいてライセンスされます。
ライセンスに準拠しない限り、このファイルを使用することはできません。
ライセンスのコピーは[LICENSE](./LICENSE)ファイルで入手できます。

適用される法律で要求されない限り、または書面で合意されない限り、本ライセンスに基づいて配布されるソフトウェアは、明示黙示を問わず、いかなる保証も条件もなしに「現状のまま」配布されます。
ライセンスの権利と制限を規定する特定の言語については、ライセンスを参照してください。
