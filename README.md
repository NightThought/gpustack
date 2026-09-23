<h1 align="center">OriginHub</h1>

<p align="center">
  AI infrastructure and LLM inference serving platform
</p>

<p align="center">
  <a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue"></a>
</p>

> **Modified from [GPUStack](https://github.com/gpustack/gpustack).**
> OriginHub is a derivative work of GPUStack, licensed under the Apache License,
> Version 2.0. Copyright (c) 2024-2026 The GPUStack authors. See
> [NOTICE](./NOTICE) for attribution and third-party components, and
> [CHANGES.md](./CHANGES.md) for what this fork changed. This project is not
> endorsed by, sponsored by, or affiliated with the GPUStack project.

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README_CN.md">简体中文</a> |
  <a href="./README_JP.md">日本語</a>
</p>

<br>

## Overview

OriginHub is an open-source GPU cluster manager for AI model serving and GPU instance provisioning. It configures and orchestrates inference engines — vLLM, SGLang, TensorRT-LLM, or your own — and lets you launch SSH-accessible GPU instances on demand. Its core features include:
- **Multi-Cluster GPU Management.** Manages GPU clusters across multiple environments. This includes on-premises servers, Kubernetes clusters, and cloud providers.
- **Pluggable Inference Engines.** Automatically configures high-performance inference engines such as vLLM, SGLang, and TensorRT-LLM. You can also add custom inference engines as needed.
- **Day 0 Model Support.** OriginHub's pluggable engine architecture enables you to deploy new models on the day they are released.
- **Performance-Optimized Configurations.** Offers pre-tuned modes for low latency or high throughput. OriginHub supports extended KV cache systems like LMCache and HiCache to reduce TTFT. It also includes built-in support for speculative decoding methods such as EAGLE3, MTP, and N-grams.
- **GPU Instances.** Launches SSH-accessible GPU instances on demand for development, fine-tuning, and interactive workloads.
- **Enterprise-Grade Operations.** Offers support for automated failure recovery, load balancing, monitoring, authentication, and access control.

## Architecture

OriginHub enables development teams, IT organizations, and service providers to deliver Model-as-a-Service at scale. It supports industry-standard APIs for LLM, voice, image, and video models. The platform includes built-in user authentication and access control, real-time monitoring of GPU performance and utilization, and detailed metering of token usage and API request rates.

The figure below illustrates how a single OriginHub server can manage multiple GPU clusters across both on-premises and cloud environments. The OriginHub scheduler allocates GPUs to maximize resource utilization and selects the appropriate inference engines for optimal performance. Administrators also gain full visibility into system health and metrics through integrated Grafana and Prometheus dashboards.

![gpustack-v2-architecture](docs/assets/gpustack-v2-architecture.png)

## Cluster Visibility at a Glance

The GPU Cluster Topology view provides a live, bird's-eye view of your entire fleet — every worker, GPU, and model deployment in one place, with real-time utilization, allocation, and health status.

![gpustack-cluster-topology](docs/assets/cluster-topology.png)

## Optimized Inference Performance

OriginHub's automated engine selection and parameter optimization deliver strong inference performance out of the box. The following figure shows throughput improvements over default vLLM configurations:

![h200-throughput-comparison](docs/assets/h200-throughput-comparison.png)

For detailed benchmarking methods and results, visit our [Inference Performance Lab](./docs/performance-lab/overview.md).

## Supported Accelerators

OriginHub supports a wide range of accelerators for AI inference:

- **NVIDIA GPU**
- **AMD GPU**
- **Ascend NPU**
- **Hygon DCU**
- **MThreads GPU**
- **Iluvatar GPU**
- **MetaX GPU**
- **Cambricon MLU**
- **T-Head PPU**

For detailed requirements and setup instructions, see the [Installation Requirements](./docs/installation/requirements.md) documentation.

## Quick Start

### Prerequisites

1. A node with at least one NVIDIA GPU. For other GPU types, please check the guidelines in the OriginHub UI when adding a worker, or refer to the [Installation documentation](./docs/installation/requirements.md) for more details.
2. Ensure the NVIDIA driver, [Docker](https://docs.docker.com/engine/install/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) are installed on the worker node.
3. (Optional) A CPU node for hosting the OriginHub server. The OriginHub server does not require a GPU and can run on a CPU-only machine. [Docker](https://docs.docker.com/engine/install/) must be installed. Docker Desktop (for Windows and macOS) is also supported. If no dedicated CPU node is available, the OriginHub server can be installed on the same machine as a GPU worker node.
4. Only Linux is supported for OriginHub worker nodes. If you use Windows, consider using WSL2 and avoid using Docker Desktop. macOS is not supported for OriginHub worker nodes.

### Install OriginHub

Run the following command to install and start the OriginHub server using Docker:

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    --volume gpustack-data:/var/lib/gpustack \
    gpustack/gpustack
```

<details>
<summary>Alternative: Use Quay Container Registry Mirror</summary>

If you cannot pull images from `Docker Hub` or the download is very slow, you can use our `Quay.io` mirror by pointing your registry to `quay.io`:

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    --volume gpustack-data:/var/lib/gpustack \
    quay.io/gpustack/gpustack \
    --system-default-container-registry quay.io
```
</details>

Check the OriginHub startup logs:

```bash
sudo docker logs -f gpustack
```

After OriginHub starts, run the following command to get the default admin password:

```bash
sudo docker exec gpustack cat /var/lib/gpustack/initial_admin_password
```

Open your browser and navigate to `http://your_host_ip` to access the OriginHub UI. Use the default username `admin` and the password you retrieved above to log in.

### Set Up a GPU Cluster

1. On the OriginHub UI, navigate to the `Clusters` page.

2. Click the `Add Cluster` button.

3. Select `Docker` as the cluster provider.

4. Fill in the `Name` and `Description` fields for the new cluster, then click the `Save` button.

5. Follow the UI guidelines to configure the new worker node. You will need to run a Docker command on the worker node to connect it to the OriginHub server. The command will look similar to the following:

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

6. Execute the command on the worker node to connect it to the OriginHub server.

7. After the worker node connects successfully, it will appear on the `Workers` page in the OriginHub UI.

### Deploy a Model

1. Navigate to the `Catalog` page in the OriginHub UI.

2. Select the `Qwen3.5-0.8B` model from the list of available models.

![deploy qwen3 from catalog](docs/assets/quick-start/quick-start-qwen3.png)

3. After the deployment compatibility checks pass, click the `Save` button to deploy the model.

4. OriginHub will start downloading the model files and deploying the model. When the deployment status shows `Running`, the model has been deployed successfully.

![model is running](docs/assets/quick-start/model-running.png)

5. Click `Playground - Chat` in the navigation menu, check that the model `qwen3.5-0.8b` is selected from the top-right `Model` dropdown. Now you can chat with the model in the UI playground.

![quick chat](docs/assets/quick-start/quick-chat.png)

### Use the model via API

1. Navigate to the `Access Control` > `API Keys` page, then click the `New API Key` button.

2. Fill in the `Name` and click the `Save` button.

3. Copy the generated API key and save it somewhere safe. Please note that you can only see it once on creation.

4. You can now use the API key to access the OpenAI-compatible API endpoints provided by OriginHub. For example, use curl as the following:

```bash
# Replace `your_api_key` and `your_gpustack_server_url`
# with your actual API key and OriginHub server URL.
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3.5-0.8b",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Tell me a joke."
      }
    ],
    "stream": true
  }'
```

## Documentation

Please see the [official docs site](./docs) for complete documentation.

## Build

1. Install [Docker](https://docs.docker.com/engine/install/).

2. Run `make package`.

## Contributing

Please read the [Contributing Guide](./docs/contributing.md) if you're interested in contributing to OriginHub.

## Support

Questions, bugs and suggestions go to this fork's [issue tracker](https://github.com/NightThought/gpustack/issues).
There is no chat community yet; when there is one, it will be linked here rather
than pointing at a project this build is not part of.

## Acknowledgement

OriginHub is built on the open-source [GPUStack](https://github.com/gpustack/gpustack)
project, licensed under the Apache License, Version 2.0.
Copyright (c) 2024-2026 The GPUStack authors.

See [NOTICE](./NOTICE) for the full attribution and for the third-party
components this product redistributes, and [CHANGES.md](./CHANGES.md) for a
summary of the modifications made here. Neither this project nor its maintainers
are endorsed by, sponsored by, or affiliated with the GPUStack project.

## License

Copyright (c) 2024-2026 The GPUStack authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at [LICENSE](./LICENSE) file for details.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
