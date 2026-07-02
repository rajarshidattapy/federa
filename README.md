# Federa

> **Privacy-first federated learning infrastructure for PyTorch.**
>
> Train machine learning models across distributed devices and organizations without centralizing data.

Federa is an open-source Python framework for building, orchestrating, and
deploying federated learning systems. Instead of shipping sensitive data to
a centralized GPU cluster, Federa brings training to where the data lives
and only ever exchanges privacy-preserving model updates.

---

## Inspiration

Federa's aggregation algorithm is a direct implementation of **FedAvg**, from
the seminal paper:

**Communication-Efficient Learning of Deep Networks from Decentralized Data**
Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y Arcas (2017).

https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf

If you use Federa in academic work, please consider citing the original paper.

---

## Why Federa?

Modern machine learning leans on two assumptions: massive centralized
compute, and centralized access to user data. Both are increasingly
expensive and hard to justify.

* 🔒 Sensitive data must otherwise be uploaded to remote servers.
* 🌐 Data residency and compliance regulations are getting stricter.
* 📉 Some of the most valuable datasets legally or practically can't be pooled.

Federa's approach:

* **Local training** -- data never leaves the device or organization.
* **Collaborative learning** -- many participants train one shared model.
* **Privacy preservation** -- only (optionally noised, clipped, quantized) model updates are ever transmitted.
* **Any PyTorch model** -- Federa only touches `state_dict()`, so it doesn't care what's inside.

---

## Features

* **FedAvg / FedProx** aggregation, weighted by each client's local dataset size
* **Differential privacy** -- Laplace and Gaussian mechanisms, gradient clipping, a privacy accountant, optional [Opacus](https://opacus.ai/) RDP accounting
* **Quantization** -- Int8 and FP16 weight compression, plus QLoRA-ready 4-bit quantization + LoRA adapter injection hooks
* **Typed, async, production-shaped** -- FastAPI + websockets + asyncio, Pydantic settings/messages, msgpack wire protocol, structured JSON logging, mypy-clean, unit + integration tests, Docker, CI

---

## Architecture

```text
┌────────────────────────────────────┐
│           Coordinator              │
│      Aggregation & Orchestration   │
└────────────────────────────────────┘
                    ▲
                    │
        GlobalModel / GradientUpdate
                    │
    ┌───────────────┼────────────────┐
    │               │                │
┌─────────┐    ┌─────────┐     ┌─────────┐
│Client A │    │Client B │ ... │Client N │
│SwarmNode│    │SwarmNode│     │SwarmNode│
└─────────┘    └─────────┘     └─────────┘
      │              │               │
 Local Data      Local Data      Local Data
 (never leaves) (never leaves) (never leaves)
```

```text
Server starts
      |
Clients connect (ClientJoin)
      |
Server broadcasts global model (GlobalModel)
      |
Clients train locally
      |
Clients send updates (GradientUpdate + TrainingMetrics)
      |
FedAvg aggregation
      |
New global model
      |
Repeat until `rounds` is reached, then the coordinator closes all connections
```

---

## Installation

```bash
pip install federa
```

From source, with dev tooling:

```bash
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install "federa[privacy]"      # Opacus RDP accounting
pip install "federa[distributed]"  # Ray, for distributed client simulation
```

---

## Quick Start

**Coordinator** (never trains -- only aggregates):

```python
from federa import Coordinator
from federa.models import wrap_model
import torch.nn as nn

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
Coordinator(wrap_model(model)).run()
```

**Client** (run on every participating device/process):

```python
from federa import SwarmNode

node = SwarmNode(
    server="ws://localhost:8000",
    model=model,
    dataset=dataset,  # any torch.utils.data.Dataset
)
node.start_training()
```

See `examples/mnist_fedavg/` for a complete runnable example (CNN on MNIST,
multiple simulated clients).

---

## Configuration

Every tunable is a `pydantic-settings` model read from environment variables
(see `federa/utils/config.py`):

```bash
export FEDERA_COORDINATOR_MIN_CLIENTS_PER_ROUND=5
export FEDERA_COORDINATOR_ROUNDS=50
export FEDERA_PRIVACY_MECHANISM=gaussian   # none | laplace | gaussian
export FEDERA_PRIVACY_EPSILON=1.0
export FEDERA_QUANT_METHOD=int8            # none | int8 | fp16
```

---

## Supported Algorithms

* **FedAvg** -- the default; sample-weighted averaging of client weights.
* **FedProx** -- FedAvg plus a proximal term during local training, for non-IID clients (`fedprox_mu` on `SwarmNode`).
* **Custom aggregation** -- implement your own strategy against `federa.training.fedavg.ClientUpdate`.

---

## Project Structure

```text
federa/
├── federa/
│   ├── client/           SwarmNode, local trainer, scheduler, websocket connection
│   ├── coordinator/      Coordinator (FastAPI server), aggregator, routing, state
│   ├── communication/    messages, msgpack protocol, websocket transport
│   ├── privacy/          Laplace/Gaussian mechanisms, clipping, accountant
│   ├── quantization/     Int8/FP16, QLoRA hooks, unified compression API
│   ├── training/         FedAvg/FedProx, checkpointing, optimizer factory
│   ├── models/           FederatedModel adapter for arbitrary nn.Module
│   └── utils/            settings, structured logging, metrics
├── examples/mnist_fedavg/
├── tests/{unit,integration}/
├── legacy/                original TypeScript prototype
├── Dockerfile / docker-compose.yml
└── pyproject.toml
```

---

## Use Cases

* 🏥 **Healthcare** -- train across hospitals without sharing patient records.
* 📱 **Mobile personalization** -- on-device recommendation/keyboard models.
* 🏢 **Enterprise ML** -- collaborative training across organizations.
* 🎓 **Research** -- federated experimentation on privacy-sensitive datasets.

---

## Development

```bash
pip install -e ".[dev]"
ruff check federa tests
mypy federa
pytest tests -v
```

## Docker

```bash
docker compose up --build
```

Starts the coordinator and two example MNIST clients (see
`examples/README.md` for details, including how to run against real data).

---

## License

MIT License.
