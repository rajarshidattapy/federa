# Getting Started

## Install

```bash
pip install -e ".[dev]"
```

## 1. Define a model

Federa works with any `torch.nn.Module` -- it never inspects the
architecture, only `state_dict()`.

```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x):
        return self.fc(x)
```

## 2. Start the coordinator

```python
# server.py
from federa import Coordinator
from federa.models import wrap_model
from federa.utils.config import CoordinatorSettings

from my_model import Net

settings = CoordinatorSettings(min_clients_per_round=2, rounds=10)
Coordinator(wrap_model(Net()), settings=settings).run()
```

```bash
python server.py
```

The coordinator listens on `ws://0.0.0.0:8000/ws` by default and exposes
`GET /health` for liveness checks.

## 3. Start clients

Each client needs its own local `Dataset`. Federa weights FedAvg by each
client's `len(dataset)`, so partition your data ahead of time (by index, by
user, by device -- whatever fits your deployment).

```python
# client.py
from federa import SwarmNode
from my_model import Net
from my_data import load_local_dataset

node = SwarmNode(
    server="ws://localhost:8000",
    model=Net(),
    dataset=load_local_dataset(),
)
node.start_training()
```

```bash
python client.py  # run once per participating device/process
```

Once `min_clients_per_round` clients have sent a `GradientUpdate` for the
current round, the coordinator aggregates, checkpoints, and broadcasts the
new global model. This repeats until `rounds` is reached, at which point the
coordinator closes every connection and `start_training()` returns.

## 4. Inspect results

Checkpoints (and a `.json` of that round's sample-weighted loss/accuracy)
land in `CoordinatorSettings.checkpoint_dir` (default `./checkpoints`):

```python
from federa.training.checkpoint import latest_checkpoint, load_checkpoint

state_dict = load_checkpoint(latest_checkpoint("./checkpoints"))
Net().load_state_dict(state_dict)
```

## 5. Add privacy and compression

Both are configured via environment variables (or `CoordinatorSettings`/
`ClientSettings` directly) -- no code changes required:

```bash
export FEDERA_PRIVACY_MECHANISM=gaussian
export FEDERA_PRIVACY_EPSILON=1.0
export FEDERA_QUANT_METHOD=int8
```

See `docs/architecture.md` for what each mechanism actually does, and
`examples/mnist_fedavg/` for a full runnable example.
