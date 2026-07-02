# Architecture

## Components

- **`federa.client.SwarmNode`** -- the client-facing API. Wraps a
  `torch.nn.Module` + `Dataset` pair, connects to a coordinator, and drives
  the training loop (`federa.client.scheduler.TrainingScheduler`) over a
  websocket (`federa.client.websocket.ClientConnection`).
- **`federa.coordinator.Coordinator`** -- the server-facing API. A FastAPI
  app exposing `GET /health` and `WS /ws`. Never trains; only aggregates
  (`federa.coordinator.aggregator.Aggregator`) and routes messages
  (`federa.coordinator.routing.ClientRegistry`).
- **`federa.communication`** -- the wire protocol. Six Pydantic message
  types (`ClientJoin`, `ClientLeave`, `Heartbeat`, `GlobalModel`,
  `GradientUpdate`, `TrainingMetrics`) form a discriminated union, packed
  with msgpack (`federa.communication.protocol`) so binary tensor payloads
  don't pay JSON/base64's ~33% size penalty. `MessageChannel`
  (`federa.communication.websocket`) is transport-agnostic: the same
  `send`/`receive` API works over Starlette's server-side `WebSocket` and
  the `websockets` client library.
- **`federa.models`** -- `FederatedModel` is the narrow interface
  (`get_weights` / `set_weights` / `forward` / ...) the rest of the
  framework depends on. `TorchModelAdapter` implements it for any
  `nn.Module` by delegating to `state_dict()` / `load_state_dict()`.
- **`federa.training`** -- aggregation algorithms and round bookkeeping,
  decoupled from networking so they're unit-testable in isolation.
- **`federa.privacy`** / **`federa.quantization`** -- operate directly on
  `state_dict`s and are composed by both the client (before sending an
  update) and, for clipping, the coordinator (before aggregating).

## One training round

```text
1. Coordinator broadcasts GlobalModel(round_number=N, weights=...)
2. Each SwarmNode:
     a. loads the global weights
     b. trains `local_epochs` epochs on its local Dataset
     c. (optional) privatizes weights (Laplace/Gaussian noise)
     d. compresses weights (none/int8/fp16)
     e. sends TrainingMetrics, then GradientUpdate
        (metrics first: the coordinator may close the round -- and every
        connection -- as soon as it has counted enough GradientUpdates, so
        each client's metrics must already be recorded by then)
3. Coordinator, per GradientUpdate:
     a. decompresses weights
     b. (optional) clips to bound sensitivity for DP
     c. buffers as a ClientUpdate(client_id, num_samples, weights)
     d. once len(buffer) >= min_clients_per_round:
          - aggregates: w_{N+1} = sum_k (n_k / n) * w_{N+1}^k   (FedAvg)
          - applies the aggregate to the global model
          - checkpoints to disk
          - if rounds remain: broadcasts GlobalModel(round_number=N+1, ...)
            else: closes every client connection
```

## Aggregation strategies

- **FedAvg** (`federa.training.fedavg.federated_average`): sample-weighted
  mean of client weights, per McMahan et al. 2017.
- **FedProx** (`federa.training.fedprox`): identical aggregation, but each
  client adds `(mu/2) * ||w_local - w_global||^2` to its local loss during
  training, keeping updates from drifting too far from the global model
  under non-IID data. Enable with `SwarmNode(..., fedprox_mu=0.01)`.

## Differential privacy

- **Laplace** (`federa.privacy.laplace.LaplaceMechanism`) and **Gaussian**
  (`federa.privacy.gaussian.GaussianMechanism`) mechanisms add calibrated
  noise to a client's weights before transmission. Because FedAvg sums many
  independently-noised updates, the noise cancels out in expectation while
  each individual client's contribution stays private.
- **Clipping** (`federa.privacy.clipping`) bounds sensitivity -- how much a
  single client's update can change the aggregate -- which both mechanisms'
  noise calibration assumes.
- **Accounting** (`federa.privacy.accountant.PrivacyAccountant`) tracks
  cumulative (epsilon, delta) spend via basic sequential composition.
  `rdp_epsilon()` gives a much tighter bound via Opacus' RDP accountant
  when many rounds are involved (`pip install federa[privacy]`).

## Quantization

- **Int8** (`federa.quantization.int8`): symmetric per-tensor quantization,
  mapping `[-max|w|, max|w|]` onto `[-127, 127]` -- a 4x bandwidth
  reduction over Float32.
- **FP16** (`federa.quantization.fp16`): a simpler 2x reduction with
  negligible accuracy loss.
- **QLoRA hooks** (`federa.quantization.qlora`): 4-bit block quantization
  plus `inject_lora_adapters()`, which freezes a model's `nn.Linear` layers
  and wraps them with trainable low-rank adapters -- so a client can ship a
  quantized frozen base model once and thereafter only train/transmit tiny
  adapter deltas each round.
- **`federa.quantization.compression`** unifies all of the above behind one
  `compress_state_dict(state_dict, method) -> CompressedPayload` /
  `decompress_state_dict(payload)` API, which is what actually populates
  `GlobalModel.weights` / `GradientUpdate.weights` on the wire.
