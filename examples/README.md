# Examples

## `mnist_fedavg/`

Trains a small CNN on MNIST across simulated clients using FedAvg.

Requires `torchvision` in addition to Federa's own dependencies:

```bash
pip install -e ".[dev]"
pip install torchvision
```

Start the coordinator (waits for 2 clients per round, runs 5 rounds):

```bash
python -m examples.mnist_fedavg.server
```

In separate terminals, start each client (MNIST is partitioned by index
modulo `--num-clients`, so every client trains on a disjoint shard):

```bash
python -m examples.mnist_fedavg.client --client-id 0 --num-clients 2
python -m examples.mnist_fedavg.client --client-id 1 --num-clients 2
```

Checkpoints for the aggregated global model are written to
`./checkpoints/round_XXXXX.pt` after every round, alongside a `.json` file
with that round's sample-weighted loss/accuracy.

To enable differential privacy or quantization, set environment variables
before starting the clients and coordinator (see `federa.utils.config`):

```bash
export FEDERA_PRIVACY_MECHANISM=gaussian
export FEDERA_PRIVACY_EPSILON=1.0
export FEDERA_QUANT_METHOD=int8
```
