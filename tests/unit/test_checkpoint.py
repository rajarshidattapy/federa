import torch

from federa.training.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    state_dict = {"w": torch.randn(2, 2)}
    path = save_checkpoint(tmp_path, round_number=1, state_dict=state_dict, metrics={"loss": 0.5})

    assert path.exists()
    loaded = load_checkpoint(path)
    assert torch.equal(loaded["w"], state_dict["w"])
    assert latest_checkpoint(tmp_path) == path


def test_latest_checkpoint_picks_highest_round(tmp_path):
    save_checkpoint(tmp_path, round_number=1, state_dict={"w": torch.zeros(1)})
    newest = save_checkpoint(tmp_path, round_number=2, state_dict={"w": torch.ones(1)})
    assert latest_checkpoint(tmp_path) == newest


def test_latest_checkpoint_none_when_missing(tmp_path):
    assert latest_checkpoint(tmp_path / "does-not-exist") is None
