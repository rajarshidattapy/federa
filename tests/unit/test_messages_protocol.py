import torch

from federa.communication.messages import ClientJoin, GradientUpdate, MessageType
from federa.communication.protocol import decode_message, encode_message
from federa.models.serialization import bytes_to_state_dict, state_dict_to_bytes


def test_encode_decode_client_join_roundtrip():
    message = ClientJoin(client_id="abc", num_samples=42)
    decoded = decode_message(encode_message(message))

    assert isinstance(decoded, ClientJoin)
    assert decoded.client_id == "abc"
    assert decoded.num_samples == 42
    assert decoded.type == MessageType.CLIENT_JOIN


def test_encode_decode_gradient_update_preserves_binary_weights():
    weights = state_dict_to_bytes({"w": torch.randn(3, 3)})
    message = GradientUpdate(client_id="c1", round_number=2, num_samples=10, weights=weights)

    decoded = decode_message(encode_message(message))

    assert isinstance(decoded, GradientUpdate)
    assert decoded.weights == weights
    restored = bytes_to_state_dict(decoded.weights)
    assert restored["w"].shape == (3, 3)


def test_decode_dispatches_to_correct_type_via_discriminator():
    join_bytes = encode_message(ClientJoin(client_id="x", num_samples=1))
    update_bytes = encode_message(
        GradientUpdate(
            client_id="x",
            round_number=0,
            num_samples=1,
            weights=state_dict_to_bytes({}),
        )
    )

    assert isinstance(decode_message(join_bytes), ClientJoin)
    assert isinstance(decode_message(update_bytes), GradientUpdate)
