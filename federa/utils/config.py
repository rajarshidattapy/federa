"""Typed application settings, loaded from environment variables or a .env file.

Every tunable in Federa (server binding, privacy budget, quantization scheme,
round sizing) lives here so both the coordinator and clients read from one
source of truth instead of scattering magic numbers across modules.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

QuantizationMethod = Literal["none", "int8", "fp16"]
PrivacyMechanism = Literal["none", "laplace", "gaussian"]
AggregationStrategy = Literal["fedavg", "fedprox"]


class PrivacySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FEDERA_PRIVACY_")

    mechanism: PrivacyMechanism = "none"
    epsilon: float = Field(default=1.0, gt=0)
    delta: float = Field(default=1e-5, gt=0, lt=1)
    sensitivity: float = Field(default=1.0, gt=0)
    max_grad_norm: float = Field(default=1.0, gt=0)


class QuantizationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FEDERA_QUANT_")

    method: QuantizationMethod = "none"


class CoordinatorSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FEDERA_COORDINATOR_")

    host: str = "0.0.0.0"
    port: int = 8000
    min_clients_per_round: int = Field(default=2, ge=1)
    rounds: int = Field(default=10, ge=1)
    round_timeout_seconds: float = Field(default=120.0, gt=0)
    aggregation_strategy: AggregationStrategy = "fedavg"
    fedprox_mu: float = Field(default=0.01, ge=0)
    checkpoint_dir: str = "./checkpoints"

    privacy: PrivacySettings = Field(default_factory=PrivacySettings)
    quantization: QuantizationSettings = Field(default_factory=QuantizationSettings)


class ClientSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FEDERA_CLIENT_")

    server_url: str = "ws://localhost:8000/ws"
    local_epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.01, gt=0)
    heartbeat_interval_seconds: float = Field(default=10.0, gt=0)
    reconnect_backoff_seconds: float = Field(default=2.0, gt=0)
    max_reconnect_attempts: int = Field(default=5, ge=0)

    privacy: PrivacySettings = Field(default_factory=PrivacySettings)
    quantization: QuantizationSettings = Field(default_factory=QuantizationSettings)


def load_coordinator_settings() -> CoordinatorSettings:
    return CoordinatorSettings()


def load_client_settings() -> ClientSettings:
    return ClientSettings()
