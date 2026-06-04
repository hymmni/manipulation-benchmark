from config import (
    Config, EnvConfig, ModelConfig, CVAEConfig,
    PlannerConfig, IDMConfig, TrainConfig, BufferConfig,
)


def test_config_default_creates():
    c = Config.default()
    assert isinstance(c, Config)


def test_nested_dataclass_access():
    c = Config.default()
    assert isinstance(c.env, EnvConfig)
    assert isinstance(c.model, ModelConfig)
    assert isinstance(c.model.cvae, CVAEConfig)
    assert isinstance(c.model.planner, PlannerConfig)
    assert isinstance(c.model.idm, IDMConfig)
    assert isinstance(c.train, TrainConfig)
    assert isinstance(c.buffer, BufferConfig)


def test_action_dim():
    c = Config.default()
    assert c.model.action_dim == 3


def test_state_dim():
    c = Config.default()
    assert c.model.state_dim == 3


def test_env_config_fields():
    c = Config.default()
    assert c.env.width == 900
    assert c.env.height == 600
    assert len(c.env.agent_start) == 3
    assert len(c.env.goal_center) == 2


def test_idm_exploration_noise():
    c = Config.default()
    assert c.model.idm.exploration_noise == 0.0


def test_planner_backbone_default():
    c = Config.default()
    assert c.model.planner.backbone == "ddpm"
    assert c.model.idm.backbone == "ddpm"


def test_train_device_default():
    c = Config.default()
    assert c.train.device == "auto"


def test_config_no_torch_import():
    """Config module must not import torch (pure stdlib dataclasses)."""
    import sys
    # config should already be imported; torch should NOT be in its module deps
    import config.config as cc
    import inspect
    src = inspect.getsource(cc)
    assert "import torch" not in src
