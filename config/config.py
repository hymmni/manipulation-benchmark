from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    width: int = 900
    height: int = 600
    svg_path: str = "references/image.svg"
    agent_start: tuple = (130.0, 300.0, 0.0)  # x, y, yaw(rad)
    goal_center: tuple = (800.0, 300.0)
    goal_radius: float = 30.0
    max_steps: int = 500
    dt: float = 0.05
    action_scale: float = 10.0  # max pixels/rad per step


@dataclass
class CVAEConfig:
    latent_dim: int = 16
    hidden_dim: int = 256


@dataclass
class PlannerConfig:
    horizon: int = 16
    n_diffusion_steps: int = 100
    hidden_dim: int = 256
    backbone: str = "ddpm"
    guidance_scale: float = 0.0
    cond_dropout_prob: float = 0.1


@dataclass
class IDMConfig:
    action_horizon: int = 4
    n_diffusion_steps: int = 100
    hidden_dim: int = 256
    backbone: str = "ddpm"
    exploration_noise: float = 0.0


@dataclass
class ModelConfig:
    state_dim: int = 3
    action_dim: int = 3
    cvae: CVAEConfig = field(default_factory=CVAEConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    idm: IDMConfig = field(default_factory=IDMConfig)


@dataclass
class TrainConfig:
    lr: float = 1e-4
    batch_size: int = 64
    pretrain_steps: int = 10000
    finetune_steps: int = 2000
    device: str = "auto"
    seed: int = 0
    use_wandb: bool = False


@dataclass
class BufferConfig:
    demo_path: str = "data_store/demos.zarr"
    self_path: str = "data_store/self_collected.zarr"
    efficiency_margin: float = 0.8  # self-collected must be <=80% of demo length


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)

    @classmethod
    def default(cls) -> "Config":
        return cls()
