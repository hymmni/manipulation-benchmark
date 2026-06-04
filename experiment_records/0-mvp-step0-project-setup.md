# Experiment Log: 2026-06-04 manipulation-benchmark

## 1. 개요 (Overview)
- **수정 대상**: 레포 루트 골격 (config/, utils/, env/, models/, data/, trainer/, tests/)
- **참조 소스**: `references/latent_diffusion_planning/env.yml` (의존성 참조만, JAX 스택 제외)
- **목표**: 0-mvp 나머지 step(1~4)이 의존할 패키지 디렉토리·dataclass config·device 유틸·conda 환경 설립

## 2. 변경 내역 (Changes)
- [x] 패키지 디렉토리 7개 생성 (config, env, models, data, trainer, utils, tests) + 각 `__init__.py`
- [x] `conftest.py` 루트 생성 (pytest sys.path 보장)
- [x] `config/config.py`: 순수 dataclass 기반 Config 계층 (Config, EnvConfig, ModelConfig, CVAEConfig, PlannerConfig, IDMConfig, TrainConfig, BufferConfig)
- [x] `utils/device.py`: `cuda → mps → cpu` fallback `get_device()` + `to_device()` 헬퍼
- [x] `environment.yml`: `trajectory-explore` conda 환경 (Python 3.10, CPU 빌드)
- [x] `tests/test_config.py`, `tests/test_device.py`: 15개 테스트 전 통과

## 3. 참조 로직 (Reference Logic)
- LDP `env.yml`의 의존성 목록에서 프레임워크와 무관한 패키지(torch, diffusers, zarr, einops 등)를 추려 `trajectory-explore` 환경에 포함.
- JAX/Flax/Mujoco/Robosuite 등 레퍼런스 고유 스택은 제외.

## 4. 가설 및 예상 결과 (Hypothesis)
- 이 골격 위에 각 step(env-2d, demo-collection, models, selfimprove-loop)이 독립적으로 쌓일 수 있음.
- `Config.default()`로 모든 하위 설정을 한 번에 초기화하고 오버라이드 가능.

## 5. 결과 기록
- `conda run -n trajectory-explore python -m pytest tests/ -q` → **15 passed in 0.55s**
- `Config.default().model.idm.exploration_noise` → **0.0**
- `get_device('auto')` → **cpu** (로컬 GPU 없음, 정상 fallback)
