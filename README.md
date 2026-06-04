# manipulation-benchmark

비효율적인 인간 데모로 출발한 2D 로봇이 **자가 성장(Self-Improvement) 루프**를 통해 스스로
더 효율적인 지름길을 발견·학습하는지 검증하는 벤치마크 + 파이프라인 **MVP**.

**파이프라인:** `CVAE` → `Latent Trajectory Diffusion Planner` (goal-conditioned, CFG) →
`Inverse Dynamics Action Diffusion` (추론 시 exploration noise로 탐색).
**자가 성장 루프:** 탐색 롤아웃 → 효율 필터(지름길 채택) → self 버퍼 누적 → 파인튜닝.

> 코드는 `feat/0-mvp` 브랜치에 있습니다. (원격 `main`은 별개 히스토리이므로 아래처럼 `-b feat/0-mvp`로 받으세요.)

---

## 1. 요구 사항

- `conda` (miniconda/anaconda)
- 학습/추론 PC: NVIDIA GPU + 최신 드라이버 (선택 — 없으면 자동 CPU fallback)
- 디스플레이 없는 서버(headless)에서는 모든 실행 앞에 `SDL_VIDEODRIVER=dummy` 를 붙입니다.

## 2. Clone

```bash
git clone -b feat/0-mvp https://github.com/hymmni/manipulation-benchmark.git
cd manipulation-benchmark
```

## 3. 환경 생성

```bash
conda env create -f environment.yml
conda activate trajectory-explore
```

`torch`/`torchvision`은 PyPI의 CUDA 빌드로 설치되어 **GPU가 있으면 자동 사용**됩니다.
드라이버가 오래되어 실패하면 `environment.yml`의 주석대로 `--index-url .../whl/cuXXX`(특정 CUDA)
또는 `.../whl/cpu`(CPU 전용)로 바꿔 다시 생성하세요.

## 4. GPU 인식 확인

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); \
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## 5. 테스트 (스모크)

```bash
SDL_VIDEODRIVER=dummy python -m pytest -q          # 기대: 113 passed
```

## 6. 자가 성장 루프 실행

```bash
# (a) 빠른 검증 — 축소 config로 1 iteration, 합성 데모 자동 생성 (CPU에서 수 초)
SDL_VIDEODRIVER=dummy python main.py --dry-run

# (b) 실제 실행 — default config, GPU 자동 사용 (장시간)
SDL_VIDEODRIVER=dummy python main.py
```

> `main.py`는 `cfg.train.device="auto"` → `cuda→mps→cpu` 순으로 디바이스를 고릅니다.
> GPU 서버에서 `(b)`를 돌리면 CUDA가 자동으로 잡힙니다.

## 7. (선택) 인간 데모 수집 GUI

디스플레이가 있는 환경에서 마우스로 경유점을 클릭해 데모 궤적을 만듭니다(자동 spline 보간 → Zarr 저장).

```bash
python collect_demos.py            # headless 환경에서는 안내 후 종료됨
```

---

## 프로젝트 구조

```
config/        # dataclass 설정 (Config/EnvConfig/ModelConfig/TrainConfig/BufferConfig)
env/           # TrajectoryExploreEnv(gym) + SVG 맵 파서 + 충돌 판정 + PyGame 렌더
models/        # CVAE / LatentTrajectoryDiffusion / InverseDynamicsDiffusion (diffusers DDPM)
data/          # spline 보간 · Zarr IO · ReplayBuffer / SelfCollectedBuffer · 데모 수집기
trainer/       # Trainer(pretrain/finetune/rollout/evaluate) + 효율 필터(지름길 판정)
utils/         # device(cuda→mps→cpu fallback) 등
tests/         # pytest (env / spline / models / config / device)
main.py            # 자가 성장 온라인 루프 진입점 (--dry-run 지원)
collect_demos.py   # 인간 데모 수집 GUI 진입점
environment.yml    # conda 환경 (trajectory-explore)
```

## 환경/디바이스 메모

- **디바이스 선택**: 모든 학습/추론은 `utils.device.get_device()`로 `cuda→mps→cpu` fallback.
  코드에 `cuda` 하드코딩 없음 — GPU 없는 PC에서도 그대로 동작.
- **맵 데이터**: env는 `references/image.svg`를 파싱하되, 파일이 없으면(예: 이 저장소 clone 시
  `references/`는 미포함) 동일한 `FALLBACK_MAP`을 사용하므로 실행에 지장이 없습니다.
- **데이터 저장**: 데모/self 버퍼는 `data_store/*.zarr`에 저장되며 git 추적에서 제외됩니다.

## 알려진 후속 개선 (MVP 범위 밖)

- CVAE 입력(원좌표 ~900)을 정규화하면 recon loss 스케일이 O(1)로 내려가 학습 품질이 개선됩니다.
  (현재는 `logvar` 클램프로 NaN만 방지 — 동작/수치 안정성에는 문제 없음)
- 파인튜닝 강화 경로: DPPO (역확산 체인을 MDP로 보는 policy-gradient 파인튜닝).
- 생성 백본 교체: flow matching / shortcut model (`cfg.*.backbone`에 교체 지점 예약됨).
