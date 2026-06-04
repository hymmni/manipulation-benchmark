# Design Decisions: 0-mvp pipeline (research-informed)

작성일: 2026-06-04. 0-mvp 스캐폴딩 착수 전, 2025–2026 최신 동향 리서치를 반영해 인터페이스 설계를 확정한 기록.

## 1. 개요 (Overview)
- **대상**: 0-mvp 스캐폴딩 (`config/`, `models/`, `trainer/`, `main.py`)
- **참조 소스**: `references/latent_diffusion_planning` (LDP, ICML'25) — 단 JAX/Flax이므로 개념만 PyTorch로 이식
- **목표**: CVAE + Latent Trajectory Diffusion Planner + Inverse Dynamics Action Diffusion 파이프라인으로 비효율 데모→자가성장(지름길 발견) 검증

## 2. 확정 결정 (Changes vs. 초기안)
- [x] **생성 백본 추상화**: `BaseGenerativeModel` + `_build_scheduler()` 한 곳에 백본 격리. MVP 기본 `"ddpm"`(diffusers), `cfg.*.backbone`로 `"flow_matching"`/`"shortcut"` 교체 지점만 예약.
- [x] **planner = goal-conditioned + CFG**: 학습 시 `cond_dropout_prob`로 조건 드롭, 샘플링 시 `guidance_scale`로 classifier-free guidance.
- [x] **IDM 탐색 = on-manifold**: `exploration_noise`를 최종 액션이 아니라 **역확산 각 스텝 posterior 샘플링**에 주입.
- [x] **효율 필터**: 성공 + 무충돌 + (cost < baseline·(1-margin)). `discriminator` 훅은 기본 None(미구현).
- [x] **config**: `PlannerConfig.{backbone,guidance_scale,cond_dropout_prob}`, `IDMConfig.backbone` 추가.

## 3. 참조 로직 (Reference Logic / 근거)
- **LDP (2504.16925)**: planner(미래 latent 상태열) + IDM 분리 → action-free·suboptimal 데이터 각각 활용. 본 프로젝트 골격의 토대.
- **AdaptDiffuser (ICML'23)**: self-evolving = generate→(reward+discriminator)filter→diffusion-loss finetune. 본 루프와 동형. **차이**: 우리는 실제 env에서 롤아웃 → feasibility가 env로 보장되어 discriminator 불필요(훅만 유지).
- **CompDiffuser/Trajectory Stitching (2503.05153)**: 지름길은 짧은 세그먼트의 조건부 합성(stitching)으로 창발. 효율은 직접 최적화가 아니라 feasibility+goal-reaching에서 간접 창발 → 짧은 horizon + closed-loop 재계획 + 버퍼 누적 방향 지지.
- **DPPO (ICLR'25)**: 역확산 체인을 MDP로 보는 PG 파인튜닝이 "구조적 on-manifold 탐색" 제공. → MVP는 filtered-BC, DPPO는 업그레이드 경로로 명시.
- **Flow matching (FlowPolicy AAAI'25 oral, MeanFlow MP1, Shortcut Models 2410.12557)**: diffusion 대비 7배+/1-step 추론. 롤아웃 대량 반복하는 자가성장 루프에서 샘플링 비용 절감 → 백본 교체 가능성을 인터페이스에 반영.

## 4. 가설 및 예상 결과 (Hypothesis)
- DDPM 기본으로 MVP feasibility를 먼저 검증하고, 루프가 동작하면 flow matching/shortcut 백본으로 갈아끼워 롤아웃 처리량을 끌어올린다.
- exploration_noise(on-manifold) + CFG planner 조합이 비효율 데모 분포 밖의 중앙 관통 지름길을 표집할 것으로 기대.

## 5. 결과 기록 (To be filled)
- (각 step 실행 후 dry-run/테스트 결과를 step별 experiment_records에 기록)
