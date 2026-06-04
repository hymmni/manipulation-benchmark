# Harness Framework: Robot Behavior Intelligence Research

## 🚀 세션 시작 체크리스트
새 세션을 시작할 때 반드시 수행하라:
1. `README.md`를 읽고 프로젝트 구조와 적용 방법을 파악한다.
2. 가상환경이 있는지 확인한다(`venv/`, `.venv/`, `conda` 환경 등). 있으면 해당 환경에서 작업한다.
   - conda: `conda activate <env>` / venv: `source venv/bin/activate`
3. `continuation_plan.md`가 있으면 읽고 이전 세션의 중단 지점부터 이어서 시작한다.

## 🧪 로봇 연구 프로토콜
- **수정 범위 (Write Scope)**: 모든 소스 코드 수정은 이 작업 레포지토리 내부에서 수행합니다. (단, `references/`는 읽기 전용 — 아래 참조)
- **환경 관리 (Environment)**: 패키지 설치 및 \`requirements.txt\`, \`environment.yml\` 생성/수정은 레포지토리 루트에서 관리합니다.
- **참조 범위 (Read Scope)**: \`references/\` 폴더의 오픈소스 코드를 분석하여 로직을 이식하되, 수정은 절대 금지합니다.
- **출처 표기**: `references/`에서 코드를 가져올 경우 `# From: references/repo_name/file.py`와 같이 주석을 남기십시오.
- **실험 기록**: 의미 있는 변화(알고리즘 교체, 핵심 파라미터 변경) 발생 시 `experiments/` 내에 기록을 남기십시오.

## 🤖 기술 스택 (Robot Learning)
- **Core**: Python, PyTorch (Device management: cuda/mps/cpu)
- **Policy**: Diffusion Policy, Flow Matching, ACT, etc.
- **Config**: Hydra (preferred), YAML
- **Data**: Zarr, HDF5, Gym/Robosuite environments
- **Experiment**: WandB, Tensorboard

## 🏗️ 아키텍처 가이드 (Robot Pipeline)
1. **Dataset/Replay Buffer**: Data loading & normalization
2. **Policy/Network**: Neural network architecture
3. **Environment Wrapper**: State/Action space mapping & observation stacking
4. **Trainer/Evaluator**: Training loops and simulation/real-world benchmarks

## 🖥️ 하드웨어 환경 (3-PC Workflow)
현재 프로젝트는 다음 3대의 환경을 오가며 개발됩니다. 코드를 설계할 때 이 환경의 제약을 반드시 고려하십시오.
1. **코딩용 PC (Local)**: GPU 없음. 코드 작성, 리팩토링, Git 관리 수행. (이 하네스가 주로 실행되는 곳)
2. **학습용 PC (Server)**: GPU 있음. 대용량 데이터 전처리 및 모델 학습 (WandB로 로깅).
3. **추론용 PC (Robot)**: GPU 있음. 로봇과 연결하여 실제 모델 추론 및 배포.
* **주의**: 코딩용 PC에는 GPU가 없으므로, 하네스 환경 내에서 코드를 테스트할 때는 CPU Fallback(`device='cuda' if torch.cuda.is_available() else 'cpu'`) 처리가 되어 있어야 합니다.

## 📝 개발 프로세스
- **Phase Execution**: `scripts/execute.py`를 사용하여 복잡한 리팩토링이나 구현 단계를 안전하게 수행하십시오.
- **Commit Message**: Scoped Conventional Commits 사용. **커밋 메시지(제목·본문)는 영어로 작성**한다. 괄호 안에 수정된 모듈 영역(`policy`, `env`, `data`, `config`, `harness` 등 베이스라인 이름이나 모듈)을 명시하고, **반드시 본문에 멀티라인(여러 줄) 상세 설명을 추가**하십시오. (예: `feat(policy): short description` + 본문 상세)
- 작업을 완료할 때마다 `experiments/`에 수정 사항 요약을 작성하십시오.

## 🤖 모델 선택 가이드
작업 복잡도에 따라 적절한 모델을 사용자에게 제안하라. 클로드는 실행 중인 세션의 모델을 변경할 수 없으므로, 모델 선택은 **세션 시작 전** 또는 **execute.py 실행 시** 이루어진다.

| 모델 | 적합한 작업 |
|---|---|
| **opus** | 신규 아키텍처 설계, 복잡한 알고리즘 구현, 다단계 추론이 필요한 phase |
| **sonnet** | 일반 코딩, 리팩토링, 대부분의 day-to-day 작업 (기본값) |
| **haiku** | 단순 수정, 문서 작성, 빠른 조회성 작업 |

execute.py로 step을 실행할 때 모델 지정:
```bash
python3 scripts/execute.py <phase_dir> --model opus   # 복잡한 phase
python3 scripts/execute.py <phase_dir> --model haiku  # 단순한 phase
```

## 🛠️ 유틸리티 명령어
- `python scripts/execute.py <phase_dir> [--model MODEL]` # 클로드의 자가 교정 실행 (하네스 내부용)
- `python scripts/merge_to_main.py <feat-branch> [--push]` # feature 브랜치를 main에 병합 (pull→rebase→`--no-ff`)
- `python scripts/schedule_continuation.py [--reset-at HH:MM]` # 세션 한도 해제 시각에 작업 재시작 자동 예약

### ⏰ 세션 연속 규칙
대화가 길어져 세션 한도에 근접하면:
1. **사용자에게 먼저 제안**한다: "세션 한도가 가까워졌습니다. 작업을 이어서 예약할까요?"
2. 사용자가 동의하면 `/schedule-continuation` 스킬을 실행한다.
3. 스킬이 지시하는 대로: `continuation_plan.md` 작성 → `schedule_continuation.py` 실행.
4. **세션 ID와 리셋 시각은 스크립트가 자동으로 감지**한다. 추측하거나 수동 입력하지 마라.

### 🔀 main 병합 규칙 (CRITICAL)
사용자가 feature 브랜치를 **main에 병합**해달라고 요청하면:
- **반드시 `scripts/merge_to_main.py` 사용을 안내하라.** `git merge`/`git rebase`를 직접 치지 마라. 이유: 이 스크립트가 `pull --ff-only` → `rebase` → `--no-ff merge` 순서와 충돌 시 자동 abort를 보장한다.
- **클로드가 직접 실행하지 마라.** main/origin을 건드리는 작업이므로, 실행할 명령어(`python3 scripts/merge_to_main.py <feat-branch>`)를 제시하고 **사용자가 직접 실행**하게 하라. (사용자가 명시적으로 "네가 실행해"라고 하면 그때만 `--yes`를 붙여 실행)
- step 압축(squash)은 feature 브랜치 내부에서만 일어나며, main 병합 시에는 각 step 커밋을 그대로 보존한다.
