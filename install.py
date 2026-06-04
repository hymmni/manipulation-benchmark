#!/usr/bin/env python3
"""
Harness 부트스트랩 설치기 — 작업 레포 안에 클론한 템플릿을 적용하고 스스로 정리한다.

기존 프로젝트에 적용 (단 두 단계):

    cd /path/to/your-project           # 적용 대상 프로젝트 루트
    git clone <template-url> harness   # 작업 레포 "안에" harness/ 로 클론
    python3 harness/install.py         # 이 한 줄로 끝

이 스크립트가 하는 일:
    1. 클론(harness/)의 harness 파일을 프로젝트 루트로 복사 (없는 파일만, 기존 파일·.git 보존)
    2. 기존 .gitignore 에 harness 규칙을 멱등하게 추가
    3. harness 디렉토리(experiments/ references/ phases/) 생성
    4. 클론 디렉토리(harness/)를 통째로 삭제해 스스로 정리

기존 파일을 절대 덮어쓰지 않는다(비파괴). 기존 CLAUDE.md 가 있으면 harness 버전을
CLAUDE.harness.md 로 남겨 수동 병합하게 한다.

from scratch로 시작하는 경우엔 이 스크립트가 필요 없다 — 템플릿을 프로젝트로 직접 클론한 뒤
`rm install.py` 로 지우면 된다.
"""

import argparse
import shutil
import sys
from pathlib import Path

CLONE_ROOT = Path(__file__).resolve().parent   # 클론된 템플릿 루트 (= harness/)
TARGET_ROOT = CLONE_ROOT.parent                # 적용 대상 프로젝트 루트

# --- .gitignore 블록 (단일 출처). 이 레포의 .gitignore 도 --print-gitignore 출력으로 생성된다. ---
GITIGNORE_BEGIN = "# >>> harness gitignore (managed) >>>"
GITIGNORE_END = "# <<< harness gitignore (managed) <<<"
GITIGNORE_BLOCK = """\
CLAUDE.md
CLAUDE.harness.md
.claude/
docs/
experiments/
phases/
references/
scripts/
continuation_plan.md
"""

# 어느 깊이에서든 복사하지 않을 잡파일 (git 메타, 캐시 등)
ALWAYS_SKIP = {".git", "__pycache__", ".pytest_cache", ".DS_Store"}
# 클론 최상위에서만 제외할 항목 (설치 도구 자체 + README, .gitignore는 append로 처리)
TOP_EXCLUDE = {"install.py", "README.md", ".gitignore"}
HARNESS_DIRS = ("experiments", "references", "phases")


def gitignore_text() -> str:
    return f"{GITIGNORE_BEGIN}\n{GITIGNORE_BLOCK}{GITIGNORE_END}\n"


def install_gitignore(target: Path) -> str:
    existing = target.read_text(encoding="utf-8") if target.exists() else ""
    if GITIGNORE_BEGIN in existing:
        return ".gitignore: 이미 harness 블록 존재 (변경 없음)"
    sep = "" if existing == "" or existing.endswith("\n") else "\n"
    gap = "\n" if existing.strip() else ""
    target.write_text(existing + sep + gap + gitignore_text(), encoding="utf-8")
    return ".gitignore: harness 블록 추가" if existing else ".gitignore: 생성 + harness 블록 추가"


def copy_tree(src: Path, dst: Path, top: bool = True) -> int:
    """src 내용을 dst로 복사. 기존 파일은 건너뛴다(비파괴). 복사한 파일 수를 반환."""
    count = 0
    for item in src.iterdir():
        if item.name in ALWAYS_SKIP or (top and item.name in TOP_EXCLUDE):
            continue
        target = dst / item.name
        if item.is_dir():
            target.mkdir(exist_ok=True)
            count += copy_tree(item, target, top=False)
        elif not target.exists():
            shutil.copy2(item, target)
            count += 1
    return count


def confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        return input(f"{prompt} [y/n] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def run_install(assume_yes: bool) -> int:
    if not (CLONE_ROOT / "CLAUDE.md").exists():
        print("  ERROR: 이 스크립트는 git clone한 harness 템플릿 안에서 실행해야 합니다.")
        print("         cd <your-project> && git clone <url> harness && python3 harness/install.py")
        return 1

    print(f"\n  Harness 설치")
    print(f"    from (clone) : {CLONE_ROOT}")
    print(f"    into (target): {TARGET_ROOT}")
    if not confirm("  위 target 에 harness를 설치하고 clone을 삭제합니다. 계속할까요?", assume_yes):
        print("  중단됨.")
        return 1

    # 기존 CLAUDE.md 충돌 시 harness 버전을 보존 (clone 삭제 후에도 병합 가능하도록)
    claude_conflict = (TARGET_ROOT / "CLAUDE.md").exists()
    if claude_conflict:
        shutil.copy2(CLONE_ROOT / "CLAUDE.md", TARGET_ROOT / "CLAUDE.harness.md")

    n = copy_tree(CLONE_ROOT, TARGET_ROOT)
    print(f"  복사: {n}개 파일 (기존 파일은 보존)")
    print(f"  {install_gitignore(TARGET_ROOT / '.gitignore')}")

    for d in HARNESS_DIRS:
        p = TARGET_ROOT / d
        if not p.exists():
            p.mkdir(parents=True)
            (p / ".gitkeep").touch()
            print(f"  디렉토리 생성: {d}/")

    shutil.rmtree(CLONE_ROOT)
    print(f"  clone 삭제: {CLONE_ROOT.name}/")

    if claude_conflict:
        print("  ⚠ 기존 CLAUDE.md 발견 — harness 버전을 CLAUDE.harness.md 로 저장했습니다.")
        print("    두 파일을 수동 병합한 뒤 CLAUDE.harness.md 를 삭제하세요.")
    print("  ✓ 완료. 이제 /harness 로 step 설계를 시작하세요.\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="작업 레포 안에 클론한 harness 템플릿을 적용한다.")
    ap.add_argument("--yes", action="store_true", help="확인 프롬프트 생략")
    ap.add_argument("--print-gitignore", action="store_true",
                    help=".gitignore 블록만 출력하고 종료 (템플릿 유지보수용)")
    args = ap.parse_args()

    if args.print_gitignore:
        sys.stdout.write(gitignore_text())
        return 0
    return run_install(args.yes)


if __name__ == "__main__":
    sys.exit(main())
