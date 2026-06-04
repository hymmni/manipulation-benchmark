#!/usr/bin/env python3
"""
현재 Claude Code 세션을 감지하고, 한도 해제 시각에 자동으로 이어서 실행을 예약한다.

Claude가 세션 한도에 근접했을 때 실행한다:
    python3 scripts/schedule_continuation.py

리셋 시각을 직접 지정하려면:
    python3 scripts/schedule_continuation.py --reset-at 14:30

프롬프트 파일을 지정하려면:
    python3 scripts/schedule_continuation.py --prompt-file continuation_plan.md
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEDULER = Path(__file__).resolve().parent / "scheduler.py"
SESSIONS_DIR = Path.home() / ".claude" / "sessions"
SESSION_LIMIT_HOURS = 5
CONTINUATION_PLAN_FILE = ROOT / "continuation_plan.md"

DEFAULT_PROMPT = """\
이전 세션에서 작업을 이어서 진행합니다.
continuation_plan.md 파일이 있으면 읽고, 중단된 지점부터 계속 진행하세요.
"""


def find_current_session() -> tuple[str | None, datetime.datetime | None]:
    """환경변수와 세션 파일에서 현재 세션 ID와 시작 시간을 찾는다."""
    session_id = os.environ.get("CLAUDE_CODE_SESSION_ID")
    if not session_id:
        return None, None

    for f in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            if data.get("sessionId") == session_id:
                started_ms = data.get("startedAt")
                if started_ms:
                    started = datetime.datetime.fromtimestamp(started_ms / 1000)
                    return session_id, started
        except Exception:
            continue

    return session_id, None


def calculate_reset_time(started_at: datetime.datetime) -> datetime.datetime:
    return started_at + datetime.timedelta(hours=SESSION_LIMIT_HOURS)


def parse_args():
    p = argparse.ArgumentParser(description="세션 한도 해제 시각에 Claude 재시작을 예약한다.")
    p.add_argument("--reset-at", metavar="HH:MM",
                   help="한도 해제 시각 (자동 계산 불가 시 직접 지정)")
    p.add_argument("--prompt", metavar="TEXT",
                   help="재시작 시 전달할 프롬프트 (기본: continuation_plan.md 읽기 안내)")
    p.add_argument("--prompt-file", metavar="PATH",
                   help="프롬프트를 읽어올 파일 경로")
    p.add_argument("--model", metavar="MODEL",
                   help="모델 (sonnet | opus | haiku)")
    p.add_argument("--permission-mode", metavar="MODE",
                   choices=["auto", "plan", "acceptEdits", "dontAsk"],
                   help="권한 모드")
    return p.parse_args()


def main():
    cli = parse_args()

    session_id, started_at = find_current_session()

    # 세션 ID
    if not session_id:
        print("  ERROR: CLAUDE_CODE_SESSION_ID 환경변수를 찾을 수 없습니다.")
        print("  Claude Code 세션 안에서 실행하세요.")
        sys.exit(1)

    # 리셋 시각 결정
    if cli.reset_at:
        t = datetime.datetime.strptime(cli.reset_at, "%H:%M").time()
        reset_time = datetime.datetime.combine(datetime.date.today(), t)
        if reset_time <= datetime.datetime.now():
            reset_time += datetime.timedelta(days=1)
        time_source = "직접 지정"
    else:
        # 자동 계산은 이 세션 기준이라 다른 세션이 먼저 토큰을 썼다면 부정확할 수 있다.
        # Claude Code UI에 표시되는 실제 리셋 시각을 --reset-at 으로 지정하는 것이 정확하다.
        if started_at:
            estimated = calculate_reset_time(started_at)
            if estimated <= datetime.datetime.now():
                estimated = datetime.datetime.now() + datetime.timedelta(hours=SESSION_LIMIT_HOURS)
            estimate_str = estimated.strftime("%H:%M")
            time_source = f"추정 (이 세션 시작 기준, 부정확할 수 있음)"
        else:
            estimated = datetime.datetime.now() + datetime.timedelta(hours=SESSION_LIMIT_HOURS)
            estimate_str = estimated.strftime("%H:%M")
            time_source = "추정 (현재 시각 기준, 부정확할 수 있음)"

        print(f"  추정 리셋 시각: {estimate_str}  ({time_source})")
        print(f"  정확한 시각은 --reset-at HH:MM 으로 지정하세요.")
        reset_time = estimated
        time_source = f"추정 ({estimate_str})"

    # 프롬프트 결정
    if cli.prompt:
        prompt = cli.prompt
    elif cli.prompt_file:
        try:
            prompt = Path(cli.prompt_file).read_text()
        except FileNotFoundError:
            print(f"  ERROR: 프롬프트 파일 {cli.prompt_file} 을 찾을 수 없습니다.")
            sys.exit(1)
    elif CONTINUATION_PLAN_FILE.exists():
        plan = CONTINUATION_PLAN_FILE.read_text().strip()
        prompt = f"이전 세션에서 작업을 이어서 진행합니다.\n\n[중단 시점 계획]\n{plan}"
    else:
        prompt = DEFAULT_PROMPT

    # scheduler.py 호출
    reset_str = reset_time.strftime("%H:%M")
    cmd = [
        sys.executable, str(SCHEDULER),
        "--time", reset_str,
        "--resume", session_id,
        "--prompt", prompt,
        "--yes",
    ]
    if cli.model:
        cmd += ["--model", cli.model]
    if cli.permission_mode:
        cmd += ["--permission-mode", cli.permission_mode]

    remaining = reset_time - datetime.datetime.now()
    h, rem = divmod(int(remaining.total_seconds()), 3600)
    m = rem // 60
    remaining_str = f"{h}시간 {m}분 후" if h else f"{m}분 후"

    print(f"\n  세션 연속 예약")
    print(f"  세션 ID : {session_id}")
    print(f"  리셋 시각: {reset_time.strftime('%H:%M')}  ({remaining_str}, {time_source})")
    if CONTINUATION_PLAN_FILE.exists():
        print(f"  계획 파일: continuation_plan.md")
    print(f"  프롬프트: {prompt[:80].strip()}{'...' if len(prompt) > 80 else ''}")
    print()

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
