#!/usr/bin/env python3
"""
예약 시간에 Claude 세션을 자동으로 시작하는 스케줄러.

사용법 A — 스크립트 직접 편집:
    아래 CONFIG 섹션을 수정한 뒤 실행한다.
    python3 scheduler.py

사용법 B — 커맨드라인 인자:
    python3 scheduler.py --time 07:00 --resume <session-id> --prompt "프롬프트"
    python3 scheduler.py --time 23:30 --cmd "claude -p" --prompt "작업 내용"
    python3 scheduler.py --time 07:00 --resume <id> --model opus --permission-mode auto --prompt "작업"
"""

import argparse
import datetime
import shlex
import subprocess
import sys
import time

# ──────────────────────────────────────────────
# CONFIG (사용법 A: 여기만 수정)
# ──────────────────────────────────────────────
TARGET_TIME = "07:00"  # 24시간제 HH:MM

# 세션 ID: 비워두면 새 세션으로 시작
SESSION_ID = ""

# 모델: "sonnet" | "opus" | "haiku"
# 비워두면 Claude Code 설정값 사용 (세션에 지정된 모델 또는 CC 기본값)
MODEL = ""

# 권한 모드: "auto"(자동 승인) | "plan"(계획만) | "acceptEdits"(편집 자동 승인) | "dontAsk"(전체 자동 승인)
# 비워두면 Claude Code 설정값 사용
PERMISSION_MODE = ""

PROMPT = """여기에 프롬프트를 입력하세요."""
# ──────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="예약 시간에 Claude를 실행한다.")
    p.add_argument("--time", metavar="HH:MM", help="실행 시각 (24시간제)")
    p.add_argument("--resume", metavar="SESSION_ID", help="이어서 실행할 세션 ID")
    p.add_argument("--cmd", metavar="COMMAND", help="실행할 CLI 명령어 (--resume 대신)")
    p.add_argument("--model", metavar="MODEL",
                   help="모델 (sonnet | opus | haiku)")
    p.add_argument("--permission-mode", metavar="MODE",
                   choices=["auto", "plan", "acceptEdits", "dontAsk"],
                   help="권한 모드 (auto=자동승인 | plan=계획만 | acceptEdits=편집승인 | dontAsk=전체승인)")
    p.add_argument("--prompt", metavar="TEXT", help="세션에 전달할 프롬프트")
    p.add_argument("--yes", action="store_true", help="대기 확인 없이 즉시 예약 (schedule_continuation.py 내부 호출용)")
    return p.parse_args()


def next_target(time_str: str) -> datetime.datetime:
    """오늘 또는 내일의 목표 datetime을 반환한다."""
    t = datetime.datetime.strptime(time_str, "%H:%M").time()
    now = datetime.datetime.now()
    candidate = datetime.datetime.combine(now.date(), t)
    if candidate <= now:
        candidate += datetime.timedelta(days=1)
    return candidate


def fmt_delta(delta: datetime.timedelta) -> str:
    total = int(delta.total_seconds())
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}시간 {m}분 후"
    if m:
        return f"{m}분 {s}초 후"
    return f"{s}초 후"


def run(command: str, prompt: str):
    args = shlex.split(command)
    proc = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
    )
    proc.communicate(input=prompt)
    return proc.returncode


def main():
    cli = parse_args()

    time_str = cli.time or TARGET_TIME
    prompt = cli.prompt or PROMPT
    model = cli.model or MODEL
    permission_mode = getattr(cli, "permission_mode", None) or PERMISSION_MODE
    session_id = cli.resume or SESSION_ID

    if cli.cmd:
        command = cli.cmd
    elif session_id:
        command = f"claude --resume {session_id}"
    else:
        command = "claude"

    if model:
        command += f" --model {model}"
    if permission_mode:
        command += f" --permission-mode {permission_mode}"

    target = next_target(time_str)
    print(f"  예약: {target.strftime('%Y-%m-%d %H:%M')}  ({fmt_delta(target - datetime.datetime.now())})")
    print(f"  명령: {command}")
    print(f"  프롬프트: {prompt[:60].strip()}{'...' if len(prompt) > 60 else ''}")
    print("  (Ctrl+C 로 취소)\n")

    try:
        while True:
            now = datetime.datetime.now()
            remaining = target - now
            if remaining.total_seconds() <= 0:
                break
            if int(remaining.total_seconds()) % 300 == 0:  # 5분마다 갱신
                print(f"  대기 중... {fmt_delta(remaining)}", flush=True)
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n  취소됨.")
        sys.exit(0)

    print(f"\n  [{datetime.datetime.now().strftime('%H:%M:%S')}] 실행합니다...")
    code = run(command, prompt)
    if code == 0:
        print("\n  완료.")
    else:
        print(f"\n  종료 코드: {code}")
    sys.exit(code)


if __name__ == "__main__":
    main()
