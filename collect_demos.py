import sys
import os

from config.config import Config
from data.demo_collector import DemoCollector, _has_display


def main() -> None:
    cfg = Config.default()
    if not _has_display():
        print(
            "No graphical display detected (DISPLAY/WAYLAND_DISPLAY not set).\n"
            "Run this script on a machine with a display or via X forwarding."
        )
        sys.exit(1)

    collector = DemoCollector(cfg.env, cfg.buffer.demo_path)
    print(f"Saving demos to: {cfg.buffer.demo_path}")
    collector.run()


if __name__ == "__main__":
    main()
