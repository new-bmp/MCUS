from __future__ import annotations

import argparse
import sys

from app.nexus_lerobot_export import add_parser


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert processed Nexus v4 data to LeRobot")
    subparsers = parser.add_subparsers(dest="command")
    converter = add_parser(subparsers)
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list or args_list[0] != "nexus-lerobot":
        args_list.insert(0, "nexus-lerobot")
    args = parser.parse_args(args_list)
    try:
        return int(args.handler(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"nexus_to_lerobot: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
