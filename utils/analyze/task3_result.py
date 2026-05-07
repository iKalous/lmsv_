#!/usr/bin/env python3

from __future__ import annotations

from .task1_result import AnalysisArtifacts, analyze_task3_run, main_cli_for_task

__all__ = ["AnalysisArtifacts", "analyze_task3_run"]


def main_cli(argv: list[str] | None = None) -> int:
    return main_cli_for_task(3, argv)


if __name__ == "__main__":
    raise SystemExit(main_cli())
