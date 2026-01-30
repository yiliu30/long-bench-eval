#!/usr/bin/env python3
"""Compatibility shim for invoking the packaged CLI from source checkouts."""

from long_bench_eval.cli import main


if __name__ == "__main__":
    main()
