"""Run unit/UI tests with external socket connections forbidden and record evidence."""
from __future__ import annotations
import io
import json
import os
import socket
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"


def main():
    output = ROOT / "docs" / "evidence"
    output.mkdir(parents=True, exist_ok=True)
    scratch = ROOT / "data" / ".cache" / "test-temp"
    scratch.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(scratch)
    stream = io.StringIO()
    started = time.perf_counter()
    with patch.object(socket.socket, "connect", side_effect=AssertionError("Offline test attempted a network connection")):
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {"recorded_at": datetime.now(timezone.utc).isoformat(), "mode": "offline_mock",
              "network": "socket.connect forbidden", "paid_api_calls": 0,
              "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "duration_seconds": time.perf_counter() - started}
    (output / "offline-tests.txt").write_text(stream.getvalue(), encoding="utf-8")
    (output / "offline-tests.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(stream.getvalue())
    print(json.dumps(report))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
