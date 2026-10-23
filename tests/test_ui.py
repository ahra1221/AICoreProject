"""Streamlit UI execution without a browser, model key or network."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest
from src.config import get_settings

ROOT = Path(__file__).resolve().parents[1]


class UiTests(unittest.TestCase):
    def test_empty_index_and_key_guidance(self):
        st.cache_resource.clear()
        fake_index = SimpleNamespace(counts=lambda: {"movie_meta": 0, "external_docs": 0},
                                     movie_collection=SimpleNamespace(get=lambda **kw: {"ids": []}))
        with patch("src.config.get_settings", return_value=get_settings({})), patch("src.indexer.MovieIndexer", return_value=fake_index):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        warnings = [element.value for element in app.warning]
        infos = [element.value for element in app.info]
        self.assertTrue(any("색인 데이터가 없습니다" in value for value in warnings))
        self.assertTrue(any("GMS_KEY" in value for value in infos))
        self._save("guidance", {"warnings": warnings, "infos": infos, "uncaught_exceptions": 0})

    def test_startup_error_does_not_expose_details(self):
        st.cache_resource.clear()
        marker = "PRIVATE_EXCEPTION_TEST_MARKER"
        with patch("src.config.get_settings", return_value=get_settings({})), patch("src.indexer.MovieIndexer", side_effect=ValueError(marker)):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        errors = [element.value for element in app.error]
        self.assertEqual(len(errors), 1)
        self.assertNotIn(marker, errors[0])
        self._save("safe-error", {"screen_errors": errors, "private_marker_visible": False, "uncaught_exceptions": 0})

    def test_generic_gms_environment_key_is_not_inherited(self):
        with patch.dict("os.environ", {"GMS_KEY": "unrelated-project-key", "MOVIELAB_GMS_KEY": ""}), patch("src.config.dotenv_values", return_value={}):
            self.assertEqual(get_settings().gms_key, "")

    @staticmethod
    def _save(name, value):
        output = ROOT / "docs/evidence"
        output.mkdir(parents=True, exist_ok=True)
        (output / f"ui-{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
