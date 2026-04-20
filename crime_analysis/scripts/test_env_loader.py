"""Unit tests for env_loader — never touches real .env file or real keys."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env_loader import load_dotenv, _parse_line  # noqa: E402


def run():
    cases = []

    # ── 1. _parse_line parses simple key=value ──
    assert _parse_line("FOO=bar") == ("FOO", "bar")
    assert _parse_line("FOO_BAR=baz qux") == ("FOO_BAR", "baz qux")
    cases.append("parse simple assignment")

    # ── 2. _parse_line strips matching quotes ──
    assert _parse_line('KEY="quoted value"') == ("KEY", "quoted value")
    assert _parse_line("KEY='single'") == ("KEY", "single")
    # mismatched quotes: kept verbatim
    assert _parse_line('KEY="mismatch') == ("KEY", '"mismatch')
    cases.append("parse quoted values")

    # ── 3. _parse_line skips comments, blanks, malformed ──
    assert _parse_line("") is None
    assert _parse_line("   ") is None
    assert _parse_line("# comment") is None
    assert _parse_line("no_equals_sign") is None
    assert _parse_line("=value") is None  # empty key
    assert _parse_line("bad key=value") is None  # space in key
    cases.append("skip comments / malformed")

    # ── 4. load_dotenv sets env vars (no override of existing) ──
    os.environ.pop("TEST_ENV_FOO", None)
    os.environ.pop("TEST_ENV_BAR", None)
    os.environ["TEST_ENV_BAR"] = "shell_wins"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".env", delete=False, encoding="utf-8"
    ) as f:
        f.write("# sample env\n")
        f.write("TEST_ENV_FOO=hello\n")
        f.write('TEST_ENV_BAR="from_file"\n')
        f.write("TEST_ENV_EMPTY=\n")
        tmp_path = Path(f.name)

    try:
        set_keys = load_dotenv(tmp_path)
        assert "TEST_ENV_FOO" in set_keys
        # TEST_ENV_BAR should NOT be in set_keys — shell win (override=False default)
        assert "TEST_ENV_BAR" not in set_keys
        assert os.environ["TEST_ENV_FOO"] == "hello"
        assert os.environ["TEST_ENV_BAR"] == "shell_wins"
        assert os.environ["TEST_ENV_EMPTY"] == ""
        cases.append("load default: shell wins")

        # ── 5. override=True flips it ──
        set_keys2 = load_dotenv(tmp_path, override=True)
        assert "TEST_ENV_BAR" in set_keys2
        assert os.environ["TEST_ENV_BAR"] == "from_file"
        cases.append("override=True beats shell")
    finally:
        tmp_path.unlink()
        for k in ("TEST_ENV_FOO", "TEST_ENV_BAR", "TEST_ENV_EMPTY"):
            os.environ.pop(k, None)

    # ── 6. load_dotenv on missing file returns empty, no crash ──
    assert load_dotenv(Path("/nonexistent/path.env")) == []
    cases.append("missing file → []")

    # ── 7. no-hardcoded-keys sanity: env_loader.py shouldn't contain AIza/sk- ──
    loader_src = (ROOT / "env_loader.py").read_text(encoding="utf-8")
    assert "AIza" not in loader_src and "sk-" not in loader_src
    cases.append("env_loader.py has no hardcoded keys")

    # ── 8. .gitignore includes .env ──
    gitignore = ROOT.parent / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text(encoding="utf-8")
        assert ".env" in content, ".gitignore must exclude .env"
        cases.append(".gitignore excludes .env")

    print(f"Ran {len(cases)} env_loader tests:")
    for c in cases:
        print(f"  ✓ {c}")
    print("\nAll assertions passed ✓")


if __name__ == "__main__":
    run()
