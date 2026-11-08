"""The CPU-only selection must run without importing MLX (which initialises Metal)."""

import subprocess
import sys

from conftest import REPO

CHECK = """
import sys, pytest
rc = pytest.main(sys.argv[1:])
loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("mlx", "mlx_lm"))
print("MLX_MODULES", loaded)
sys.exit(rc or (3 if loaded else 0))
"""


def test_cpu_selection_does_not_import_mlx():
    proc = subprocess.run(
        [
            sys.executable, "-c", CHECK, str(REPO / "tests"), "-q", "-p", "no:cacheprovider",
            "-m", "not mlx and not integration", f"--ignore={__file__}",
        ],  # fmt: skip
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "MLX_MODULES []" in proc.stdout
