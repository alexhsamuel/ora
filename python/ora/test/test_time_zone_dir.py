import os
from   pathlib import Path
import pytest
import subprocess
import sys

try:
    import tzdata
except ImportError:
    tzdata = None

# Note: if PYTHONTZPATH is the empty string, `zoneinfo.TZPATH` contains a single
# entry, `.`.  However, `zoneinfo` ignores relative dir paths in `TZPATH`.

#-------------------------------------------------------------------------------

def run(script, zoneinfo_path):
    """
    Runs `script` in a Python subprocess with env var PYTHONTZPATH set to
    `zoneinfo_path` or none.
    """
    env_var = "PYTHONTZPATH"

    env = os.environ.copy()
    if zoneinfo_path is None:
        env.pop(env_var, None)
    else:
        env[env_var] = str(zoneinfo_path)

    proc = subprocess.run(
        [sys.executable, "-c", script],
        env     =env,
        stdout  =subprocess.PIPE,
        text    =True,
        check   =True,
    )
    return proc.stdout


@pytest.mark.parametrize("zoneinfo_path", (None, ""))
def test_sytem_zoneinfo_dir(zoneinfo_path):
    # Get zoneinfo's path.
    path = run(
        "import zoneinfo; print(':'.join(zoneinfo.TZPATH))",
        zoneinfo_path
    )
    path = [ Path(p) for p in path.strip().split(":") if Path(p).is_absolute() ]

    # Find the first entry that exists.
    try:
        zoneinfo_dir = next( p for p in path if p.is_dir() )
    except StopIteration:
        if tzdata is None:
            zoneinfo_dir = ora._INTERNAL_ZONEINFO_DIR
        else:
            zoneinfo_dir = Path(tzdata.__file__).parent / "zoneinfo"
    assert zoneinfo_dir.is_dir()

    # Get Ora's zoneinfo dir.
    ora_dir = run("import ora; print(ora.get_zoneinfo_dir())", zoneinfo_path)
    ora_dir = Path(ora_dir.strip())
    assert ora_dir == zoneinfo_dir


