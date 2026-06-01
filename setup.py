#!/usr/bin/env python
import os
import subprocess
import time

from setuptools import setup

version_file = "hat/version.py"


def get_git_hash():
    def _minimal_ext_cmd(cmd):
        env = {}
        for k in ["SYSTEMROOT", "PATH", "HOME"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        sha = out.strip().decode("ascii")
    except OSError:
        sha = "unknown"

    return sha


def get_hash():
    if os.path.exists(".git"):
        sha = get_git_hash()[:7]
    else:
        sha = "unknown"
    return sha


def write_version_py():
    content = """# GENERATED VERSION FILE
# TIME: {}
__version__ = '{}'
__gitsha__ = '{}'
version_info = ({})
"""
    sha = get_hash()
    with open("VERSION", "r") as f:
        SHORT_VERSION = f.read().strip()
    VERSION_INFO = ", ".join([x if x.isdigit() else f'"{x}"' for x in SHORT_VERSION.split(".")])

    version_file_str = content.format(time.asctime(), SHORT_VERSION, sha, VERSION_INFO)
    with open(version_file, "w") as f:
        f.write(version_file_str)


def get_version():
    ns = {}
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"), ns)
    return ns["__version__"]


write_version_py()
setup(version=get_version())
