"""Flat-layout compatibility packaging with an explicit resource allowlist."""
import json
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py

ROOT=Path(__file__).parent
FILES=json.loads((ROOT/'installer/core-files.json').read_text())['files']


class BuildCore(build_py):
    def run(self):
        super().run()
        for relative in FILES:
            source=ROOT/relative
            target=Path(self.build_lib)/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            self.copy_file(str(source),str(target))


setup(py_modules=sorted({Path(p).stem for p in FILES if Path(p).parent==Path('.') and p.endswith('.py') and Path(p).stem.isidentifier()}|{'abu_runtime'}),cmdclass={'build_py':BuildCore})
