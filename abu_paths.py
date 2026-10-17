"""Per-user mutable state; installed files and source geometry stay immutable."""
import os
from pathlib import Path
import sys


def user_data_dir():
    if os.environ.get('ABU_DATA_HOME'):return Path(os.environ['ABU_DATA_HOME']).expanduser()
    if sys.platform=='win32':return Path(os.environ.get('LOCALAPPDATA',Path.home()/'AppData/Local'))/'AbuAlign'
    if sys.platform=='darwin':return Path.home()/'Library/Application Support/AbuAlign'
    return Path(os.environ.get('XDG_DATA_HOME',Path.home()/'.local/share'))/'abu-align'


def state_dir(name):
    root=user_data_dir()/name;root.mkdir(parents=True,exist_ok=True);return root
