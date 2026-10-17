"""Frozen desktop and allowlisted worker entry point; never installs dependencies."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def resource_root():
    return Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))


def self_test(gui=False):
    """Exercise bundled CPU geometry and worker dependencies with synthetic data."""
    import numpy as np
    import scipy
    import cv2
    import PIL
    import trimesh
    import viser
    import pycolmap
    from scipy.spatial.transform import Rotation
    from abu_m54 import SceneTransform
    from manual_camera_geometry import Camera
    from control_points_model import ControlDocument
    from spherical_refinement import triangulate
    assert np.allclose(Rotation.from_rotvec([.1,.2,.3]).as_matrix().T @ Rotation.from_rotvec([.1,.2,.3]).as_matrix(), np.eye(3))
    assert pycolmap.CameraModelId.EQUIRECTANGULAR is not None
    # Exercise native libraries rather than just checking package metadata.
    assert cv2.resize(np.zeros((8,16,3),np.uint8),(8,4)).shape == (4,8,3)
    report={'version':(resource_root()/'VERSION.txt').read_text().strip(),
            'frozen':bool(getattr(sys,'frozen',False)), 'numpy':np.__version__,
            'scipy':scipy.__version__, 'pycolmap':pycolmap.__version__,
            'cuda':bool(pycolmap.has_cuda), 'gui':False}
    if gui:
        import tkinter as tk
        from abu_align_gui import AbuAlignApp
        # Do not query upstream update services during an installation test.
        from unittest.mock import patch
        with patch.object(AbuAlignApp,'_start_background_engine_check',lambda self:None):
            root=tk.Tk();root.withdraw()
            try:
                app=AbuAlignApp(root);root.update_idletasks()
                if getattr(sys,'frozen',False):
                    assert app.engine_var.get()=='native_colmap'
                    assert not app.auto_train_var.get(), 'Core must not default to unbundled training'
                from control_points_workspace import open_editor
                # Main construction covers host adapters without reading a dataset.
                assert root.winfo_exists();report['gui']=True
            finally:root.destroy()
    return report


def dispatch(arguments=None):
    args=list(sys.argv[1:] if arguments is None else arguments)
    if os.environ.get('ABU_OFFLINE_TEST')=='1':
        import socket
        original=socket.socket.connect
        def local_only(sock,address):
            if isinstance(address,tuple) and address[0] not in ('127.0.0.1','::1','localhost'):
                raise RuntimeError('External network disabled for installation verification')
            return original(sock,address)
        socket.socket.connect=local_only
    if args and args[0]=='--worker-smoke-test':
        from installer.smoke_core import worker_test
        print(json.dumps(worker_test()),flush=True);return 0
    if args and args[0]=='--viewer-smoke-test':
        from installer.smoke_core import viewer_test
        print(json.dumps(viewer_test()),flush=True);return 0
    if args and args[0] in ('--self-test','--gui-smoke-test'):
        print(json.dumps(self_test(args[0]=='--gui-smoke-test')),flush=True);return 0
    if args and args[0]=='--doctor':
        from abu_components import capabilities
        print(json.dumps(capabilities(),indent=2));return 0
    if args and args[0]=='--m54':
        sys.argv=['abu_m54_app.py',*args[1:]]
        runpy.run_path(str(resource_root()/'abu_m54_app.py'),run_name='__main__');return 0
    # Existing application subprocesses use [sys.executable, '-B', script, ...].
    # A frozen runtime must explicitly dispatch these instead of relaunching its GUI.
    if args and args[0]=='-B':args.pop(0)
    if args and args[0].endswith('.py'):
        manifest=json.loads((resource_root()/'installer/core-files.json').read_text())
        requested=Path(args[0]).resolve()
        expected=resource_root()/requested.name
        allowed={Path(p).name for p in manifest['files'] if p.endswith('.py')}
        if requested.name not in allowed or requested != expected.resolve():
            raise ValueError('Only bundled application workers can run in the installed runtime.')
        sys.argv=[str(expected),*args[1:]]
        runpy.run_path(str(expected),run_name='__main__');return 0
    if args:raise ValueError('Unknown runtime option. Use --doctor or --m54 PATH.')
    from abu_align_gui import main
    main();return 0


if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(dispatch())
