"""Synthetic visual QA scene; run explicitly, never loads user data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import numpy as np
import viser
from viewer_workspace import InspectionWorkspace


if __name__ == '__main__':
    rng = np.random.default_rng(12)
    points = rng.uniform([-4, -3, 0], [4, 3, 4], (35000, 3))
    faces = rng.integers(0, 3, len(points))
    points[np.arange(len(points)), faces] = np.where(rng.random(len(points)) > .5,
        np.array([4, 3, 4])[faces], np.array([-4, -3, 0])[faces])
    colors = np.clip(np.array([120, 180, 170])+points*15, 0, 255).astype(np.uint8)
    cameras = [dict(name=f'images/frame_{i:03d}.jpg', pos=np.array([6*np.cos(i*.25), 6*np.sin(i*.25), 2.]),
                    wxyz=np.array([1., 0, 0, 0])) for i in range(24)]
    def load(path):
        offset = np.array([.18, 0, .08]) if path == 'b' else np.zeros(3)
        return dict(points=(points+offset).astype(np.float32), colors=colors,
            cameras=[dict(c, pos=c['pos']+offset) for c in cameras],
            source='Synthetic review fixture', asset='Demonstration only')
    runs = {'01 · Reference alignment': 'a', '02 · Comparison alignment': 'b'}
    server = viser.ViserServer(host='127.0.0.1', port=8898)
    server.gui.configure_theme(control_layout='collapsible', control_width='medium', dark_mode=True,
        brand_color=(112, 219, 192), show_logo=False, show_share_button=True)
    workspace = InspectionWorkspace(server, runs, next(iter(runs)), 'Workspace design review', load, lambda: runs)
    try:
        time.sleep(300)
    finally:
        server.stop()
