"""
Unit tests for NVIDIA PyCuSFM engine adapter in Abu Align
"""
import os
import sys
import json
import numpy as np

# Ensure root and Abu_Align are in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Abu_Align.pycusfm_engine import (
    is_wsl_available,
    to_wsl_path,
    to_win_path,
    quaternion_to_axis_angle,
    rotation_matrix_to_axis_angle,
    build_frames_meta_json,
    run_wsl_cmd
)

def test_wsl_detection():
    available = is_wsl_available()
    print(f"[TEST 1] WSL Availability: {available}")
    assert available, "WSL with Ubuntu must be available"

def test_path_translation():
    win_p = r"C:\Users\USer\datasets\scan_01\images"
    wsl_p = to_wsl_path(win_p)
    print(f"[TEST 2] Windows -> WSL: {win_p} -> {wsl_p}")
    assert wsl_p == "/mnt/c/Users/USer/datasets/scan_01/images"

    round_win = to_win_path(wsl_p)
    print(f"[TEST 2] WSL -> Windows: {wsl_p} -> {round_win}")
    assert round_win.lower().startswith("c:/")

def test_rotation_math():
    # Identity rotation
    R_eye = np.eye(3, dtype=np.float64)
    aa_eye = rotation_matrix_to_axis_angle(R_eye)
    print(f"[TEST 3] Identity Axis-Angle: {aa_eye}")
    assert abs(aa_eye["angle_degrees"]) < 1e-5

    # 90 degree rotation around Z axis
    theta = np.pi / 2.0
    R_z90 = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0]
    ])
    aa_z90 = rotation_matrix_to_axis_angle(R_z90)
    print(f"[TEST 3] 90deg Z-Rot Axis-Angle: {aa_z90}")
    assert abs(aa_z90["angle_degrees"] - 90.0) < 1e-3
    assert abs(aa_z90["z"] - 1.0) < 1e-3

def test_frames_meta_generation(tmp_path):
    os.makedirs(tmp_path, exist_ok=True)
    out_meta = os.path.join(tmp_path, "frames_meta.json")

    images = ["img_001.jpg", "img_002.jpg", "img_003.jpg"]
    poses = [
        (np.eye(3), np.array([0.0, 0.0, 0.0])),
        (np.eye(3), np.array([0.1, 0.0, 0.0])),
        (np.eye(3), np.array([0.2, 0.0, 0.0])),
    ]

    build_frames_meta_json(
        image_paths=images,
        poses_c2w=poses,
        img_width=1920,
        img_height=1080,
        fx=1500.0,
        fy=1500.0,
        cx=960.0,
        cy=540.0,
        output_meta_path=out_meta
    )

    assert os.path.isfile(out_meta)
    with open(out_meta, "r") as f:
        data = json.load(f)

    assert data["initial_pose_type"] == "EGO_MOTION"
    assert len(data["keyframes_metadata"]) == 3
    assert data["camera_params_id_to_camera_params"]["0"]["calibration_parameters"]["image_width"] == 1920
    print(f"[TEST 4] Successfully generated valid frames_meta.json ({len(data['keyframes_metadata'])} keyframes)")

def test_cusfm_cli_wsl():
    res = run_wsl_cmd("cusfm_cli --help")
    print(f"[TEST 5] cusfm_cli --help returncode: {res.returncode}")
    assert res.returncode == 0, f"cusfm_cli failed: {res.stderr}"

def test_coarse_pose_extraction():
    import pycolmap
    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera(
        model="PINHOLE", width=1920, height=1080, params=[1500.0, 1500.0, 960.0, 540.0], camera_id=1
    )
    if hasattr(rec, "add_camera_with_trivial_rig"):
        rec.add_camera_with_trivial_rig(cam)
    else:
        rec.add_camera(cam)

    images = ["frame_001.jpg", "frame_002.jpg"]
    for idx, name in enumerate(images):
        im = pycolmap.Image(name=name, camera_id=1, image_id=idx + 1)
        rigid = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.array([0.0, 0.0, float(idx)]))
        if hasattr(rec, "add_image_with_trivial_frame"):
            rec.add_image_with_trivial_frame(im, rigid)
        else:
            im.cam_from_world = rigid
            im.registered = True
            rec.add_image(im)

    # Test the exact extraction loop from align_cli.py
    initial_poses = []
    name_to_im = {im.name: im for im in rec.images.values()}
    for p in images:
        bname = os.path.basename(p)
        assert bname in name_to_im
        im = name_to_im[bname]
        cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        R_w2c = cfw.rotation.matrix()
        t_w2c = cfw.translation
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c
        initial_poses.append((R_c2w, t_c2w))

    assert len(initial_poses) == 2
    assert np.allclose(initial_poses[0][0], np.eye(3))
    print(f"[TEST 6] Successfully extracted coarse poses from pycolmap Reconstruction ({len(initial_poses)} frames)")

if __name__ == "__main__":
    print("=== Running PyCuSFM Adapter Test Suite ===")
    test_wsl_detection()
    test_path_translation()
    test_rotation_math()
    scratch_dir = r"C:\Users\USer\.gemini\antigravity\scratch\General Stuff 01\vggt\.test_tmp"
    test_frames_meta_generation(scratch_dir)
    test_cusfm_cli_wsl()
    test_coarse_pose_extraction()
    if os.path.exists(scratch_dir):
        import shutil
        shutil.rmtree(scratch_dir)
    print("=== ALL 6 PYCUSFM TESTS PASSED SUCCESSFULLY! ===")
