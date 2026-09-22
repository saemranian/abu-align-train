import os, sys
cur = os.path.dirname(os.path.abspath(__file__))
for p in [cur, os.path.dirname(cur), os.path.dirname(os.path.dirname(cur)), os.path.dirname(os.path.dirname(os.path.dirname(cur)))]:
    if p and p not in sys.path:
        sys.path.insert(0, p)
"""
Abu Align Version 0.0.0.2
Professional Graphical User Interface for Neural Photogrammetry Alignment
Built on VGGT & VGGT-Omega with PyCeres Bundle Adjustment
"""

import os
import sys
import time
import subprocess
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

class AbuAlignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Abu Align - Version 0.0.0.2 (Neural Photogrammetry)")
        self.root.geometry("980x760")
        self.root.minsize(860, 640)

        # Style configuration (Modern Dark Palette)
        self.bg_dark = "#181825"
        self.panel_bg = "#1e1e2e"
        self.accent_blue = "#3b82f6"
        self.accent_green = "#10b981"
        self.accent_red = "#ef4444"
        self.text_primary = "#f8fafc"
        self.text_secondary = "#94a3b8"
        self.console_bg = "#11111b"

        self.root.configure(bg=self.bg_dark)

        self.conda_python = r"C:\Users\USer\miniconda3\envs\vggt\python.exe"
        self.cli_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "align_cli.py")
        self.worker_process = None
        self.output_queue = queue.Queue()
        self.is_running = False
        self.latest_output_dir = None

        self._build_ui()
        self._check_queue()

    def _build_ui(self):
        # 1. Header Banner
        header = tk.Frame(self.root, bg=self.panel_bg, height=65)
        header.pack(fill=tk.X, padx=12, pady=(12, 6))

        title_lbl = tk.Label(
            header,
            text="ABU ALIGN",
            font=("Segoe UI", 16, "bold"),
            fg=self.accent_blue,
            bg=self.panel_bg
        )
        title_lbl.pack(side=tk.LEFT, padx=16, pady=8)

        version_badge = tk.Label(
            header,
            text="v0.0.0.2",
            font=("Segoe UI", 9, "bold"),
            fg="#ffffff",
            bg=self.accent_blue,
            padx=6,
            pady=2
        )
        version_badge.pack(side=tk.LEFT, pady=12)

        subtitle_lbl = tk.Label(
            header,
            text="|   Neural Photogrammetry & Fast Lens Self-Calibration Engine",
            font=("Segoe UI", 10),
            fg=self.text_secondary,
            bg=self.panel_bg
        )
        subtitle_lbl.pack(side=tk.LEFT, padx=10, pady=12)

        # 2. Main Content Notebook / Grid
        content_frame = tk.Frame(self.root, bg=self.bg_dark)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        # Top Control Card
        control_card = tk.LabelFrame(
            content_frame,
            text=" Alignment Configuration ",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_primary,
            bg=self.panel_bg,
            padx=14,
            pady=10
        )
        control_card.pack(fill=tk.X, pady=(0, 8))

        # Dataset Path Selection
        path_row = tk.Frame(control_card, bg=self.panel_bg)
        path_row.pack(fill=tk.X, pady=4)

        tk.Label(
            path_row,
            text="Dataset Folder:",
            font=("Segoe UI", 9, "bold"),
            fg=self.text_primary,
            bg=self.panel_bg,
            width=14,
            anchor="w"
        ).pack(side=tk.LEFT)

        self.dataset_var = tk.StringVar(value=r"C:\Users\USer\Documents\Works\Clients\Schemata\Container_03\Sony\101MSDCF\CC\Interior\RedBox\SEL Copy")
        self.path_entry = tk.Entry(
            path_row,
            textvariable=self.dataset_var,
            font=("Segoe UI", 9),
            bg=self.bg_dark,
            fg=self.text_primary,
            insertbackground=self.text_primary,
            relief=tk.FLAT
        )
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8), ipady=3)

        browse_btn = tk.Button(
            path_row,
            text="Browse...",
            font=("Segoe UI", 9),
            bg=self.accent_blue,
            fg="#ffffff",
            activebackground="#2563eb",
            relief=tk.FLAT,
            padx=12,
            command=self._browse_dataset
        )
        browse_btn.pack(side=tk.RIGHT)

        # Options Grid: 4 Clean Columns
        opts_grid = tk.Frame(control_card, bg=self.panel_bg)
        opts_grid.pack(fill=tk.X, pady=(8, 4))

        # Column 1: Engine & Canvas
        col1 = tk.Frame(opts_grid, bg=self.panel_bg)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        tk.Label(col1, text="Engine & Canvas:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        self.engine_var = tk.StringVar(value="vggt_omega")
        rb_omega = tk.Radiobutton(col1, text="VGGT-Omega (Fast, Low VRAM)", variable=self.engine_var, value="vggt_omega", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_omega.pack(anchor="w")
        rb_vggt = tk.Radiobutton(col1, text="VGGT Standard 1B", variable=self.engine_var, value="vggt", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_vggt.pack(anchor="w")

        sub_row = tk.Frame(col1, bg=self.panel_bg)
        sub_row.pack(anchor="w", pady=(4, 0))
        tk.Label(sub_row, text="Res:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        self.res_var = tk.StringVar(value="1024")
        res_cb = ttk.Combobox(sub_row, textvariable=self.res_var, values=["1024", "1536", "2048"], state="readonly", width=6)
        res_cb.pack(side=tk.LEFT, padx=(2, 6))

        tk.Label(sub_row, text="Stride:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        self.stride_var = tk.StringVar(value="1")
        stride_cb = ttk.Combobox(sub_row, textvariable=self.stride_var, values=["1", "2", "3", "4", "5"], state="readonly", width=4)
        stride_cb.pack(side=tk.LEFT, padx=2)

        # Column 2: Point Cloud Density (Quality)
        col2 = tk.Frame(opts_grid, bg=self.panel_bg)
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        tk.Label(col2, text="Point Cloud Density:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        self.density_var = tk.StringVar(value="high")
        rb_q1 = tk.Radiobutton(col2, text="Preview (100K pts, Conf 2.2)", variable=self.density_var, value="preview", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_q1.pack(anchor="w")
        rb_q2 = tk.Radiobutton(col2, text="High Quality (300K pts, Conf 1.8)", variable=self.density_var, value="high", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_q2.pack(anchor="w")
        rb_q3 = tk.Radiobutton(col2, text="Ultra Dense (800K pts, Conf 1.5)", variable=self.density_var, value="ultra", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_q3.pack(anchor="w")
        rb_q4 = tk.Radiobutton(col2, text="Max Scan Grade (1.5M pts, Conf 1.2)", variable=self.density_var, value="max", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_q4.pack(anchor="w")

        # Column 3: Camera Alignment Accuracy
        col3 = tk.Frame(opts_grid, bg=self.panel_bg)
        col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        tk.Label(col3, text="Camera Alignment Accuracy:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        self.align_var = tk.StringVar(value="ultra")
        rb_a1 = tk.Radiobutton(col3, text="Feedforward (Fast, ~25s)", variable=self.align_var, value="none", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_a1.pack(anchor="w")
        rb_a2 = tk.Radiobutton(col3, text="Standard Ceres BA (~1m)", variable=self.align_var, value="standard", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_a2.pack(anchor="w")
        rb_a3 = tk.Radiobutton(col3, text="Ultra Sub-Pixel BA (Cauchy+Polish)", variable=self.align_var, value="ultra", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_a3.pack(anchor="w")

        # Column 4: Lens Distortion Calibration
        col4 = tk.Frame(opts_grid, bg=self.panel_bg)
        col4.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        tk.Label(col4, text="Lens Distortion Model:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        self.lens_var = tk.StringVar(value="brown")
        rb_l1 = tk.Radiobutton(col4, text="Pinhole (Ideal, zero distortion)", variable=self.lens_var, value="pinhole", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_l1.pack(anchor="w")
        rb_l2 = tk.Radiobutton(col4, text="Brown Calibration (k1-k2, t1-t2)", variable=self.lens_var, value="brown", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_l2.pack(anchor="w")

        # 3. Action Buttons & Progress Bar
        action_bar = tk.Frame(content_frame, bg=self.bg_dark)
        action_bar.pack(fill=tk.X, pady=(0, 6))

        self.start_btn = tk.Button(
            action_bar,
            text="Gû¦  START ALIGNMENT",
            font=("Segoe UI", 11, "bold"),
            bg=self.accent_green,
            fg="#ffffff",
            activebackground="#059669",
            relief=tk.FLAT,
            padx=18,
            pady=6,
            command=self._start_alignment
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.stop_btn = tk.Button(
            action_bar,
            text="Gûá  STOP",
            font=("Segoe UI", 10, "bold"),
            bg=self.accent_red,
            fg="#ffffff",
            activebackground="#dc2626",
            relief=tk.FLAT,
            padx=14,
            pady=6,
            state=tk.DISABLED,
            command=self._stop_alignment
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.open_folder_btn = tk.Button(
            action_bar,
            text="=ƒôü Open Results",
            font=("Segoe UI", 10),
            bg=self.panel_bg,
            fg=self.text_primary,
            activebackground="#33334d",
            relief=tk.FLAT,
            padx=12,
            pady=6,
            command=self._open_results_folder
        )
        self.open_folder_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.viewer_btn = tk.Button(
            action_bar,
            text="=ƒîÉ 3D Browser Viewer",
            font=("Segoe UI", 10),
            bg=self.panel_bg,
            fg=self.text_primary,
            activebackground="#33334d",
            relief=tk.FLAT,
            padx=12,
            pady=6,
            command=self._launch_viewer
        )
        self.viewer_btn.pack(side=tk.LEFT)

        # 4. Live Progress Console Card
        console_card = tk.LabelFrame(
            content_frame,
            text=" Live Telemetry & Process Logs ",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_primary,
            bg=self.panel_bg,
            padx=10,
            pady=8
        )
        console_card.pack(fill=tk.BOTH, expand=True)

        self.console = ScrolledText(
            console_card,
            bg=self.console_bg,
            fg="#a6accd",
            insertbackground="#ffffff",
            font=("Consolas", 9),
            relief=tk.FLAT,
            wrap=tk.WORD
        )
        self.console.pack(fill=tk.BOTH, expand=True)
        self._log("Abu Align Version 0.0.0.2 Ready.\nSelect a photoscan directory and click 'START ALIGNMENT'.\n")

    def _browse_dataset(self):
        folder = filedialog.askdirectory(initialdir=self.dataset_var.get())
        if folder:
            self.dataset_var.set(folder)

    def _log(self, text):
        self.console.insert(tk.END, text)
        self.console.see(tk.END)

    def _start_alignment(self):
        if self.is_running:
            return

        dataset_path = self.dataset_var.get().strip()
        if not os.path.isdir(dataset_path):
            messagebox.showerror("Error", f"Dataset folder not found:\n{dataset_path}")
            return

        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.console.delete(1.0, tk.END)

        cmd = [
            self.conda_python,
            "-u",
            self.cli_script,
            "--input", dataset_path,
            "--engine", self.engine_var.get(),
            "--stride", str(self.stride_var.get()),
            "--load_res", str(self.res_var.get()),
            "--align_mode", self.align_var.get(),
            "--lens_model", self.lens_var.get(),
        ]

        density = self.density_var.get()
        if density == "preview":
            cmd.extend(["--conf_thresh", "2.2", "--max_points", "100000"])
        elif density == "high":
            cmd.extend(["--conf_thresh", "1.8", "--max_points", "300000"])
        elif density == "ultra":
            cmd.extend(["--conf_thresh", "1.5", "--max_points", "800000"])
        elif density == "max":
            cmd.extend(["--conf_thresh", "1.2", "--max_points", "1500000"])

        self._log(f"Executing: {' '.join(cmd)}\n\n")

        thread = threading.Thread(target=self._run_process, args=(cmd,), daemon=True)
        thread.start()

    def _run_process(self, cmd):
        try:
            self.worker_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )

            for line in iter(self.worker_process.stdout.readline, ""):
                if line:
                    self.output_queue.put(line)
                    if "Output Directory:" in line:
                        parts = line.split("Output Directory:")
                        if len(parts) > 1:
                            self.latest_output_dir = parts[1].strip()

            self.worker_process.stdout.close()
            self.worker_process.wait()
        except Exception as e:
            self.output_queue.put(f"\n[ERROR] Process failed: {str(e)}\n")
        finally:
            self.is_running = False
            self.output_queue.put("__PROCESS_FINISHED__")

    def _check_queue(self):
        try:
            while True:
                msg = self.output_queue.get_nowait()
                if msg == "__PROCESS_FINISHED__":
                    self.start_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                    self._log("\n[STATUS] Alignment Run Completed.\n")
                else:
                    self._log(msg)
        except queue.Empty:
            pass
        self.root.after(80, self._check_queue)

    def _stop_alignment(self):
        if self.worker_process and self.is_running:
            try:
                self.worker_process.terminate()
                self._log("\n[ABORT] User cancelled alignment process.\n")
            except Exception:
                pass
            self.is_running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def _open_results_folder(self):
        target = self.latest_output_dir
        if not target or not os.path.isdir(target):
            dataset = self.dataset_var.get().strip()
            target = os.path.join(dataset, "results")
        if os.path.isdir(target):
            os.startfile(target)
        else:
            messagebox.showinfo("Info", f"Results folder does not exist yet:\n{target}")

    def _launch_viewer(self):
        dataset = self.dataset_var.get().strip()
        if not os.path.isdir(dataset):
            messagebox.showerror("Error", "Please select a valid dataset folder first.")
            return
        cmd = [self.conda_python, "-u", self.cli_script, "--input", dataset, "--view"]
        subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0)

def main():
    root = tk.Tk()
    app = AbuAlignApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

