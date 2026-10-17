using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

namespace AbuAlignLauncher
{
    static class Program
    {
        [STAThread]
        static void Main()
        {
            try
            {
                string appDir = AppDomain.CurrentDomain.BaseDirectory;
                string bundled = Path.Combine(appDir, "runtime", "AbuRuntime.exe");
                if (File.Exists(bundled))
                {
                    var installed = new ProcessStartInfo(bundled);
                    installed.WorkingDirectory = appDir;
                    installed.UseShellExecute = false;
                    installed.CreateNoWindow = true;
                    Process.Start(installed);
                    return;
                }
                string scriptPath = Path.Combine(appDir, "abu_align_gui.py");
                if (!File.Exists(scriptPath))
                {
                    MessageBox.Show("GUI script not found at:\n" + scriptPath,
                        "Abu Align - Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }

                string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                string[] candidatePythons = new string[]
                {
                    Path.Combine(appDir, "env", "python.exe"),
                    Path.Combine(appDir, "env", "Scripts", "python.exe"),
                    Path.Combine(localAppData, @"AbuAlign\env\python.exe"),
                    Path.Combine(localAppData, @"AbuAlign\env\Scripts\python.exe"),
                    Path.Combine(userProfile, @"miniconda3\envs\abu_align\python.exe"),
                    Path.Combine(userProfile, @"miniconda3\envs\vggt\python.exe"),
                    Path.Combine(userProfile, @"anaconda3\envs\abu_align\python.exe"),
                    Path.Combine(userProfile, @"anaconda3\envs\vggt\python.exe")
                };

                List<string> existingPythons = new List<string>();
                foreach (string candidate in candidatePythons)
                {
                    if (File.Exists(candidate))
                    {
                        existingPythons.Add(candidate);
                    }
                }

                if (existingPythons.Count == 0)
                {
                    string installerBat = Path.Combine(appDir, "install_abu_align.bat");
                    if (File.Exists(installerBat))
                    {
                        DialogResult dr = MessageBox.Show(
                            "Welcome to Abu Align & Train!\n\nFirst-time setup is required to configure the Python & GPU runtime.\nWould you like to run the automated setup now?",
                            "Abu Align & Train - First-Time Setup",
                            MessageBoxButtons.YesNo,
                            MessageBoxIcon.Information);
                        if (dr == DialogResult.Yes)
                        {
                            ProcessStartInfo setupPsi = new ProcessStartInfo();
                            setupPsi.FileName = installerBat;
                            setupPsi.WorkingDirectory = appDir;
                            setupPsi.UseShellExecute = true;
                            Process setupProc = Process.Start(setupPsi);
                            if (setupProc != null)
                            {
                                setupProc.WaitForExit();
                            }
                            foreach (string candidate in candidatePythons)
                            {
                                if (File.Exists(candidate))
                                {
                                    existingPythons.Add(candidate);
                                }
                            }
                        }
                    }

                    if (existingPythons.Count == 0)
                    {
                        MessageBox.Show("Python runtime not found.\nPlease run install_abu_align.bat to complete the runtime setup.",
                            "Abu Align - Setup Required", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        return;
                    }
                }

                string lastError = "";
                foreach (string pyExe in existingPythons)
                {
                    ProcessStartInfo psi = new ProcessStartInfo();
                    psi.FileName = pyExe;
                    psi.Arguments = "\"" + scriptPath + "\"";
                    psi.WorkingDirectory = appDir;
                    psi.UseShellExecute = false;
                    psi.CreateNoWindow = true;
                    psi.RedirectStandardError = true;

                    Process proc = Process.Start(psi);
                    if (proc != null)
                    {
                        bool exitedQuickly = proc.WaitForExit(2000);
                        if (!exitedQuickly || proc.ExitCode == 0)
                        {
                            return;
                        }
                        lastError = proc.StandardError.ReadToEnd();
                    }
                }

                MessageBox.Show(
                    "Abu Align & Train encountered a startup error:\n\n" +
                    (string.IsNullOrWhiteSpace(lastError) ? "Python process exited unexpectedly." : lastError),
                    "Abu Align & Train - Startup Diagnostic",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Failed to launch Abu Align:\n" + ex.Message,
                    "Abu Align - Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
    }
}
