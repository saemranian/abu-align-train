using System;
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
                string scriptPath = Path.Combine(appDir, "abu_align_gui.py");
                
                string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                string[] candidatePythons = new string[]
                {
                    Path.Combine(appDir, "env", "pythonw.exe"),
                    Path.Combine(appDir, "env", "python.exe"),
                    Path.Combine(appDir, "env", "Scripts", "pythonw.exe"),
                    Path.Combine(appDir, "env", "Scripts", "python.exe"),
                    Path.Combine(localAppData, @"AbuAlign\env\pythonw.exe"),
                    Path.Combine(localAppData, @"AbuAlign\env\python.exe"),
                    Path.Combine(localAppData, @"AbuAlign\env\Scripts\pythonw.exe"),
                    Path.Combine(localAppData, @"AbuAlign\env\Scripts\python.exe"),
                    Path.Combine(userProfile, @"miniconda3\envs\abu_align\pythonw.exe"),
                    Path.Combine(userProfile, @"miniconda3\envs\abu_align\python.exe"),
                    Path.Combine(userProfile, @"miniconda3\envs\vggt\pythonw.exe"),
                    Path.Combine(userProfile, @"miniconda3\envs\vggt\python.exe"),
                    Path.Combine(userProfile, @"anaconda3\envs\abu_align\pythonw.exe"),
                    Path.Combine(userProfile, @"anaconda3\envs\abu_align\python.exe"),
                    Path.Combine(userProfile, @"anaconda3\envs\vggt\pythonw.exe"),
                    Path.Combine(userProfile, @"anaconda3\envs\vggt\python.exe")
                };

                string pythonPath = null;
                foreach (string candidate in candidatePythons)
                {
                    if (File.Exists(candidate))
                    {
                        pythonPath = candidate;
                        break;
                    }
                }

                if (string.IsNullOrEmpty(pythonPath))
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
                                    pythonPath = candidate;
                                    break;
                                }
                            }
                        }
                    }

                    if (string.IsNullOrEmpty(pythonPath))
                    {
                        MessageBox.Show("Python runtime not found.\nPlease run install_abu_align.bat to complete the runtime setup.",
                            "Abu Align - Setup Required", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        return;
                    }
                }

                if (!File.Exists(scriptPath))
                {
                    MessageBox.Show("GUI script not found at:\n" + scriptPath,
                        "Abu Align - Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }

                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = pythonPath;
                psi.Arguments = "\"" + scriptPath + "\"";
                psi.WorkingDirectory = appDir;
                psi.UseShellExecute = false;
                psi.CreateNoWindow = true;

                Process.Start(psi);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Failed to launch Abu Align:\n" + ex.Message,
                    "Abu Align - Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
    }
}
