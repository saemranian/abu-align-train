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
                string[] candidatePythons = new string[]
                {
                    Path.Combine(appDir, "env", "pythonw.exe"),
                    Path.Combine(appDir, "env", "python.exe"),
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
                    MessageBox.Show("Python executable not found in standard Conda environments (abu_align or vggt).\nPlease run install_abu_align.bat or set up the conda environment.",
                        "Abu Align - Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
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
