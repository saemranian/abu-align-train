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
                string pythonPath = @"C:\Users\USer\miniconda3\envs\vggt\pythonw.exe";

                if (!File.Exists(pythonPath))
                {
                    pythonPath = @"C:\Users\USer\miniconda3\envs\vggt\python.exe";
                }

                if (!File.Exists(pythonPath))
                {
                    MessageBox.Show("Conda environment python not found at:\n" + pythonPath,
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
