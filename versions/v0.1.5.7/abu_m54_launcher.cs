using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Windows.Forms;

class AbuM54Launcher {
    static string Quote(string text) {
        var result = new StringBuilder("\""); int slashes = 0;
        foreach (char c in text) {
            if (c == '\\') { slashes++; continue; }
            if (c == '"') { result.Append('\\', slashes * 2 + 1); result.Append('"'); }
            else { result.Append('\\', slashes); result.Append(c); }
            slashes = 0;
        }
        result.Append('\\', slashes * 2); return result.Append('"').ToString();
    }
    [STAThread] static void Main(string[] args) {
        string directory = AppDomain.CurrentDomain.BaseDirectory;
        string python = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "miniconda3", "envs", "vggt", "pythonw.exe");
        string script = Path.Combine(directory, "abu_m54_app.py");
        if (!File.Exists(python) || !File.Exists(script)) { MessageBox.Show("Abu M54 needs the local vggt environment and abu_m54_app.py beside this launcher.", "Abu M54"); return; }
        var command = new StringBuilder(Quote(script)); foreach (string arg in args) command.Append(" ").Append(Quote(arg));
        var info = new ProcessStartInfo(python, command.ToString()); info.WorkingDirectory = directory; info.UseShellExecute = false; info.CreateNoWindow = true;
        try { Process.Start(info); } catch (Exception error) { MessageBox.Show(error.Message, "Abu M54 could not start"); }
    }
}
