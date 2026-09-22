Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\USer\.gemini\antigravity\scratch\General Stuff 01\vggt"
WshShell.Run """C:\Users\USer\miniconda3\envs\vggt\pythonw.exe"" ""abu_align_gui.py""", 0, False
Set WshShell = Nothing
