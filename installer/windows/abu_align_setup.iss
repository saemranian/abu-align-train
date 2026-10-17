#ifndef MyAppVersion
  #define MyAppVersion "0.1.5.8"
#endif

[Setup]
AppId={{8A4E92B1-5C7D-4E3F-9A1B-2D6F8E4C0B92}
AppName=Abu Align & Train
AppVersion={#MyAppVersion}
AppVerName=Abu Align & Train v{#MyAppVersion}
AppPublisher=Abu Photogrammetry & 3DGS Studio
AppPublisherURL=https://github.com/saemranian/abu-align-train
AppSupportURL=https://github.com/saemranian/abu-align-train/issues
AppUpdatesURL=https://github.com/saemranian/abu-align-train/releases
DefaultDirName={localappdata}\AbuAlign
DefaultGroupName=Abu Align & Train
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist
OutputBaseFilename=Abu_Align_and_Train_Setup_Windows_x64
SetupIconFile=..\..\abu_icon.ico
UninstallDisplayIcon={app}\Abu_Align_and_Train.exe
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\..\*.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\*.exe"; DestDir: "{app}"; Flags: ignoreversion; Excludes: "*_v0.*"
Source: "..\..\*.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\*.sh"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\*.cs"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\*.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\*.png"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\*.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\*.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\pyproject.toml"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\vggt\*"; DestDir: "{app}\vggt"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__\*,*.pyc"
Source: "..\..\training\*"; DestDir: "{app}\training"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__\*,*.pyc"
Source: "..\..\docs\*"; DestDir: "{app}\docs"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{app}"; Permissions: users-modify
Name: "{app}\logs"; Permissions: users-modify
Name: "{app}\presets"; Permissions: users-modify
Name: "{app}\checkpoints"; Permissions: users-modify

[Icons]
Name: "{autoprograms}\Abu Align & Train"; Filename: "{app}\Abu_Align_and_Train.exe"; WorkingDir: "{app}"; IconFilename: "{app}\abu_icon.ico"
Name: "{autodesktop}\Abu Align & Train"; Filename: "{app}\Abu_Align_and_Train.exe"; WorkingDir: "{app}"; IconFilename: "{app}\abu_icon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\Abu_Align_and_Train.exe"; Description: "Launch Abu Align & Train Studio"; Flags: nowait postinstall skipifsilent
Filename: "{app}\install_abu_align.bat"; Description: "Run Fresh Python/CUDA Environment Provisioner (Only needed if Python/Conda is not installed yet)"; Flags: postinstall shellexec waituntilterminated unchecked
