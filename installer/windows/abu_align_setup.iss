#ifndef MyAppVersion
  #define MyAppVersion "0.1.6.1"
#endif
#ifndef MyAppId
  #define MyAppId "{{8A4E92B1-5C7D-4E3F-9A1B-2D6F8E4C0B92}"
#endif
[Setup]
AppId={#MyAppId}
AppName=Abu Align & Train CPU Core Candidate
AppVersion={#MyAppVersion}
AppPublisher=Abu Align contributors
AppPublisherURL=https://github.com/saemranian/abu-align-train
DefaultDirName={localappdata}\Programs\AbuAlign
DefaultGroupName=Abu Align & Train
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=Abu_Align_and_Train_Setup_Windows_x64
SetupIconFile=..\..\abu_icon.ico
UninstallDisplayIcon={app}\Abu_Align_and_Train.exe
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
[Files]
Source: "..\dist\AbuRuntime\*"; DestDir: "{app}\runtime"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\Abu_Align_and_Train.exe"; DestDir: "{app}"; Flags: ignoreversion
[Icons]
Name: "{autoprograms}\Abu Align & Train"; Filename: "{app}\Abu_Align_and_Train.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\Abu Align & Train"; Filename: "{app}\Abu_Align_and_Train.exe"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{autoprograms}\Abu M54"; Filename: "{app}\runtime\AbuRuntime.exe"; Parameters: "--m54"; WorkingDir: "{app}"
[Run]
Filename: "{app}\Abu_Align_and_Train.exe"; Description: "Launch Abu CPU core candidate"; Flags: nowait postinstall skipifsilent
