#ifndef MyAppVersion
  #define MyAppVersion "0.1.6.6"
#endif
#ifdef FullPacksDir
  #ifdef FullGpuOnline
    #error Choose either offline packs or online GPU setup, not both
  #endif
#endif
#ifndef MyAppId
  #define MyAppId "{{8A4E92B1-5C7D-4E3F-9A1B-2D6F8E4C0B92}"
#endif
[Setup]
AppId={#MyAppId}
#ifdef FullPacksDir
AppName=Abu Align & Train Full GPU Candidate
#else
#ifdef FullGpuOnline
AppName=Abu Align & Train Full GPU Candidate
#else
AppName=Abu Align & Train CPU Core Candidate
#endif
#endif
AppVersion={#MyAppVersion}
AppPublisher=Abu Align contributors
AppPublisherURL=https://github.com/saemranian/abu-align-train
LicenseFile=..\license-summary.txt
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
#ifdef FullPacksDir
Source: "{#FullPacksDir}\*"; DestDir: "{app}\engine-packs"; Flags: ignoreversion
#endif
[Icons]
Name: "{autoprograms}\Abu Align & Train"; Filename: "{app}\Abu_Align_and_Train.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\Abu Align & Train"; Filename: "{app}\Abu_Align_and_Train.exe"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{autoprograms}\Abu M54"; Filename: "{app}\runtime\AbuRuntime.exe"; Parameters: "--m54"; WorkingDir: "{app}"
[Run]
Filename: "{app}\Abu_Align_and_Train.exe"; Description: "Launch Abu Align & Train"; Flags: nowait postinstall skipifsilent
#ifdef FullPacksDir
[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then begin
    if not Exec(ExpandConstant('{app}\runtime\AbuRuntime.exe'),
      '--install-bundled-components "' + ExpandConstant('{app}\engine-packs') + '"',
      ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, ResultCode) then
      RaiseException('Could not start GPU component installation.');
    if ResultCode <> 0 then
      RaiseException('GPU component installation failed. Check free disk space and retry setup.');
  end;
end;
#endif
#ifdef FullGpuOnline
[Code]
procedure GPUSetupLog(const S: String; const Error, FirstLine: Boolean);
begin
  Log(S);
  WizardForm.StatusLabel.Caption := S;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then begin
    WizardForm.StatusLabel.Caption := 'Installing GPU alignment and training components...';
    if not ExecAndLogOutput(ExpandConstant('{app}\runtime\AbuRuntime.exe'),
      '--install-default-components', ExpandConstant('{app}'), SW_SHOWNORMAL,
      ewWaitUntilTerminated, ResultCode, @GPUSetupLog) then
      RaiseException('Could not start GPU component setup.');
    if ResultCode <> 0 then
      RaiseException('GPU setup did not finish. Check Internet access and disk space, then retry setup; partial downloads are retained.');
  end;
end;
#endif
