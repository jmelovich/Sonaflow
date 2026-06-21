; Sonaflow OFFLINE installer (Inno Setup 6.1+).
;
; Embeds application runtime ZIPs only. Public HuggingFace models download on
; first launch. TensorRT engines are built on the target GPU at first launch.
;
; Compile via tools\Build-OfflineInstaller.ps1

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

#ifndef CudaSuffix
  #define CudaSuffix ""
#endif

#ifndef PayloadDir
  #pragma error "PayloadDir must be set at compile time (see Build-OfflineInstaller.ps1)."
#endif

#define AppName "Sonaflow"
#define AppPublisher "Sonaflow"
#define AppExe "Sonaflow.exe"

[Setup]
AppId={{B4F2B1B0-7B3E-4E2A-9C1D-5F6A7E8D9C01}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppVerName={#AppName} {#AppVersion}
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputBaseFilename=Sonaflow-Setup-{#AppVersion}{#CudaSuffix}-offline
OutputDir=build\dist\Sonaflow-Setup-{#AppVersion}{#CudaSuffix}-offline
WizardStyle=modern
DiskSpanning=yes
Compression=none
SolidCompression=no
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\{#AppExe}
InfoAfterFile=docs\OPTIONAL_SETUP.txt
; Runtime zips ~8 GB + headroom for models/engines downloaded on first launch.
ExtraDiskSpaceRequired=12884901888

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Components]
Name: "core"; Description: "Sonaflow core (app + Python runtime)"; Types: full compact custom; Flags: fixed
#ifdef IncludeQwen
Name: "qwen"; Description: "Qwen2-Audio captions (sidecar runtime; model downloads on first launch)"; Types: full custom
#endif

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[UninstallDelete]
; Payload zips are extracted in [Code]; Inno does not track those files individually.
Type: filesandordirs; Name: "{app}"
; Models, TensorRT engines, and settings live outside the program directory.
Type: filesandordirs; Name: "{localappdata}\Sonaflow"

[Files]
Source: "{#PayloadDir}\core-runtime-{#AppVersion}{#CudaSuffix}.zip"; DestDir: "{tmp}"; DestName: "core-runtime.zip"; Flags: ignoreversion deleteafterinstall
#ifdef IncludeModels
Source: "{#PayloadDir}\models-core-{#AppVersion}{#CudaSuffix}.zip"; DestDir: "{tmp}"; DestName: "models-core.zip"; Flags: ignoreversion deleteafterinstall
#endif
#ifdef IncludeQwen
Source: "{#PayloadDir}\qwen-runtime-{#AppVersion}{#CudaSuffix}.zip"; DestDir: "{tmp}"; DestName: "qwen-runtime.zip"; Flags: ignoreversion deleteafterinstall; Components: qwen
#endif
Source: "docs\OPTIONAL_SETUP.txt"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "scripts\write_packaged_settings.py"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "scripts\extract_zip.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "redist\vc_redist.x64.exe"; DestDir: "{app}\redist"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Optional setup notes"; Filename: "{app}\docs\OPTIONAL_SETUP.txt"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\redist\vc_redist.x64.exe"; Parameters: "/quiet /norestart"; StatusMsg: "Installing Visual C++ runtime..."; Flags: waituntilterminated; Check: ShouldInstallVCRedist
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Code]
const
  SM_CXSCREEN = 0;
  SM_CYSCREEN = 1;

function GetSystemMetrics(nIndex: Integer): Integer;
  external 'GetSystemMetrics@user32.dll stdcall';

var
  ResPage: TInputQueryWizardPage;
  DisplayW, DisplayH, GenW, GenH: Integer;

function SnapWidth(DisplayW, DisplayH: Integer): Integer;
var
  W: Integer;
begin
  if DisplayH <= 0 then DisplayH := 1;
  { Inno integer division truncates (3440/1440=2); force floating-point math. }
  W := Round((512.0 * DisplayW / DisplayH) / 64.0) * 64;
  if W < 320 then W := 320;
  if W > 1216 then W := 1216;
  Result := W;
end;
function ResTag: String;
begin
  Result := IntToStr(GenW) + 'x' + IntToStr(GenH);
end;

function ExtractZip(ZipPath, Dest: String): Boolean;
var
  Cmd, Args, ScriptPath: String;
  ResultCode: Integer;
begin
  ForceDirectories(Dest);
  ScriptPath := ExpandConstant('{app}\scripts\extract_zip.ps1');
  Cmd := ExpandConstant('{cmd}');
  Args := '/C powershell -NoProfile -ExecutionPolicy Bypass -File "' + ScriptPath + '" ' +
          '"' + ZipPath + '" "' + Dest + '"';
  Result := Exec(Cmd, Args, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
end;

function QueryNvidia(var Info: String): Boolean;
var
  TmpFile: String;
  ResultCode: Integer;
  Lines: TArrayOfString;
begin
  Result := False;
  Info := '';
  TmpFile := ExpandConstant('{tmp}\nvsmi.txt');
  if Exec(ExpandConstant('{cmd}'),
          '/C nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "' + TmpFile + '" 2>nul',
          '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    if (ResultCode = 0) and LoadStringsFromFile(TmpFile, Lines) and (GetArrayLength(Lines) > 0) then
    begin
      Info := Trim(Lines[0]);
      Result := (Info <> '');
    end;
  end;
end;

function InitializeSetup(): Boolean;
var
  Info: String;
begin
  Result := True;
  if not QueryNvidia(Info) then
  begin
    if MsgBox('No NVIDIA GPU/driver was detected (nvidia-smi unavailable).' + #13#10 +
              'Sonaflow requires an NVIDIA RTX GPU with a recent driver.' + #13#10#13#10 +
              'Continue anyway?', mbConfirmation, MB_YESNO) = IDNO then
      Result := False;
  end
  else
  begin
    Log('Detected GPU: ' + Info);
    if (Pos('RTX 50', Uppercase(Info)) = 0) and (Pos('RTX 50', Info) = 0) then
      MsgBox('Detected GPU: ' + Info + #13#10#13#10 +
             'This build was validated on RTX 50-series GPUs. TensorRT engines will be ' +
             'rebuilt on first run on other GPUs.', mbInformation, MB_OK);
  end;
end;

procedure InitializeWizard();
begin
  ResPage := CreateInputQueryPage(wpSelectDir,
    'Display resolution',
    'Choose the resolution Sonaflow should target.',
    'Sonaflow generates at 512 px height and snaps width to your display aspect ratio. ' +
    'TensorRT engines for this size are built on first launch (~5-15 min, then cached). ' +
    'AI models download from Hugging Face on first launch (internet required).');
  ResPage.Add('Display width (px):', False);
  ResPage.Add('Display height (px):', False);
  ResPage.Values[0] := IntToStr(GetSystemMetrics(SM_CXSCREEN));
  ResPage.Values[1] := IntToStr(GetSystemMetrics(SM_CYSCREEN));
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  DW, DH: Integer;
begin
  Result := True;
  if CurPageID = ResPage.ID then
  begin
    DW := StrToIntDef(Trim(ResPage.Values[0]), 0);
    DH := StrToIntDef(Trim(ResPage.Values[1]), 0);
    if (DW <= 0) or (DH <= 0) then
    begin
      MsgBox('Please enter a valid display width and height.', mbError, MB_OK);
      Result := False;
      Exit;
    end;
    DisplayW := DW;
    DisplayH := DH;
    GenH := 512;
    GenW := SnapWidth(DW, DH);
  end;
end;

function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo, MemoTypeInfo,
  MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
begin
  Result := MemoDirInfo + NewLine + NewLine +
            'Generation resolution:' + NewLine + Space + ResTag + NewLine + NewLine +
            'AI models: download on first launch' + NewLine +
            'TensorRT engines: built on first launch' + NewLine + NewLine +
            MemoComponentsInfo + NewLine + MemoTasksInfo;
end;

function VCRedistInstalled(): Boolean;
var
  Installed: Cardinal;
begin
  Result := RegQueryDWordValue(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', Installed) and (Installed = 1);
  if not Result then
    Result := RegQueryDWordValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', Installed) and (Installed = 1);
end;

function ShouldInstallVCRedist(): Boolean;
begin
  Result := FileExists(ExpandConstant('{app}\redist\vc_redist.x64.exe')) and not VCRedistInstalled();
end;

procedure WritePackagedSettings();
var
  MainPy, ScriptPy, DataDir, SettingsPath, Cmd, Args, QwenFlag: String;
  ResultCode: Integer;
begin
  MainPy := ExpandConstant('{app}\runtime\main-env\python.exe');
  ScriptPy := ExpandConstant('{app}\scripts\write_packaged_settings.py');
  if not FileExists(MainPy) or not FileExists(ScriptPy) then
    RaiseException('Cannot write settings: bundled Python or write_packaged_settings.py missing.');

  DataDir := ExpandConstant('{localappdata}\Sonaflow');
  ForceDirectories(DataDir);
  ForceDirectories(DataDir + '\models');
  ForceDirectories(DataDir + '\engines');
  SettingsPath := DataDir + '\visualizer_settings.json';

  QwenFlag := '';
#ifdef IncludeQwen
  if WizardIsComponentSelected('qwen') then
    QwenFlag := ' --qwen';
#endif

  Cmd := ExpandConstant('{cmd}');
  Args := '/C set PYTHONUTF8=1 && "' + MainPy + '" "' + ScriptPy + '" ' +
          IntToStr(DisplayW) + ' ' + IntToStr(DisplayH) + ' "' + SettingsPath + '"' + QwenFlag;
  if not Exec(Cmd, Args, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) or (ResultCode <> 0) then
    RaiseException('Failed to write visualizer settings.');
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Tmp, DataDir: String;
begin
  if CurStep = ssPostInstall then
  begin
    Tmp := ExpandConstant('{tmp}');
    DataDir := ExpandConstant('{localappdata}\Sonaflow');

    WizardForm.StatusLabel.Caption := 'Extracting application (this may take several minutes)...';
    WizardForm.ProgressGauge.Style := npbstMarquee;
    if not ExtractZip(Tmp + '\core-runtime.zip', ExpandConstant('{app}')) then
      RaiseException('Failed to extract core runtime.');

#ifdef IncludeModels
    WizardForm.StatusLabel.Caption := 'Extracting bundled models...';
    if FileExists(Tmp + '\models-core.zip') then
      if not ExtractZip(Tmp + '\models-core.zip', DataDir + '\models') then
        RaiseException('Failed to extract models.');
#endif

#ifdef IncludeQwen
    if WizardIsComponentSelected('qwen') and FileExists(Tmp + '\qwen-runtime.zip') then
    begin
      WizardForm.StatusLabel.Caption := 'Extracting Qwen sidecar runtime...';
      if not ExtractZip(Tmp + '\qwen-runtime.zip', ExpandConstant('{app}')) then
        RaiseException('Failed to extract Qwen sidecar runtime.');
    end;
#endif

    WritePackagedSettings();
    WizardForm.ProgressGauge.Style := npbstNormal;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  AppDir, DataDir: String;
begin
  { [UninstallDelete] handles both trees; this is a failsafe if anything remains. }
  if CurUninstallStep = usPostUninstall then
  begin
    AppDir := ExpandConstant('{app}');
    DataDir := ExpandConstant('{localappdata}\Sonaflow');
    if DirExists(AppDir) then
      DelTree(AppDir, True, True, True);
    if DirExists(DataDir) then
      DelTree(DataDir, True, True, True);
  end;
end;
