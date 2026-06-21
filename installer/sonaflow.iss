; Sonaflow web-bootstrapper installer (Inno Setup 6.1+).
;
; NOTE: Requires a remote release host with uploaded payloads. Until you have one,
; use sonaflow-offline.iss + tools\Build-OfflineInstaller.ps1 instead.
;
; Payload metadata is embedded at compile time via generated\payloads.iss +
; generated\engines.iss (produced by tools\Build-Payloads.ps1). When those are
; absent (editing/standalone compile), safe placeholder defaults are used.

#ifexist "generated\payloads.iss"
  #include "generated\payloads.iss"
#else
  #pragma message "generated\payloads.iss not found - using DEV placeholder payload metadata."
  #define AppVersion "0.0.0-dev"
  #define PayloadBaseUrl "https://downloads.example.com/sonaflow/dev"
  #define CoreRuntimeUrl "https://downloads.example.com/sonaflow/dev/core-runtime.zip"
  #define CoreRuntimeSha ""
  #define CoreRuntimeSize "0"
#endif

#define AppName "Sonaflow"
#define AppPublisher "Sonaflow"
#define AppExe "Sonaflow.exe"

[Setup]
AppId={{B4F2B1B0-7B3E-4E2A-9C1D-5F6A7E8D9C01}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
; Per-user install: no admin/UAC. Program files under LocalAppData\Programs.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputBaseFilename=Sonaflow-Setup-{#AppVersion}
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\{#AppExe}
; The installer itself stays tiny; payloads are downloaded at run time.

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Components]
Name: "core";  Description: "Sonaflow core (runtime, app, SD-Turbo + CLAP models)"; Types: full compact custom; Flags: fixed
#ifdef HaveQwen
Name: "qwen";  Description: "Qwen2-Audio captions (large: adds the audio captioning model, ~16 GB)"; Types: full
#endif

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Icons]
Name: "{group}\{#AppName}";        Filename: "{app}\{#AppExe}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\{#AppName}";  Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Payload files extracted in [Code] are not tracked individually.
Type: filesandordirs; Name: "{app}"
Type: filesandordirs; Name: "{localappdata}\Sonaflow"

[Code]
const
  SM_CXSCREEN = 0;
  SM_CYSCREEN = 1;
  VC_REDIST_URL = 'https://aka.ms/vs/17/release/vc_redist.x64.exe';

function GetSystemMetrics(nIndex: Integer): Integer;
  external 'GetSystemMetrics@user32.dll stdcall';

var
  ResPage: TInputQueryWizardPage;
  GenW, GenH: Integer;
  DownloadPage: TDownloadWizardPage;

{ Engine lookup table (generated). Fallback stub if no engines were published. }
#ifexist "generated\engines.iss"
  #include "generated\engines.iss"
#else
function FindPrebuiltEngine(Res: String; var Url, Sha: String): Boolean;
begin
  Result := False;
end;
#endif

{ ---- resolution snap: mirrors music_visualizer/utils/resolution.py ---- }
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

{ ---- prereq: NVIDIA driver presence/version via nvidia-smi ---- }
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
              'Sonaflow requires an NVIDIA RTX GPU with a recent driver for real-time generation.' + #13#10#13#10 +
              'Continue anyway?', mbConfirmation, MB_YESNO) = IDNO then
      Result := False;
  end
  else
  begin
    Log('Detected GPU: ' + Info);
    { v1 was validated on RTX 50-series; warn (do not block) otherwise. }
    if (Pos('RTX 50', Uppercase(Info)) = 0) and (Pos('RTX 50', Info) = 0) then
      MsgBox('Detected GPU: ' + Info + #13#10#13#10 +
             'Sonaflow''s accelerated (TensorRT) path was validated on RTX 50-series GPUs. ' +
             'On other GPUs the prebuilt engine may not load and will be rebuilt on first run, ' +
             'which can take longer or may be unsupported.', mbInformation, MB_OK);
  end;
end;

function VCRedistInstalled(): Boolean;
var
  Installed: Cardinal;
begin
  Result := RegQueryDWordValue(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', Installed) and (Installed = 1);
  if not Result then
    Result := RegQueryDWordValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', Installed) and (Installed = 1);
end;

procedure InitializeWizard();
begin
  ResPage := CreateInputQueryPage(wpSelectComponents,
    'Display resolution',
    'Choose the resolution Sonaflow should target.',
    'Enter your display resolution. Sonaflow generates at a fixed 512px height and ' +
    'snaps the width (to a multiple of 64) to match your aspect ratio. ' +
    'A matching prebuilt graphics engine is downloaded when available; otherwise it ' +
    'is built once on first launch.');
  ResPage.Add('Display width (px):', False);
  ResPage.Add('Display height (px):', False);
  ResPage.Values[0] := IntToStr(GetSystemMetrics(SM_CXSCREEN));
  ResPage.Values[1] := IntToStr(GetSystemMetrics(SM_CYSCREEN));

  DownloadPage := CreateDownloadPage(SetupMessage(msgWizardPreparing), SetupMessage(msgPreparingDesc), nil);
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
    GenH := 512;
    GenW := SnapWidth(DW, DH);
    MsgBox('Sonaflow will generate at ' + ResTag + ' for your ' + IntToStr(DW) + 'x' + IntToStr(DH) +
           ' display.', mbInformation, MB_OK);
  end;
end;

function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo, MemoTypeInfo,
  MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
begin
  Result := MemoDirInfo + NewLine + NewLine +
            'Generation resolution:' + NewLine + Space + ResTag + NewLine + NewLine +
            MemoComponentsInfo + NewLine + MemoTasksInfo;
end;

{ Queue downloads onto the download page based on chosen components + resolution. }
procedure QueueDownloads();
var
  EngUrl, EngSha: String;
begin
  DownloadPage.Clear;
  DownloadPage.Add('{#CoreRuntimeUrl}', 'core-runtime.zip', '{#CoreRuntimeSha}');
#ifdef HaveModelsCore
  DownloadPage.Add('{#ModelsCoreUrl}', 'models-core.zip', '{#ModelsCoreSha}');
#endif
#ifdef HaveQwen
  if IsComponentSelected('qwen') then
  begin
    DownloadPage.Add('{#QwenRuntimeUrl}', 'qwen-runtime.zip', '{#QwenRuntimeSha}');
    #ifdef QwenModelUrl
    DownloadPage.Add('{#QwenModelUrl}', 'qwen-model.zip', '{#QwenModelSha}');
    #endif
  end;
#endif
  if FindPrebuiltEngine(ResTag, EngUrl, EngSha) then
    DownloadPage.Add(EngUrl, 'engine.zip', EngSha);

  if not VCRedistInstalled() then
    DownloadPage.Add(VC_REDIST_URL, 'vc_redist.x64.exe', '');
end;

{ Stream-extract a zip into Dest using .NET via PowerShell (handles multi-GB). }
function ExtractZip(ZipPath, Dest: String): Boolean;
var
  Cmd, Args: String;
  ResultCode: Integer;
begin
  ForceDirectories(Dest);
  Cmd := ExpandConstant('{cmd}');
  Args := '/C powershell -NoProfile -ExecutionPolicy Bypass -Command ' +
          '"Add-Type -AssemblyName System.IO.Compression.FileSystem; ' +
          '[System.IO.Compression.ZipFile]::ExtractToDirectory(''' + ZipPath + ''', ''' + Dest + ''', $true)"';
  Result := Exec(Cmd, Args, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
end;

procedure WritePackagedSettings();
var
  DataDir, SettingsPath, Json: String;
begin
  DataDir := ExpandConstant('{localappdata}\Sonaflow');
  ForceDirectories(DataDir);
  ForceDirectories(DataDir + '\models');
  ForceDirectories(DataDir + '\engines');
  SettingsPath := DataDir + '\visualizer_settings.json';
  { Minimal overrides; the app merges these onto its built-in defaults. width/height
    drive the (resolution-specific) TensorRT engine; acceleration enables it. }
  Json := '{' + #13#10 +
          '  "width": ' + IntToStr(GenW) + ',' + #13#10 +
          '  "height": ' + IntToStr(GenH) + ',' + #13#10 +
          '  "acceleration": "tensorrt"' + #13#10 +
          '}' + #13#10;
  if not FileExists(SettingsPath) then
    SaveStringToFile(SettingsPath, Json, False);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Tmp, DataDir: String;
  RC: Integer;
begin
  if CurStep = ssInstall then
  begin
    { Download phase: must complete before files are laid down. }
    QueueDownloads();
    DownloadPage.Show;
    try
      try
        DownloadPage.Download;
      except
        if DownloadPage.AbortedByUser then
          RaiseException('Download cancelled.')
        else
          RaiseException('Download failed: ' + GetExceptionMessage);
      end;
    finally
      DownloadPage.Hide;
    end;
  end
  else if CurStep = ssPostInstall then
  begin
    Tmp := ExpandConstant('{tmp}');
    DataDir := ExpandConstant('{localappdata}\Sonaflow');

    // Core runtime -> install dir
    if not ExtractZip(Tmp + '\core-runtime.zip', ExpandConstant('{app}')) then
      RaiseException('Failed to extract core runtime.');

    // Core models -> per-user data\models
    if FileExists(Tmp + '\models-core.zip') then
      ExtractZip(Tmp + '\models-core.zip', DataDir + '\models');

    // Qwen (optional)
    if FileExists(Tmp + '\qwen-runtime.zip') then
      ExtractZip(Tmp + '\qwen-runtime.zip', ExpandConstant('{app}'));
    if FileExists(Tmp + '\qwen-model.zip') then
      ExtractZip(Tmp + '\qwen-model.zip', DataDir + '\models');

    // Prebuilt engine (optional) -> per-user data\engines
    if FileExists(Tmp + '\engine.zip') then
      ExtractZip(Tmp + '\engine.zip', DataDir + '\engines');

    // VC++ runtime if we downloaded it
    if FileExists(Tmp + '\vc_redist.x64.exe') then
      Exec(Tmp + '\vc_redist.x64.exe', '/quiet /norestart', '', SW_SHOW, ewWaitUntilTerminated, RC);

    WritePackagedSettings();
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  AppDir, DataDir: String;
begin
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
