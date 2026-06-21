/*
 * Sonaflow launcher stub.
 *
 * A small console executable that makes a packaged install behave like a normal
 * app: it locates the install tree relative to its own path, points the app at a
 * writable per-user data directory, and runs the frozen interpreter on main.py,
 * forwarding any extra command-line arguments verbatim.
 *
 * Console subsystem on purpose: the visualizer accepts live prompt injection by
 * typing into the terminal, so a console window is the intended UX.
 *
 * Layout assumed (relative to this exe):
 *     <root>\Sonaflow.exe                         (this)
 *     <root>\runtime\main-env\python.exe
 *     <root>\app\music_visualizer\main.py
 *     <root>\streamdiffusion\                      (cwd for the child)
 * Mutable data lives at %LOCALAPPDATA%\Sonaflow.
 */

#include <windows.h>
#include <shlobj.h>
#include <stdio.h>
#include <wchar.h>

#pragma comment(lib, "shell32.lib")

static void join(wchar_t *dst, size_t cap, const wchar_t *base, const wchar_t *tail) {
    _snwprintf_s(dst, cap, _TRUNCATE, L"%s\\%s", base, tail);
}

/* Return a pointer into the raw command line just past argv[0] (the exe). */
static const wchar_t *args_after_exe(const wchar_t *cmdline) {
    const wchar_t *p = cmdline;
    if (*p == L'"') {
        p++;
        while (*p && *p != L'"') p++;
        if (*p == L'"') p++;
    } else {
        while (*p && *p != L' ' && *p != L'\t') p++;
    }
    while (*p == L' ' || *p == L'\t') p++;
    return p;
}

int wmain(void) {
    wchar_t exePath[MAX_PATH];
    if (!GetModuleFileNameW(NULL, exePath, MAX_PATH)) {
        fwprintf(stderr, L"Sonaflow: cannot resolve executable path.\n");
        return 1;
    }
    /* Strip the file name to get the install root. */
    wchar_t *slash = wcsrchr(exePath, L'\\');
    if (slash) *slash = L'\0';
    const wchar_t *root = exePath;

    wchar_t pythonExe[MAX_PATH], mainPy[MAX_PATH], sdPath[MAX_PATH];
    join(pythonExe, MAX_PATH, root, L"runtime\\main-env\\python.exe");
    join(mainPy, MAX_PATH, root, L"app\\music_visualizer\\main.py");
    join(sdPath, MAX_PATH, root, L"streamdiffusion");

    if (GetFileAttributesW(pythonExe) == INVALID_FILE_ATTRIBUTES) {
        fwprintf(stderr, L"Sonaflow: runtime not found at %s\n", pythonExe);
        return 1;
    }

    /* Writable per-user data dir: %LOCALAPPDATA%\Sonaflow */
    wchar_t dataDir[MAX_PATH], modelsDir[MAX_PATH], enginesDir[MAX_PATH];
    const wchar_t *localApp = _wgetenv(L"LOCALAPPDATA");
    if (!localApp) {
        fwprintf(stderr, L"Sonaflow: LOCALAPPDATA is not set.\n");
        return 1;
    }
    _snwprintf_s(dataDir, MAX_PATH, _TRUNCATE, L"%s\\Sonaflow", localApp);
    join(modelsDir, MAX_PATH, dataDir, L"models");
    join(enginesDir, MAX_PATH, dataDir, L"engines");
    SHCreateDirectoryExW(NULL, dataDir, NULL);
    SHCreateDirectoryExW(NULL, modelsDir, NULL);
    SHCreateDirectoryExW(NULL, enginesDir, NULL);

    SetEnvironmentVariableW(L"PYTHONUTF8", L"1");
    SetEnvironmentVariableW(L"SONAFLOW_DATA_DIR", dataDir);
    SetEnvironmentVariableW(L"HF_HOME", modelsDir);
    SetEnvironmentVariableW(L"STREAMDIFFUSION_PATH", sdPath);
    SetEnvironmentVariableW(L"SONAFLOW_ENGINE_DIR", enginesDir);

    /* Optional Qwen sidecar (separate component); only point at it if present. */
    wchar_t qwenPy[MAX_PATH];
    join(qwenPy, MAX_PATH, root, L"runtime\\qwen-env\\python.exe");
    if (GetFileAttributesW(qwenPy) != INVALID_FILE_ATTRIBUTES) {
        SetEnvironmentVariableW(L"SONAFLOW_QWEN_PYTHON", qwenPy);
    }

    /* Build child command line: "python.exe" "main.py" <forwarded args...> */
    const wchar_t *forwarded = args_after_exe(GetCommandLineW());
    size_t cap = wcslen(pythonExe) + wcslen(mainPy) + wcslen(forwarded) + 16;
    wchar_t *cmd = (wchar_t *)malloc(cap * sizeof(wchar_t));
    if (!cmd) { fwprintf(stderr, L"Sonaflow: out of memory.\n"); return 1; }
    if (*forwarded)
        _snwprintf_s(cmd, cap, _TRUNCATE, L"\"%s\" \"%s\" %s", pythonExe, mainPy, forwarded);
    else
        _snwprintf_s(cmd, cap, _TRUNCATE, L"\"%s\" \"%s\"", pythonExe, mainPy);

    SetCurrentDirectoryW(sdPath);

    STARTUPINFOW si; PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si)); si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    if (!CreateProcessW(NULL, cmd, NULL, NULL, TRUE, 0, NULL, sdPath, &si, &pi)) {
        fwprintf(stderr, L"Sonaflow: failed to launch interpreter (error %lu).\n", GetLastError());
        free(cmd);
        return 1;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    free(cmd);
    return (int)code;
}
