// glpresent_bridge.cpp
//
// GPU presenter with Dear ImGui settings overlay (CUDA->GL interop + Win32/WGL).
// See header comments in the original design; UI exports are documented below.

#include <windows.h>
#include <GL/gl.h>
#include <commdlg.h>

#include <cuda.h>
#include <cudaGL.h>

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_opengl2.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>

#define EXPORT extern "C" __declspec(dllexport)

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif
#ifndef GL_BGRA
#define GL_BGRA 0x80E1
#endif

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(
    HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace {

char g_error[512] = "";

HWND   g_hwnd  = nullptr;
HDC    g_hdc   = nullptr;
HGLRC  g_hglrc = nullptr;
ATOM   g_class = 0;
HINSTANCE g_hinst = nullptr;

CUcontext g_ctx = nullptr;

GLuint                 g_tex = 0;
CUgraphicsResource     g_res = nullptr;
unsigned int           g_texW = 0, g_texH = 0;

volatile bool g_close = false;
volatile bool g_toggle_overlay = false;
volatile bool g_toggle_ui = false;
bool g_fullscreen = false;
RECT g_restoredWnd{};
LONG g_restoredStyle = 0;
LONG g_restoredExStyle = 0;
int g_clientW = 0, g_clientH = 0;
int g_pixel_format = 0;

const wchar_t* kClassName = L"MV_GLPresentWindow";

struct WglViewportData {
    HDC hDC = nullptr;
};

static bool wgl_setup_viewport_dc(HWND hwnd, WglViewportData* data) {
    if (!hwnd || !data) return false;
    HDC hdc = GetDC(hwnd);
    if (!hdc) return false;
    PIXELFORMATDESCRIPTOR pfd{};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 0;
    pfd.iLayerType = PFD_MAIN_PLANE;
    int pf = g_pixel_format;
    if (pf == 0) pf = ChoosePixelFormat(hdc, &pfd);
    if (pf == 0 || !SetPixelFormat(hdc, pf, &pfd)) {
        ReleaseDC(hwnd, hdc);
        return false;
    }
    data->hDC = hdc;
    return true;
}

static void wgl_release_viewport_dc(HWND hwnd, WglViewportData* data) {
    if (!data || !data->hDC) return;
    wglMakeCurrent(nullptr, nullptr);
    ReleaseDC(hwnd, data->hDC);
    data->hDC = nullptr;
}

static void Hook_Renderer_CreateWindow(ImGuiViewport* viewport) {
    IM_ASSERT(viewport->RendererUserData == nullptr);
    auto* data = IM_NEW(WglViewportData);
    if (!wgl_setup_viewport_dc((HWND)viewport->PlatformHandle, data)) {
        IM_DELETE(data);
        return;
    }
    viewport->RendererUserData = data;
}

static void Hook_Renderer_DestroyWindow(ImGuiViewport* viewport) {
    if (viewport->RendererUserData == nullptr) return;
    auto* data = (WglViewportData*)viewport->RendererUserData;
    wgl_release_viewport_dc((HWND)viewport->PlatformHandle, data);
    IM_DELETE(data);
    viewport->RendererUserData = nullptr;
}

static void Hook_Platform_RenderWindow(ImGuiViewport* viewport, void*) {
    if (auto* data = (WglViewportData*)viewport->RendererUserData)
        wglMakeCurrent(data->hDC, g_hglrc);
}

static void Hook_Renderer_SwapBuffers(ImGuiViewport* viewport, void*) {
    if (auto* data = (WglViewportData*)viewport->RendererUserData)
        SwapBuffers(data->hDC);
}

static void ui_init_viewport_hooks() {
    ImGuiIO& io = ImGui::GetIO();
    if (!(io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)) return;
    ImGuiPlatformIO& platform_io = ImGui::GetPlatformIO();
    platform_io.Renderer_CreateWindow = Hook_Renderer_CreateWindow;
    platform_io.Renderer_DestroyWindow = Hook_Renderer_DestroyWindow;
    platform_io.Renderer_SwapBuffers = Hook_Renderer_SwapBuffers;
    platform_io.Platform_RenderWindow = Hook_Platform_RenderWindow;
}

// --- Settings UI -------------------------------------------------------------

enum UiParamType {
    UI_BOOL = 0,
    UI_INT = 1,
    UI_FLOAT = 2,
    UI_ENUM = 3,
    UI_TEXT = 4,
};

enum UiAction {
    UI_ACTION_SAVE_DEFAULTS = 1,
    UI_ACTION_EXPORT = 2,
    UI_ACTION_IMPORT = 4,
    UI_ACTION_RESTART = 8,
    UI_ACTION_POP_OUT = 16,
};

struct UiParamDef {
    const wchar_t* label;
    const wchar_t* group;
    const wchar_t* tooltip;
    int type;
    double min_val;
    double max_val;
    double step;
    const wchar_t* enum_csv;
    int requires_restart;
};

struct UiParamState {
    UiParamDef def{};
    // Owned copies of the wide strings. The UiParamDef pointers come from
    // ctypes buffers that Python frees after set_params returns, so we must
    // not hold onto them -- copy into these and read from them instead.
    std::wstring label_w;
    std::wstring group_w;
    std::wstring tooltip_w;
    std::vector<std::wstring> enum_opts;
    std::vector<std::string> enum_labels_utf8;
    std::vector<const char*> enum_label_ptrs;
    double num_val = 0.0;
    double default_num = 0.0;
    std::wstring text_val;
    std::wstring default_text;
    bool type_edit = false;
    bool type_edit_focus = false;
    bool dirty = false;
    bool user_changed = false;
};

std::vector<UiParamState> g_params;
std::unordered_map<std::wstring, bool> g_group_collapsed;  // true = collapsed
struct GroupCollapseChange {
    std::wstring group;
    bool collapsed;
};
std::vector<GroupCollapseChange> g_pending_group_collapse;
std::unordered_set<std::wstring> g_group_collapse_force;
bool g_panel_open = false;       // F1 toggles the docked settings panel
bool g_popped_out = false;       // panel in separate viewport window
bool g_restart_pending = false;
int g_pending_actions = 0;
bool g_imgui_ready = false;

void set_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_error, sizeof(g_error), fmt, args);
    va_end(args);
}

bool cu_ok(CUresult r, const char* what) {
    if (r == CUDA_SUCCESS) return true;
    const char* msg = nullptr;
    cuGetErrorString(r, &msg);
    set_error("CUDA error in %s: %s", what, msg ? msg : "unknown");
    return false;
}

void update_client_size() {
    if (!g_hwnd) return;
    RECT cli{};
    GetClientRect(g_hwnd, &cli);
    g_clientW = cli.right - cli.left;
    g_clientH = cli.bottom - cli.top;
}

void toggle_fullscreen() {
    if (!g_hwnd) return;

    if (!g_fullscreen) {
        GetWindowRect(g_hwnd, &g_restoredWnd);
        g_restoredStyle = GetWindowLong(g_hwnd, GWL_STYLE);
        g_restoredExStyle = GetWindowLong(g_hwnd, GWL_EXSTYLE);

        MONITORINFO mi{};
        mi.cbSize = sizeof(mi);
        HMONITOR mon = MonitorFromWindow(g_hwnd, MONITOR_DEFAULTTONEAREST);
        GetMonitorInfo(mon, &mi);
        const RECT& r = mi.rcMonitor;

        SetWindowLong(g_hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE);
        SetWindowPos(
            g_hwnd, HWND_TOP, r.left, r.top,
            r.right - r.left, r.bottom - r.top,
            SWP_FRAMECHANGED | SWP_SHOWWINDOW);
        g_fullscreen = true;
    } else {
        SetWindowLong(g_hwnd, GWL_STYLE, g_restoredStyle);
        SetWindowLong(g_hwnd, GWL_EXSTYLE, g_restoredExStyle);
        SetWindowPos(
            g_hwnd, HWND_TOP,
            g_restoredWnd.left, g_restoredWnd.top,
            g_restoredWnd.right - g_restoredWnd.left,
            g_restoredWnd.bottom - g_restoredWnd.top,
            SWP_FRAMECHANGED | SWP_SHOWWINDOW);
        g_fullscreen = false;
    }
    update_client_size();
}

void pump_messages() {
    MSG msg;
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}

static std::string wide_to_utf8(const wchar_t* ws) {
    if (!ws) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, ws, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) return {};
    std::vector<char> buf((size_t)n);
    WideCharToMultiByte(CP_UTF8, 0, ws, -1, buf.data(), n, nullptr, nullptr);
    return std::string(buf.data());
}

static std::vector<std::wstring> split_enum_csv(const wchar_t* csv) {
    std::vector<std::wstring> out;
    if (!csv) return out;
    std::wstring s(csv);
    size_t start = 0;
    while (start <= s.size()) {
        size_t bar = s.find(L'|', start);
        if (bar == std::wstring::npos) bar = s.size();
        std::wstring part = s.substr(start, bar - start);
        if (!part.empty()) out.push_back(part);
        start = bar + 1;
    }
    return out;
}

static int enum_index_from_value(const std::wstring& val, const std::vector<std::wstring>& opts) {
    for (size_t i = 0; i < opts.size(); ++i) {
        if (opts[i] == val) return (int)i;
    }
    return -1;
}

static void mark_param_changed(int idx) {
    if (idx < 0 || idx >= (int)g_params.size()) return;
    g_params[idx].user_changed = true;
    g_params[idx].dirty = false;
    if (g_params[idx].def.requires_restart) {
        g_restart_pending = true;
    }
}

static void ui_init_imgui() {
    if (g_imgui_ready) return;
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    io.IniFilename = nullptr;

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.Alpha = 0.92f;
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        // Platform windows need opaque bg + square corners to match OS chrome.
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    ImGui_ImplWin32_InitForOpenGL(g_hwnd);
    ImGui_ImplOpenGL2_Init();
    ui_init_viewport_hooks();
    g_imgui_ready = true;
}

static void ui_shutdown_imgui() {
    if (!g_imgui_ready) return;
    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    g_imgui_ready = false;
}

static void clamp_int(int& v, int vmin, int vmax) {
    if (v < vmin) v = vmin;
    if (v > vmax) v = vmax;
}

static bool imgui_int_param(const char* widget_id, int& v, int vmin, int vmax) {
    if (vmin > vmax) std::swap(vmin, vmax);
    clamp_int(v, vmin, vmax);

    // ImGui::SliderInt asserts when min/max exceed half the ImS32 range.
    constexpr int kImSafeMin = -1073741824;
    constexpr int kImSafeMax = 1073741823;
    const long long range = (long long)vmax - (long long)vmin;
    const bool use_input = range > 100000
        || vmax > kImSafeMax
        || vmin < kImSafeMin;

    if (use_input) {
        if (!ImGui::InputInt(widget_id, &v, 1, 10)) return false;
        clamp_int(v, vmin, vmax);
        return true;
    }

    int smin = vmin > kImSafeMin ? vmin : kImSafeMin;
    int smax = vmax < kImSafeMax ? vmax : kImSafeMax;
    if (smin > smax) smax = smin;
    if (!ImGui::SliderInt(widget_id, &v, smin, smax)) return false;
    return true;
}

static float settings_panel_width() {
    ImGuiViewport* vp = ImGui::GetMainViewport();
    const float min_w = 540.0f;
    const float max_w = 720.0f;
    const float frac = vp->WorkSize.x * 0.40f;
    if (frac < min_w) return min_w;
    if (frac > max_w) return max_w;
    return frac;
}

static void param_label_row(const std::string& lbl, bool requires_restart = false) {
    if (requires_restart) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.72f, 0.35f, 1.0f));
    }
    const float wrap_x = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x;
    ImGui::PushTextWrapPos(wrap_x);
    ImGui::TextUnformatted(lbl.c_str());
    ImGui::PopTextWrapPos();
    if (requires_restart) {
        ImGui::PopStyleColor();
    }
    ImGui::SetNextItemWidth(-1.0f);
}

static void reset_param_to_default(UiParamState& p) {
    p.num_val = p.default_num;
    p.text_val = p.default_text;
    p.type_edit = false;
    p.type_edit_focus = false;
}

static void handle_value_gestures(int idx, UiParamState& p, bool allow_type_edit) {
    if (!ImGui::IsItemHovered()) return;
    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        reset_param_to_default(p);
        mark_param_changed(idx);
    } else if (allow_type_edit && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        p.type_edit = true;
        p.type_edit_focus = false;
    }
}

static void begin_restart_widget_style(bool requires_restart) {
    if (requires_restart) {
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.38f, 0.24f, 0.08f, 0.65f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.48f, 0.30f, 0.10f, 0.75f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.55f, 0.34f, 0.12f, 0.85f));
    }
}

static void end_restart_widget_style(bool requires_restart) {
    if (requires_restart) {
        ImGui::PopStyleColor(3);
    }
}

static void cache_enum_labels(UiParamState& st) {
    st.enum_labels_utf8.clear();
    st.enum_label_ptrs.clear();
    st.enum_labels_utf8.reserve(st.enum_opts.size());
    for (const auto& w : st.enum_opts) {
        st.enum_labels_utf8.push_back(wide_to_utf8(w.c_str()));
    }
    st.enum_label_ptrs.reserve(st.enum_labels_utf8.size());
    for (const auto& s : st.enum_labels_utf8) {
        st.enum_label_ptrs.push_back(s.c_str());
    }
}

static void render_settings_panel(bool as_popup) {
    const float panel_w = settings_panel_width();
    ImGuiViewport* vp = ImGui::GetMainViewport();

    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoSavedSettings;

    if (as_popup) {
        ImGuiWindowClass wc{};
        wc.ClassId = 0x4d765310;
        wc.ViewportFlagsOverrideSet = ImGuiViewportFlags_NoAutoMerge;
        ImGui::SetNextWindowClass(&wc);
        ImGui::SetNextWindowPos(
            ImVec2(vp->WorkPos.x + vp->WorkSize.x * 0.5f, vp->WorkPos.y + 40.0f),
            ImGuiCond_FirstUseEver,
            ImVec2(0.5f, 0.0f));
        ImGui::SetNextWindowSize(ImVec2(panel_w, 720.0f), ImGuiCond_FirstUseEver);
        flags |= ImGuiWindowFlags_NoDocking;
        bool open = g_panel_open;
        if (!ImGui::Begin(u8"Music Visualizer Settings", &open, flags)) {
            ImGui::End();
            if (!open) {
                g_panel_open = false;
                g_popped_out = false;
            }
            return;
        }
        if (!open) {
            g_panel_open = false;
            g_popped_out = false;
            ImGui::End();
            return;
        }
    } else {
        ImGui::SetNextWindowPos(
            ImVec2(vp->WorkPos.x + vp->WorkSize.x - panel_w, vp->WorkPos.y),
            ImGuiCond_Always);
        ImGui::SetNextWindowSize(
            ImVec2(panel_w, vp->WorkSize.y),
            ImGuiCond_Always);
        flags |= ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar;
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.10f, 0.82f));
        if (!ImGui::Begin("##SettingsPanel", nullptr, flags)) {
            ImGui::End();
            ImGui::PopStyleColor();
            return;
        }
    }

    if (g_restart_pending) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.85f, 0.3f, 1.0f));
        ImGui::TextWrapped(
            "Some settings require a restart to take effect. "
            "Click Restart when ready (TensorRT changes may take several minutes).");
        ImGui::PopStyleColor();
        if (ImGui::Button("Restart Now")) {
            g_pending_actions |= UI_ACTION_RESTART;
            g_restart_pending = false;
        }
        ImGui::Separator();
    }

    if (ImGui::Button("Save Defaults")) g_pending_actions |= UI_ACTION_SAVE_DEFAULTS;
    ImGui::SameLine();
    if (ImGui::Button("Export JSON")) g_pending_actions |= UI_ACTION_EXPORT;
    ImGui::SameLine();
    if (ImGui::Button("Import JSON")) g_pending_actions |= UI_ACTION_IMPORT;
    ImGui::SameLine();
    if (as_popup) {
        if (ImGui::Button("Dock")) {
            g_popped_out = false;
            g_panel_open = true;
        }
    } else {
        if (ImGui::Button("Pop Out")) {
            g_popped_out = true;
            g_pending_actions |= UI_ACTION_POP_OUT;
        }
    }
    ImGui::Separator();

    std::wstring current_group;
    int group_start = 0;
    auto render_group = [&](int start, int end, const std::wstring& name) {
        std::string hdr = wide_to_utf8(name.c_str());
        auto collapsed_it = g_group_collapsed.find(name);
        const bool was_collapsed =
            collapsed_it == g_group_collapsed.end() ? true : collapsed_it->second;
        // CollapsingHeader(label, bool* p_open) hides the header when *p_open is
        // false (visibility, not collapse). Use SetNextItemOpen for collapse state.
        if (g_group_collapse_force.count(name)) {
            ImGui::SetNextItemOpen(!was_collapsed, ImGuiCond_Always);
            g_group_collapse_force.erase(name);
        }
        const bool open = ImGui::CollapsingHeader(hdr.c_str());
        const bool collapsed = !open;
        if (collapsed != was_collapsed) {
            g_group_collapsed[name] = collapsed;
            g_pending_group_collapse.push_back({name, collapsed});
        }
        if (!open)
            return;
        ImGui::Indent(4.0f);
        for (int i = start; i < end; ++i) {
            UiParamState& p = g_params[i];
            std::string lbl = !p.label_w.empty()
                ? wide_to_utf8(p.label_w.c_str())
                : ("param" + std::to_string(i));
            std::string tip = wide_to_utf8(p.tooltip_w.c_str());

            ImGui::PushID(i);
            bool changed = false;
            const bool restart = p.def.requires_restart != 0;
            switch (p.def.type) {
            case UI_BOOL: {
                bool v = p.num_val != 0.0;
                param_label_row(lbl, restart);
                begin_restart_widget_style(restart);
                if (ImGui::Checkbox("##value", &v)) {
                    p.num_val = v ? 1.0 : 0.0;
                    changed = true;
                }
                end_restart_widget_style(restart);
                handle_value_gestures(i, p, false);
                break;
            }
            case UI_INT: {
                int v = (int)std::llround(p.num_val);
                const int vmin = (int)std::llround(p.def.min_val);
                const int vmax = (int)std::llround(p.def.max_val);
                float step = p.def.step > 0 ? (float)p.def.step : 1.0f;
                param_label_row(lbl, restart);
                begin_restart_widget_style(restart);
                if (p.type_edit) {
                    if (!p.type_edit_focus) {
                        ImGui::SetKeyboardFocusHere();
                        p.type_edit_focus = true;
                    }
                    if (ImGui::InputInt("##value", &v, (int)step, (int)(step * 10.0f))) {
                        clamp_int(v, vmin, vmax);
                        p.num_val = (double)v;
                        changed = true;
                    }
                    if (!ImGui::IsItemActive()) {
                        p.type_edit = false;
                        p.type_edit_focus = false;
                    }
                } else if (imgui_int_param("##value", v, vmin, vmax)) {
                    p.num_val = (double)v;
                    changed = true;
                }
                end_restart_widget_style(restart);
                if (!p.type_edit) handle_value_gestures(i, p, true);
                break;
            }
            case UI_FLOAT: {
                float v = (float)p.num_val;
                float step = p.def.step > 0 ? (float)p.def.step : 0.001f;
                param_label_row(lbl, restart);
                begin_restart_widget_style(restart);
                if (p.type_edit) {
                    if (!p.type_edit_focus) {
                        ImGui::SetKeyboardFocusHere();
                        p.type_edit_focus = true;
                    }
                    if (ImGui::InputFloat("##value", &v, step, step * 10.0f, "%.6f")) {
                        p.num_val = (double)v;
                        changed = true;
                    }
                    if (!ImGui::IsItemActive()) {
                        p.type_edit = false;
                        p.type_edit_focus = false;
                    }
                } else if (ImGui::SliderFloat("##value", &v, (float)p.def.min_val,
                                               (float)p.def.max_val, "%.4f")) {
                    p.num_val = (double)v;
                    changed = true;
                }
                end_restart_widget_style(restart);
                if (!p.type_edit) handle_value_gestures(i, p, true);
                break;
            }
            case UI_ENUM: {
                const auto& opts = p.enum_opts;
                int cur = enum_index_from_value(p.text_val, opts);
                if (cur < 0 || cur >= (int)p.enum_label_ptrs.size()) cur = 0;
                param_label_row(lbl, restart);
                begin_restart_widget_style(restart);
                if (p.type_edit) {
                    char buf[512] = {};
                    WideCharToMultiByte(CP_UTF8, 0, p.text_val.c_str(), -1, buf, sizeof(buf), nullptr, nullptr);
                    if (!p.type_edit_focus) {
                        ImGui::SetKeyboardFocusHere();
                        p.type_edit_focus = true;
                    }
                    if (ImGui::InputText("##value", buf, sizeof(buf))) {
                        wchar_t wbuf[512] = {};
                        MultiByteToWideChar(CP_UTF8, 0, buf, -1, wbuf, 512);
                        p.text_val = wbuf;
                        changed = true;
                    }
                    if (!ImGui::IsItemActive()) {
                        p.type_edit = false;
                        p.type_edit_focus = false;
                    }
                } else if (!p.enum_label_ptrs.empty()) {
                    if (ImGui::Combo("##value", &cur, p.enum_label_ptrs.data(),
                                     (int)p.enum_label_ptrs.size())) {
                        if (cur >= 0 && cur < (int)opts.size()) {
                            p.text_val = opts[(size_t)cur];
                            changed = true;
                        }
                    }
                }
                end_restart_widget_style(restart);
                if (!p.type_edit) handle_value_gestures(i, p, true);
                break;
            }
            case UI_TEXT: {
                char buf[1024] = {};
                WideCharToMultiByte(CP_UTF8, 0, p.text_val.c_str(), -1, buf, sizeof(buf), nullptr, nullptr);
                param_label_row(lbl, restart);
                begin_restart_widget_style(restart);
                const bool enter = ImGui::InputText(
                    "##value", buf, sizeof(buf), ImGuiInputTextFlags_EnterReturnsTrue);
                {
                    wchar_t wbuf[1024] = {};
                    MultiByteToWideChar(CP_UTF8, 0, buf, -1, wbuf, 1024);
                    std::wstring next(wbuf);
                    if (p.text_val != next) {
                        p.text_val = std::move(next);
                        if (enter || ImGui::IsItemDeactivatedAfterEdit()) {
                            changed = true;
                        }
                    } else if (enter) {
                        changed = true;
                    }
                }
                end_restart_widget_style(restart);
                handle_value_gestures(i, p, false);
                break;
            }
            default:
                break;
            }
            if (!tip.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                ImGui::SetTooltip("%s", tip.c_str());
            }
            if (changed) mark_param_changed(i);
            ImGui::PopID();
            ImGui::Spacing();
        }
        ImGui::Unindent(4.0f);
    };

    // Build group ranges
    for (int i = 0; i <= (int)g_params.size(); ++i) {
        std::wstring grp = (i < (int)g_params.size() && !g_params[i].group_w.empty())
            ? g_params[i].group_w : L"General";
        if (i == (int)g_params.size() || grp != current_group) {
            if (i > group_start) {
                render_group(group_start, i, current_group.empty() ? L"General" : current_group);
            }
            group_start = i;
            current_group = grp;
        }
    }

    ImGui::End();
    if (!as_popup) ImGui::PopStyleColor();
}

static void render_ui() {
    if (!g_imgui_ready) return;

    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    // Settings panel (F1 toggles when docked; pop-out window ignores F1)
    if (g_popped_out) {
        if (g_panel_open) render_settings_panel(true);
    } else if (g_panel_open) {
        render_settings_panel(false);
    }

    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

    ImGuiIO& io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        wglMakeCurrent(g_hdc, g_hglrc);
    }
}

LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (g_imgui_ready) {
        if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wp, lp))
            return true;
    }

    ImGuiIO* io_ptr = g_imgui_ready ? &ImGui::GetIO() : nullptr;

    switch (msg) {
        case WM_CLOSE:
            g_close = true;
            return 0;
        case WM_DESTROY:
            g_close = true;
            PostQuitMessage(0);
            return 0;
        case WM_KEYDOWN:
            // F1 always toggles the settings panel (even when ImGui has focus).
            if (wp == VK_F1) {
                g_toggle_ui = true;
                if (!g_popped_out) g_panel_open = !g_panel_open;
                return 0;
            }
            if (!io_ptr || !io_ptr->WantCaptureKeyboard) {
                if (wp == VK_ESCAPE || wp == 'Q') g_close = true;
                else if (wp == VK_F7) g_toggle_overlay = true;
                else if (wp == VK_F11) toggle_fullscreen();
            }
            return 0;
        case WM_SYSCOMMAND:
            // Alt alone activates the system menu and blocks inside DefWindowProc
            // until dismissed -- freezing the present loop on the render thread.
            if ((wp & 0xfff0) == SC_KEYMENU)
                return 0;
            break;
        case WM_SIZE:
            g_clientW = LOWORD(lp);
            g_clientH = HIWORD(lp);
            return 0;
        case WM_ERASEBKGND:
            return 1;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

// Path to the .ico file; set by glpresent_set_icon_path() before glpresent_init().
static wchar_t g_icon_path[MAX_PATH] = {};

// Load the application icon.
// 1. If an .ico path was provided via glpresent_set_icon_path, load from file.
// 2. Otherwise fall back to resource ID 1 embedded in the process exe
//    (the packaged Sonaflow.exe embeds the icon via launcher.rc).
// desired_w/h == 0 means system default size.
static HICON load_app_icon(int desired_w, int desired_h) {
    UINT flags = (desired_w == 0) ? LR_DEFAULTSIZE : 0;
    if (g_icon_path[0]) {
        HICON h = (HICON)LoadImageW(nullptr, g_icon_path, IMAGE_ICON,
                                    desired_w, desired_h, LR_LOADFROMFILE | flags);
        if (h) return h;
    }
    // Packaged build: icon is embedded in Sonaflow.exe as resource ID 1.
    return (HICON)LoadImageW(GetModuleHandleW(nullptr), MAKEINTRESOURCEW(1),
                             IMAGE_ICON, desired_w, desired_h, flags);
}

bool create_window(const wchar_t* title, int w, int h) {
    g_hinst = GetModuleHandleW(nullptr);

    HICON hIconLg = load_app_icon(0, 0);
    HICON hIconSm = load_app_icon(16, 16);

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.style = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = g_hinst;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hIcon   = hIconLg;
    wc.hIconSm = hIconSm;
    wc.lpszClassName = kClassName;
    g_class = RegisterClassExW(&wc);
    if (!g_class) {
        set_error("RegisterClassExW failed (err %lu)", GetLastError());
        return false;
    }

    RECT rc{0, 0, w, h};
    DWORD style = WS_OVERLAPPEDWINDOW;
    AdjustWindowRect(&rc, style, FALSE);

    g_hwnd = CreateWindowExW(
        0, kClassName, title ? title : L"Visualizer", style,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rc.right - rc.left, rc.bottom - rc.top,
        nullptr, nullptr, g_hinst, nullptr);
    if (!g_hwnd) {
        set_error("CreateWindowExW failed (err %lu)", GetLastError());
        return false;
    }

    // Explicitly push the icon onto the window in case the class registration
    // race or a pre-existing class registration left a stale icon.
    if (hIconLg) SendMessage(g_hwnd, WM_SETICON, ICON_BIG,   (LPARAM)hIconLg);
    if (hIconSm) SendMessage(g_hwnd, WM_SETICON, ICON_SMALL, (LPARAM)hIconSm);

    g_hdc = GetDC(g_hwnd);

    PIXELFORMATDESCRIPTOR pfd{};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 0;
    pfd.iLayerType = PFD_MAIN_PLANE;
    int pf = ChoosePixelFormat(g_hdc, &pfd);
    if (!pf || !SetPixelFormat(g_hdc, pf, &pfd)) {
        set_error("pixel format setup failed (err %lu)", GetLastError());
        return false;
    }
    g_pixel_format = pf;

    g_hglrc = wglCreateContext(g_hdc);
    if (!g_hglrc || !wglMakeCurrent(g_hdc, g_hglrc)) {
        set_error("wglCreateContext/MakeCurrent failed (err %lu)", GetLastError());
        return false;
    }

    ShowWindow(g_hwnd, SW_SHOW);
    SetForegroundWindow(g_hwnd);
    g_clientW = w;
    g_clientH = h;
    return true;
}

bool ensure_texture(unsigned int w, unsigned int h) {
    if (g_tex && w == g_texW && h == g_texH) return true;

    if (g_res) { cuGraphicsUnregisterResource(g_res); g_res = nullptr; }
    if (g_tex) { glDeleteTextures(1, &g_tex); g_tex = 0; }

    glGenTextures(1, &g_tex);
    glBindTexture(GL_TEXTURE_2D, g_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    CUresult r = cuGraphicsGLRegisterImage(
        &g_res, g_tex, GL_TEXTURE_2D,
        CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
    if (!cu_ok(r, "cuGraphicsGLRegisterImage")) {
        glDeleteTextures(1, &g_tex);
        g_tex = 0;
        return false;
    }

    g_texW = w;
    g_texH = h;
    return true;
}

bool upload_to_texture(unsigned long long dptr, unsigned int w, unsigned int h) {
    if (!cu_ok(cuGraphicsMapResources(1, &g_res, 0), "cuGraphicsMapResources"))
        return false;

    CUarray array = nullptr;
    CUresult r = cuGraphicsSubResourceGetMappedArray(&array, g_res, 0, 0);
    if (!cu_ok(r, "cuGraphicsSubResourceGetMappedArray")) {
        cuGraphicsUnmapResources(1, &g_res, 0);
        return false;
    }

    CUDA_MEMCPY2D cp{};
    cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cp.srcDevice     = (CUdeviceptr)dptr;
    cp.srcPitch      = (size_t)w * 4;
    cp.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    cp.dstArray      = array;
    cp.WidthInBytes  = (size_t)w * 4;
    cp.Height        = h;
    bool ok = cu_ok(cuMemcpy2D(&cp), "cuMemcpy2D(present)");

    cuGraphicsUnmapResources(1, &g_res, 0);
    return ok;
}

void draw_quad() {
    int cw = g_clientW > 0 ? g_clientW : (int)g_texW;
    int ch = g_clientH > 0 ? g_clientH : (int)g_texH;

    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    double sx = (double)cw / g_texW;
    double sy = (double)ch / g_texH;
    double s = sx < sy ? sx : sy;
    int vw = (int)(g_texW * s + 0.5);
    int vh = (int)(g_texH * s + 0.5);
    int vx = (cw - vw) / 2;
    int vy = (ch - vh) / 2;
    glViewport(vx, vy, vw, vh);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_tex);
    glColor4f(1.f, 1.f, 1.f, 1.f);

    glBegin(GL_QUADS);
    glTexCoord2f(0.f, 1.f); glVertex2f(0.f, 0.f);
    glTexCoord2f(1.f, 1.f); glVertex2f(1.f, 0.f);
    glTexCoord2f(1.f, 0.f); glVertex2f(1.f, 1.f);
    glTexCoord2f(0.f, 0.f); glVertex2f(0.f, 1.f);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}

} // namespace

EXPORT const char* glpresent_last_error() {
    return g_error;
}

// Set the path to the application .ico file before calling glpresent_init().
// Python passes the resolved path so the DLL doesn't have to guess it.
EXPORT void glpresent_set_icon_path(const wchar_t* path) {
    if (path && path[0])
        wcscpy_s(g_icon_path, MAX_PATH, path);
    else
        g_icon_path[0] = L'\0';
}

EXPORT int glpresent_init(const wchar_t* title, int tex_w, int tex_h, int gpu) {
    g_error[0] = '\0';
    g_close = false;

    if (tex_w <= 0 || tex_h <= 0) {
        set_error("invalid texture size %dx%d", tex_w, tex_h);
        return 0;
    }

    if (!cu_ok(cuInit(0), "cuInit")) return 0;

    if (!cu_ok(cuCtxGetCurrent(&g_ctx), "cuCtxGetCurrent")) return 0;
    if (g_ctx == nullptr) {
        CUdevice dev;
        if (!cu_ok(cuDeviceGet(&dev, gpu), "cuDeviceGet")) return 0;
        if (!cu_ok(cuDevicePrimaryCtxRetain(&g_ctx, dev), "cuDevicePrimaryCtxRetain"))
            return 0;
        if (!cu_ok(cuCtxSetCurrent(g_ctx), "cuCtxSetCurrent")) return 0;
    }

    if (!create_window(title, tex_w, tex_h)) return 0;
    if (!ensure_texture((unsigned)tex_w, (unsigned)tex_h)) return 0;

    ui_init_imgui();
    pump_messages();
    return 1;
}

EXPORT int glpresent_present(unsigned long long rgba_dptr, int w, int h) {
    if (!g_hwnd) { set_error("glpresent not initialized"); return 0; }

    pump_messages();
    if (g_close) return 1;

    cuCtxSetCurrent(g_ctx);

    if (!ensure_texture((unsigned)w, (unsigned)h)) return 0;
    if (!upload_to_texture(rgba_dptr, (unsigned)w, (unsigned)h)) return 0;

    draw_quad();

    // UI draws in full window coordinates
    glViewport(0, 0, g_clientW > 0 ? g_clientW : w, g_clientH > 0 ? g_clientH : h);
    render_ui();

    SwapBuffers(g_hdc);
    return 1;
}

EXPORT int glpresent_should_close() {
    pump_messages();
    return g_close ? 1 : 0;
}

EXPORT int glpresent_take_overlay_toggle() {
    if (!g_toggle_overlay) return 0;
    g_toggle_overlay = false;
    return 1;
}

EXPORT int glpresent_take_ui_toggle() {
    if (!g_toggle_ui) return 0;
    g_toggle_ui = false;
    return 1;
}

EXPORT int glpresent_client_size(int* w, int* h) {
    if (w) *w = g_clientW;
    if (h) *h = g_clientH;
    return (g_clientW > 0 && g_clientH > 0) ? 1 : 0;
}

EXPORT void glpresent_shutdown() {
    ui_shutdown_imgui();
    g_params.clear();

    if (g_res) { cuGraphicsUnregisterResource(g_res); g_res = nullptr; }
    if (g_tex) { glDeleteTextures(1, &g_tex); g_tex = 0; }
    g_texW = g_texH = 0;

    if (g_hglrc) {
        wglMakeCurrent(nullptr, nullptr);
        wglDeleteContext(g_hglrc);
        g_hglrc = nullptr;
    }
    if (g_hwnd && g_hdc) { ReleaseDC(g_hwnd, g_hdc); g_hdc = nullptr; }
    if (g_hwnd) { DestroyWindow(g_hwnd); g_hwnd = nullptr; }
    if (g_class) { UnregisterClassW(kClassName, g_hinst); g_class = 0; }
    g_fullscreen = false;
    pump_messages();
    g_close = false;
}

// --- Settings UI exports -----------------------------------------------------

EXPORT int glpresent_ui_set_params(const UiParamDef* arr, int n) {
    g_params.clear();
    if (!arr || n <= 0) return 0;
    g_params.reserve((size_t)n);
    for (int i = 0; i < n; ++i) {
        UiParamState st{};
        st.def = arr[i];
        // Deep-copy the wide strings into owned storage; the source pointers
        // are owned by Python's ctypes buffers and are freed after this call.
        st.label_w = arr[i].label ? arr[i].label : L"";
        st.group_w = arr[i].group ? arr[i].group : L"General";
        st.tooltip_w = arr[i].tooltip ? arr[i].tooltip : L"";
        st.enum_opts = split_enum_csv(arr[i].enum_csv);
        cache_enum_labels(st);
        // Drop the dangling pointers so nothing reads them by accident.
        st.def.label = nullptr;
        st.def.group = nullptr;
        st.def.tooltip = nullptr;
        st.def.enum_csv = nullptr;
        st.num_val = st.def.min_val;
        g_params.push_back(std::move(st));
    }
    return (int)g_params.size();
}

EXPORT void glpresent_ui_set_value(int idx, double val) {
    if (idx < 0 || idx >= (int)g_params.size()) return;
    g_params[idx].num_val = val;
    g_params[idx].user_changed = false;
}

EXPORT void glpresent_ui_set_text(int idx, const wchar_t* text) {
    if (idx < 0 || idx >= (int)g_params.size()) return;
    g_params[idx].text_val = text ? text : L"";
    g_params[idx].user_changed = false;
}

EXPORT void glpresent_ui_set_default_value(int idx, double val) {
    if (idx < 0 || idx >= (int)g_params.size()) return;
    g_params[idx].default_num = val;
}

EXPORT void glpresent_ui_set_default_text(int idx, const wchar_t* text) {
    if (idx < 0 || idx >= (int)g_params.size()) return;
    g_params[idx].default_text = text ? text : L"";
}

EXPORT int glpresent_ui_take_changes(int* idx_buf, double* val_buf, int max_n) {
    int count = 0;
    for (int i = 0; i < (int)g_params.size() && count < max_n; ++i) {
        UiParamState& p = g_params[i];
        if (!p.user_changed) continue;
        // Skip text-backed params: those are drained via param_changed_text /
        // get_text. Consuming them here would clear user_changed and report a
        // bogus numeric value (num_val == 0), clobbering the real string.
        if (p.def.type == UI_TEXT || p.def.type == UI_ENUM) continue;
        p.user_changed = false;
        if (idx_buf) idx_buf[count] = i;
        if (val_buf) val_buf[count] = p.num_val;
        ++count;
    }
    return count;
}

EXPORT int glpresent_ui_param_changed_text(int idx) {
    if (idx < 0 || idx >= (int)g_params.size()) return 0;
    UiParamState& p = g_params[idx];
    if (!p.user_changed) return 0;
    if (p.def.type != UI_TEXT && p.def.type != UI_ENUM) return 0;
    p.user_changed = false;
    return 1;
}

EXPORT const wchar_t* glpresent_ui_get_text(int idx) {
    if (idx < 0 || idx >= (int)g_params.size()) return L"";
    return g_params[idx].text_val.c_str();
}

EXPORT int glpresent_ui_take_actions() {
    int a = g_pending_actions;
    g_pending_actions = 0;
    return a;
}

EXPORT void glpresent_ui_set_restart_pending(int pending) {
    g_restart_pending = pending != 0;
}

EXPORT void glpresent_ui_set_group_collapsed(const wchar_t* group, int collapsed) {
    if (!group) return;
    g_group_collapsed[group] = collapsed != 0;
    g_group_collapse_force.insert(group);
}

EXPORT int glpresent_ui_take_group_collapsed_changes(
    wchar_t* group_out, int group_cap, int* collapsed_out) {
    if (g_pending_group_collapse.empty()) return 0;
    GroupCollapseChange ch = std::move(g_pending_group_collapse.front());
    g_pending_group_collapse.erase(g_pending_group_collapse.begin());
    if (group_out && group_cap > 0) {
        wcsncpy_s(group_out, (size_t)group_cap, ch.group.c_str(), _TRUNCATE);
    }
    if (collapsed_out) *collapsed_out = ch.collapsed ? 1 : 0;
    return 1;
}

EXPORT int glpresent_ui_open_file_dialog(int save_mode, wchar_t* out_path, int cap) {
    if (!out_path || cap <= 0) return 0;
    out_path[0] = L'\0';

    wchar_t file[MAX_PATH] = L"visualizer_settings.json";
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = g_hwnd;
    ofn.lpstrFilter = L"JSON Settings\0*.json\0All Files\0*.*\0";
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrDefExt = L"json";
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;
    if (!save_mode) ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;

    BOOL ok = save_mode ? GetSaveFileNameW(&ofn) : GetOpenFileNameW(&ofn);
    if (!ok) return 0;
    wcsncpy_s(out_path, (size_t)cap, file, _TRUNCATE);
    return 1;
}

EXPORT int glpresent_ui_is_visible() {
    return g_panel_open ? 1 : 0;
}

EXPORT void glpresent_ui_set_visible(int visible) {
    if (!g_popped_out) g_panel_open = visible != 0;
}

EXPORT void glpresent_set_fullscreen(int on) {
    if (!g_hwnd) return;
    const bool want = on != 0;
    if (want != g_fullscreen) toggle_fullscreen();
}
