// rtxvsr_bridge.cpp
//
// Minimal C bridge over the NVIDIA RTX Video SDK (NGX) Video Super Resolution
// feature, exposing a tiny ctypes-friendly surface for the Python visualizer:
//
//   rtxvsr_init(app_data_path, gpu)   -> retain CUDA primary ctx + create VSR feature
//   rtxvsr_upscale(in, inW,inH, out, outW,outH, quality)
//                                     -> VSR upscale one frame, device ptr -> device ptr
//   rtxvsr_shutdown()
//   rtxvsr_last_error()               -> human-readable last failure
//
// Pixel format (matches the SDK's CUDA "ARGB" path): 8-bit, 4 channels, packed
// as a little-endian 0xAARRGGBB DWORD, i.e. in-memory byte order B,G,R,A. Pitch
// is exactly 4*width (tightly packed), which is what the deviceptr path requires.
//
// The context is the device's *primary* context (cuDevicePrimaryCtxRetain), which
// is the same context the CUDA runtime (and therefore PyTorch) uses. That makes
// the device pointers from torch tensors directly valid here -- no IPC, no copy
// across contexts.

#include <cuda.h>

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_defs.h>
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_vsr.h>
#include <nvsdk_ngx_defs_vsr.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>

#define APP_ID 0xA1B2C3D4ULL

#define EXPORT extern "C" __declspec(dllexport)

namespace {

char g_error[512] = "";

CUdevice    g_device     = 0;
CUcontext   g_context    = nullptr;
bool        g_ctxRetained = false;

NVSDK_NGX_Parameter* g_params  = nullptr;
NVSDK_NGX_Handle*    g_vsr     = nullptr;

// Persistent CUDA arrays / objects, reallocated only when a dimension changes.
CUarray      g_srcArray = nullptr;
CUtexObject  g_srcTex   = 0;
unsigned int g_srcW = 0, g_srcH = 0;

CUarray      g_dstArray = nullptr;
CUsurfObject g_dstSurf  = 0;
unsigned int g_dstW = 0, g_dstH = 0;

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

bool ngx_ok(NVSDK_NGX_Result r, const char* what) {
    if (NVSDK_NGX_SUCCEED(r)) return true;
    set_error("NGX error in %s: 0x%08x", what, (unsigned)r);
    return false;
}

void destroy_src() {
    if (g_srcTex)   { cuTexObjectDestroy(g_srcTex); g_srcTex = 0; }
    if (g_srcArray) { cuArrayDestroy(g_srcArray);   g_srcArray = nullptr; }
    g_srcW = g_srcH = 0;
}

void destroy_dst() {
    if (g_dstSurf)  { cuSurfObjectDestroy(g_dstSurf); g_dstSurf = 0; }
    if (g_dstArray) { cuArrayDestroy(g_dstArray);     g_dstArray = nullptr; }
    g_dstW = g_dstH = 0;
}

// (Re)create the source texture-backed array (8-bit, 4 channels).
bool ensure_src(unsigned int w, unsigned int h) {
    if (g_srcArray && w == g_srcW && h == g_srcH) return true;
    destroy_src();

    CUDA_ARRAY_DESCRIPTOR desc{};
    desc.Width       = w;
    desc.Height      = h;
    desc.Format      = CU_AD_FORMAT_UNSIGNED_INT8;
    desc.NumChannels = 4;
    if (!cu_ok(cuArrayCreate(&g_srcArray, &desc), "cuArrayCreate(src)")) return false;

    CUDA_RESOURCE_DESC res{};
    res.resType = CU_RESOURCE_TYPE_ARRAY;
    res.res.array.hArray = g_srcArray;

    CUDA_TEXTURE_DESC tex{};
    tex.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
    tex.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    tex.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
    tex.filterMode     = CU_TR_FILTER_MODE_LINEAR;
    tex.flags          = CU_TRSF_NORMALIZED_COORDINATES;
    if (!cu_ok(cuTexObjectCreate(&g_srcTex, &res, &tex, nullptr), "cuTexObjectCreate(src)"))
        return false;

    g_srcW = w; g_srcH = h;
    return true;
}

// (Re)create the destination surface-backed array (8-bit, 4 channels).
bool ensure_dst(unsigned int w, unsigned int h) {
    if (g_dstArray && w == g_dstW && h == g_dstH) return true;
    destroy_dst();

    CUDA_ARRAY_DESCRIPTOR desc{};
    desc.Width       = w;
    desc.Height      = h;
    desc.Format      = CU_AD_FORMAT_UNSIGNED_INT8;
    desc.NumChannels = 4;
    if (!cu_ok(cuArrayCreate(&g_dstArray, &desc), "cuArrayCreate(dst)")) return false;

    CUDA_RESOURCE_DESC res{};
    res.resType = CU_RESOURCE_TYPE_ARRAY;
    res.res.array.hArray = g_dstArray;
    if (!cu_ok(cuSurfObjectCreate(&g_dstSurf, &res), "cuSurfObjectCreate(dst)"))
        return false;

    g_dstW = w; g_dstH = h;
    return true;
}

} // namespace

EXPORT const char* rtxvsr_last_error() {
    return g_error;
}

EXPORT int rtxvsr_init(const wchar_t* app_data_path, int gpu) {
    g_error[0] = '\0';

    if (!cu_ok(cuInit(0), "cuInit")) return 0;

    int count = 0;
    if (!cu_ok(cuDeviceGetCount(&count), "cuDeviceGetCount")) return 0;
    if (gpu < 0 || gpu >= count) {
        set_error("invalid gpu index %d (have %d device(s))", gpu, count);
        return 0;
    }
    if (!cu_ok(cuDeviceGet(&g_device, gpu), "cuDeviceGet")) return 0;

    // Share PyTorch's context: the runtime API uses the device primary context.
    if (!cu_ok(cuDevicePrimaryCtxRetain(&g_context, g_device), "cuDevicePrimaryCtxRetain"))
        return 0;
    g_ctxRetained = true;
    if (!cu_ok(cuCtxSetCurrent(g_context), "cuCtxSetCurrent")) return 0;

    const wchar_t* path = (app_data_path && app_data_path[0]) ? app_data_path : L".";

    // NGX only searches the application folder (python.exe's dir) for feature
    // snippets by default. Add the SDK's snippet dir (where nvngx_vsr.dll lives)
    // to the search list so VSR is discoverable.
    static std::wstring pathStore;
    pathStore = path;
    const wchar_t* paths[1] = { pathStore.c_str() };
    NVSDK_NGX_FeatureCommonInfo info = {};
    info.PathListInfo.Path   = paths;
    info.PathListInfo.Length = 1;

    if (!ngx_ok(NVSDK_NGX_CUDA_Init(APP_ID, path, &info), "NVSDK_NGX_CUDA_Init"))
        return 0;

    if (!ngx_ok(NVSDK_NGX_CUDA_GetCapabilityParameters(&g_params),
                "NVSDK_NGX_CUDA_GetCapabilityParameters"))
        return 0;

    int available = 0;
    if (!ngx_ok(g_params->Get(NVSDK_NGX_Parameter_VSR_Available, &available),
                "Get(VSR_Available)"))
        return 0;
    if (!available) {
        int needsDriver = 0, major = 0, minor = 0;
        g_params->Get(NVSDK_NGX_Parameter_VSR_NeedsUpdatedDriver, &needsDriver);
        g_params->Get(NVSDK_NGX_Parameter_VSR_MinDriverVersionMajor, &major);
        g_params->Get(NVSDK_NGX_Parameter_VSR_MinDriverVersionMinor, &minor);
        if (needsDriver) {
            set_error("VSR unavailable: driver too old (need >= %d.%d).", major, minor);
        } else {
            set_error("VSR unavailable: nvngx_vsr.dll not found in the search path "
                      "(looked in '%ls' and the app folder), or this GPU is "
                      "unsupported.", path);
        }
        return 0;
    }

    NVSDK_NGX_CUDA_VSR_Create_Params create{};
    create.InCUContext = g_context;
    create.InCUStream  = nullptr;  // default stream; Python synchronizes around calls
    if (!ngx_ok(NGX_CUDA_CREATE_VSR(&g_vsr, g_params, &create), "NGX_CUDA_CREATE_VSR"))
        return 0;

    return 1;
}

EXPORT int rtxvsr_upscale(unsigned long long in_dptr,
                          unsigned int in_w, unsigned int in_h,
                          unsigned long long out_dptr,
                          unsigned int out_w, unsigned int out_h,
                          unsigned int quality) {
    if (!g_vsr) { set_error("rtxvsr not initialized"); return 0; }

    // Re-asserting the context is cheap and makes us robust to being called from
    // a different host thread than rtxvsr_init() (contexts are per-thread).
    if (!cu_ok(cuCtxSetCurrent(g_context), "cuCtxSetCurrent")) return 0;

    if (!ensure_src(in_w, in_h)) return 0;
    if (!ensure_dst(out_w, out_h)) return 0;

    // Upload input device buffer (pitch = 4*width) into the texture array.
    CUDA_MEMCPY2D up{};
    up.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    up.srcDevice     = (CUdeviceptr)in_dptr;
    up.srcPitch      = (size_t)in_w * 4;
    up.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    up.dstArray      = g_srcArray;
    up.WidthInBytes  = (size_t)in_w * 4;
    up.Height        = in_h;
    if (!cu_ok(cuMemcpy2D(&up), "cuMemcpy2D(upload)")) return 0;

    CUtexObject  inTex  = g_srcTex;
    CUsurfObject outSurf = g_dstSurf;

    NVSDK_NGX_CUDA_VSR_Eval_Params eval{};
    eval.pInput  = &inTex;
    eval.pOutput = &outSurf;
    eval.InputSubrectBase.X      = 0;
    eval.InputSubrectBase.Y      = 0;
    eval.InputSubrectSize.Width  = in_w;
    eval.InputSubrectSize.Height = in_h;
    eval.OutputSubrectBase.X     = 0;
    eval.OutputSubrectBase.Y     = 0;
    eval.OutputSubrectSize.Width  = out_w;
    eval.OutputSubrectSize.Height = out_h;
    eval.QualityLevel = (NVSDK_NGX_VSR_QualityLevel)quality;

    if (!ngx_ok(NGX_CUDA_EVALUATE_VSR(g_vsr, g_params, &eval), "NGX_CUDA_EVALUATE_VSR"))
        return 0;

    // Download result array into the output device buffer (pitch = 4*width).
    CUDA_MEMCPY2D down{};
    down.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    down.srcArray      = g_dstArray;
    down.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    down.dstDevice     = (CUdeviceptr)out_dptr;
    down.dstPitch      = (size_t)out_w * 4;
    down.WidthInBytes  = (size_t)out_w * 4;
    down.Height        = out_h;
    if (!cu_ok(cuMemcpy2D(&down), "cuMemcpy2D(download)")) return 0;

    return 1;
}

EXPORT void rtxvsr_shutdown() {
    destroy_src();
    destroy_dst();
    if (g_vsr)    { NVSDK_NGX_CUDA_ReleaseFeature(g_vsr); g_vsr = nullptr; }
    if (g_params) { NVSDK_NGX_CUDA_DestroyParameters(g_params); g_params = nullptr; }
    NVSDK_NGX_CUDA_Shutdown();
    if (g_ctxRetained) {
        cuDevicePrimaryCtxRelease(g_device);
        g_ctxRetained = false;
        g_context = nullptr;
    }
}
