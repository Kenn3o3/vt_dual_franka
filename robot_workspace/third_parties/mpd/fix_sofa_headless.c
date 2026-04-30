/*
 * fix_sofa_headless.c
 *
 * LD_PRELOAD library for headless SOFA rendering on H100 / Mesa EGL.
 *
 * Key fixes:
 *   1. Redirect GLEW's glXGetProcAddressARB to eglGetProcAddress so SOFA
 *      gets EGL-aware function pointers.  When called after eglMakeCurrent,
 *      these pointers use the active EGL context via Mesa's unified dispatch.
 *   2. Stub out SOFA's shadow-map / FBO code that crashes without GPU hardware.
 *   3. Leave LightManager::drawScene() intact so the scene rendering visitor
 *      still traverses OglModel objects.
 *
 * Compile:
 *   gcc -shared -fPIC -O2 -o fix_sofa_headless.so fix_sofa_headless.c \
 *       -ldl -lEGL -lOSMesa
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <EGL/egl.h>
#include <GL/gl.h>

typedef void (*GL_FUNC_PTR)(void);
extern GL_FUNC_PTR OSMesaGetProcAddress(const char *funcName);

/* -----------------------------------------------------------------------
 * Redirect GLEW's GLX proc address queries to eglGetProcAddress.
 * eglGetProcAddress returns EGL-context-aware function pointers that use
 * Mesa's unified dispatch table set by eglMakeCurrent.
 * ----------------------------------------------------------------------- */

GL_FUNC_PTR glXGetProcAddressARB(const unsigned char *procName) {
    if (!procName) return NULL;
    const char *name = (const char *)procName;

    /*
     * GLX functions (glX*, wgl*, etc.) must NOT go through eglGetProcAddress.
     * SOFA's GLEW calls glXQueryVersion(NULL display) during glewInit, which
     * would SIGSEGV if it gets a real, callable GLX function pointer but has
     * no display.  Returning the dlsym pointer keeps them as harmless no-ops
     * (Mesa's GLX functions silently fail with a NULL display).
     */
    if (strncmp(name, "glX", 3) == 0) {
        GL_FUNC_PTR addr = (GL_FUNC_PTR)dlsym(RTLD_DEFAULT, name);
        return addr;
    }

    /*
     * All other GL/GLES functions go through eglGetProcAddress so that SOFA's
     * GLEW gets EGL-context-aware function pointers.  These use Mesa's unified
     * dispatch table set by eglMakeCurrent and therefore actually render into
     * the EGL PBuffer surface instead of being no-ops.
     */
    GL_FUNC_PTR addr = (GL_FUNC_PTR)eglGetProcAddress(name);
    if (addr) return addr;
    addr = (GL_FUNC_PTR)dlsym(RTLD_DEFAULT, name);
    if (addr) return addr;
    addr = OSMesaGetProcAddress(name);
    return addr;
}

GL_FUNC_PTR glXGetProcAddress(const unsigned char *procName) {
    return glXGetProcAddressARB(procName);
}

const char *glXGetClientString(void *dpy, int name) {
    (void)dpy; (void)name;
    return "";
}

const char *glXQueryExtensionsString(void *dpy, int screen) {
    (void)dpy; (void)screen;
    return "";
}

/* -----------------------------------------------------------------------
 * Override glGetString / glGetStringi so SOFA's GLEW version detection
 * uses the EGL-aware dispatch instead of libGL.so.1's GLX-path (which
 * returns NULL without a GLX context).  GLEW uses these return values to
 * decide which extensions (VBOs, shaders, etc.) are available.
 * ----------------------------------------------------------------------- */

typedef unsigned int GLenum;
typedef unsigned char GLubyte;
typedef unsigned int GLuint;

const GLubyte *glGetString(GLenum name) {
    typedef const GLubyte *(*fn_t)(GLenum);
    static fn_t _egl_fn = NULL;
    if (!_egl_fn) {
        _egl_fn = (fn_t)eglGetProcAddress("glGetString");
        if (!_egl_fn) _egl_fn = (fn_t)dlsym(RTLD_NEXT, "glGetString");
    }
    if (_egl_fn) return _egl_fn(name);
    return NULL;
}

const GLubyte *glGetStringi(GLenum name, GLuint index) {
    typedef const GLubyte *(*fn_t)(GLenum, GLuint);
    static fn_t _egl_fn = NULL;
    if (!_egl_fn) {
        _egl_fn = (fn_t)eglGetProcAddress("glGetStringi");
        if (!_egl_fn) _egl_fn = (fn_t)dlsym(RTLD_NEXT, "glGetStringi");
    }
    if (_egl_fn) return _egl_fn(name, index);
    return NULL;
}

/* -----------------------------------------------------------------------
 * FrameBufferObject::getCurrentFramebufferID() → always return 0
 * This prevents a stack-smash when the viewport is unset.
 * ----------------------------------------------------------------------- */
int _ZN4sofa2gl17FrameBufferObject23getCurrentFramebufferIDEv(void *self) {
    (void)self;
    return 0;
}

/* -----------------------------------------------------------------------
 * Light::computeShadowMapSize() → set a safe fixed 512×512 size.
 * The original calls glGetIntegerv(GL_VIEWPORT) which stack-smashes.
 * ----------------------------------------------------------------------- */
void _ZN4sofa2gl9component6shader5Light20computeShadowMapSizeEv(void *self) {
    if (!self) return;
    volatile int *p = (int *)((char *)self + 0x80);
    *p = 512;
    *(p + 1) = 512;
}

/* -----------------------------------------------------------------------
 * Light::initVisual() variants → no-op.
 * The FBO / shader objects they try to create are not available in
 * software headless mode; zeroing them prevents null-deref in draw.
 * ----------------------------------------------------------------------- */
void _ZN4sofa2gl9component6shader5Light10initVisualEv(void *s) { (void)s; }
void _ZN4sofa2gl9component6shader16DirectionalLight10initVisualEv(void *s) { (void)s; }
void _ZN4sofa2gl9component6shader11PositionalLight10initVisualEv(void *s) { (void)s; }
void _ZN4sofa2gl9component6shader9SpotLight10initVisualEv(void *s) { (void)s; }

/* LightManager::initVisual() → no-op (shadow FBO setup crashes) */
void _ZN4sofa2gl9component6shader12LightManager10initVisualEv(void *s) { (void)s; }

/* -----------------------------------------------------------------------
 * LightManager shadow-pass methods → no-op.
 * preDrawScene / postDrawScene manage shadow-map FBOs; skipping them
 * means no shadows but also no crashes.
 * NOTE: drawScene() is NOT stubbed here – it drives OglModel rendering!
 * ----------------------------------------------------------------------- */
void _ZN4sofa2gl9component6shader12LightManager12preDrawSceneEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader12LightManager13postDrawSceneEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }

/* restoreDefaultLight resets GL_LIGHT0..7 – keep it so lighting works */
/* (not stubbed) */

/* LightManager::draw(VisualParams const*) – uses const pointer */
void _ZN4sofa2gl9component6shader12LightManager4drawEPKNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }

/* fwdDraw / bwdDraw */
void _ZN4sofa2gl9component6shader12LightManager7fwdDrawEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader12LightManager7bwdDrawEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }

/* -----------------------------------------------------------------------
 * Individual Light draw / shadow methods → no-op.
 * These would try to render into shadow-map FBOs.
 * ----------------------------------------------------------------------- */
void _ZN4sofa2gl9component6shader5Light9drawLightEv(void *s) { (void)s; }
void _ZN4sofa2gl9component6shader5Light13preDrawShadowEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader5Light14postDrawShadowEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader5Light4drawEPKNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader5Light10drawSourceEPKNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }

void _ZN4sofa2gl9component6shader16DirectionalLight9drawLightEv(void *s) { (void)s; }
void _ZN4sofa2gl9component6shader16DirectionalLight13preDrawShadowEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader16DirectionalLight14postDrawShadowEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader16DirectionalLight4drawEPKNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader16DirectionalLight10drawSourceEPKNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }

void _ZN4sofa2gl9component6shader11PositionalLight9drawLightEv(void *s) { (void)s; }
void _ZN4sofa2gl9component6shader11PositionalLight13preDrawShadowEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader11PositionalLight14postDrawShadowEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader11PositionalLight4drawEPKNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader11PositionalLight10drawSourceEPKNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }

void _ZN4sofa2gl9component6shader9SpotLight9drawLightEv(void *s) { (void)s; }
void _ZN4sofa2gl9component6shader9SpotLight13preDrawShadowEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader9SpotLight14postDrawShadowEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader9SpotLight4drawEPKNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader9SpotLight10drawSourceEPKNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
