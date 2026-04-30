/*
 * fix_sofa_egl.c
 *
 * LD_PRELOAD library for headless SOFA rendering on H100 with NVIDIA EGL.
 *
 * Problem: SOFA's OglModel::internalDraw() calls legacy GL functions
 * (glColor3f, glLightModeli, glBegin/End, etc.) via its PLT entries,
 * which resolve to libOpenGL.so.0 (GLVND dispatch stubs).  In some
 * environments the GLVND dispatch table for these stubs is not properly
 * connected to the NVIDIA EGL vendor, causing SIGSEGV inside
 * libnvidia-eglcore.so with a NULL per-context state pointer.
 *
 * Root cause: eglGetProcAddress returns different (working) pointers than
 * libOpenGL.so.0's GLVND dispatch stubs for the same function.  Replacing
 * SOFA's PLT resolutions with eglGetProcAddress-obtained pointers (via
 * LD_PRELOAD interception) fixes the crash.
 *
 * This library intercepts:
 *   1. All legacy OpenGL functions that SOFA calls directly (via PLT).
 *   2. GLEW's glXGetProcAddressARB → eglGetProcAddress (for GLEW init).
 *   3. glGetString / glGetStringi (for correct version detection).
 *   4. SOFA lighting / shadow FBO stubs (to prevent FBO crashes).
 *
 * Compile:
 *   gcc -shared -fPIC -O2 -o fix_sofa_egl.so fix_sofa_egl.c -ldl
 *   (no -lEGL to avoid eager NVIDIA EGL initialization in parent)
 */

#define _GNU_SOURCE
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>

typedef void (*GL_FUNC_PTR)(void);
typedef void *(*egl_get_proc_t)(const char *);

/* -----------------------------------------------------------------------
 * Lazily open the SYSTEM (GLVND) libEGL and cache eglGetProcAddress.
 * We always load the system libEGL to get NVIDIA EGL's dispatch table,
 * not conda's copy.
 * ----------------------------------------------------------------------- */
static egl_get_proc_t get_egl_proc(void) {
    static egl_get_proc_t fn = NULL;
    if (!fn) {
        void *h = dlopen("/usr/lib/x86_64-linux-gnu/libEGL.so.1",
                         RTLD_LAZY | RTLD_GLOBAL);
        if (!h) h = dlopen("libEGL.so.1", RTLD_LAZY | RTLD_GLOBAL);
        if (h) fn = (egl_get_proc_t)dlsym(h, "eglGetProcAddress");
    }
    return fn;
}

/* Get a GL function pointer via eglGetProcAddress (with RTLD_DEFAULT fallback) */
static GL_FUNC_PTR egl_fn(const char *name) {
    egl_get_proc_t ep = get_egl_proc();
    GL_FUNC_PTR p = NULL;
    if (ep) p = (GL_FUNC_PTR)ep(name);
    if (!p) p = (GL_FUNC_PTR)dlsym(RTLD_DEFAULT, name);
    return p;
}

/* -----------------------------------------------------------------------
 * Redirect GLEW's GLX proc address queries to eglGetProcAddress.
 * ----------------------------------------------------------------------- */
GL_FUNC_PTR glXGetProcAddressARB(const unsigned char *procName) {
    if (!procName) return NULL;
    const char *name = (const char *)procName;
    if (strncmp(name, "glX", 3) == 0)
        return (GL_FUNC_PTR)dlsym(RTLD_DEFAULT, name);
    return egl_fn(name);
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
 * Intercept all direct GL calls from SOFA's OglModel and other components.
 * These bypass GLVND's dispatch stubs (which can fail with EGL contexts)
 * and go directly through eglGetProcAddress-obtained function pointers.
 *
 * GL types are defined inline to avoid pulling in GL headers at compile time.
 * ----------------------------------------------------------------------- */
typedef unsigned int   GLenum;
typedef unsigned int   GLbitfield;
typedef unsigned int   GLuint;
typedef int            GLint;
typedef int            GLsizei;
typedef unsigned char  GLboolean;
typedef signed char    GLbyte;
typedef short          GLshort;
typedef unsigned char  GLubyte;
typedef unsigned short GLushort;
typedef unsigned long  GLulong;
typedef float          GLfloat;
typedef float          GLclampf;
typedef double         GLdouble;
typedef double         GLclampd;
typedef void           GLvoid;

/* Macro: define an EGL-backed wrapper for a GL function.
 * PARAMS is a parenthesised list of "type name" pairs.
 * ARGS   is a parenthesised list of argument names.
 * RTYPE  is the return type (use void for void functions).
 */
#define GL_WRAP(RTYPE, NAME, PARAMS, ARGS) \
RTYPE NAME PARAMS { \
    typedef RTYPE (*fn_t) PARAMS; \
    static fn_t _fn; \
    if (!_fn) _fn = (fn_t)egl_fn(#NAME); \
    if (_fn) return _fn ARGS; \
    return (RTYPE)0; \
}

/* void variant that doesn't return a value */
#define GL_WRAP_V(NAME, PARAMS, ARGS) \
void NAME PARAMS { \
    typedef void (*fn_t) PARAMS; \
    static fn_t _fn; \
    if (!_fn) _fn = (fn_t)egl_fn(#NAME); \
    if (_fn) _fn ARGS; \
}

/* ----- glGetString / glGetStringi (return const GLubyte*) ----- */
const GLubyte *glGetString(GLenum name) {
    typedef const GLubyte *(*fn_t)(GLenum);
    static fn_t _fn;
    if (!_fn) _fn = (fn_t)egl_fn("glGetString");
    return _fn ? _fn(name) : NULL;
}

const GLubyte *glGetStringi(GLenum name, GLuint index) {
    typedef const GLubyte *(*fn_t)(GLenum, GLuint);
    static fn_t _fn;
    if (!_fn) _fn = (fn_t)egl_fn("glGetStringi");
    return _fn ? _fn(name, index) : NULL;
}

/* ----- GLboolean-returning ----- */
GL_WRAP(GLboolean, glIsEnabled, (GLenum cap), (cap))

/* ----- void functions (zero or more args) ----- */
GL_WRAP_V(glBegin,              (GLenum mode),                       (mode))
GL_WRAP_V(glEnd,                (void),                              ())
GL_WRAP_V(glClear,              (GLbitfield mask),                   (mask))
GL_WRAP_V(glClearColor,         (GLclampf r, GLclampf g, GLclampf b, GLclampf a), (r,g,b,a))
GL_WRAP_V(glEnable,             (GLenum cap),                        (cap))
GL_WRAP_V(glDisable,            (GLenum cap),                        (cap))
GL_WRAP_V(glEnableClientState,  (GLenum arr),                        (arr))
GL_WRAP_V(glDisableClientState, (GLenum arr),                        (arr))
GL_WRAP_V(glBlendFunc,          (GLenum sfactor, GLenum dfactor),    (sfactor, dfactor))
GL_WRAP_V(glCullFace,           (GLenum mode),                       (mode))
GL_WRAP_V(glDepthMask,          (GLboolean flag),                    (flag))
GL_WRAP_V(glHint,               (GLenum target, GLenum mode),        (target, mode))
GL_WRAP_V(glLineWidth,          (GLfloat width),                     (width))
GL_WRAP_V(glPointSize,          (GLfloat size),                      (size))
GL_WRAP_V(glPolygonMode,        (GLenum face, GLenum mode),          (face, mode))
GL_WRAP_V(glScissor,            (GLint x, GLint y, GLsizei w, GLsizei h), (x,y,w,h))
GL_WRAP_V(glViewport,           (GLint x, GLint y, GLsizei w, GLsizei h), (x,y,w,h))

/* Matrix operations */
GL_WRAP_V(glMatrixMode,         (GLenum mode),                       (mode))
GL_WRAP_V(glLoadIdentity,       (void),                              ())
GL_WRAP_V(glPushMatrix,         (void),                              ())
GL_WRAP_V(glPopMatrix,          (void),                              ())
GL_WRAP_V(glPushAttrib,         (GLbitfield mask),                   (mask))
GL_WRAP_V(glPopAttrib,          (void),                              ())
GL_WRAP_V(glLoadMatrixd,        (const GLdouble *m),                 (m))
GL_WRAP_V(glMultMatrixf,        (const GLfloat *m),                  (m))

/* Vertex / normal / color / texcoord */
GL_WRAP_V(glColor3f,            (GLfloat r, GLfloat g, GLfloat b),  (r,g,b))
GL_WRAP_V(glColor4f,            (GLfloat r, GLfloat g, GLfloat b, GLfloat a), (r,g,b,a))
GL_WRAP_V(glColor4fv,           (const GLfloat *v),                  (v))
GL_WRAP_V(glNormal3fv,          (const GLfloat *v),                  (v))
GL_WRAP_V(glVertex3f,           (GLfloat x, GLfloat y, GLfloat z),  (x,y,z))
GL_WRAP_V(glVertex3fv,          (const GLfloat *v),                  (v))
GL_WRAP_V(glVertex3d,           (GLdouble x, GLdouble y, GLdouble z),(x,y,z))
GL_WRAP_V(glVertex3dv,          (const GLdouble *v),                 (v))
GL_WRAP_V(glTexCoord2f,         (GLfloat s, GLfloat t),              (s,t))
GL_WRAP_V(glTexCoord3f,         (GLfloat s, GLfloat t, GLfloat r),  (s,t,r))

/* Vertex arrays / draw calls */
GL_WRAP_V(glVertexPointer,      (GLint sz, GLenum type, GLsizei stride, const GLvoid *ptr), (sz,type,stride,ptr))
GL_WRAP_V(glNormalPointer,      (GLenum type, GLsizei stride, const GLvoid *ptr), (type,stride,ptr))
GL_WRAP_V(glTexCoordPointer,    (GLint sz, GLenum type, GLsizei stride, const GLvoid *ptr), (sz,type,stride,ptr))
GL_WRAP_V(glDrawArrays,         (GLenum mode, GLint first, GLsizei count), (mode,first,count))
GL_WRAP_V(glDrawElements,       (GLenum mode, GLsizei count, GLenum type, const GLvoid *idx), (mode,count,type,idx))

/* Textures */
GL_WRAP_V(glGenTextures,        (GLsizei n, GLuint *textures),       (n,textures))
GL_WRAP_V(glBindTexture,        (GLenum target, GLuint texture),     (target,texture))
GL_WRAP_V(glTexParameteri,      (GLenum target, GLenum pname, GLint param), (target,pname,param))
GL_WRAP_V(glTexEnvf,            (GLenum target, GLenum pname, GLfloat param), (target,pname,param))
GL_WRAP_V(glTexEnvi,            (GLenum target, GLenum pname, GLint param), (target,pname,param))
GL_WRAP_V(glTexImage2D,         (GLenum target, GLint level, GLint ifmt,
                                  GLsizei w, GLsizei h, GLint border,
                                  GLenum fmt, GLenum type, const GLvoid *data),
                                 (target,level,ifmt,w,h,border,fmt,type,data))

/* Lighting (legacy fixed-function) */
GL_WRAP_V(glLightModeli,        (GLenum pname, GLint param),         (pname,param))
GL_WRAP_V(glMaterialf,          (GLenum face, GLenum pname, GLfloat param), (face,pname,param))
GL_WRAP_V(glMaterialfv,         (GLenum face, GLenum pname, const GLfloat *params), (face,pname,params))

/* Clip planes */
GL_WRAP_V(glClipPlane,          (GLenum plane, const GLdouble *eq),  (plane,eq))

/* Query */
GL_WRAP_V(glGetFloatv,          (GLenum pname, GLfloat *data),       (pname,data))

/* ----- GetClipPlane (returns void but has output param) ----- */
GL_WRAP_V(glGetClipPlane,       (GLenum plane, GLdouble *eq),        (plane,eq))


/* -----------------------------------------------------------------------
 * FrameBufferObject::getCurrentFramebufferID() → always return 0.
 * ----------------------------------------------------------------------- */
int _ZN4sofa2gl17FrameBufferObject23getCurrentFramebufferIDEv(void *self) {
    (void)self;
    return 0;
}

/* -----------------------------------------------------------------------
 * Light::computeShadowMapSize() → set 512×512.
 * ----------------------------------------------------------------------- */
void _ZN4sofa2gl9component6shader5Light20computeShadowMapSizeEv(void *self) {
    if (!self) return;
    volatile int *p = (int *)((char *)self + 0x80);
    *p = 512; *(p + 1) = 512;
}

/* -----------------------------------------------------------------------
 * Light/LightManager initVisual → no-op (prevents shadow FBO creation).
 * ----------------------------------------------------------------------- */
void _ZN4sofa2gl9component6shader5Light10initVisualEv(void *s) { (void)s; }
void _ZN4sofa2gl9component6shader16DirectionalLight10initVisualEv(void *s) { (void)s; }
void _ZN4sofa2gl9component6shader11PositionalLight10initVisualEv(void *s) { (void)s; }
void _ZN4sofa2gl9component6shader9SpotLight10initVisualEv(void *s) { (void)s; }
void _ZN4sofa2gl9component6shader12LightManager10initVisualEv(void *s) { (void)s; }

/* Shadow-pass methods → no-op. */
void _ZN4sofa2gl9component6shader12LightManager12preDrawSceneEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader12LightManager13postDrawSceneEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader12LightManager4drawEPKNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader12LightManager7fwdDrawEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader12LightManager7bwdDrawEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }

/* Shadow render passes for individual lights → no-op. */
void _ZN4sofa2gl9component6shader5Light13preDrawShadowEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader5Light14postDrawShadowEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader16DirectionalLight13preDrawShadowEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader16DirectionalLight14postDrawShadowEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader11PositionalLight13preDrawShadowEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader11PositionalLight14postDrawShadowEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader9SpotLight13preDrawShadowEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader9SpotLight14postDrawShadowEPNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }

/* Light draw/drawSource → no-op. */
void _ZN4sofa2gl9component6shader5Light4drawEPKNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader5Light10drawSourceEPKNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader16DirectionalLight4drawEPKNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader16DirectionalLight10drawSourceEPKNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader11PositionalLight4drawEPKNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader11PositionalLight10drawSourceEPKNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader9SpotLight4drawEPKNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
void _ZN4sofa2gl9component6shader9SpotLight10drawSourceEPKNS_4core6visual12VisualParamsE(void *s, void *v) { (void)s; (void)v; }
