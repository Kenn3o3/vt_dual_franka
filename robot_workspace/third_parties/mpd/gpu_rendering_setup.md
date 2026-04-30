# SOFA 任务 GPU 无头渲染配置文档

> 适用平台：NVIDIA H100（驱动 ≥ 550.x）+ Ubuntu 20.04/22.04 + SOFA v23.12.01  
> 解决的问题：SOFA 任务在多进程训练时无法使用 GPU 渲染（原始代码强依赖 Mesa llvmpipe 软件渲染）

---

## 1. 背景与根因分析

### 原始方案的问题

原始代码通过 `fix_sofa_headless.so`（Mesa llvmpipe）实现无头渲染，存在以下问题：

1. **CPU 软件渲染**：llvmpipe 全程用 CPU 模拟 OpenGL，渲染速度慢，无法利用 H100 GPU。
2. **多进程 fork 崩溃**：NVIDIA EGL 驱动在 `fork()` 后的子进程中状态不一致，导致 `SIGSEGV`。
3. **GL 调用路径错误**：SOFA 的 `OglModel::internalDraw()` 通过 PLT（过程链接表）直接调用 `glColor3f`、`glLightModeli`、`glBegin` 等 legacy GL 函数。这些 PLT 条目解析到 `libOpenGL.so.0`（GLVND dispatch stubs），在 NVIDIA EGL 上下文中 GLVND 的 dispatch table 未被正确连接，导致 `libnvidia-eglcore.so` 内部因 per-context 状态指针为 NULL 而 `SIGSEGV`。

### 关键发现

```
SOFA PLT → libOpenGL.so.0 (GLVND stub) → [GLVND dispatch table 未连接 NVIDIA EGL] → SIGSEGV

vs.

eglGetProcAddress("glColor3f") → 直接返回 NVIDIA EGL 内部的函数指针 → 正常工作
```

`eglGetProcAddress` 和 `libOpenGL.so.0` 的 GLVND stub 对同一函数返回**不同的指针**：前者可用，后者在 EGL 上下文中崩溃。

---

## 2. 修改文件总览

| 文件 | 位置 | 说明 |
|------|------|------|
| `fix_sofa_egl.c` | `mpd/` 根目录 | 新增。LD_PRELOAD 库，拦截 SOFA 所有直接 GL 调用并重定向至 `eglGetProcAddress` |
| `fix_sofa_egl.so` | `mpd/` 根目录 | `fix_sofa_egl.c` 的编译产物，**需在目标机器上编译**（见第 3 节） |
| `train_sofa_tasks.sh` | `mpd/` 根目录 | 修改渲染环境变量：移除 llvmpipe 相关，改用 NVIDIA EGL |
| `dependencies/sofa_env/sofa_env/base.py` | git submodule 内 | 修改 Pyglet EGL 初始化路径和 `_EGLGLWrapper`（见第 4 节） |

> **所有修改均位于工程路径内**（`/home/hasac_cover/gjn/mpd`），无需修改系统文件或 conda 包。  
> 唯一的外部要求是系统级 NVIDIA EGL 库（`/usr/lib/x86_64-linux-gnu/libEGL.so.1`），这是系统包，不随代码迁移。

---

## 3. fix_sofa_egl.so 编译

### 依赖的系统包

```bash
# Ubuntu 20.04 / 22.04
sudo apt-get install -y gcc libgl-dev libegl-dev
```

| 系统包 | 提供的内容 | 用途 |
|--------|-----------|------|
| `gcc` | C 编译器 | 编译 `.c` → `.so` |
| `libgl-dev` | `/usr/include/GL/gl.h` 等头文件 | GL 类型定义 |
| `libegl-dev` | `/usr/include/EGL/egl.h` 等头文件 | EGL 函数声明 |

> **不需要** `-lEGL` 链接标志——`fix_sofa_egl.c` 使用 `dlopen()` 延迟加载 `libEGL.so.1`，  
> 避免父进程初始化 NVIDIA EGL 导致 `fork()` 后子进程状态损坏。

### 编译命令

```bash
cd /home/hasac_cover/gjn/mpd   # 或任意工程路径
gcc -shared -fPIC -O2 -o fix_sofa_egl.so fix_sofa_egl.c -ldl
```

编译后验证：

```bash
nm -D fix_sofa_egl.so | grep -E "glColor3f|eglGetProcAddress|egl_fn"
# 应看到 T glColor3f, T egl_fn, U dlopen 等符号
```

### 迁移到新机器的步骤

```bash
# 1. 复制源文件（.so 无需复制，在目标机器重新编译）
scp mpd/fix_sofa_egl.c  new-server:/path/to/mpd/

# 2. 在目标机器编译
ssh new-server
cd /path/to/mpd
sudo apt-get install -y gcc libgl-dev libegl-dev   # 如果缺少
gcc -shared -fPIC -O2 -o fix_sofa_egl.so fix_sofa_egl.c -ldl
```

---

## 4. base.py 的关键修改

文件位置：`dependencies/sofa_env/sofa_env/base.py`（在 git submodule `sofa_env` 内）

### 修改 1：`_EGLGLWrapper` 类（第 15 行起）

新增了一个 EGL-aware 的 OpenGL 包装器，替代原来直接使用 `OpenGL.GL`（PyOpenGL）的方式。

**核心思路**：通过 `eglGetProcAddress` 获取 GL 函数指针，而不是通过 `libGL.so.1`（GLVND dispatch），确保所有 Python 侧的 GL 调用都走 EGL 上下文的正确 dispatch 路径。

```python
# dependencies/sofa_env/sofa_env/base.py
class _EGLGLWrapper:
    def __init__(self):
        # 显式加载系统 GLVND libEGL.so.1（而非 conda 环境内的同名库）
        egl_path = "/usr/lib/x86_64-linux-gnu/libEGL.so.1"
        try:
            egl = ctypes.CDLL(egl_path)
        except OSError:
            egl = ctypes.CDLL("libEGL.so.1")   # fallback
        ...
```

### 修改 2：`_init_pyglet_window()` 的 libEGL patch（第 543 行起）

在 Pyglet 加载 EGL 库**之前**，将 `pyglet.lib.load_library` monkey-patch 为强制返回系统 `libEGL.so.1`：

```python
# 确保 Pyglet、_EGLGLWrapper、fix_sofa_egl.so 三者使用同一个 libEGL 实例
if self.pyglet.options.get("headless", False):
    _system_egl = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libEGL.so.1", mode=1)
    _orig_load = self.pyglet.lib.load_library
    def _patched_load(name, *args, **kwargs):
        if name == "EGL":
            return _system_egl
        return _orig_load(name, *args, **kwargs)
    self.pyglet.lib.load_library = _patched_load
```

**为什么需要这个 patch**：conda 环境中 Pyglet 可能加载 conda 内部的 `libEGL.so.1` 副本，与 `fix_sofa_egl.so` 使用的系统 `libEGL.so.1` 是两个不同的实例，导致 EGL dispatch table 不一致，出现渲染错误。

### 修改 3：OpenGL Context 请求方式（第 580 行起）

明确请求 **Compatibility Profile**（`forward_compatible=False`），而非 Core Profile：

```python
for major, minor in [(2, 1), (2, 0), (3, 1), (3, 0)]:
    req = self.pyglet.gl.Config(
        depth_size=24, double_buffer=True,
        major_version=major, minor_version=minor,
        forward_compatible=False,   # Compatibility Profile 保留 legacy GL
    )
```

**原因**：SOFA 的 `OglModel` 使用 `glBegin/glEnd`、矩阵栈等 legacy fixed-function API，Core Profile 中这些 API 被移除，会导致渲染全黑。

---

## 5. train_sofa_tasks.sh 的渲染相关配置

```bash
# 移除了原来的 llvmpipe 相关变量（LIBGL_ALWAYS_SOFTWARE 等）
# 新增以下配置：

export PYGLET_HEADLESS=1
export LD_PRELOAD="${SCRIPT_DIR}/fix_sofa_egl.so"

# PYGLET_HEADLESS_DEVICE 在每个 run_task() 调用时动态设置，
# 值等于 CUDA_VISIBLE_DEVICES 中该 worker 实际使用的物理 GPU 编号
# 例如：PYGLET_HEADLESS_DEVICE=7 表示使用 /dev/dri/renderD* 对应的第 7 块 GPU
```

### 多进程并行的关键设置

SOFA + NVIDIA EGL 在 `fork()` 后的子进程中会崩溃（NVIDIA 驱动不支持 `fork()` 后在子进程中初始化 EGL）。解决方案是使用 `spawn` 模式而非 `fork`：

```python
# 在 workspace 代码中（如 sofa_vector_workspace.py）
env = AsyncVectorEnv(
    env_fns,
    context="spawn",   # 必须是 "spawn"，不能是 "fork"
)
```

---

## 6. 系统级依赖说明（无法迁移进工程，需手动确认）

这些是目标机器必须满足的外部条件，**无法打包进代码仓库**：

| 依赖项 | 版本要求 | 验证命令 |
|--------|---------|---------|
| NVIDIA 驱动 | ≥ 525.x（建议 550.x） | `nvidia-smi \| grep "Driver Version"` |
| GLVND 系统 libEGL | 存在 `/usr/lib/x86_64-linux-gnu/libEGL.so.1` | `ls -la /usr/lib/x86_64-linux-gnu/libEGL.so*` |
| gcc | 任意现代版本 | `gcc --version` |
| libgl-dev | Ubuntu 包 | `dpkg -l libgl-dev` |
| libegl-dev | Ubuntu 包 | `dpkg -l libegl-dev` |
| CUDA | ≥ 11.x | `nvcc --version` 或 `nvidia-smi` |

> **关于 libEGL 硬编码路径**：`fix_sofa_egl.c` 和 `base.py` 中均硬编码了  
> `/usr/lib/x86_64-linux-gnu/libEGL.so.1`。如果目标机器的路径不同（如 ARM 服务器），  
> 需相应修改这两处路径，然后重新编译 `fix_sofa_egl.so`。

---

## 7. 完整迁移 Checklist

迁移到新机器时，按顺序执行：

```bash
# Step 1: 确认系统依赖
nvidia-smi                                         # 确认 NVIDIA 驱动已安装
ls /usr/lib/x86_64-linux-gnu/libEGL.so.1          # 确认系统 libEGL 存在
sudo apt-get install -y gcc libgl-dev libegl-dev  # 安装编译依赖

# Step 2: 编译 fix_sofa_egl.so（在工程目录内）
cd /path/to/mpd
gcc -shared -fPIC -O2 -o fix_sofa_egl.so fix_sofa_egl.c -ldl

# Step 3: 验证编译产物
nm -D fix_sofa_egl.so | grep -c "^[0-9a-f]* T gl"   # 应输出 ≥ 40（拦截的 GL 函数数量）

# Step 4: 冒烟测试（单进程）
cd /path/to/mpd
bash -c "
  PYGLET_HEADLESS=1 PYGLET_HEADLESS_DEVICE=0 \
  LD_PRELOAD=./fix_sofa_egl.so \
  python -c \"
import sys; sys.path.insert(0, 'dependencies/sofa_env')
# ... 运行最小 SOFA headless 渲染测试
print('smoke test placeholder')
\"
"

# Step 5: 多进程测试（spawn 模式）
bash train_sofa_tasks.sh -t rope_threading -g 0 epochs=1
# 检查 results/ 目录下是否有非全黑的 .gif 文件

# Step 6: 确认 GPU 使用
nvidia-smi   # 观察对应 GPU 的 MiB 使用量是否 > 0
```

---

## 8. 已知限制与注意事项

1. **`.so` 不能直接跨架构复用**：`fix_sofa_egl.so` 必须在目标机器上重新编译，不能从 x86 复制到 ARM。
2. **NVIDIA 驱动 < 525 可能不支持**：EGL 对 legacy OpenGL（Compatibility Profile）的支持在旧驱动中存在 bug，建议使用 550.x 版本。
3. **submodule 修改需单独提交**：`dependencies/sofa_env/sofa_env/base.py` 的修改在 `sofa_env` 这个 git submodule 内，需要在该 submodule 中单独 `git add` 和 `git commit`，主仓库只记录 submodule 的 commit hash。
4. **`fix_sofa_egl.so` 建议不要 commit 到 git**：编译产物通常不应放入版本控制，应在 `.gitignore` 中忽略 `*.so`，只提交源文件 `fix_sofa_egl.c`。

---

## 9. 参考：fix_sofa_egl.c 的拦截机制

`fix_sofa_egl.c` 通过 LD_PRELOAD 的符号覆盖机制，拦截约 50 个 SOFA 直接调用的 GL 函数：

```
SOFA OglModel (PLT) → [LD_PRELOAD 拦截] → eglGetProcAddress → NVIDIA EGL 内部指针
                           ↑
                    fix_sofa_egl.so
```

主要拦截类别：

| 类别 | 函数示例 | 拦截原因 |
|------|---------|---------|
| 立即模式绘制 | `glBegin`, `glEnd`, `glVertex3fv` | OglModel immediate-mode fallback |
| 颜色与材质 | `glColor3f`, `glMaterialf`, `glMaterialfv` | 每顶点颜色、光照材质属性 |
| 灯光 | `glLightModeli`, `glLightfv`, `glEnable(GL_LIGHTING)` | SOFA 光照模型 |
| 矩阵变换 | `glPushMatrix`, `glPopMatrix`, `glLoadIdentity` | 场景变换栈 |
| 顶点数组 | `glVertexPointer`, `glNormalPointer`, `glDrawElements` | VBO/VAO 回退路径 |
| GLEW 初始化 | `glXGetProcAddressARB` → `eglGetProcAddress` | GLEW 用 GLX API 查询函数，需重定向到 EGL |
| 版本查询 | `glGetString(GL_VERSION)` | 防止 SOFA 因版本字符串判断走错路径 |
| 光照/FBO | `Light::initVisual`, `LightManager::initVisual` | 避免 shadow map FBO 在 EGL 上崩溃 |
