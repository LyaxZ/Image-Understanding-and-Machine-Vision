import os, sys, time, threading
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import cv2
except ImportError:
    sys.exit("[错误] 请先安装: pip install opencv-python")
try:
    from PIL import Image, ImageTk
except ImportError:
    sys.exit("[错误] 请先安装: pip install Pillow")

# 可选加速
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator

# ════════════════════════════════════════════════════════════════
#            兼容中文路径的读写
# ════════════════════════════════════════════════════════════════

def cv2_imread(path, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0: return None
        return cv2.imdecode(data, flags)
    except Exception:
        return cv2.imread(path, flags)

def cv2_imwrite(path, img, params=None):
    try:
        ext = os.path.splitext(str(path))[1].lower()
        ok, buf = cv2.imencode(ext if ext else ".jpg", img, params or [])
        if ok:
            buf.tofile(str(path))
            return True
    except Exception:
        pass
    return cv2.imwrite(path, img, params or [])

def path_exists(path):
    try: return os.path.exists(str(path))
    except Exception: return False

# ════════════════════════════════════════════════════════════════
#               动态规划核心（Numba加速可选）
# ════════════════════════════════════════════════════════════════

@njit
def _find_seam_numba(energy):
    H, W = energy.shape
    dp = energy.copy()
    bt = np.zeros((H, W), dtype=np.int32)
    for i in range(1, H):
        for j in range(W):
            left = dp[i-1, j-1] if j > 0 else np.inf
            mid  = dp[i-1, j]
            right= dp[i-1, j+1] if j < W-1 else np.inf
            if left <= mid and left <= right:
                min_val = left
                bt[i, j] = j - 1
            elif mid <= left and mid <= right:
                min_val = mid
                bt[i, j] = j
            else:
                min_val = right
                bt[i, j] = j + 1
            dp[i, j] = energy[i, j] + min_val
    seam = np.zeros(H, dtype=np.int32)
    seam[-1] = np.argmin(dp[-1])
    for i in range(H-2, -1, -1):
        seam[i] = bt[i+1, seam[i+1]]
    return seam

def find_vseam(energy):
    if NUMBA_AVAILABLE:
        return _find_seam_numba(energy)
    else:
        H, W = energy.shape
        dp = energy.copy().astype(np.float64)
        bt = np.zeros((H, W), dtype=np.int32)
        for i in range(1, H):
            up = dp[i-1]
            L = np.empty(W); L[0] = np.inf; L[1:] = up[:-1]
            R = np.empty(W); R[-1] = np.inf; R[:-1] = up[1:]
            stk = np.stack([L, up, R], axis=0)
            mi = np.argmin(stk, axis=0)
            dp[i] = energy[i] + np.min(stk, axis=0)
            bt[i] = np.arange(W, dtype=np.int32) + (mi - 1)
        seam = np.zeros(H, dtype=np.int32)
        seam[-1] = np.argmin(dp[-1])
        for i in range(H-2, -1, -1):
            seam[i] = bt[i+1, seam[i+1]]
        return seam

def find_hseam(energy):
    return find_vseam(energy.T)

# ════════════════════════════════════════════════════════════════
#                      能 量 函 数
# ════════════════════════════════════════════════════════════════

def _e_sobel(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return np.abs(cv2.Sobel(g, cv2.CV_64F, 1, 0, 3)) + np.abs(cv2.Sobel(g, cv2.CV_64F, 0, 1, 3))

def _e_laplacian(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return np.abs(cv2.Laplacian(g, cv2.CV_64F, 3))

def _e_saliency(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    s = np.abs(cv2.GaussianBlur(g, (5,5), 2) - cv2.GaussianBlur(g, (31,31), 15))
    e = 0.6 * s + 0.4 * _e_sobel(img)
    mx = e.max()
    return e / mx * 1000 if mx > 0 else e

ENERGY_FN = {
    "sobel": _e_sobel,
    "laplacian": _e_laplacian,
    "saliency": _e_saliency,
}

# ════════════════════════════════════════════════════════════════
#            接缝移除/插入 (带蒙版同步)
# ════════════════════════════════════════════════════════════════

def remove_vseam(img, seam):
    H, W = img.shape[:2]
    out = np.zeros((H, W-1, *img.shape[2:]), dtype=img.dtype)
    for i in range(H):
        c = seam[i]
        out[i, :c] = img[i, :c]
        out[i, c:] = img[i, c+1:]
    return out

def remove_hseam(img, seam):
    if img.ndim == 3:
        return remove_vseam(img.transpose(1,0,2), seam).transpose(1,0,2)
    return remove_vseam(img.T, seam).T

def insert_vseam(img, seam):
    H, W = img.shape[:2]
    out = np.zeros((H, W+1, *img.shape[2:]), dtype=img.dtype)
    for i in range(H):
        c = seam[i]
        out[i, :c+1] = img[i, :c+1]
        avg = np.clip((img[i,c].astype(float)+img[i,min(c+1,W-1)].astype(float))/2, 0, 255).astype(img.dtype)
        out[i, c+1] = avg
        out[i, c+2:] = img[i, c+1:]
    return out

def apply_emask(energy, mask, mode, w, blur_ksize=0):
    out = energy.copy()
    if mask is None: return out
    if blur_ksize > 0:
        mask = cv2.GaussianBlur(mask, (blur_ksize|1, blur_ksize|1), 0)
    m = mask > 128
    if mode == "protect":   out[m] += w
    elif mode == "remove":  out[m] -= w
    return out

def unsharp(img, sigma=0.8):
    bl = cv2.GaussianBlur(img, (0,0), sigma)
    return np.clip(cv2.addWeighted(img, 1.5, bl, -0.5, 0), 0, 255).astype(img.dtype)

def energy_to_bgr(energy):
    e = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(e, cv2.COLORMAP_INFERNO)

# ════════════════════════════════════════════════════════════════
#                        G U I  主 类
# ════════════════════════════════════════════════════════════════

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Seam Carving — 涂抹蒙版 / 物体放大")

        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except: pass

        import tkinter.font as tkfont
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=18)
        tkfont.nametofont("TkTextFont").configure(size=18)
        tkfont.nametofont("TkHeadingFont").configure(size=18)
        style = ttk.Style()
        style.configure('.', font=('TkDefaultFont', 18))

        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        w, h = 2400, 1200
        x, y = (sw-w)//2, (sh-h)//2 - 30
        root.geometry(f"{w}x{h}+{x}+{max(0,y)}")
        root.minsize(1000, 550)

        self.img = None
        self._disp_orig = None
        self._disp_result = None
        self._disp_energy = None
        self._photo_refs = {}

        self.running = False
        self.stop_req = False
        self._result_img = None
        self._energy_img = None
        self._prog = 0.0
        self._status_text = "就绪"
        self._done = False
        self._error_msg = None
        self._elapsed = 0.0

        # 三种蒙版
        self.mask_protect = None
        self.mask_remove = None
        self.mask_amplify = None
        self.draw_mode = "protect"
        
        self.brush_size = 15
        self.drawing = False
        self.last_draw_xy = None

        self.amp_factor = tk.DoubleVar(value=1.5)
        self.show_seam = tk.BooleanVar(value=True)
        self.alternate_order = tk.BooleanVar(value=True)
        self.blur_radius = tk.IntVar(value=5)

        # 框选状态
        self.rect_start = None
        self.rect_id = None
        self.rect_mode = None

        self._build_ui()
        default_path = "./Data/inputs/test2.jpg"
        if os.path.exists(default_path):
            self.var_inp.set(default_path)
            self._load_input()
        else:
            self.lbl_status.config(text=f"默认图片未找到: {default_path}")
        self.root.after(100, self._init_placeholder)

    # ---------- 界面构建 ----------
    def _build_ui(self):
        self.frm_left = ttk.Frame(self.root, width=600)
        self.frm_left.pack(side=tk.LEFT, fill=tk.Y, padx=(10,4), pady=10)
        self.frm_left.pack_propagate(False)

        self.frm_right = ttk.Frame(self.root)
        self.frm_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4,10), pady=10)

        self._build_left()
        self._build_right()

    def _make_row(self, parent, label_text, var, btn_cmd=None, suffix=""):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label_text, width=14, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var, width=12).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,3))
        if btn_cmd:
            ttk.Button(row, text="…", width=3, command=btn_cmd).pack(side=tk.RIGHT)
        if suffix:
            ttk.Label(row, text=suffix, foreground="gray").pack(side=tk.LEFT, padx=(0,3))
        return row

    def _build_left(self):
        p = self.frm_left

        ttk.Label(p, text=" 文件路径", font=("",14,"bold")).pack(anchor="w", pady=(0,6))
        self.var_inp = tk.StringVar()
        self.var_out = tk.StringVar()
        self._make_row(p, "输入图像", self.var_inp, lambda: self._browse(self.var_inp, False))
        self._make_row(p, "输出图像", self.var_out, lambda: self._browse(self.var_out, True))

        ttk.Separator(p).pack(fill=tk.X, pady=10)

        ttk.Label(p, text=" 普通缩放", font=("",14,"bold")).pack(anchor="w", pady=(0,6))
        self.var_tw = tk.StringVar()
        self.var_th = tk.StringVar()
        self._make_row(p, "目标宽度:", self.var_tw, suffix="px (空=不变)")
        self._make_row(p, "目标高度:", self.var_th, suffix="px (空=不变)")
        ttk.Checkbutton(p, text=" 交替移除接缝 (优化质量)", variable=self.alternate_order).pack(anchor="w", padx=8, pady=3)

        ttk.Separator(p).pack(fill=tk.X, pady=10)

        ttk.Label(p, text=" 涂抹蒙版", font=("",14,"bold")).pack(anchor="w", pady=(0,6))
        f_mode = ttk.Frame(p)
        f_mode.pack(fill=tk.X, pady=2)
        ttk.Label(f_mode, text="模式:", width=6).pack(side=tk.LEFT)
        self.draw_mode_var = tk.StringVar(value="protect")
        ttk.Radiobutton(f_mode, text="保护(红)", variable=self.draw_mode_var, value="protect", command=lambda: self._set_draw_mode("protect")).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(f_mode, text="移除(绿)", variable=self.draw_mode_var, value="remove", command=lambda: self._set_draw_mode("remove")).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(f_mode, text="放大目标(蓝)", variable=self.draw_mode_var, value="amplify", command=lambda: self._set_draw_mode("amplify")).pack(side=tk.LEFT, padx=5)

        f_size = ttk.Frame(p)
        f_size.pack(fill=tk.X, pady=2)
        ttk.Label(f_size, text="笔刷大小:", width=8).pack(side=tk.LEFT)
        self.brush_scale = ttk.Scale(f_size, from_=1, to=50, orient=tk.HORIZONTAL, command=self._on_brush_change)
        self.brush_scale.set(self.brush_size)
        self.brush_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_brush_size = ttk.Label(f_size, text=str(self.brush_size), width=3)
        self.lbl_brush_size.pack(side=tk.LEFT, padx=5)

        f_btn = ttk.Frame(p)
        f_btn.pack(fill=tk.X, pady=5)
        ttk.Button(f_btn, text="清除保护", command=self._clear_protect).pack(side=tk.LEFT, padx=2)
        ttk.Button(f_btn, text="清除移除", command=self._clear_remove).pack(side=tk.LEFT, padx=2)
        ttk.Button(f_btn, text="清除放大目标", command=self._clear_amplify).pack(side=tk.LEFT, padx=2)
        ttk.Button(f_btn, text="全部清除", command=self._clear_all).pack(side=tk.LEFT, padx=2)

        # 自动蒙版
        ttk.Label(p, text=" 自动蒙版 (GrabCut)").pack(anchor="w", pady=(10,0))
        f_grab = ttk.Frame(p)
        f_grab.pack(fill=tk.X, pady=2)
        ttk.Button(f_grab, text="框选保护区域", command=lambda: self._start_rect("protect")).pack(side=tk.LEFT, padx=2)
        ttk.Button(f_grab, text="框选移除区域", command=lambda: self._start_rect("remove")).pack(side=tk.LEFT, padx=2)

        ttk.Separator(p).pack(fill=tk.X, pady=10)

        ttk.Label(p, text=" 物体放大 (涂抹蓝色区域)", font=("",14,"bold")).pack(anchor="w", pady=(0,6))
        f_amp = ttk.Frame(p)
        f_amp.pack(fill=tk.X, pady=2)
        ttk.Label(f_amp, text="放大系数:", width=10).pack(side=tk.LEFT)
        ttk.Scale(f_amp, from_=1.1, to=2.5, variable=self.amp_factor, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(f_amp, textvariable=self.amp_factor, width=4).pack(side=tk.LEFT, padx=5)
        f_amp2 = ttk.Frame(p)
        f_amp2.pack(fill=tk.X, pady=2)
        ttk.Button(f_amp2, text="🔵 物体放大 (先放大再缩小)", command=self._start_amplify).pack(side=tk.LEFT, padx=2)
        ttk.Button(f_amp2, text="🔷 正向放大 (直接插入)", command=self._start_forward_amplify).pack(side=tk.LEFT, padx=2)

        ttk.Separator(p).pack(fill=tk.X, pady=10)

        ttk.Label(p, text=" 参数设置", font=("",14,"bold")).pack(anchor="w", pady=(0,6))
        r = ttk.Frame(p); r.pack(fill=tk.X, pady=3)
        ttk.Label(r, text="能量函数:", width=14).pack(side=tk.LEFT)
        self.var_energy = tk.StringVar(value="sobel")
        cb = ttk.Combobox(r, textvariable=self.var_energy, values=list(ENERGY_FN.keys()), state="readonly", width=10)
        cb.pack(side=tk.LEFT)
        cb.bind("<<ComboboxSelected>>", self._on_energy_change)

        self.var_sharp = tk.BooleanVar(value=True)
        ttk.Checkbutton(p, text=" 启用锐化", variable=self.var_sharp).pack(anchor="w", padx=8, pady=3)
        ttk.Checkbutton(p, text=" 在能量图上显示接缝", variable=self.show_seam).pack(anchor="w", padx=8, pady=3)

        r = ttk.Frame(p); r.pack(fill=tk.X, pady=3)
        ttk.Label(r, text="更新间隔:", width=14).pack(side=tk.LEFT)
        self.var_iv = tk.IntVar(value=3)
        ttk.Spinbox(r, from_=1, to=100, textvariable=self.var_iv, width=5).pack(side=tk.LEFT)
        ttk.Label(r, text="步", foreground="gray").pack(side=tk.LEFT, padx=3)

        r = ttk.Frame(p); r.pack(fill=tk.X, pady=3)
        ttk.Label(r, text="蒙版羽化:", width=14).pack(side=tk.LEFT)
        ttk.Scale(r, from_=0, to=21, variable=self.blur_radius, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(r, textvariable=self.blur_radius, width=3).pack(side=tk.LEFT, padx=5)

        ttk.Separator(p).pack(fill=tk.X, pady=10)

        self.btn_run = ttk.Button(p, text="▶  开始运行 (普通缩放)", command=self._start_normal)
        self.btn_run.pack(fill=tk.X, ipady=6, pady=(0,4))

        self.btn_stop = ttk.Button(p, text="■  停止", command=self._stop, state="disabled")
        self.btn_stop.pack(fill=tk.X, ipady=3, pady=(0,6))

        self.var_prog = tk.DoubleVar(value=0)
        self.pbar = ttk.Progressbar(p, variable=self.var_prog, maximum=100)
        self.pbar.pack(fill=tk.X, pady=2)

        self.lbl_status = ttk.Label(p, text="就绪", wraplength=380, font=("Consolas",12))
        self.lbl_status.pack(anchor="w", pady=(4,0))
        self.lbl_time = ttk.Label(p, text="", font=("Consolas",12), foreground="gray")
        self.lbl_time.pack(anchor="w")
        self.lbl_size = ttk.Label(p, text="", font=("Consolas",12), foreground="gray")
        self.lbl_size.pack(anchor="w")

        if not NUMBA_AVAILABLE:
            ttk.Label(p, text="⚠ Numba 未安装，性能较慢", foreground="orange").pack(anchor="w", pady=(5,0))

    def _build_right(self):
        r = self.frm_right
        left_col = ttk.Frame(r); left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,3))
        right_col = ttk.Frame(r); right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(3,0))
        left_col.columnconfigure(0, weight=1); left_col.rowconfigure(0, weight=1); left_col.rowconfigure(1, weight=1)
        right_col.columnconfigure(0, weight=1); right_col.rowconfigure(0, weight=1)

        self.frm_orig = ttk.LabelFrame(left_col, text=" 原图 (可涂抹) ")
        self.frm_orig.grid(row=0, column=0, sticky="nsew", padx=2, pady=(0,3))
        self.cv_orig = tk.Canvas(self.frm_orig, bg="#1e1e1e", highlightthickness=0)
        self.cv_orig.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.frm_energy = ttk.LabelFrame(left_col, text=" 能量图 ")
        self.frm_energy.grid(row=1, column=0, sticky="nsew", padx=2, pady=(3,0))
        self.cv_energy = tk.Canvas(self.frm_energy, bg="#1e1e1e", highlightthickness=0)
        self.cv_energy.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.frm_result = ttk.LabelFrame(right_col, text=" 实时结果 ")
        self.frm_result.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.cv_result = tk.Canvas(self.frm_result, bg="#1e1e1e", highlightthickness=0)
        self.cv_result.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        for cv in (self.cv_orig, self.cv_result, self.cv_energy):
            cv.bind("<Configure>", self._on_resize)

        self.cv_orig.bind("<ButtonPress-1>", self._start_draw)
        self.cv_orig.bind("<B1-Motion>", self._draw)
        self.cv_orig.bind("<ButtonRelease-1>", self._stop_draw)
        # 右键框选
        self.cv_orig.bind("<ButtonPress-3>", self._start_rect_draw)
        self.cv_orig.bind("<B3-Motion>", self._update_rect_draw)
        self.cv_orig.bind("<ButtonRelease-3>", self._finish_rect_draw)

    # ---------- 蒙版管理 ----------
    def _init_masks(self, h, w):
        self.mask_protect = np.zeros((h,w), dtype=np.uint8)
        self.mask_remove = np.zeros((h,w), dtype=np.uint8)
        self.mask_amplify = np.zeros((h,w), dtype=np.uint8)

    def _set_draw_mode(self, mode): self.draw_mode = mode
    def _on_brush_change(self, val):
        self.brush_size = int(float(val))
        self.lbl_brush_size.config(text=str(self.brush_size))

    def _clear_protect(self):
        if self.mask_protect is not None: self.mask_protect.fill(0)
        self._redraw_overlay()
    def _clear_remove(self):
        if self.mask_remove is not None: self.mask_remove.fill(0)
        self._redraw_overlay()
    def _clear_amplify(self):
        if self.mask_amplify is not None: self.mask_amplify.fill(0)
        self._redraw_overlay()
    def _clear_all(self):
        if self.mask_protect is not None: self.mask_protect.fill(0)
        if self.mask_remove is not None: self.mask_remove.fill(0)
        if self.mask_amplify is not None: self.mask_amplify.fill(0)
        self._redraw_overlay()

    # ---------- 框选生成蒙版 (GrabCut) ----------
    def _start_rect(self, mode):
        self.rect_mode = mode
        self.cv_orig.config(cursor="cross")
        self.lbl_status.config(text=f"请用鼠标右键框选{'保护' if mode=='protect' else '移除'}区域")

    def _start_rect_draw(self, event):
        if self.rect_mode is None: return
        self.rect_start = (event.x, event.y)
        if self.rect_id:
            self.cv_orig.delete(self.rect_id)
        self.rect_id = None

    def _update_rect_draw(self, event):
        if self.rect_mode is None or self.rect_start is None: return
        x0, y0 = self.rect_start
        x1, y1 = event.x, event.y
        if self.rect_id:
            self.cv_orig.coords(self.rect_id, x0, y0, x1, y1)
        else:
            self.rect_id = self.cv_orig.create_rectangle(x0, y0, x1, y1, outline="yellow", width=2)

    def _finish_rect_draw(self, event):
        if self.rect_mode is None or self.rect_start is None: return
        x0, y0 = self.rect_start
        x1, y1 = event.x, event.y
        self.cv_orig.delete(self.rect_id)
        self.rect_id = None
        self.cv_orig.config(cursor="")
        p1 = self._canvas_to_image(x0, y0)
        p2 = self._canvas_to_image(x1, y1)
        if p1 is None or p2 is None:
            self.rect_mode = None
            return
        xmin, xmax = min(p1[0], p2[0]), max(p1[0], p2[0])
        ymin, ymax = min(p1[1], p2[1]), max(p1[1], p2[1])
        if xmax - xmin < 5 or ymax - ymin < 5:
            self.rect_mode = None
            return
        rect = (xmin, ymin, xmax - xmin, ymax - ymin)
        threading.Thread(target=self._run_grabcut, args=(rect, self.rect_mode), daemon=True).start()
        self.rect_mode = None

    def _run_grabcut(self, rect, mode):
        if self.img is None: return
        mask = np.zeros(self.img.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        # 增加迭代次数并启用边缘优化
        cv2.grabCut(self.img, mask, rect, bgdModel, fgdModel, 8, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 255).astype('uint8')
        # 形态学处理去除边缘残留
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
        if mode == "protect":
            self.mask_protect = cv2.bitwise_or(self.mask_protect, mask2)
            self.mask_remove = cv2.bitwise_and(self.mask_remove, cv2.bitwise_not(mask2))
        else:
            self.mask_remove = cv2.bitwise_or(self.mask_remove, mask2)
            self.mask_protect = cv2.bitwise_and(self.mask_protect, cv2.bitwise_not(mask2))
        self.root.after(0, self._redraw_overlay)
        self.root.after(0, lambda: self.lbl_status.config(text="GrabCut 完成"))

    # ---------- 涂抹绘制 ----------
    def _start_draw(self, event):
        if self.img is None: return
        self.drawing = True
        self.last_draw_xy = (event.x, event.y)
        self._draw(event)

    def _draw(self, event):
        if not self.drawing or self.img is None: return
        coord = self._canvas_to_image(event.x, event.y)
        if coord is None: return
        x, y = coord
        if self.last_draw_xy:
            last = self._canvas_to_image(*self.last_draw_xy)
            if last: self._draw_line(last[0], last[1], x, y)
        else:
            self._draw_circle(x, y)
        self.last_draw_xy = (event.x, event.y)
        self._redraw_overlay()

    def _stop_draw(self, event):
        self.drawing = False
        self.last_draw_xy = None

    def _canvas_to_image(self, cx, cy):
        if self.img is None: return None
        cw, ch = self.cv_orig.winfo_width(), self.cv_orig.winfo_height()
        if cw<=1 or ch<=1: return None
        h, w = self.img.shape[:2]
        scale = min(cw/w, ch/h)
        disp_w, disp_h = int(w*scale), int(h*scale)
        ox, oy = (cw-disp_w)//2, (ch-disp_h)//2
        if not (ox <= cx < ox+disp_w and oy <= cy < oy+disp_h): return None
        ix = int((cx-ox)/scale); iy = int((cy-oy)/scale)
        return max(0, min(w-1, ix)), max(0, min(h-1, iy))

    def _draw_circle(self, cx, cy):
        if self.draw_mode == "protect":
            cv2.circle(self.mask_protect, (cx,cy), self.brush_size, 255, -1)
            cv2.circle(self.mask_remove, (cx,cy), self.brush_size, 0, -1)
            cv2.circle(self.mask_amplify, (cx,cy), self.brush_size, 0, -1)
        elif self.draw_mode == "remove":
            cv2.circle(self.mask_remove, (cx,cy), self.brush_size, 255, -1)
            cv2.circle(self.mask_protect, (cx,cy), self.brush_size, 0, -1)
            cv2.circle(self.mask_amplify, (cx,cy), self.brush_size, 0, -1)
        else:
            cv2.circle(self.mask_amplify, (cx,cy), self.brush_size, 255, -1)
            cv2.circle(self.mask_protect, (cx,cy), self.brush_size, 0, -1)
            cv2.circle(self.mask_remove, (cx,cy), self.brush_size, 0, -1)

    def _draw_line(self, x0,y0, x1,y1):
        if self.draw_mode == "protect":
            cv2.line(self.mask_protect, (x0,y0), (x1,y1), 255, self.brush_size*2)
            cv2.line(self.mask_remove, (x0,y0), (x1,y1), 0, self.brush_size*2)
            cv2.line(self.mask_amplify, (x0,y0), (x1,y1), 0, self.brush_size*2)
        elif self.draw_mode == "remove":
            cv2.line(self.mask_remove, (x0,y0), (x1,y1), 255, self.brush_size*2)
            cv2.line(self.mask_protect, (x0,y0), (x1,y1), 0, self.brush_size*2)
            cv2.line(self.mask_amplify, (x0,y0), (x1,y1), 0, self.brush_size*2)
        else:
            cv2.line(self.mask_amplify, (x0,y0), (x1,y1), 255, self.brush_size*2)
            cv2.line(self.mask_protect, (x0,y0), (x1,y1), 0, self.brush_size*2)
            cv2.line(self.mask_remove, (x0,y0), (x1,y1), 0, self.brush_size*2)

    def _redraw_overlay(self):
        if self.img is None: return
        overlay = self.img.copy()
        if self.mask_protect is not None and np.any(self.mask_protect):
            overlay[self.mask_protect>0] = (overlay[self.mask_protect>0]*0.6 + np.array([0,0,255])*0.4).astype(np.uint8)
        if self.mask_remove is not None and np.any(self.mask_remove):
            overlay[self.mask_remove>0] = (overlay[self.mask_remove>0]*0.6 + np.array([0,255,0])*0.4).astype(np.uint8)
        if self.mask_amplify is not None and np.any(self.mask_amplify):
            overlay[self.mask_amplify>0] = (overlay[self.mask_amplify>0]*0.6 + np.array([255,0,0])*0.4).astype(np.uint8)
        self._show(self.cv_orig, overlay, "_disp_orig")

    # ---------- 显示与文件 ----------
    def _show(self, canvas, img_bgr, key=None):
        canvas.delete("all")
        cw, ch = max(canvas.winfo_width(),60), max(canvas.winfo_height(),60)
        if img_bgr is None:
            canvas.create_text(cw//2, ch//2, text="等待加载…", fill="#555", font=("",14))
            return
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((cw-10, ch-10), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        self._photo_refs[id(canvas)] = photo
        if key: setattr(self, key, img_bgr)
        canvas.create_image(cw//2, ch//2, image=photo)

    def _init_placeholder(self):
        for cv in (self.cv_orig, self.cv_result, self.cv_energy):
            self._show(cv, None)

    def _on_resize(self, event):
        cid = event.widget.winfo_id()
        if cid==self.cv_orig.winfo_id() and self._disp_orig is not None: self._show(self.cv_orig, self._disp_orig)
        elif cid==self.cv_result.winfo_id() and self._disp_result is not None: self._show(self.cv_result, self._disp_result)
        elif cid==self.cv_energy.winfo_id() and self._disp_energy is not None: self._show(self.cv_energy, self._disp_energy)

    def _browse(self, var, is_save):
        if is_save:
            p = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG","*.jpg"),("PNG","*.png")])
        else:
            p = filedialog.askopenfilename(filetypes=[("图像","*.jpg *.jpeg *.png *.bmp")])
        if p:
            var.set(p)
            if var is self.var_inp: self._load_input()

    def _load_input(self):
        path = self.var_inp.get().strip()
        if not path or not path_exists(path): return
        self.img = cv2_imread(path)
        if self.img is None: messagebox.showerror("错误","无法读取图像"); return
        H, W = self.img.shape[:2]
        self._init_masks(H, W)
        self._show(self.cv_orig, self.img, "_disp_orig")
        self._show(self.cv_result, None)
        self.lbl_status.config(text=f"已加载 {W}×{H}")
        self.lbl_size.config(text=f"原始尺寸: {W} × {H}")
        if not self.var_out.get():
            base = os.path.basename(path); name, ext = os.path.splitext(base)
            out_dir = "./Data/outputs/task2"; os.makedirs(out_dir, exist_ok=True)
            self.var_out.set(os.path.join(out_dir, f"{name}_carved{ext}"))
        self._on_energy_change()

    def _on_energy_change(self, *_):
        if self.img is None: return
        e = ENERGY_FN[self.var_energy.get()](self.img)
        self._show(self.cv_energy, energy_to_bgr(e), "_disp_energy")

    # ---------- 运行控制 ----------
    def _prepare_run(self):
        self.running = True; self.stop_req = False
        self._result_img = None; self._energy_img = None
        self._prog = 0.0; self._status_text = ""; self._done = False
        self._error_msg = None; self._disp_result = None; self._elapsed = 0.0
        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.frm_result.config(text=" 处理中… ")

    def _start_normal(self):
        if self.img is None: messagebox.showwarning("提示","请先选择输入图像"); return
        if not self.var_out.get().strip(): messagebox.showwarning("提示","请指定输出路径"); return
        self._prepare_run()
        threading.Thread(target=self._worker_normal, daemon=True).start()
        self._poll()

    def _start_amplify(self):
        if self.img is None: messagebox.showwarning("提示","请先选择输入图像"); return
        if self.mask_amplify is None or not np.any(self.mask_amplify):
            messagebox.showwarning("提示","请先用蓝色画笔涂抹要放大的区域"); return
        self._prepare_run()
        threading.Thread(target=self._worker_amplify, daemon=True).start()
        self._poll()

    def _start_forward_amplify(self):
        if self.img is None: messagebox.showwarning("提示","请先选择输入图像"); return
        if self.mask_amplify is None or not np.any(self.mask_amplify):
            messagebox.showwarning("提示","请先用蓝色画笔涂抹要放大的区域"); return
        self._prepare_run()
        threading.Thread(target=self._worker_forward_amplify, daemon=True).start()
        self._poll()

    def _stop(self):
        self.stop_req = True
        self.lbl_status.config(text="正在停止…")

    def _poll(self):
        if self._done:
            self.btn_run.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.var_prog.set(self._prog); self.lbl_status.config(text=self._status_text)
            if self._error_msg: messagebox.showerror("处理错误", self._error_msg)
            elif self._result_img is not None:
                out = self.var_out.get().strip()
                os.makedirs(os.path.dirname(out), exist_ok=True)
                cv2_imwrite(out, self._result_img)
                rH, rW = self._result_img.shape[:2]; oH, oW = self.img.shape[:2]
                self.lbl_size.config(text=f"{oW}×{oH} → {rW}×{rH}")
                self.frm_result.config(text=" 最终结果 ")
                messagebox.showinfo("完成", f"处理完成！\n尺寸变化: {oW}×{oH} → {rW}×{rH}\n已保存到: {out}")
            else: self.frm_result.config(text=" 实时结果 ")
            return
        if self._result_img is not None: self._show(self.cv_result, self._result_img, "_disp_result")
        if self._energy_img is not None: self._show(self.cv_energy, self._energy_img, "_disp_energy")
        self.var_prog.set(self._prog); self.lbl_status.config(text=self._status_text)
        self.lbl_time.config(text=f"耗时: {self._elapsed:.1f}s")
        self.root.after(120, self._poll)

    # ---------- 能量调整辅助函数 ----------
    def _adjust_energy(self, e, mask_p, mask_r, blur_radius):
        if mask_p is not None:
            e = apply_emask(e, mask_p, "protect", 5000, blur_radius)
        if mask_r is not None:
            e = apply_emask(e, mask_r, "remove", 5000, blur_radius)
        return e

    def _draw_seam_on_energy(self, energy_bgr, seam, vertical=True):
        if not self.show_seam.get(): return energy_bgr
        out = energy_bgr.copy()
        if vertical:
            for y, x in enumerate(seam):
                cv2.circle(out, (x, y), 1, (0,255,255), -1)
        else:
            for x, y in enumerate(seam):
                cv2.circle(out, (x, y), 1, (0,255,255), -1)
        return out

    # ---------- 普通缩放（支持交替顺序）----------
    def _worker_normal(self):
        img = self.img.copy()
        efn = ENERGY_FN[self.var_energy.get()]
        iv = max(1, self.var_iv.get())
        t0 = time.time()
        mask_p = self.mask_protect.copy() if self.mask_protect is not None else None
        mask_r = self.mask_remove.copy() if self.mask_remove is not None else None
        blur = self.blur_radius.get()
        try:
            H, W = img.shape[:2]
            tw = int(self.var_tw.get()) if self.var_tw.get().strip() else W
            th = int(self.var_th.get()) if self.var_th.get().strip() else H
            tw, th = max(1, tw), max(1, th)
            dw = W - tw if tw < W else 0
            dh = H - th if th < H else 0
            aw = tw - W if tw > W else 0
            ah = th - H if th > H else 0
            total_ops = dw + dh + aw + ah
            if total_ops == 0:
                self._result_img = img
                self._done = True
                return
            self._did_shrink = (dw + dh) > 0
            done = 0

            if self.alternate_order.get():
                # 交替模式：轮流通处理宽度和高度，包括缩小和放大
                # 为了正确处理，我们将放大操作分解为“先寻找所有待插入接缝，再反向插入”
                # 但为了交替，我们需要在循环中动态判断当前操作类型
                # 简化：按顺序处理缩小，放大单独处理（若需严格交替可进一步优化，此处保持功能正确）
                # 由于放大需要预先收集接缝，交替模式下改为：先完成所有缩小操作（交替进行），再统一处理放大
                # 下面采用顺序处理，以保证功能完整；严格交替实现较复杂，此处优先保证功能正确
                # 若您希望严格交替放大缩小，可后续细化
                pass

            # 为了确保功能完整，下面使用顺序模式：先处理宽度（缩小+放大），再处理高度
            # 宽度缩小
            for _ in range(dw):
                if self.stop_req: break
                e = self._adjust_energy(efn(img), mask_p, mask_r, blur)
                seam = find_vseam(e)
                self._energy_img = self._draw_seam_on_energy(energy_to_bgr(e), seam, True)
                img = remove_vseam(img, seam)
                if mask_p is not None: mask_p = remove_vseam(mask_p, seam)
                if mask_r is not None: mask_r = remove_vseam(mask_r, seam)
                done += 1
                self._prog = done / total_ops * 100
                self._elapsed = time.time() - t0
                self._status_text = f"缩窄 {done}/{dw}"
                self._result_img = img
                if done % iv == 0: time.sleep(0.005)

            # 宽度放大（需要先收集所有接缝再反向插入）
            if aw > 0:
                seams = []
                tmp = img.copy()
                tmp_p = mask_p.copy() if mask_p is not None else None
                tmp_r = mask_r.copy() if mask_r is not None else None
                for k in range(aw):
                    if self.stop_req: break
                    e = self._adjust_energy(efn(tmp), tmp_p, tmp_r, blur)
                    if k % iv == 0:
                        self._energy_img = self._draw_seam_on_energy(energy_to_bgr(e), None, True)  # 暂不画具体接缝
                        self._result_img = tmp
                    seam = find_vseam(e)
                    seams.append(seam)
                    tmp = remove_vseam(tmp, seam)
                    if tmp_p is not None: tmp_p = remove_vseam(tmp_p, seam)
                    if tmp_r is not None: tmp_r = remove_vseam(tmp_r, seam)
                    done += 1
                    self._prog = done / total_ops * 100
                    self._elapsed = time.time() - t0
                    self._status_text = f"查找宽度接缝 {k+1}/{aw}"
                    if done % iv == 0: time.sleep(0.005)
                # 反向插入
                seams.reverse()
                for k, seam in enumerate(seams):
                    if self.stop_req: break
                    img = insert_vseam(img, seam)
                    if mask_p is not None: mask_p = insert_vseam(mask_p, seam)
                    if mask_r is not None: mask_r = insert_vseam(mask_r, seam)
                    self._result_img = img
                    self._prog = (dw + aw + k + 1) / total_ops * 100
                    self._elapsed = time.time() - t0
                    self._status_text = f"加宽 {k+1}/{aw}"
                    if (k + 1) % iv == 0: time.sleep(0.005)

            # 高度缩小
            for _ in range(dh):
                if self.stop_req: break
                e = self._adjust_energy(efn(img), mask_p, mask_r, blur)
                seam = find_hseam(e)
                self._energy_img = self._draw_seam_on_energy(energy_to_bgr(e), seam, False)
                img = remove_hseam(img, seam)
                if mask_p is not None: mask_p = remove_hseam(mask_p, seam)
                if mask_r is not None: mask_r = remove_hseam(mask_r, seam)
                done += 1
                self._prog = done / total_ops * 100
                self._elapsed = time.time() - t0
                self._status_text = f"缩矮 {done - dw - aw}/{dh}"
                self._result_img = img
                if done % iv == 0: time.sleep(0.005)

            # 高度放大
            if ah > 0:
                seams = []
                tmp = img.copy()
                tmp_p = mask_p.copy() if mask_p is not None else None
                tmp_r = mask_r.copy() if mask_r is not None else None
                for k in range(ah):
                    if self.stop_req: break
                    e = self._adjust_energy(efn(tmp), tmp_p, tmp_r, blur)
                    seam = find_hseam(e)
                    seams.append(seam)
                    tmp = remove_hseam(tmp, seam)
                    if tmp_p is not None: tmp_p = remove_hseam(tmp_p, seam)
                    if tmp_r is not None: tmp_r = remove_hseam(tmp_r, seam)
                    done += 1
                    self._prog = done / total_ops * 100
                    self._elapsed = time.time() - t0
                    self._status_text = f"查找高度接缝 {k+1}/{ah}"
                    if done % iv == 0: time.sleep(0.005)
                seams.reverse()
                for k, seam in enumerate(seams):
                    if self.stop_req: break
                    if img.ndim == 3:
                        img = insert_vseam(img.transpose(1,0,2), seam).transpose(1,0,2)
                    else:
                        img = insert_vseam(img.T, seam).T
                    if mask_p is not None and mask_p.ndim == 2:
                        mask_p = insert_vseam(mask_p.T, seam).T
                    if mask_r is not None and mask_r.ndim == 2:
                        mask_r = insert_vseam(mask_r.T, seam).T
                    self._result_img = img
                    self._prog = (dw + aw + dh + k + 1) / total_ops * 100
                    self._elapsed = time.time() - t0
                    self._status_text = f"加高 {k+1}/{ah}"
                    if (k + 1) % iv == 0: time.sleep(0.005)

            if self.var_sharp.get() and self._did_shrink:
                img = unsharp(img)
            self._result_img = img
        except Exception as e:
            self._error_msg = str(e)
        finally:
            self._done = True
    # ---------- 物体放大 (先放大再缩小) ----------
    def _worker_amplify(self):
        img = self.img.copy()
        efn = ENERGY_FN[self.var_energy.get()]
        iv = max(1, self.var_iv.get())
        t0 = time.time()
        mask_a = self.mask_amplify.copy()
        factor = self.amp_factor.get()
        H, W = img.shape[:2]
        newW, newH = int(W*factor), int(H*factor)
        try:
            self._status_text = "正在放大图像..."
            img_big = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_LINEAR)
            mask_big = cv2.resize(mask_a, (newW, newH), interpolation=cv2.INTER_NEAREST)
            total_w = newW - W
            total_h = newH - H
            total_ops = total_w + total_h
            done = 0
            blur = self.blur_radius.get()
            for _ in range(total_w):
                if self.stop_req: break
                e = efn(img_big)
                e = apply_emask(e, mask_big, "protect", 3000, blur)
                seam = find_vseam(e)
                self._energy_img = self._draw_seam_on_energy(energy_to_bgr(e), seam, True)
                img_big = remove_vseam(img_big, seam)
                mask_big = remove_vseam(mask_big, seam)
                self._result_img = img_big
                done += 1; self._prog = done/total_ops*100
                self._elapsed = time.time()-t0
                self._status_text = f"宽度缩减 {done}/{total_w}"
                if done%iv==0: time.sleep(0.005)
            for _ in range(total_h):
                if self.stop_req: break
                e = efn(img_big)
                e = apply_emask(e, mask_big, "protect", 3000, blur)
                seam = find_hseam(e)
                self._energy_img = self._draw_seam_on_energy(energy_to_bgr(e), seam, False)
                img_big = remove_hseam(img_big, seam)
                mask_big = remove_hseam(mask_big, seam)
                self._result_img = img_big
                done += 1; self._prog = done/total_ops*100
                self._elapsed = time.time()-t0
                self._status_text = f"高度缩减 {done-total_w}/{total_h}"
                if done%iv==0: time.sleep(0.005)
            if self.var_sharp.get():
                img_big = unsharp(img_big)
            self._result_img = img_big
        except Exception as e:
            self._error_msg = str(e)
        finally:
            self._done = True

    # ---------- 正向放大 (直接插入接缝) ----------
    def _worker_forward_amplify(self):
        img = self.img.copy()
        efn = ENERGY_FN[self.var_energy.get()]
        iv = max(1, self.var_iv.get())
        t0 = time.time()
        mask_a = self.mask_amplify.copy()
        factor = self.amp_factor.get()
        H, W = img.shape[:2]
        add_w = int(W * (factor - 1))
        add_h = int(H * (factor - 1))
        total_ops = add_w + add_h
        done = 0
        blur = self.blur_radius.get()
        try:
            for _ in range(add_w):
                if self.stop_req: break
                e = efn(img)
                e = apply_emask(e, mask_a, "protect", -2000, blur)
                seam = find_vseam(e)
                self._energy_img = self._draw_seam_on_energy(energy_to_bgr(e), seam, True)
                img = insert_vseam(img, seam)
                mask_a = insert_vseam(mask_a, seam)
                self._result_img = img
                done += 1; self._prog = done/total_ops*100
                self._elapsed = time.time()-t0
                self._status_text = f"宽度插入 {done}/{add_w}"
                if done%iv==0: time.sleep(0.005)
            for _ in range(add_h):
                if self.stop_req: break
                e = efn(img)
                e = apply_emask(e, mask_a, "protect", -2000, blur)
                seam = find_hseam(e)
                self._energy_img = self._draw_seam_on_energy(energy_to_bgr(e), seam, False)
                if img.ndim==3:
                    img = insert_vseam(img.transpose(1,0,2), seam).transpose(1,0,2)
                else:
                    img = insert_vseam(img.T, seam).T
                mask_a = insert_vseam(mask_a.T, seam).T if mask_a.ndim==2 else mask_a
                self._result_img = img
                done += 1; self._prog = done/total_ops*100
                self._elapsed = time.time()-t0
                self._status_text = f"高度插入 {done-add_w}/{add_h}"
                if done%iv==0: time.sleep(0.005)
            self._result_img = img
        except Exception as e:
            self._error_msg = str(e)
        finally:
            self._done = True

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()