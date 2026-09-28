from __future__ import annotations
import os
import traceback
import shutil
import tempfile
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from skimage.segmentation import find_boundaries
#
# Load custom function
import segmentation_algorithms as alg
from csv_workflow import CSVWorkflow, Case
#
class ImageSegmenterApp(tk.Tk):
    """Desktop Python Image Segmenter workflow."""
    #
    def __init__(self, workflow: CSVWorkflow):
        super().__init__()
        self.workflow = workflow
        self.title("Simple2DSeg - Python Image Segmenter")
        # All application resources are kept beside segmenter_app.py.
        self._base_dir = Path(__file__).resolve().parent
        # Tkinter title-bar/taskbar icon and dashboard branding assets.
        # These are UI-only additions; segmentation behavior is unchanged.
        # (a) Window/title-bar icon.
        self._set_window_icon()
        # (b) Load images used by the common dashboard header.
        self._load_brand_assets()
        # Do not force a fixed pixel size here.  A geometry such as 1420x900
        # is larger than the usable desktop on common 1366x768 displays and
        # can extend behind the Windows taskbar.  The window is maximized in
        # ``_fit_window_to_screen`` after the widgets have been created.
        self.minsize(900, 600)
        #
        self.case: Case | None = None
        self.image: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.preview_mask: np.ndarray | None = None
        self.preview_desc: str = ""
        self.history: list[tuple[str, np.ndarray]] = []
        self.history_index = -1
        #        
        # Rotation is for visualization/editing only.
        self._original_image: np.ndarray | None = None
        self.image_rotation_deg = 0
        self.mask_rotation_deg = 0
        #
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._tk_image = None
        self._display_geom = None  # x0,y0,w,h,scale
        self.mouse_mode = "none"
        self.drag_points: list[tuple[int, int]] = []
        self.drag_start: tuple[int, int] | None = None
        self.polygon_points: list[tuple[int, int]] = []
        self._pan_anchor = None
        #
        self.graph_fg = None
        self.graph_bg = None
        self.graph_labels = None
        self.grab_fg = None
        self.grab_bg = None
        self.grab_roi = None
        #
        self._build_ui()
        self.bind("1", lambda e: self.save_and_next())
        self.bind("0", lambda e: self.exit_without_saving())
        self.protocol("WM_DELETE_WINDOW", self.exit_without_saving)
        #
        # Let Tk finish calculating the requested widget sizes first, then
        # maximize to the operating-system work area.  On Windows this is
        # equivalent to pressing the maximize button and therefore respects
        # the taskbar and display scaling settings.
        self.after_idle(self._fit_window_to_screen)
        self.after(100, lambda: self.load_next_case(0))

    def _fit_window_to_screen(self):
        """Open the application maximized, with a portable fallback.

        ``state('zoomed')`` is the native Windows/Tk maximize operation and is
        preferable to setting geometry to ``winfo_screenwidth()`` x
        ``winfo_screenheight()`` because the latter includes the taskbar area.

        On Tk/window managers that do not support the ``zoomed`` state, try
        the X11 ``-zoomed`` attribute.  As a final fallback, open at roughly
        95% x 90% of the reported screen and center the window.
        """
        self.update_idletasks()

        # Windows and several Tk builds.
        try:
            self.state("zoomed")
            self.after(50, self.redraw)
            return
        except tk.TclError:
            pass

        # Common X11/Linux Tk builds.
        try:
            self.attributes("-zoomed", True)
            self.after(50, self.redraw)
            return
        except (tk.TclError, TypeError):
            pass

        # Portable fallback (also useful on window managers without zoomed).
        sw = max(1, int(self.winfo_screenwidth()))
        sh = max(1, int(self.winfo_screenheight()))
        width = max(900, int(sw * 0.95))
        height = max(600, int(sh * 0.90))
        width = min(width, sw)
        height = min(height, sh)
        x = max(0, (sw - width) // 2)
        y = max(0, (sh - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.after(50, self.redraw)

    # ------------------------------ UI ------------------------------
    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        self.tabs = ttk.Notebook(self)
        self.tabs.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.seg_tab = ttk.Frame(self.tabs)
        self.threshold_tab = ttk.Frame(self.tabs)
        self.flood_tab = ttk.Frame(self.tabs)
        self.morph_tab = ttk.Frame(self.tabs)
        self.active_tab = ttk.Frame(self.tabs)
        self.graph_tab = ttk.Frame(self.tabs)
        self.circles_tab = ttk.Frame(self.tabs)
        self.grab_tab = ttk.Frame(self.tabs)
        for tab, name in [
            (self.seg_tab, "SEGMENTATION"), (self.threshold_tab, "THRESHOLD"),
            (self.flood_tab, "FLOOD FILL"), (self.morph_tab, "MORPHOLOGY"),
            (self.active_tab, "ACTIVE CONTOURS"), (self.graph_tab, "GRAPH CUT"),
            (self.circles_tab, "FIND CIRCLES"), (self.grab_tab, "GRABCUT")]:
            self.tabs.add(tab, text=name)

        self._build_segmentation_tab()
        self._build_threshold_tab()
        self._build_flood_tab()
        self._build_morph_tab()
        self._build_active_tab()
        self._build_graph_tab()
        self._build_circles_tab()
        self._build_grab_tab()
        #
        # Using place() means this does not change the existing segmentation
        # control layouts.
        for tab in (self.seg_tab, self.threshold_tab, self.flood_tab, self.morph_tab, self.active_tab, self.graph_tab, self.circles_tab, self.grab_tab):
            self._add_brand_panel(tab)
        #
        left = ttk.Frame(self, width=250)
        left.grid(row=1, column=0, sticky="nsw")
        left.grid_propagate(False)
        ttk.Label(left, text="Data Browser", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=6, pady=(6, 2))
        self.case_label = ttk.Label(left, text="No case loaded", wraplength=235, justify="left")
        self.case_label.pack(fill="x", padx=6, pady=4)
        ttk.Separator(left).pack(fill="x", pady=4)
        ttk.Label(left, text="History", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=6)
        self.history_list = tk.Listbox(left, exportselection=False)
        self.history_list.pack(fill="both", expand=True, padx=6, pady=4)
        self.history_list.bind("<<ListboxSelect>>", self._history_selected)
        hb = ttk.Frame(left); hb.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(hb, text="Undo", command=self.undo).pack(side="left", expand=True, fill="x")
        ttk.Button(hb, text="Delete", command=self.delete_history_selection).pack(side="left", expand=True, fill="x", padx=(4, 0))
        #
        center = ttk.Frame(self)
        center.grid(row=1, column=1, sticky="nsew")
        center.rowconfigure(0, weight=1); center.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(center, bg="#252525", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._wheel_linux(1))
        self.canvas.bind("<Button-5>", lambda e: self._wheel_linux(-1))
        #
        status = ttk.Frame(center)
        status.grid(row=1, column=0, sticky="ew")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status, textvariable=self.status_var).pack(side="left", padx=6, pady=3)
        self.progress_var = tk.StringVar(value="")
        ttk.Label(status, textvariable=self.progress_var).pack(side="right", padx=6)
        
    def _set_window_icon(self):
        """
        Set the Simple2DSeg icon in the native Tkinter title bar.
        Windows may restore the default Tk icon when the window is
        first mapped/maximized, so this function is also called again
        shortly after startup.
        """
        ico_path = self._base_dir / "app_icon.ico"
        png_path = self._base_dir / "app_icon.png"
        if not hasattr(self, "_window_icon_tk"):
            self._window_icon_tk = None
        #
        # ----------------------------------------------------------
        # Windows
        # ----------------------------------------------------------
        if os.name == "nt":
            #
            icon_candidates = []
            # First use the supplied .ico file.
            if ico_path.is_file():
                icon_candidates.append(ico_path)
            # Also create a known-valid multi-resolution ICO from PNG.
            # This is useful if the supplied .ico is not accepted by Tk/Windows.
            if png_path.is_file():
                try:
                    runtime_ico = (Path(tempfile.gettempdir()) / "app_icon.ico")
                    #
                    with Image.open(png_path) as im:
                        im = im.convert("RGBA")
                        im.save(runtime_ico, format="ICO", sizes=[(16, 16), (20, 20), (24, 24),
                                                                  (32, 32), (40, 40), (48, 48),
                                                                  (64, 64), (128, 128), (256, 256)])
                    #
                    icon_candidates.append(runtime_ico)
                    # Keep path referenced for the lifetime of the app.
                    self._runtime_ico_path = runtime_ico
                except Exception:
                    pass
                #
            # Apply directly to the CURRENT Tk top-level window.
            for candidate in icon_candidates:
                try:
                    self.wm_iconbitmap(str(candidate))
                    self.iconbitmap(str(candidate))
                    return
                except tk.TclError:
                    continue
                #
            # Last-resort PNG fallback.
            if png_path.is_file():
                try:
                    self._window_icon_tk = tk.PhotoImage(file=str(png_path))
                    self.iconphoto(True, self._window_icon_tk)
                except tk.TclError:
                    self._window_icon_tk = None
                #
            return
        # ----------------------------------------------------------
        # Linux / other Tk platforms
        # ----------------------------------------------------------
        if png_path.is_file():
            try:
                self._window_icon_tk = tk.PhotoImage(file=str(png_path))
                self.iconphoto(True, self._window_icon_tk)
            except tk.TclError:
                self._window_icon_tk = None
            #
        #
    #
    def _load_brand_assets(self):
        """
        Load app_icon.png and Help_picture.png once.
        References are kept on self so Tkinter does not garbage-collect
        the PhotoImage objects.
        """
        self._brand_icon_tk = None
        self._help_picture_tk = None
        # Pillow compatibility.
        resample = getattr(Image, "Resampling", Image).LANCZOS
        # ----------------------------------------------------------
        # Simple2DSeg application icon
        # ----------------------------------------------------------
        try:
            icon_path = self._base_dir / "app_icon.png"
            icon = Image.open(icon_path).convert("RGBA")
            icon = icon.resize((58, 58), resample)
            self._brand_icon_tk = ImageTk.PhotoImage(icon)
        except Exception:
            self._brand_icon_tk = None
        #
        # ----------------------------------------------------------
        # Help image
        # ----------------------------------------------------------
        try:
            help_path = self._base_dir / "Help_picture.png"
            help_img = Image.open(help_path).convert("RGBA")
            help_img = help_img.resize((44, 58), resample)
            self._help_picture_tk = ImageTk.PhotoImage(help_img)
        except Exception:
            self._help_picture_tk = None
        #
    #
    def _add_brand_panel(self, parent):
        """
        (b) Add the common branding section shown on the right-hand
        side of all eight Tkinter dashboard tabs.
        The panel uses place() deliberately so that the existing
        segmentation controls are not repacked or otherwise changed.
        """
        # ----------------------------------------------------------
        # Text styles
        # ----------------------------------------------------------
        style = ttk.Style(self)
        style.configure("Simple2DSeg.BrandTitle.TLabel", font=("Comic Sans MS", 20, "bold"))
        style.configure("Simple2DSeg.BrandSub.TLabel", font=("Comic Sans MS", 13), foreground="#0072C6")
        style.configure("Simple2DSeg.BrandInfo.TLabel", font=("Comic Sans MS", 12), justify="left")
        style.configure("Simple2DSeg.Help.TLabel", font=("Comic Sans MS", 12), justify="center")
        # ----------------------------------------------------------
        # Entire right-side panel
        # ----------------------------------------------------------
        panel = ttk.Frame(parent)
        panel.place(relx=1.0, x=-14, y=8, anchor="ne", width=310, height=165)
        #
        # ==========================================================
        # TOP:
        #
        # [app icon]  Simple2DSeg App
        #             A 2D mask creation app
        # ==========================================================
        top = ttk.Frame(panel)
        top.pack(anchor="e")
        if self._brand_icon_tk is not None:
            ttk.Label(top, image=self._brand_icon_tk).pack(side="left", padx=(0, 10), pady=(0, 2))
        #
        title_box = ttk.Frame(top)
        title_box.pack(side="left", anchor="n")
        ttk.Label(title_box, text="Simple2DSeg App", style="Simple2DSeg.BrandTitle.TLabel").pack(anchor="w")
        ttk.Label(title_box, text="A 2D mask creation app", style="Simple2DSeg.BrandSub.TLabel").pack(anchor="w")
        # ==========================================================
        # BOTTOM:
        #
        # This code is created with          [Help_picture]
        # help of OpenAI                         Help
        # ==========================================================
        bottom = ttk.Frame(panel)
        bottom.pack(fill="x", pady=(28, 0))
        ttk.Label(bottom, text="This app is created with\n the help of OpenAI", style="Simple2DSeg.BrandInfo.TLabel").pack(side="left", padx=(12, 16), anchor="n")
        # ----------------------------------------------------------
        # Clickable Help area
        # ----------------------------------------------------------
        help_box = ttk.Frame(bottom, cursor="hand2")
        help_box.pack(side="right", padx=(0, 5), anchor="n")
        if self._help_picture_tk is not None:
            help_image = ttk.Label(help_box, image=self._help_picture_tk, cursor="hand2")
            help_image.pack()
            # (c) Clicking Help_picture downloads/saves the PDF.
            help_image.bind("<Button-1>", self._download_help_pdf)
        #
        help_text = ttk.Label(help_box, text="Help", cursor="hand2", style="Simple2DSeg.Help.TLabel")
        help_text.pack()
        # Make the Help text clickable as well.
        help_text.bind("<Button-1>", self._download_help_pdf)
        help_box.bind("<Button-1>", self._download_help_pdf)


    def _download_help_pdf(self, _event=None):
        """
        (c) Clicking Help_picture opens a Save-As dialog so the
        user can save/download help_segment_app.pdf locally.
        """
        src = self._base_dir / "help_segment_app.pdf"
        if not src.is_file():
            messagebox.showerror("Help file not found",
                                 f"Could not find:\n{src}")
            return
        #
        dst = filedialog.asksaveasfilename(parent=self, title="Save Simple2DSeg Help", initialfile="help_segment_app.pdf",
                                           defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        #
        # User pressed Cancel.
        if not dst:
            return
        #
        try:
            shutil.copy2(src, dst)
            messagebox.showinfo(f"Saved Help PDF saved to:\n{dst}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))
        #
    #
    def _group(self, parent, title):
        f = ttk.LabelFrame(parent, text=title)
        f.pack(side="left", fill="y", padx=3, pady=3)
        return f

    def _build_segmentation_tab(self):
        """Build the SEGMENTATION ribbon in three rows so no tools are clipped.

        The original implementation placed every group in one horizontal row.
        Tkinter's ``pack(side="left")`` does not wrap widgets when the window is
        narrower than their combined requested width, so the right-most controls
        were pushed off-screen on displays such as 1366x768.

        This layout keeps the same tools/callbacks while arranging them as:
            Row 1: LOAD | TEXTURE | CREATE MASK | ROTATE
            Row 2: ADD TO MASK | REFINE MASK
            Row 3: ZOOM / VIEW | EXPORT
        """

        # Three explicit ribbon rows.  Each row fills the available width and
        # remains independent, so groups never rely on automatic wrapping.
        row1 = ttk.Frame(self.seg_tab)
        row1.pack(fill="x", anchor="w", padx=2, pady=(2, 0))

        row2 = ttk.Frame(self.seg_tab)
        row2.pack(fill="x", anchor="w", padx=2, pady=0)

        row3 = ttk.Frame(self.seg_tab)
        row3.pack(fill="x", anchor="w", padx=2, pady=(0, 2))

        # --------------------------------------------------------------
        # Row 1: LOAD | TEXTURE | CREATE MASK
        # --------------------------------------------------------------
        load = self._group(row1, "LOAD")
        ttk.Button(load, text="Load Mask", command=self.load_mask_dialog).pack(side="left", padx=3, pady=8)
        ttk.Button(load, text="New Segmentation", command=self.new_segmentation).pack(side="left", padx=3, pady=8)

        texture = self._group(row1, "TEXTURE")
        self.use_texture = tk.BooleanVar(value=False)
        ttk.Checkbutton(texture, text="Include Texture\nFeatures", variable=self.use_texture).pack(side="left", padx=8, pady=4)

        create = self._group(row1, "CREATE MASK")
        ttk.Button(create, text="Threshold", command=lambda: self.tabs.select(self.threshold_tab)).pack(side="left", padx=2, pady=8)
        ttk.Button(create, text="Graph Cut", command=lambda: self.tabs.select(self.graph_tab)).pack(side="left", padx=2, pady=8)
        ttk.Button(create, text="K-means", command=self.run_kmeans).pack(side="left", padx=2, pady=8)
        ttk.Button(create, text="Find Circles", command=lambda: self.tabs.select(self.circles_tab)).pack(side="left", padx=2, pady=8)
        
        rotate = self._group(row1, "ROTATE")
        rotate_select = ttk.Frame(rotate)
        rotate_select.pack(side="left", padx=(5, 2), pady=3)
        self.rotate_image_var = tk.BooleanVar(value=False)
        self.rotate_mask_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(rotate_select, text="image", variable=self.rotate_image_var).pack(anchor="w")
        ttk.Checkbutton(rotate_select, text="mask", variable=self.rotate_mask_var).pack(anchor="w")
        rotate_buttons = ttk.Frame(rotate)
        rotate_buttons.pack(side="left", padx=(2, 5), pady=3)
        ttk.Button(rotate_buttons, text="90 deg", width=9, command=lambda: self.rotate_selected(90)).pack(fill="x", pady=(0, 2))
        ttk.Button(rotate_buttons, text="180 deg", width=9, command=lambda: self.rotate_selected(180)).pack(fill="x")

        # --------------------------------------------------------------
        # Row 2: ADD TO MASK | REFINE MASK
        # --------------------------------------------------------------
        add = self._group(row2, "ADD TO MASK")
        ttk.Button(add, text="GrabCut", command=lambda: self.tabs.select(self.grab_tab)).pack(side="left", padx=2, pady=8)
        ttk.Button(add, text="Flood Fill", command=lambda: self.tabs.select(self.flood_tab)).pack(side="left", padx=2, pady=8)
        ttk.Button(add, text="Freehand", command=lambda: self.set_mouse_mode("freehand")).pack(side="left", padx=2, pady=8)
        ttk.Button(add, text="Rectangle", command=lambda: self.set_mouse_mode("rectangle")).pack(side="left", padx=2, pady=8)
        ttk.Button(add, text="Ellipse", command=lambda: self.set_mouse_mode("ellipse")).pack(side="left", padx=2, pady=8)
        ttk.Button(add, text="Polygon", command=lambda: self.set_mouse_mode("polygon")).pack(side="left", padx=2, pady=8)

        refine = self._group(row2, "REFINE MASK")
        ttk.Button(refine, text="Morphology", command=lambda: self.tabs.select(self.morph_tab)).pack(side="left", padx=2, pady=8)
        ttk.Button(refine, text="Active Contours", command=lambda: self.tabs.select(self.active_tab)).pack(side="left", padx=2, pady=8)
        ttk.Button(refine, text="Clear Border", command=self.clear_border).pack(side="left", padx=2, pady=8)
        ttk.Button(refine, text="Fill Holes", command=self.fill_holes).pack(side="left", padx=2, pady=8)
        ttk.Button(refine, text="Invert Mask", command=self.invert_mask).pack(side="left", padx=2, pady=8)

        # --------------------------------------------------------------
        # Row 3: ZOOM / VIEW | EXPORT
        # --------------------------------------------------------------
        view = self._group(row3, "ZOOM / VIEW")
        ttk.Button(view, text="Zoom +", command=lambda: self.set_zoom(self.zoom * 1.25)).pack(side="left", padx=2, pady=8)
        ttk.Button(view, text="Zoom -", command=lambda: self.set_zoom(self.zoom / 1.25)).pack(side="left", padx=2, pady=8)
        ttk.Button(view, text="Pan", command=lambda: self.set_mouse_mode("pan")).pack(side="left", padx=2, pady=8)

        self.show_binary = tk.BooleanVar(value=False)
        ttk.Checkbutton(view, text="Show Binary", variable=self.show_binary, command=self._toggle_show_binary).pack(side="left", padx=(8, 4), pady=8)

        # Show the product of the original image and the currently selected
        # history mask.  This is equivalent to displaying only the pixels
        # inside the selected mask while setting the background to black.
        self.show_subtraction = tk.BooleanVar(value=False)
        ttk.Checkbutton(view, text="Show Subtraction", variable=self.show_subtraction, command=self._toggle_show_subtraction).pack(side="left", padx=(4, 4), pady=8)
        ttk.Label(view, text="Mask Opacity:").pack(side="left", padx=(10, 2), pady=8)
        self.opacity = tk.DoubleVar(value=0.45)
        ttk.Scale(view, from_=0.0, to=1.0, variable=self.opacity, orient="horizontal", length=140, 
                  command=lambda _=None: self.redraw()).pack(side="left", padx=(2, 8), pady=8)

        exp = self._group(row3, "EXPORT")
        ttk.Button(exp, text="Save & Next [1]", command=self.save_and_next).pack(side="left", padx=4, pady=8)
        ttk.Button(exp, text="Exit [0]", command=self.exit_without_saving).pack(side="left", padx=4, pady=8)
        
    # ---------------------------- rotate ----------------------------
    @staticmethod
    def _rotate_array_clockwise(arr, angle: int):
        """
        Rotate a NumPy array clockwise.
        Supports:
            90 degrees
            180 degrees
            270 degrees
        """
        if arr is None:
            return None
        #
        angle = int(angle) % 360
        if angle == 0:
            return np.asarray(arr).copy()
        #
        if angle not in (90, 180, 270):
            raise ValueError("Rotation angle must be 90, 180, or 270 degrees.")
        #
        # np.rot90 uses positive k for counter-clockwise.
        # Negative k therefore gives clockwise rotation.
        return np.rot90(arr, k=-(angle // 90)).copy()
    #
    @staticmethod
    def _shape_after_rotation(shape, angle: int):
        """
        Determine H, W after a rotation.
        """
        h = int(shape[0])
        w = int(shape[1])
        #
        if int(angle) % 180 == 90:
            return (w, h)
        #
        return (h, w)
    #
    def rotate_selected(self, angle: int):
        """
        Rotate selected image and/or mask clockwise.
        IMPORTANT:
            This only changes visualization/editing orientation.
            The original image on disk is never changed.
            When Save & Next is pressed, the current mask is converted
            back to the original source-image orientation before saving.
        """
        if self.image is None or self.mask is None:
            return
        #
        angle = int(angle) % 360
        if angle not in (90, 180):
            raise ValueError("ROTATE supports 90 or 180 degrees.")
        #
        rotate_image = bool(self.rotate_image_var.get())
        rotate_mask = bool(self.rotate_mask_var.get())
        # ----------------------------------------------------------
        # Nothing selected
        # ----------------------------------------------------------
        if not rotate_image and not rotate_mask:
            messagebox.showinfo("Rotate", "Select image and/or mask before choosing a rotation angle.")
            return
        #
        # ----------------------------------------------------------
        # Check resulting dimensions
        # ----------------------------------------------------------
        image_shape = self.image.shape[:2]
        mask_shape = self.mask.shape[:2]
        if rotate_image:
            next_image_shape = self._shape_after_rotation(image_shape, angle)
        else:
            next_image_shape = image_shape
        #
        if rotate_mask:
            next_mask_shape = self._shape_after_rotation(mask_shape, angle)
        else:
            next_mask_shape = mask_shape
        #
        # Existing segmentation tools expect image/mask to share
        # the same pixel coordinate system.
        if next_image_shape != next_mask_shape:
            messagebox.showwarning("Rotate", "This rotation would give the image and mask different dimensions.\n\n"
                                   "For a non-square image, select both image and mask for a 90-degree rotation.\n\n"
                                   "Independent 180-degree rotation is supported.")
            #
            return
        #
        # ----------------------------------------------------------
        # Clear unfinished drawing geometry
        # ----------------------------------------------------------
        self.drag_points = []
        self.drag_start = None
        self.polygon_points = []
        self._pan_anchor = None
        # ==========================================================
        # ROTATE IMAGE
        # ==========================================================
        if rotate_image:
            self.image = self._rotate_array_clockwise(self.image, angle)
            self.image_rotation_deg = (self.image_rotation_deg + angle) % 360
            # These arrays live in IMAGE coordinates,
            # therefore they follow image rotation.
            for attr in ("graph_fg", "graph_bg", "graph_labels", "grab_fg", "grab_bg", "grab_roi"):
                value = getattr(self, attr, None)
                if value is not None:
                    setattr(self, attr, self._rotate_array_clockwise(value, angle))
                #
            #
        #
        # ==========================================================
        # ROTATE MASK
        # ==========================================================
        if rotate_mask:
            # History contains complete cumulative mask snapshots.
            # Rotate every snapshot so History / Undo / Delete remain
            # synchronized with the current displayed mask orientation.
            if self.history:
                rotated_history = []
                for desc, hist_mask in self.history:
                    rotated_mask = self._rotate_array_clockwise(hist_mask, angle)
                    rotated_history.append((desc, rotated_mask))
                #
                self.history = rotated_history
                if (0 <= self.history_index < len(self.history)):
                    self.mask = (self.history[self.history_index][1].copy())
                else:
                    self.mask = (self._rotate_array_clockwise(self.mask, angle))
                #
            else:
                self.mask = (self._rotate_array_clockwise(self.mask, angle))
            #
            # Preview also belongs to mask coordinates.
            if self.preview_mask is not None:
                self.preview_mask = (self._rotate_array_clockwise(self.preview_mask, angle))
            #
            self.mask_rotation_deg = (self.mask_rotation_deg + angle) % 360
            self._refresh_history_list()
        #
        # ----------------------------------------------------------
        # Reset viewport after rotation
        # ----------------------------------------------------------
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        targets = []
        if rotate_image:
            targets.append("image")
        #
        if rotate_mask:
            targets.append("mask")
        #
        self.status(f'Rotated {" and ".join(targets)} clockwise by {angle} degrees.')
        self.redraw()
    #
    def _mask_to_original_orientation(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert the current working mask back to the
        original source-image orientation.
        """
        m = np.asarray(mask, bool)
        # Current mask was rotated CLOCKWISE.
        #
        # To restore it, rotate COUNTER-CLOCKWISE by
        # the accumulated amount.
        k = (self.mask_rotation_deg % 360) // 90
        if k:
            m = np.rot90(m, k=k).copy()
        else:
            m = m.copy()
        #
        return m
    #
    def _apply_close_buttons(self, parent, apply_cmd):
        b = ttk.Frame(parent); b.pack(side="left", fill="y", padx=8)
        ttk.Button(b, text="Preview", command=apply_cmd).pack(fill="x", pady=(7,2))
        ttk.Button(b, text="Apply", command=self.apply_preview).pack(fill="x", pady=2)
        ttk.Button(b, text="Close", command=self.close_preview_tab).pack(fill="x", pady=2)

    def _build_threshold_tab(self):
        f = self._group(self.threshold_tab, "Threshold")
        self.threshold_method = tk.StringVar(value="Global")
        ttk.Label(f, text="Method").grid(row=0,column=0,padx=3,pady=3)
        ttk.Combobox(f, textvariable=self.threshold_method, values=["Global","Manual","Adaptive"], width=10, state="readonly").grid(row=1,column=0,padx=3)
        self.threshold_value = tk.DoubleVar(value=0.5)
        ttk.Label(f, text="Threshold").grid(row=0,column=1,padx=3)
        ttk.Entry(f, textvariable=self.threshold_value, width=8).grid(row=1,column=1,padx=3)
        self.adapt_sensitivity = tk.DoubleVar(value=0.5)
        ttk.Label(f, text="Sensitivity").grid(row=0,column=2,padx=3)
        ttk.Entry(f, textvariable=self.adapt_sensitivity, width=8).grid(row=1,column=2,padx=3)
        self.foreground_polarity = tk.StringVar(value="bright")
        ttk.Label(f, text="Foreground Polarity").grid(row=0,column=3,padx=3)
        ttk.Combobox(f, textvariable=self.foreground_polarity, values=["bright","dark"], width=8, state="readonly").grid(row=1,column=3,padx=3)
        self._apply_close_buttons(self.threshold_tab, self.preview_threshold)

    def _build_flood_tab(self):
        f = self._group(self.flood_tab, "Flood Fill")
        self.flood_metric = tk.StringVar(value="Euclidean")
        ttk.Label(f, text="Distance Metric").grid(row=0,column=0,padx=4,pady=3)
        ttk.Combobox(f, textvariable=self.flood_metric, values=["Euclidean","Geodesic"], width=10, state="readonly").grid(row=1,column=0,padx=4)
        self.flood_tol = tk.DoubleVar(value=0.05)
        ttk.Label(f, text="Tolerance").grid(row=0,column=1,padx=4)
        ttk.Entry(f, textvariable=self.flood_tol, width=8).grid(row=1,column=1,padx=4)
        ttk.Button(f, text="Select Seed", command=lambda: self.set_mouse_mode("flood")).grid(row=1,column=2,padx=6)
        self._apply_close_buttons(self.flood_tab, lambda: self.status("Click a seed point in the image."))

    def _build_morph_tab(self):
        f = self._group(self.morph_tab, "Morphology")
        self.morph_op = tk.StringVar(value="dilate")
        self.morph_shape = tk.StringVar(value="disk")
        self.morph_radius = tk.IntVar(value=3); self.morph_length = tk.IntVar(value=3)
        self.morph_angle = tk.DoubleVar(value=0); self.morph_width = tk.IntVar(value=3)
        ttk.Label(f,text="Operation").grid(row=0,column=0); ttk.Combobox(f,textvariable=self.morph_op,values=["dilate","erode","open","close"],state="readonly",width=8).grid(row=1,column=0,padx=3)
        ttk.Label(f,text="Shape").grid(row=0,column=1); ttk.Combobox(f,textvariable=self.morph_shape,values=["disk","diamond","line","octagon","square","rectangle"],state="readonly",width=10).grid(row=1,column=1,padx=3)
        for j,(label,var) in enumerate([("Radius",self.morph_radius),("Length",self.morph_length),("Degrees",self.morph_angle),("Width",self.morph_width)],start=2):
            ttk.Label(f,text=label).grid(row=0,column=j); ttk.Entry(f,textvariable=var,width=7).grid(row=1,column=j,padx=3)
        self._apply_close_buttons(self.morph_tab, self.preview_morphology)

    def _build_active_tab(self):
        f = self._group(self.active_tab, "Active Contours")
        self.active_method = tk.StringVar(value="Chan-Vese")
        self.active_iterations = tk.IntVar(value=100)
        ttk.Label(f,text="Method").grid(row=0,column=0); ttk.Combobox(f,textvariable=self.active_method,values=["Chan-Vese","edge"],state="readonly",width=10).grid(row=1,column=0,padx=4)
        ttk.Label(f,text="Iterations").grid(row=0,column=1); ttk.Entry(f,textvariable=self.active_iterations,width=8).grid(row=1,column=1,padx=4)
        self._apply_close_buttons(self.active_tab, self.preview_active_contours)

    def _build_graph_tab(self):
        draw = self._group(self.graph_tab, "Draw")
        ttk.Button(draw,text="Foreground",command=lambda:self.set_mouse_mode("graph_fg")).pack(side="left",padx=3,pady=8)
        ttk.Button(draw,text="Background",command=lambda:self.set_mouse_mode("graph_bg")).pack(side="left",padx=3)
        ttk.Button(draw,text="Erase",command=lambda:self.set_mouse_mode("graph_erase")).pack(side="left",padx=3)
        self.marker_size = tk.IntVar(value=9)
        ttk.Label(draw,text="Marker Size").pack(side="left",padx=(8,2)); ttk.Spinbox(draw,from_=1,to=51,textvariable=self.marker_size,width=4).pack(side="left")
        clear = self._group(self.graph_tab,"Clear")
        ttk.Button(clear,text="Clear Foreground",command=lambda:self._clear_scribble("graph_fg")).pack(padx=4,pady=(4,1))
        ttk.Button(clear,text="Clear Background",command=lambda:self._clear_scribble("graph_bg")).pack(padx=4,pady=1)
        superp = self._group(self.graph_tab,"Superpixel Settings")
        self.superpixel_density = tk.IntVar(value=50)
        self.show_superpixels = tk.BooleanVar(value=False)
        ttk.Checkbutton(superp,text="Show Boundaries",variable=self.show_superpixels,command=self.redraw).pack(side="left",padx=4)
        ttk.Label(superp,text="Density").pack(side="left"); ttk.Scale(superp,from_=0,to=100,variable=self.superpixel_density,orient="horizontal").pack(side="left",padx=4)
        self._apply_close_buttons(self.graph_tab, self.preview_graphcut)

    def _build_circles_tab(self):
        f = self._group(self.circles_tab,"Find Circles")
        self.circle_min = tk.IntVar(value=50); self.circle_max=tk.IntVar(value=150)
        self.circle_polarity=tk.StringVar(value="bright"); self.circle_sens=tk.DoubleVar(value=0.85)
        for j,(label,var) in enumerate([("Min Diameter",self.circle_min),("Max Diameter",self.circle_max)]):
            ttk.Label(f,text=label).grid(row=0,column=j); ttk.Entry(f,textvariable=var,width=8).grid(row=1,column=j,padx=3)
        ttk.Label(f,text="Object Polarity").grid(row=0,column=2); ttk.Combobox(f,textvariable=self.circle_polarity,values=["bright","dark"],state="readonly",width=8).grid(row=1,column=2,padx=3)
        ttk.Label(f,text="Sensitivity").grid(row=0,column=3); ttk.Entry(f,textvariable=self.circle_sens,width=8).grid(row=1,column=3,padx=3)
        self._apply_close_buttons(self.circles_tab, self.preview_circles)

    def _build_grab_tab(self):
        roi = self._group(self.grab_tab,"ROI")
        self.grab_roi_style = tk.StringVar(value="Rectangle")
        ttk.Combobox(roi,textvariable=self.grab_roi_style,values=["Rectangle","Polygon"],state="readonly",width=10).pack(side="left",padx=3,pady=8)
        ttk.Button(roi,text="Draw ROI",command=self.start_grab_roi).pack(side="left",padx=3)
        draw = self._group(self.grab_tab,"Draw")
        ttk.Button(draw,text="Foreground",command=lambda:self.set_mouse_mode("grab_fg")).pack(side="left",padx=3,pady=8)
        ttk.Button(draw,text="Background",command=lambda:self.set_mouse_mode("grab_bg")).pack(side="left",padx=3)
        ttk.Button(draw,text="Erase",command=lambda:self.set_mouse_mode("grab_erase")).pack(side="left",padx=3)
        ttk.Button(draw,text="Clear",command=self.clear_grabcut_marks).pack(side="left",padx=3)
        self._apply_close_buttons(self.grab_tab, self.preview_grabcut)

    # -------------------------- workflow ----------------------------
    def load_next_case(self, start: int):
        case = self.workflow.next_unprocessed(start)
        if case is None:
            messagebox.showinfo("Complete", "All unflagged cases have been processed.")
            self.destroy()
            return
        #
        self.case = case
        try:
            image, existing, warning = (self.workflow.load_case(case))
        except Exception as e:
            messagebox.showerror("Load error",
                                 f"Could not load row {case.row_index + 1}:\n{e}")
            self.load_next_case(case.row_index + 1)
            return
        #
        # ==========================================================
        # ORIGINAL IMAGE
        # ==========================================================
        # This copy must NEVER be rotated.
        # It is used when writing the final:
        #     image * mask
        # output.
        self._original_image = image.copy()
        # Working/display image.
        self.image = image.copy()
        # Every new case starts in source orientation.
        self.image_rotation_deg = 0
        self.mask_rotation_deg = 0
        # Reset checkboxes.
        if hasattr(self, "rotate_image_var"):
            self.rotate_image_var.set(False)
        #
        if hasattr(self, "rotate_mask_var"):
            self.rotate_mask_var.set(False)
        #
        # ==========================================================
        # MASK
        # ==========================================================
        if existing is not None:
            self.mask = existing.copy()
        else:
            self.mask = np.zeros(image.shape[:2], bool)
        #
        self.preview_mask = None
        # ==========================================================
        # HISTORY
        # ==========================================================
        self.history = [(("Loaded existing mask" if existing is not None else "Load Image"), self.mask.copy())]
        self.history_index = 0
        self._reset_aux_masks()
        # Reset viewing controls.
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._refresh_history_list()
        self.case_label.config(text=(f"Row {case.row_index + 1}/"
                                     f"{len(self.workflow.df)}\n"
                                     f"Image: {case.image_path}\n"
                                     f"Mask: {case.mask_save_path}"))
        self.progress_var.set(f"Row {case.row_index + 1} of {len(self.workflow.df)}")
        if warning:
            messagebox.showwarning("Mask not loaded", warning)
        #
        self.status("Existing mask loaded." if existing is not None else "New empty segmentation.")
        self.redraw()
    #
    def save_and_next(self):
        if (self.image is None or self.mask is None or self.case is None):
            return
        # ----------------------------------------------------------
        # Apply active preview first
        # ----------------------------------------------------------
        if self.preview_mask is not None:
            if not messagebox.askyesno("Unapplied preview", "A preview is active. Apply it before saving?"):
                return
            #
            self.apply_preview()
        #
        try:
            # ======================================================
            # RESTORE ORIGINAL MASK ORIENTATION
            # ======================================================
            save_mask = (self._mask_to_original_orientation(self.mask))
            # ======================================================
            # USE ORIGINAL IMAGE
            # ======================================================
            # Never save the rotated visualization image.
            if self._original_image is not None:
                source_image = (self._original_image)
            else:
                # Safety fallback.
                source_image = self.image
            # ======================================================
            # DIMENSION CHECK
            # ======================================================
            if (save_mask.shape != source_image.shape[:2]):
                raise ValueError("The mask could not be restored to the original image dimensions before saving.")
            #
            # ======================================================
            # ORIGINAL-ORIENTATION IMAGE * MASK
            # ======================================================
            seg = alg.masked_image(source_image, save_mask)
            self.workflow.save_case(self.case, save_mask, seg)
        except Exception as e:
            messagebox.showerror("Save error", str(e))
            return
        #
        idx = self.case.row_index
        self.status(f"Saved mask and segmented image; Segment=1 for row {idx + 1}.")
        self.load_next_case(idx + 1)
    #
    def exit_without_saving(self):
        if self.case is not None:
            ok = messagebox.askyesno("Exit", "Exit without saving the current row? It will remain unflagged for resume.")
            if not ok: return
        self.destroy()

    # --------------------------- history ----------------------------
    def commit_mask(self, new_mask: np.ndarray, description: str):
        if self.mask is None: return
        m = np.asarray(new_mask, bool)
        if m.shape != self.mask.shape:
            raise ValueError("New mask shape does not match image.")
        self.history = self.history[:self.history_index+1]
        self.history.append((description, m.copy()))
        self.history_index += 1
        self.mask = m.copy(); self.preview_mask = None; self.preview_desc = ""
        self._refresh_history_list(); self.redraw(); self.status(description)

    def _refresh_history_list(self):
        self.history_list.delete(0, "end")
        for i,(desc,_) in enumerate(self.history):
            self.history_list.insert("end", f"{i+1}. {desc}")
        if self.history:
            self.history_list.selection_set(self.history_index); self.history_list.see(self.history_index)

    def _history_selected(self, _evt=None):
        sel = self.history_list.curselection()
        if not sel: return
        idx = sel[0]
        if 0 <= idx < len(self.history):
            self.history_index = idx; self.mask = self.history[idx][1].copy(); self.preview_mask=None; self.redraw()

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.mask = self.history[self.history_index][1].copy()
            self.preview_mask = None
            self.preview_desc = ""
            self._refresh_history_list()
            self.redraw()

    def delete_history_selection(self):
        """Delete the selected mask edit and revert to the state before it.

        Each history item stores a *complete cumulative mask snapshot*, not an
        independent drawing object.  Therefore, if an older history item is
        deleted while newer items remain, those newer snapshots can still
        contain the pixels introduced by the deleted edit.

        To guarantee that the selected mistake is removed completely, this
        function removes the selected history entry AND every later entry, then
        restores the mask snapshot immediately preceding the deleted item.

        The first history item (loaded/existing mask or initial empty image) is
        kept as the base state and cannot be deleted.
        """
        if not self.history:
            return

        selected = self.history_list.curselection()
        if selected:
            idx = int(selected[0])
        else:
            idx = self.history_index

        if idx <= 0:
            messagebox.showinfo(
                "Delete history",
                "The first history item is the base mask state and cannot be deleted.",
            )
            return

        if idx >= len(self.history):
            return

        desc = self.history[idx][0]
        later_count = len(self.history) - idx - 1

        if later_count > 0:
            prompt = (
                f'Delete \"{desc}\" and the {later_count} later history '
                f'item{"s" if later_count != 1 else ""}?\n\n'
                "Later history states are cumulative mask snapshots, so they must "
                "also be removed to completely undo this selection."
            )
        else:
            prompt = f'Delete \"{desc}\" and restore the previous mask state?'

        if not messagebox.askyesno("Delete history", prompt):
            return

        # Remove the selected edit and any dependent snapshots after it.
        del self.history[idx:]

        self.history_index = idx - 1
        self.mask = self.history[self.history_index][1].copy()
        self.preview_mask = None
        self.preview_desc = ""

        # Remove any unfinished interactive geometry so the deleted selection
        # is not still drawn as a temporary yellow outline on the canvas.
        self.drag_points = []
        self.drag_start = None
        self.polygon_points = []

        self._refresh_history_list()
        self.redraw()
        self.status(f'Deleted history item: {desc}')

    def set_preview(self, m, desc):
        self.preview_mask = np.asarray(m, bool); self.preview_desc = desc; self.redraw(); self.status(f"Preview: {desc}")
    def apply_preview(self):
        if self.preview_mask is not None:
            self.commit_mask(self.preview_mask, self.preview_desc or "Applied preview")
    def close_preview_tab(self):
        self.preview_mask = None; self.preview_desc=""; self.tabs.select(self.seg_tab); self.redraw()

    # ----------------------- segmentation ops -----------------------
    def preview_threshold(self):
        if self.image is None: return
        try:
            method=self.threshold_method.get()
            if method=="Global": m=alg.global_threshold(self.image)
            elif method=="Manual": m=alg.manual_threshold(self.image,self.threshold_value.get())
            else: m=alg.adaptive_threshold(self.image,self.adapt_sensitivity.get(),self.foreground_polarity.get())
            self.set_preview(m, f"Threshold ({method})")
        except Exception as e: self._show_op_error(e)

    def run_kmeans(self):
        if self.image is None: return
        try:
            self.status("Running K-means..."); self.update_idletasks()
            m=alg.kmeans_mask(self.image,self.use_texture.get()); self.commit_mask(m,"K-means Clustering")
        except Exception as e: self._show_op_error(e)

    def preview_morphology(self):
        if self.mask is None: return
        try:
            se=alg.structure_element(self.morph_shape.get(),self.morph_radius.get(),self.morph_length.get(),self.morph_angle.get(),self.morph_width.get())
            m=alg.morphology_operation(self.mask,self.morph_op.get(),se)
            self.set_preview(m,f"Morphology: {self.morph_op.get()} ({self.morph_shape.get()})")
        except Exception as e:self._show_op_error(e)

    def preview_active_contours(self):
        if self.image is None or self.mask is None:return
        try:
            self.status("Running Active Contours..."); self.update_idletasks()
            m=alg.active_contour(self.image,self.mask,self.active_iterations.get(),self.active_method.get(),self.use_texture.get())
            self.set_preview(m,f"Active Contours ({self.active_method.get()})")
        except Exception as e:self._show_op_error(e)

    def clear_border(self):
        if self.mask is not None:self.commit_mask(alg.clear_border_mask(self.mask),"Clear Border")
    def fill_holes(self):
        if self.mask is not None:self.commit_mask(alg.fill_holes(self.mask),"Fill Holes")
    def invert_mask(self):
        if self.mask is not None:self.commit_mask(alg.invert_mask(self.mask),"Invert Mask")

    def preview_circles(self):
        if self.image is None:return
        try:
            minr=max(1,self.circle_min.get()//2); maxr=max(minr,self.circle_max.get()//2)
            centers,radii=alg.find_circles(self.image,minr,maxr,self.circle_polarity.get(),self.circle_sens.get())
            m=alg.circles_to_mask(self.image.shape[:2],centers,radii)
            self.set_preview(m,f"Find Circles ({len(radii)} circles)")
        except Exception as e:self._show_op_error(e)

    def requested_superpixels(self):
        if self.image is None:return 100
        h,w=self.image.shape[:2]; val=float(self.superpixel_density.get())
        # Exact R2018a GraphCutBaseTab formula: round(((numel/100)*(val/100))+100)
        return int(round(((h*w/100.0)*(val/100.0))+100))

    def preview_graphcut(self):
        if self.image is None:return
        try:
            self.status("Running Graph Cut / Lazy Snapping..."); self.update_idletasks()
            m,labels=alg.lazy_snapping(self.image,self.graph_fg,self.graph_bg,self.requested_superpixels(),self.use_texture.get())
            self.graph_labels=labels; self.set_preview(m,"Graph Cut / Lazy Snapping")
        except Exception as e:self._show_op_error(e)

    def preview_grabcut(self):
        if self.image is None:return
        try:
            m=alg.grabcut(self.image,self.grab_roi,self.grab_fg,self.grab_bg)
            self.set_preview(self.mask | m,"GrabCut (add to mask)")
        except Exception as e:self._show_op_error(e)

    # ------------------------- load/new mask ------------------------
    def load_mask_dialog(self):
        if self.image is None:
            return
        #
        p = filedialog.askopenfilename(title="Load Mask", filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")])
        if not p:
            return
        #
        try:
            raw = np.array(Image.open(p).convert("L"))
            m = (raw.astype(np.float32)/ 255.0 > 0.5)
            # A mask loaded from disk is always expected
            # to have the ORIGINAL source-image dimensions.
            if self._original_image is not None:
                source_shape = (self._original_image.shape[:2])
            else:
                source_shape = (self.image.shape[:2])
            #
            if m.shape != source_shape:
                raise ValueError("Mask dimensions must match the original source image.")
            #
            # ------------------------------------------------------
            # Convert loaded mask to CURRENT editing orientation
            # ------------------------------------------------------
            if self.mask_rotation_deg:
                m = self._rotate_array_clockwise(m, self.mask_rotation_deg)
            #
            self.commit_mask(m, "Load Mask")
        except Exception as e:
            self._show_op_error(e)
        #
    #
    def new_segmentation(self):
        if self.mask is None:
            return
        # Use current working-mask dimensions rather than source-image
        # dimensions, because the user may currently be editing a
        # 90-degree rotated non-square mask.
        empty_mask = np.zeros(self.mask.shape, bool)
        self.commit_mask(empty_mask, "New Segmentation")
    #
    # ------------------------- mouse tools --------------------------
    def set_mouse_mode(self, mode):
        self.mouse_mode=mode; self.drag_points=[]; self.drag_start=None
        if mode not in {"polygon","grab_roi_poly"}: self.polygon_points=[]
        self.status(f"Tool: {mode.replace('_',' ').title()}")

    def start_grab_roi(self):
        self.set_mouse_mode("grab_roi_rect" if self.grab_roi_style.get()=="Rectangle" else "grab_roi_poly")

    def on_mouse_down(self,e):
        if self.image is None:return
        if self.mouse_mode=="pan": self._pan_anchor=(e.x,e.y,self.pan_x,self.pan_y); return
        pt=self.canvas_to_image(e.x,e.y)
        if pt is None:return
        if self.mouse_mode in {"freehand","graph_fg","graph_bg","graph_erase","grab_fg","grab_bg","grab_erase"}:
            self.drag_points=[pt]; self._paint_scribble(pt,pt,temporary=True)
        elif self.mouse_mode in {"rectangle","ellipse","grab_roi_rect"}:
            self.drag_start=pt; self.drag_points=[pt]
        elif self.mouse_mode in {"polygon","grab_roi_poly"}:
            self.polygon_points.append(pt); self.redraw()
        elif self.mouse_mode=="flood": self._do_flood(pt)

    def on_mouse_drag(self,e):
        if self.image is None:return
        if self.mouse_mode=="pan" and self._pan_anchor:
            x0,y0,px,py=self._pan_anchor; self.pan_x=px+(e.x-x0); self.pan_y=py+(e.y-y0); self.redraw(); return
        pt=self.canvas_to_image(e.x,e.y)
        if pt is None:return
        if self.mouse_mode in {"freehand","graph_fg","graph_bg","graph_erase","grab_fg","grab_bg","grab_erase"} and self.drag_points:
            prev=self.drag_points[-1]; self.drag_points.append(pt)
            if self.mouse_mode!="freehand": self._paint_scribble(prev,pt,temporary=False)
            self.redraw()
        elif self.mouse_mode in {"rectangle","ellipse","grab_roi_rect"} and self.drag_start:
            self.drag_points=[self.drag_start,pt]; self.redraw()

    def on_mouse_up(self,e):
        if self.image is None:return
        if self.mouse_mode=="pan": self._pan_anchor=None; return
        pt=self.canvas_to_image(e.x,e.y)
        if pt is None:return
        if self.mouse_mode=="freehand" and len(self.drag_points)>=3:
            self.drag_points.append(pt); addition=alg.polygon_mask(self.image.shape[:2],self.drag_points)
            self.commit_mask(self.mask|addition,"Freehand") ; self.drag_points=[]
        elif self.mouse_mode in {"rectangle","ellipse"} and self.drag_start:
            add=alg.rectangle_mask(self.image.shape[:2],self.drag_start,pt) if self.mouse_mode=="rectangle" else alg.ellipse_mask(self.image.shape[:2],self.drag_start,pt)
            self.commit_mask(self.mask|add,self.mouse_mode.title()); self.drag_start=None; self.drag_points=[]
        elif self.mouse_mode=="grab_roi_rect" and self.drag_start:
            self.grab_roi=alg.rectangle_mask(self.image.shape[:2],self.drag_start,pt); self.drag_start=None; self.drag_points=[]; self.redraw(); self.status("GrabCut ROI set.")

    def on_double_click(self,e):
        if self.image is None:return
        if self.mouse_mode in {"polygon","grab_roi_poly"} and len(self.polygon_points)>=3:
            m=alg.polygon_mask(self.image.shape[:2],self.polygon_points)
            if self.mouse_mode=="polygon": self.commit_mask(self.mask|m,"Polygon")
            else: self.grab_roi=m; self.redraw(); self.status("GrabCut polygon ROI set.")
            self.polygon_points=[]

    def _paint_scribble(self,p0,p1,temporary=False):
        if self.image is None:return
        radius=max(1,int(self.marker_size.get()))
        mode=self.mouse_mode
        if mode.startswith("graph_"):
            fg,bg=self.graph_fg,self.graph_bg
        elif mode.startswith("grab_"):
            fg,bg=self.grab_fg,self.grab_bg
        else:return
        if temporary and p0==p1:
            pass
        x0,y0=p0; x1,y1=p1
        if mode.endswith("fg"):
            cv2.line(fg,(x0,y0),(x1,y1),1,radius); cv2.line(bg,(x0,y0),(x1,y1),0,radius)
        elif mode.endswith("bg"):
            cv2.line(bg,(x0,y0),(x1,y1),1,radius); cv2.line(fg,(x0,y0),(x1,y1),0,radius)
        elif mode.endswith("erase"):
            cv2.line(fg,(x0,y0),(x1,y1),0,radius); cv2.line(bg,(x0,y0),(x1,y1),0,radius)

    def _do_flood(self,pt):
        try:
            tol=float(self.flood_tol.get())
            if self.flood_metric.get()=="Euclidean": add=alg.flood_fill_euclidean(self.image,pt,tol,self.use_texture.get())
            else: add=alg.flood_fill_geodesic(self.image,pt,tol,self.use_texture.get())
            self.set_preview(self.mask|add,f"Flood Fill ({self.flood_metric.get()})")
        except Exception as e:self._show_op_error(e)

    def _clear_scribble(self,which):
        if which=="graph_fg":self.graph_fg[:]=0
        elif which=="graph_bg":self.graph_bg[:]=0
        self.redraw()
    def clear_grabcut_marks(self):
        if self.grab_fg is not None:self.grab_fg[:]=0
        if self.grab_bg is not None:self.grab_bg[:]=0
        self.grab_roi=None; self.redraw()

    def _reset_aux_masks(self):
        if self.image is None:return
        shape=self.image.shape[:2]
        self.graph_fg=np.zeros(shape,np.uint8); self.graph_bg=np.zeros(shape,np.uint8)
        self.grab_fg=np.zeros(shape,np.uint8); self.grab_bg=np.zeros(shape,np.uint8); self.grab_roi=None; self.graph_labels=None

    # ------------------------- view/render --------------------------
    def _toggle_show_binary(self):
        """Toggle binary-mask view.

        Binary and subtraction are mutually exclusive display modes.
        Turning binary view on automatically turns subtraction view off.
        """
        if self.show_binary.get():
            self.show_subtraction.set(False)
        self.redraw()

    def _toggle_show_subtraction(self):
        """Toggle masked/subtraction view for the selected history state.

        The displayed image is the pixel-wise product of the original image
        and ``self.mask``.  ``self.mask`` is always updated when a History
        item is selected, so this view follows the currently selected history
        state rather than an unapplied preview.
        """
        if self.show_subtraction.get():
            self.show_binary.set(False)
        self.redraw()

    def redraw(self):
        if self.image is None or not self.canvas.winfo_exists():return
        cw=max(10,self.canvas.winfo_width()); ch=max(10,self.canvas.winfo_height())
        h,w=self.image.shape[:2]
        fit=min(cw/w,ch/h)
        scale=max(0.02,fit*self.zoom)
        dw=max(1,int(round(w*scale))); dh=max(1,int(round(h*scale)))
        x0=(cw-dw)/2+self.pan_x; y0=(ch-dh)/2+self.pan_y
        self._display_geom=(x0,y0,dw,dh,scale)

        # Normal overlay/binary mode may show an active preview, preserving
        # the existing application behavior.  Subtraction mode intentionally
        # uses ``self.mask`` because that is the mask snapshot corresponding
        # to the currently selected item in the History panel.
        mask=self.preview_mask if self.preview_mask is not None else self.mask

        if self.show_binary.get() and mask is not None:
            rgb=np.repeat((mask.astype(np.uint8)*255)[...,None],3,axis=2)

        elif self.show_subtraction.get() and self.mask is not None:
            # Pixel-wise product: original image * selected binary mask.
            # Pixels outside the selected mask become black.
            rgb=self._display_rgb(self.image)
            selected_mask=self.mask.astype(bool)
            rgb=(rgb * selected_mask[...,None]).astype(np.uint8)

        else:
            rgb=self._display_rgb(self.image)
            if mask is not None:
                alpha=float(self.opacity.get()); m=mask.astype(bool)
                overlay=rgb.astype(np.float32)
                overlay[m]=overlay[m]*(1-alpha)+np.array([255,0,0],np.float32)*alpha
                rgb=np.clip(overlay,0,255).astype(np.uint8)

        # Auxiliary marks foreground/background overlays.
        if self.graph_fg is not None:
            rgb[self.graph_fg.astype(bool)]=[0,255,0]; rgb[self.graph_bg.astype(bool)]=[255,0,0]
        if self.tabs.index(self.tabs.select())==self.tabs.index(self.grab_tab) and self.grab_fg is not None:
            rgb[self.grab_fg.astype(bool)]=[0,255,0]; rgb[self.grab_bg.astype(bool)]=[255,0,0]
            if self.grab_roi is not None:
                b=find_boundaries(self.grab_roi,mode="outer"); rgb[b]=[255,255,0]
        if self.show_superpixels.get() and self.graph_labels is not None:
            b=find_boundaries(self.graph_labels); rgb[b]=[255,255,0]

        pil=Image.fromarray(rgb).resize((dw,dh),Image.Resampling.BILINEAR)
        self._tk_image=ImageTk.PhotoImage(pil)
        self.canvas.delete("all"); self.canvas.create_image(x0,y0,image=self._tk_image,anchor="nw")
        self._draw_active_geometry()

    def _draw_active_geometry(self):
        if not self._display_geom:return
        def cpt(p):
            x0,y0,_,_,s=self._display_geom; return (x0+p[0]*s,y0+p[1]*s)
        if self.drag_points:
            pts=[]
            for p in self.drag_points: pts.extend(cpt(p))
            if len(pts)>=4:self.canvas.create_line(*pts,fill="yellow",width=2)
        if self.drag_start and len(self.drag_points)>=2:
            p0=cpt(self.drag_start); p1=cpt(self.drag_points[-1])
            if self.mouse_mode in {"rectangle","grab_roi_rect"}: self.canvas.create_rectangle(*p0,*p1,outline="yellow",width=2)
            elif self.mouse_mode=="ellipse": self.canvas.create_oval(*p0,*p1,outline="yellow",width=2)
        if self.polygon_points:
            pts=[]
            for p in self.polygon_points:pts.extend(cpt(p))
            if len(pts)>=4:self.canvas.create_line(*pts,fill="yellow",width=2)
            for p in self.polygon_points:
                x,y=cpt(p); self.canvas.create_oval(x-2,y-2,x+2,y+2,fill="yellow",outline="yellow")

    def _display_rgb(self,a):
        f=alg.to_float01(a)
        if f.ndim==2:return np.repeat((f*255).astype(np.uint8)[...,None],3,axis=2)
        return (f[...,:3]*255).astype(np.uint8)

    def canvas_to_image(self,cx,cy):
        if self._display_geom is None or self.image is None:return None
        x0,y0,dw,dh,s=self._display_geom
        x=int((cx-x0)/s); y=int((cy-y0)/s)
        h,w=self.image.shape[:2]
        if 0<=x<w and 0<=y<h:return (x,y)
        return None

    def set_zoom(self,z):
        self.zoom=float(np.clip(z,0.1,20.0)); self.redraw()
    def on_mouse_wheel(self,e):self.set_zoom(self.zoom*(1.15 if e.delta>0 else 1/1.15))
    def _wheel_linux(self,d):self.set_zoom(self.zoom*(1.15 if d>0 else 1/1.15))

    def status(self,msg):self.status_var.set(msg)
    def _show_op_error(self,e):
        self.status(f"Error: {e}"); messagebox.showerror("Segmentation operation", f"{e}")
    #
#
def run_app(csv_path: str):
    workflow=CSVWorkflow(csv_path)
    app=ImageSegmenterApp(workflow)
    app.mainloop()
