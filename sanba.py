import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import ImageTk, Image
import os
from natsort import natsorted
from obspy import read, Trace, Stream
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.signal import hilbert, correlate, fftconvolve, detrend
import numpy as np
from numpy import isnan, isinf
import datetime
from obspy.signal.regression import linear_regression
from obspy.signal.invsim import cosine_taper
import pandas as pd
from tqdm import tqdm, trange
import warnings
import gc
from obspy.io.mseed.headers import InternalMSEEDWarning
from pandas.plotting import register_matplotlib_converters
from scipy.interpolate import griddata
from threading import Thread
import zoneinfo

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None

        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return

        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        frame = ttk.Frame(tw, style="Tooltip.TFrame")
        frame.pack()

        label = ttk.Label(frame, text=self.text, style="Tooltip.TLabel")
        label.pack(padx=6, pady=3)

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None
            
class PSVM(ttk.Frame):
    def __init__(self, parent: tk.Tk):
        super().__init__(parent)
        self.parent = parent
        self.version = "v1.0.0"

        self._init_state()
        self._configure_window()
        self._configure_styles()
        self._load_assets()
        self._build_menu()
        self._build_toolbar()
        self._build_plot_area()
        self._build_status_bar()
        self._set_default_parameters()

    # ------------------------------------------------------------------
    # INITIALIZATION HELPERS
    # ------------------------------------------------------------------
    def _init_state(self):
        self.current_project_path = None
        self.pairs = None
        self.status_var = tk.StringVar(value="Welcome to SANBA!")

    def _configure_window(self):
        self.parent.overrideredirect(False)
        self.parent.title(f"SANBA {self.version}")
        self.parent.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.parent.after(10, self._maximize_window)

    def _maximize_window(self):
        # Windows
        try:
            self.parent.state("zoomed")
            return
        except tk.TclError:
            pass

        # Many Linux window managers
        try:
            self.parent.attributes("-zoomed", True)
            return
        except tk.TclError:
            pass

        # Fallback: manually size to screen
        try:
            width = self.parent.winfo_screenwidth()
            height = self.parent.winfo_screenheight()
            self.parent.geometry(f"{width}x{height}+0+0")
        except tk.TclError:
            pass

    def _configure_styles(self):
        self.style = ttk.Style(self.parent)
        self.style.theme_use("vista")

        default_font = ("Segoe UI", 10)
        self.parent.option_add("*Font", default_font)

        self.style.configure("Toolbar.TFrame", padding=(8, 6))
        self.style.configure("Toolbar.TButton", padding=(8, 6))
        self.style.configure("Status.TLabel", padding=(10, 6))
        self.style.configure("Tooltip.TFrame", relief="solid", borderwidth=1)
        self.style.configure("Tooltip.TLabel")

    def _load_assets(self):
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        icons_dir = os.path.join(self.script_dir, "icons")

        self.window_ico = ImageTk.PhotoImage(
            Image.open(os.path.join(icons_dir, "ico_sanba.png"))
        )
        self.parent.iconphoto(False, self.window_ico)

        self.icons = {
            "create": ImageTk.PhotoImage(Image.open(os.path.join(icons_dir, "ico_new.png"))),
            "open": ImageTk.PhotoImage(Image.open(os.path.join(icons_dir, "ico_load.png"))),
            "pairs": ImageTk.PhotoImage(Image.open(os.path.join(icons_dir, "ico_pair.png"))),
            "corr": ImageTk.PhotoImage(Image.open(os.path.join(icons_dir, "ico_corr.png"))),
            "stack": ImageTk.PhotoImage(Image.open(os.path.join(icons_dir, "ico_stack.png"))),
            "mwcs": ImageTk.PhotoImage(Image.open(os.path.join(icons_dir, "ico_mwcs.png"))),
            "plot_dvv": ImageTk.PhotoImage(Image.open(os.path.join(icons_dir, "ico_dvv.png"))),
            "options": ImageTk.PhotoImage(Image.open(os.path.join(icons_dir, "ico_options.png"))),
        }

    def _build_menu(self):
        menubar = tk.Menu(self.parent)
        self.parent.config(menu=menubar)

        self._build_file_menu(menubar)
        self._build_processing_menu(menubar)
        self._build_plot_menu(menubar)
        self._build_options_menu(menubar)

    def _build_file_menu(self, menubar):
        file_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Create new project path", command=self.create_project)
        file_menu.add_command(label="Load a project path", command=self.load_project)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

    def _build_processing_menu(self, menubar):
        processing_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="Processing", menu=processing_menu)
        processing_menu.add_command(label="Get pairs", command=self.get_pairs)
        processing_menu.add_command(label="Run correlation", command=self.correlation)
        processing_menu.add_command(label="Run stacking", command=self.stack)
        processing_menu.add_command(label="Run MWCS", command=self.compute_dvv)
        processing_menu.add_separator()
        processing_menu.add_command(label="Run all steps", command=self.run_all)

    def _build_plot_menu(self, menubar):
        plotting_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="Plot", menu=plotting_menu)
        plotting_menu.add_command(label="Plot dv/v", command=self.plot_dvv)

    def _build_options_menu(self, menubar):
        options_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="Options", menu=options_menu)
        options_menu.add_command(label="Settings", command=self.options)

    def _build_toolbar(self):
        toolbar_frame = ttk.Frame(self.parent, style="Toolbar.TFrame")
        toolbar_frame.pack(fill="x")
        self.toolbar_frame = toolbar_frame

        buttons = [
            ("create_project_button", "create", self.create_project, "Create a new project"),
            ("load_project_button", "open", self.load_project, "Load an existing project"),
            ("find_pairs_button", "pairs", self.get_pairs, "Select station pairs"),
            ("corr_button", "corr", lambda: Thread(target=self.correlation).start(), "Run correlation"),
            ("stack_button", "stack", lambda: Thread(target=self.stack).start(), "Run stacking"),
            ("mwcs_button", "mwcs", lambda: Thread(target=self.compute_dvv).start(), "Run MWCS analysis"),
            ("plot_dvv_button", "plot_dvv", lambda: Thread(target=self.plot_dvv).start(), "Plot dv/v results"),
            ("options_button", "options", self.options, "Open settings"),
        ]

        for attr_name, icon_key, command, tooltip_text in buttons:
            button = ttk.Button(
                toolbar_frame,
                image=self.icons[icon_key],
                command=command,
                style="Toolbar.TButton"
            )
            button.pack(side="left")

            ToolTip(button, tooltip_text)  # 👈 add this line

            setattr(self, attr_name, button)

        ttk.Label(toolbar_frame, text="Progress: ").pack(side="left", padx=(10, 4))

        self.progress = ttk.Progressbar(
            toolbar_frame,
            length=220,
            mode="determinate"
        )
        self.progress.pack(side="left")

    def _build_plot_area(self):
        frame_plot = ttk.Frame(self.parent)
        frame_plot.pack(fill="both", expand=True)
        self.frame_plot = frame_plot

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax2 = self.ax.twinx()

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_plot)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        self.plot_toolbar = NavigationToolbar2Tk(self.canvas, frame_plot)
        self.plot_toolbar.update()

    def _build_status_bar(self):
        status_frame = ttk.Frame(self.parent)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            relief="groove",
            style="Status.TLabel"
        )
        self.status_label.pack(fill="x")

    def _set_default_parameters(self):
        # ---------------------------
        # Correlation / stacking
        # ---------------------------
        self.network_code = "AM"
        self.channel_code = "EHZ.D"
        self.do_crosscomponent_analysis = False
        self.corr_sorting_type = "pairs"#both#pairs#individual
        self.correlation_method = "pcc"#pcc#cc

        self.corr_remove_mean = True
        self.corr_remove_trend = True
        self.corr_taper = True
        self.corr_bandpass_filter = True
        self.corr_onebit_norm = False
        self.corr_spectral_whitening = False

        self.corr_window_size = 3600
        self.corr_overlap = 0
        self.corr_min_freq = 3
        self.corr_max_freq = 12
        self.corr_resample_rate = self.corr_max_freq * 2
        self.corr_max_lag = 5
        self.corr_snr_threshold = 0
        self.stack_window_length_days = 1

        # ---------------------------
        # MWCS
        # ---------------------------
        self.mwcs_reference = "following"#mean#following#static
        self.mwcs_freq_min = 4
        self.mwcs_freq_max = 10
        self.mwcs_window_length = 1
        self.mwcs_window_step = self.mwcs_window_length / 10
        self.mwcs_moving_start = -self.corr_max_lag

        self.mwcs_coherency_min = 0.5
        self.mwcs_error_max = 0.2
        self.mwcs_lagtime_ballistic = 1
        self.mwcs_lagtime_max = self.corr_max_lag
        self.mwcs_abs_delay_time_limit = 0.1

        self.mwcs_do_similarity_analysis = False
        self.mwcs_similarity_method = "zero_lag_pcc"

        # ---------------------------
        # Plotting
        # ---------------------------
        self.corr_plot = True
        self.stack_plot = True
        self.mwcs_plot = False
        self.output_timezone = "America/Sao_Paulo"

    def on_closing(self):
        if messagebox.askyesno("SANBA", "Exit?"):
            self.parent.destroy()
            sys.exit()

    def options(self):

        # Close old options window if it already exists
        try:
            if self.top_options.winfo_exists():
                self.top_options.destroy()
        except:
            pass

        self.top_options = tk.Toplevel(self)
        self.top_options.title("SANBA - Settings")
        self.top_options.geometry("620x820")
        self.top_options.resizable(False, False)
        self.top_options.transient(self)
        self.top_options.grab_set()

        main_frame = ttk.Frame(self.top_options, padding=10)
        main_frame.pack(fill="both", expand=True)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill="both", expand=True, pady=(0, 10))

        # ------------------------------------------------------------------
        # TAB 1 - CORRELATION AND STACKING
        # ------------------------------------------------------------------
        tab_corr_outer = ttk.Frame(notebook)
        notebook.add(tab_corr_outer, text="Correlation & Stacking")

        corr_canvas = tk.Canvas(tab_corr_outer, highlightthickness=0)
        corr_scrollbar = ttk.Scrollbar(tab_corr_outer, orient="vertical", command=corr_canvas.yview)
        corr_scrollable_frame = ttk.Frame(corr_canvas)

        corr_scrollable_frame.bind(
            "<Configure>",
            lambda e: corr_canvas.configure(scrollregion=corr_canvas.bbox("all"))
        )

        corr_canvas.create_window((0, 0), window=corr_scrollable_frame, anchor="nw")
        corr_canvas.configure(yscrollcommand=corr_scrollbar.set)

        corr_canvas.pack(side="left", fill="both", expand=True)
        corr_scrollbar.pack(side="right", fill="y")

        def _on_corr_mousewheel(event):
            corr_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        corr_canvas.bind_all("<MouseWheel>", _on_corr_mousewheel)

        corr_padx = 8
        corr_pady = 4

        ttk.Label(corr_scrollable_frame, text="General", font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=corr_padx, pady=(8, 6)
        )

        ttk.Label(corr_scrollable_frame, text="Network code:").grid(row=1, column=0, sticky="w", padx=corr_padx, pady=corr_pady)
        entry_network_code = ttk.Entry(corr_scrollable_frame, width=25)
        entry_network_code.grid(row=1, column=1, sticky="ew", padx=corr_padx, pady=corr_pady)
        entry_network_code.insert(0, self.network_code)

        ttk.Label(corr_scrollable_frame, text="Channel code:").grid(row=2, column=0, sticky="w", padx=corr_padx, pady=corr_pady)
        entry_channel_code = ttk.Entry(corr_scrollable_frame, width=25)
        entry_channel_code.grid(row=2, column=1, sticky="ew", padx=corr_padx, pady=corr_pady)
        entry_channel_code.insert(0, self.channel_code)

        do_crosscomponent_analysis_var = tk.BooleanVar(value=self.do_crosscomponent_analysis)
        ttk.Checkbutton(
            corr_scrollable_frame,
            text="Do cross-component analysis",
            variable=do_crosscomponent_analysis_var
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=corr_padx, pady=(6, 10))

        ttk.Label(corr_scrollable_frame, text="Station sorting", font=("TkDefaultFont", 10, "bold")).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=corr_padx, pady=(8, 6)
        )

        sorting_type_var = tk.StringVar(value=self.corr_sorting_type)
        ttk.Radiobutton(
            corr_scrollable_frame, text="Pairs (cross-correlations)",
            variable=sorting_type_var, value="pairs"
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=corr_padx, pady=2)

        ttk.Radiobutton(
            corr_scrollable_frame, text="Individual (auto-correlations)",
            variable=sorting_type_var, value="individual"
        ).grid(row=6, column=0, columnspan=2, sticky="w", padx=corr_padx, pady=2)

        ttk.Radiobutton(
            corr_scrollable_frame, text="Both",
            variable=sorting_type_var, value="both"
        ).grid(row=7, column=0, columnspan=2, sticky="w", padx=corr_padx, pady=(2, 10))

        ttk.Label(corr_scrollable_frame, text="Parameters and pre-processing for correlation", font=("TkDefaultFont", 10, "bold")).grid(
            row=8, column=0, columnspan=2, sticky="w", padx=corr_padx, pady=(8, 6)
        )

        ttk.Label(corr_scrollable_frame, text="Resample rate (Hz):").grid(row=9, column=0, sticky="w", padx=corr_padx, pady=corr_pady)
        entry_xcorr_resample = ttk.Entry(corr_scrollable_frame, width=25)
        entry_xcorr_resample.grid(row=9, column=1, sticky="ew", padx=corr_padx, pady=corr_pady)
        entry_xcorr_resample.insert(0, self.corr_resample_rate)

        ttk.Label(corr_scrollable_frame, text="Window length (s):").grid(row=10, column=0, sticky="w", padx=corr_padx, pady=corr_pady)
        entry_xcorr_length = ttk.Entry(corr_scrollable_frame, width=25)
        entry_xcorr_length.grid(row=10, column=1, sticky="ew", padx=corr_padx, pady=corr_pady)
        entry_xcorr_length.insert(0, self.corr_window_size)

        ttk.Label(corr_scrollable_frame, text="Window overlap (0.5 = 50%):").grid(row=11, column=0, sticky="w", padx=corr_padx, pady=corr_pady)
        entry_xcorr_overlap = ttk.Entry(corr_scrollable_frame, width=25)
        entry_xcorr_overlap.grid(row=11, column=1, sticky="ew", padx=corr_padx, pady=corr_pady)
        entry_xcorr_overlap.insert(0, self.corr_overlap)

        ttk.Label(corr_scrollable_frame, text="Pre-processing", font=("TkDefaultFont", 10, "bold")).grid(
            row=12, column=0, columnspan=2, sticky="w", padx=corr_padx, pady=(8, 6)
        )

        remove_mean_var = tk.BooleanVar(value=self.corr_remove_mean)
        remove_trend_var = tk.BooleanVar(value=self.corr_remove_trend)
        taper_var = tk.BooleanVar(value=self.corr_taper)
        bandpass_filter_var = tk.BooleanVar(value=self.corr_bandpass_filter)
        spectral_whitening_var = tk.BooleanVar(value=self.corr_spectral_whitening)
        onebit_norm_var = tk.BooleanVar(value=self.corr_onebit_norm)

        ttk.Checkbutton(corr_scrollable_frame, text="Remove mean", variable=remove_mean_var).grid(
            row=13, column=0, columnspan=2, sticky="w", padx=corr_padx, pady=2
        )
        ttk.Checkbutton(corr_scrollable_frame, text="Remove trend", variable=remove_trend_var).grid(
            row=14, column=0, columnspan=2, sticky="w", padx=corr_padx, pady=2
        )
        ttk.Checkbutton(corr_scrollable_frame, text="Taper", variable=taper_var).grid(
            row=15, column=0, columnspan=2, sticky="w", padx=corr_padx, pady=2
        )
        ttk.Checkbutton(corr_scrollable_frame, text="Bandpass filter", variable=bandpass_filter_var).grid(
            row=16, column=0, columnspan=2, sticky="w", padx=corr_padx, pady=2
        )
        ttk.Checkbutton(corr_scrollable_frame, text="Spectral whitening", variable=spectral_whitening_var).grid(
            row=17, column=0, columnspan=2, sticky="w", padx=corr_padx, pady=2
        )
        ttk.Checkbutton(corr_scrollable_frame, text="1-bit normalization", variable=onebit_norm_var).grid(
            row=18, column=0, columnspan=2, sticky="w", padx=corr_padx, pady=(2, 10)
        )

        ttk.Label(corr_scrollable_frame, text="Minimum frequency (Hz):").grid(row=19, column=0, sticky="w", padx=corr_padx, pady=corr_pady)
        entry_xcorr_min_freq = ttk.Entry(corr_scrollable_frame, width=25)
        entry_xcorr_min_freq.grid(row=19, column=1, sticky="ew", padx=corr_padx, pady=corr_pady)
        entry_xcorr_min_freq.insert(0, self.corr_min_freq)

        ttk.Label(corr_scrollable_frame, text="Maximum frequency (Hz):").grid(row=20, column=0, sticky="w", padx=corr_padx, pady=corr_pady)
        entry_xcorr_max_freq = ttk.Entry(corr_scrollable_frame, width=25)
        entry_xcorr_max_freq.grid(row=20, column=1, sticky="ew", padx=corr_padx, pady=corr_pady)
        entry_xcorr_max_freq.insert(0, self.corr_max_freq)

        ttk.Label(corr_scrollable_frame, text="Maximum absolute time lag (s):").grid(row=21, column=0, sticky="w", padx=corr_padx, pady=corr_pady)
        entry_xcorr_max_lag = ttk.Entry(corr_scrollable_frame, width=25)
        entry_xcorr_max_lag.grid(row=21, column=1, sticky="ew", padx=corr_padx, pady=corr_pady)
        entry_xcorr_max_lag.insert(0, self.corr_max_lag)

        ttk.Label(corr_scrollable_frame, text="Signal extraction method:").grid(row=22, column=0, sticky="w", padx=corr_padx, pady=corr_pady)
        correlation_method_var = tk.StringVar(value=self.correlation_method)
        method_frame = ttk.Frame(corr_scrollable_frame)
        method_frame.grid(row=22, column=1, sticky="w", padx=corr_padx, pady=corr_pady)
        ttk.Radiobutton(method_frame, text="Cross-correlation", variable=correlation_method_var, value="cc").pack(anchor="w")
        ttk.Radiobutton(method_frame, text="Phase cross-correlation", variable=correlation_method_var, value="pcc").pack(anchor="w")

        ttk.Label(corr_scrollable_frame, text="Stacking", font=("TkDefaultFont", 10, "bold")).grid(
            row=23, column=0, columnspan=2, sticky="w", padx=corr_padx, pady=(8, 6)
        )
        
        ttk.Label(corr_scrollable_frame, text="Number of days for moving-window stacking:").grid(row=24, column=0, sticky="w", padx=corr_padx, pady=corr_pady)
        entry_stack_ndays = ttk.Entry(corr_scrollable_frame, width=25)
        entry_stack_ndays.grid(row=24, column=1, sticky="ew", padx=corr_padx, pady=(corr_pady, 12))
        entry_stack_ndays.insert(0, self.stack_window_length_days)

        corr_scrollable_frame.columnconfigure(1, weight=1)

        # ------------------------------------------------------------------
        # TAB 2 - MWCS
        # ------------------------------------------------------------------
        tab_mwcs = ttk.Frame(notebook, padding=10)
        notebook.add(tab_mwcs, text="MWCS")

        mwcs_padx = 8
        mwcs_pady = 4

        ttk.Label(tab_mwcs, text="Reference function", font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=mwcs_padx, pady=(4, 6)
        )

        mwcs_reference_type_var = tk.StringVar(value=self.mwcs_reference)
        ttk.Radiobutton(tab_mwcs, text="Static (first stack)", variable=mwcs_reference_type_var, value="static").grid(
            row=1, column=0, columnspan=2, sticky="w", padx=mwcs_padx, pady=2
        )
        ttk.Radiobutton(tab_mwcs, text="Mean of all stacks", variable=mwcs_reference_type_var, value="mean").grid(
            row=2, column=0, columnspan=2, sticky="w", padx=mwcs_padx, pady=2
        )
        ttk.Radiobutton(tab_mwcs, text="Following behind moving correlation", variable=mwcs_reference_type_var, value="following").grid(
            row=3, column=0, columnspan=2, sticky="w", padx=mwcs_padx, pady=(2, 10)
        )

        ttk.Label(tab_mwcs, text="MWCS parameters", font=("TkDefaultFont", 10, "bold")).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=mwcs_padx, pady=(4, 6)
        )

        ttk.Label(tab_mwcs, text="Minimum frequency (Hz):").grid(row=5, column=0, sticky="w", padx=mwcs_padx, pady=mwcs_pady)
        entry_mwcs_min_freq = ttk.Entry(tab_mwcs, width=25)
        entry_mwcs_min_freq.grid(row=5, column=1, sticky="ew", padx=mwcs_padx, pady=mwcs_pady)
        entry_mwcs_min_freq.insert(0, self.mwcs_freq_min)

        ttk.Label(tab_mwcs, text="Maximum frequency (Hz):").grid(row=6, column=0, sticky="w", padx=mwcs_padx, pady=mwcs_pady)
        entry_mwcs_max_freq = ttk.Entry(tab_mwcs, width=25)
        entry_mwcs_max_freq.grid(row=6, column=1, sticky="ew", padx=mwcs_padx, pady=mwcs_pady)
        entry_mwcs_max_freq.insert(0, self.mwcs_freq_max)

        ttk.Label(tab_mwcs, text="Moving window length (s):").grid(row=7, column=0, sticky="w", padx=mwcs_padx, pady=mwcs_pady)
        entry_mwcs_window = ttk.Entry(tab_mwcs, width=25)
        entry_mwcs_window.grid(row=7, column=1, sticky="ew", padx=mwcs_padx, pady=mwcs_pady)
        entry_mwcs_window.insert(0, self.mwcs_window_length)

        ttk.Label(tab_mwcs, text="Moving window step (s):").grid(row=8, column=0, sticky="w", padx=mwcs_padx, pady=mwcs_pady)
        entry_mwcs_step = ttk.Entry(tab_mwcs, width=25)
        entry_mwcs_step.grid(row=8, column=1, sticky="ew", padx=mwcs_padx, pady=mwcs_pady)
        entry_mwcs_step.insert(0, self.mwcs_window_step)

        ttk.Label(tab_mwcs, text="Start time lag for moving window (s):").grid(row=9, column=0, sticky="w", padx=mwcs_padx, pady=mwcs_pady)
        entry_mwcs_start_time = ttk.Entry(tab_mwcs, width=25)
        entry_mwcs_start_time.grid(row=9, column=1, sticky="ew", padx=mwcs_padx, pady=mwcs_pady)
        entry_mwcs_start_time.insert(0, self.mwcs_moving_start)

        ttk.Label(tab_mwcs, text="Delay times filtering thresholds", font=("TkDefaultFont", 10, "bold")).grid(
            row=10, column=0, columnspan=2, sticky="w", padx=mwcs_padx, pady=(10, 6)
        )

        ttk.Label(tab_mwcs, text="Minimum coherency:").grid(row=11, column=0, sticky="w", padx=mwcs_padx, pady=mwcs_pady)
        entry_mwcs_min_coh_filter = ttk.Entry(tab_mwcs, width=25)
        entry_mwcs_min_coh_filter.grid(row=11, column=1, sticky="ew", padx=mwcs_padx, pady=mwcs_pady)
        entry_mwcs_min_coh_filter.insert(0, self.mwcs_coherency_min)

        ttk.Label(tab_mwcs, text="Maximum error:").grid(row=12, column=0, sticky="w", padx=mwcs_padx, pady=mwcs_pady)
        entry_mwcs_max_err_filter = ttk.Entry(tab_mwcs, width=25)
        entry_mwcs_max_err_filter.grid(row=12, column=1, sticky="ew", padx=mwcs_padx, pady=mwcs_pady)
        entry_mwcs_max_err_filter.insert(0, self.mwcs_error_max)

        ttk.Label(tab_mwcs, text="Maximum absolute time lag (s):").grid(row=13, column=0, sticky="w", padx=mwcs_padx, pady=mwcs_pady)
        entry_mwcs_max_time_filter = ttk.Entry(tab_mwcs, width=25)
        entry_mwcs_max_time_filter.grid(row=13, column=1, sticky="ew", padx=mwcs_padx, pady=mwcs_pady)
        entry_mwcs_max_time_filter.insert(0, self.mwcs_lagtime_max)

        ttk.Label(tab_mwcs, text="Ballistic arrival exclusion absolute time lag (s):").grid(row=14, column=0, sticky="w", padx=mwcs_padx, pady=mwcs_pady)
        entry_mwcs_min_time_filter = ttk.Entry(tab_mwcs, width=25)
        entry_mwcs_min_time_filter.grid(row=14, column=1, sticky="ew", padx=mwcs_padx, pady=mwcs_pady)
        entry_mwcs_min_time_filter.insert(0, self.mwcs_lagtime_ballistic)

        ttk.Label(tab_mwcs, text="Absolute time axis limit (s):").grid(row=15, column=0, sticky="w", padx=mwcs_padx, pady=mwcs_pady)
        entry_mwcs_max_dt_filter = ttk.Entry(tab_mwcs, width=25)
        entry_mwcs_max_dt_filter.grid(row=15, column=1, sticky="ew", padx=mwcs_padx, pady=mwcs_pady)
        entry_mwcs_max_dt_filter.insert(0, self.mwcs_abs_delay_time_limit)

        ttk.Label(tab_mwcs, text="Waveform convergence", font=("TkDefaultFont", 10, "bold")).grid(
            row=16, column=0, columnspan=2, sticky="w", padx=mwcs_padx, pady=(10, 6)
        )

        do_similarity_analysis_var = tk.BooleanVar(value=self.mwcs_do_similarity_analysis)
        ttk.Checkbutton(
            tab_mwcs,
            text="Run similarity analysis",
            variable=do_similarity_analysis_var
        ).grid(row=17, column=0, columnspan=2, sticky="w", padx=mwcs_padx, pady=(0, 6))

        ttk.Label(tab_mwcs, text="Similarity extraction method:").grid(row=18, column=0, sticky="w", padx=mwcs_padx, pady=mwcs_pady)
        similarity_method_var = tk.StringVar(value=self.mwcs_similarity_method)
        similarity_frame = ttk.Frame(tab_mwcs)
        similarity_frame.grid(row=18, column=1, sticky="w", padx=mwcs_padx, pady=mwcs_pady)
        ttk.Radiobutton(similarity_frame, text="Zero-lag CCG", variable=similarity_method_var, value="zero_lag_cc").pack(anchor="w")
        ttk.Radiobutton(similarity_frame, text="Zero-lag PCC", variable=similarity_method_var, value="zero_lag_pcc").pack(anchor="w")

        tab_mwcs.columnconfigure(1, weight=1)

        # ------------------------------------------------------------------
        # TAB 3 - PLOTTING
        # ------------------------------------------------------------------
        tab_plot = ttk.Frame(notebook, padding=10)
        notebook.add(tab_plot, text="Plotting")

        ttk.Label(tab_plot, text="Plot generation", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(4, 8))

        plot_corr_var = tk.BooleanVar(value=self.corr_plot)
        plot_stack_var = tk.BooleanVar(value=self.stack_plot)
        plot_mwcs_var = tk.BooleanVar(value=self.mwcs_plot)

        ttk.Checkbutton(
            tab_plot,
            text="Plot image of correlation functions over time",
            variable=plot_corr_var
        ).pack(anchor="w", pady=4)

        ttk.Checkbutton(
            tab_plot,
            text="Plot image of stack functions over time",
            variable=plot_stack_var
        ).pack(anchor="w", pady=4)

        ttk.Checkbutton(
            tab_plot,
            text="Plot images of delay-time least-squares regression (MWCS)",
            variable=plot_mwcs_var
        ).pack(anchor="w", pady=4)

        ttk.Label(tab_plot, text="Time settings", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(10, 6))

        frame_tz = ttk.Frame(tab_plot)
        frame_tz.pack(fill="x", pady=4)
        ttk.Label(frame_tz, text="Output timezone (Region/City):").pack(side="left", padx=(0, 8))
        entry_timezone = ttk.Entry(frame_tz, width=30)
        entry_timezone.pack(side="left", fill="x", expand=True)
        entry_timezone.insert(0, self.output_timezone)

        # ------------------------------------------------------------------
        # BOTTOM BUTTONS
        # ------------------------------------------------------------------
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x")

        def done():
            try:
                # Correlation / stacking
                self.network_code = str(entry_network_code.get()).strip()
                self.channel_code = str(entry_channel_code.get()).strip()
                self.do_crosscomponent_analysis = do_crosscomponent_analysis_var.get()
                self.corr_sorting_type = sorting_type_var.get()

                self.corr_resample_rate = float(entry_xcorr_resample.get())
                self.corr_window_size = float(entry_xcorr_length.get())
                self.corr_overlap = float(entry_xcorr_overlap.get())

                self.corr_remove_mean = remove_mean_var.get()
                self.corr_remove_trend = remove_trend_var.get()
                self.corr_taper = taper_var.get()
                self.corr_bandpass_filter = bandpass_filter_var.get()
                self.corr_spectral_whitening = spectral_whitening_var.get()
                self.corr_onebit_norm = onebit_norm_var.get()

                self.corr_min_freq = float(entry_xcorr_min_freq.get())
                self.corr_max_freq = float(entry_xcorr_max_freq.get())
                self.corr_max_lag = float(entry_xcorr_max_lag.get())

                self.correlation_method = correlation_method_var.get()
                self.stack_window_length_days = float(entry_stack_ndays.get())

                # MWCS
                self.mwcs_reference = mwcs_reference_type_var.get()
                self.mwcs_freq_min = float(entry_mwcs_min_freq.get())
                self.mwcs_freq_max = float(entry_mwcs_max_freq.get())
                self.mwcs_window_length = float(entry_mwcs_window.get())
                self.mwcs_window_step = float(entry_mwcs_step.get())
                self.mwcs_moving_start = float(entry_mwcs_start_time.get())
                self.mwcs_coherency_min = float(entry_mwcs_min_coh_filter.get())
                self.mwcs_error_max = float(entry_mwcs_max_err_filter.get())
                self.mwcs_lagtime_max = float(entry_mwcs_max_time_filter.get())
                self.mwcs_lagtime_ballistic = float(entry_mwcs_min_time_filter.get())
                self.mwcs_abs_delay_time_limit = float(entry_mwcs_max_dt_filter.get())
                self.mwcs_do_similarity_analysis = do_similarity_analysis_var.get()
                self.mwcs_similarity_method = similarity_method_var.get()

                # Plotting
                self.corr_plot = plot_corr_var.get()
                self.stack_plot = plot_stack_var.get()
                self.mwcs_plot = plot_mwcs_var.get()
                self.output_timezone = str(entry_timezone.get()).strip()

                # Basic validation
                if self.corr_min_freq >= self.corr_max_freq:
                    raise ValueError("Correlation minimum frequency must be smaller than maximum frequency.")

                if self.mwcs_freq_min >= self.mwcs_freq_max:
                    raise ValueError("MWCS minimum frequency must be smaller than maximum frequency.")

                if self.corr_overlap < 0 or self.corr_overlap >= 1:
                    raise ValueError("Correlation overlap must be between 0 and 1.")

                if self.corr_window_size <= 0 or self.corr_resample_rate <= 0:
                    raise ValueError("Correlation window size and resample rate must be positive.")

                if self.stack_window_length_days <= 0:
                    raise ValueError("Stack window length must be positive.")

                if self.mwcs_window_length <= 0 or self.mwcs_window_step <= 0:
                    raise ValueError("MWCS window length and step must be positive.")

                if self.mwcs_lagtime_ballistic < 0 or self.mwcs_lagtime_max < 0:
                    raise ValueError("MWCS lag times must be non-negative.")

                if self.mwcs_lagtime_ballistic > self.mwcs_lagtime_max:
                    raise ValueError("Ballistic exclusion time lag cannot be greater than maximum time lag.")

                try:
                    zoneinfo.ZoneInfo(self.output_timezone)
                except Exception:
                    raise ValueError(f"Invalid timezone: {self.output_timezone}\nExample of valid values: Europe/Paris, America/New_York, Asia/Tokyo...")

                self.status_var.set("New settings saved successfully.")
                messagebox.showinfo("SANBA", "Settings saved successfully.")
                self.top_options.destroy()

            except ValueError as e:
                messagebox.showwarning("SANBA", f"Invalid inputs:\n{e}")
                self.top_options.lift()
                self.top_options.focus_force()

        def cancel():
            self.top_options.destroy()

        ttk.Button(button_frame, text="Cancel", command=cancel, width=18).pack(side="right", padx=(6, 0))
        ttk.Button(button_frame, text="Save settings", command=done, width=18).pack(side="right")

    def run_all(self):
        
        if self.current_project_path == None:
            messagebox.showwarning("SANBA", "No project path detected. Create or load a project to continue.")
            return
            
        if self.pairs == None:
            messagebox.showwarning("SANBA", "No pair(s) of station(s) detected. Select stations to continue.")
            return

        self.correlation()
        self.stack()
        self.compute_dvv()
        self.plot_dvv()

    def create_project(self):
        directory = filedialog.askdirectory()

        if directory:
            project_name = simpledialog.askstring("SANBA", "Enter the name of the new project:")

            if project_name:
                proj_dir = os.path.join(directory, project_name)

                if os.path.exists(proj_dir):
                    messagebox.showwarning(
                        "SANBA",
                        "This project already exists, please enter a different name."
                    )
                    return

                os.makedirs(proj_dir, exist_ok=True)

                data_dir = os.path.join(proj_dir, "data")
                out_dir = os.path.join(proj_dir, "out")

                os.makedirs(data_dir, exist_ok=True)
                os.makedirs(out_dir, exist_ok=True)
                os.makedirs(os.path.join(out_dir, "corr"), exist_ok=True)
                os.makedirs(os.path.join(out_dir, "stack"), exist_ok=True)
                os.makedirs(os.path.join(out_dir, "dvv"), exist_ok=True)

                self.current_project_path = os.path.abspath(proj_dir)
                self.status_var.set("Finished creating project.")
                messagebox.showinfo("SANBA", "Project created successfully.")

    def load_project(self):

        project_dir = tk.filedialog.askdirectory(initialdir="projects")

        if project_dir:
            if os.path.exists(project_dir+"/out/corr") and os.path.exists(project_dir+"/out/stack") and os.path.exists(project_dir+"/out/dvv"):
                self.current_project_path = project_dir
                self.status_var.set("Finished loading project.")
                messagebox.showinfo("SANBA", "Project loaded successfully.")
            else:
                messagebox.showwarning("SANBA", "The selected directory is not a valid project.")
                return

    def get_pairs(self):

        if self.current_project_path:

            data_dir, fmt = os.path.join(self.current_project_path, "data"), self.corr_sorting_type
            
            # Get a list of all directories in data_dir
            all_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
          
            # Filter directories that start with 'AM.'
            am_dirs = [d for d in all_dirs if d.startswith(f'{self.network_code}.')]

            if am_dirs:
                
                self.top_get_pairs = tk.Toplevel(self)
                self.top_get_pairs.title("PSVM - Get pairs of stations")

                station_vars = []
                
                for sta in am_dirs:
                    station_var = tk.BooleanVar()
                    station_var.set(True)
                    ttk.Checkbutton(self.top_get_pairs, text=sta, variable=station_var).pack()
                    station_vars.append(station_var)

                def done():

                    stations2use = [station for station, var in zip(am_dirs, station_vars) if var.get() == True]

                    if self.corr_sorting_type == "pairs" or self.corr_sorting_type == "both":
                        if len(stations2use) < 2:
                            messagebox.showwarning("SANBA", "Current setting for sorting of stations is set to 'pairs' or 'both'. Select at least two stations to continue.")
                            return
                    elif self.corr_sorting_type == "individual":
                        if len(stations2use) < 1:
                            messagebox.showwarning("SANBA", "Current setting for sorting of stations is set to 'individual'. Select at least one station to continue.")
                            return
                        
                    pairs = []
                    for i in range(len(stations2use)):
                        if fmt in ["pairs", "both"]:
                            for j in range(i+1, len(stations2use)):
                                pairs.append((stations2use[i], stations2use[j]))
                        if fmt in ["individual", "both"]:
                            pairs.append((stations2use[i], stations2use[i]))

                    #print(pairs)
                    self.pairs = pairs
                
                    self.status_var.set("Finished getting pairs of stations.")
                    messagebox.showinfo("SANBA", f"A total of {len(pairs)} pair(s) defined for a total of {len(stations2use)} selected station(s)")
                    self.top_get_pairs.destroy()
                
                ttk.Button(self.top_get_pairs, text="Get pairs", command=done, width=35).pack(pady=5)

            else:
                messagebox.showwarning("SANBA", "No stations were found in the 'data' directory. Add these folders and waveform files to continue.")
            
        else:
            messagebox.showwarning("SANBA", "No project path detected. Create or load a project to continue.")

    '''def spectral_whitening(self, signal, dt, f1, f2):
        # Number of samples in the signal
        n = len(signal)
        # FFT of the signal
        fft_signal = np.fft.fft(signal)
        # Frequency array
        freqs = np.fft.fftfreq(n, d=dt)
        # Magnitude spectrum
        magnitude = np.abs(fft_signal)
        # Phase spectrum
        phase = np.angle(fft_signal)
        # Identify indices within the frequency range f1 to f2
        idx = np.where((freqs >= f1) & (freqs <= f2))
        # Flatten the magnitude spectrum between f1 and f2
        magnitude[idx] = 1.0
        # Create the whitened FFT signal
        whitened_fft = magnitude * np.exp(1j * phase)
        # Inverse FFT to get the whitened time domain signal
        whitened_signal = np.fft.ifft(whitened_fft).real
        return whitened_signal'''

    def spectral_whitening(self, signal, dt, f1, f2, napod=100):
        signal = np.asarray(signal, dtype=float)

        if signal.ndim != 1:
            raise ValueError("signal must be a 1D array.")
        if len(signal) == 0:
            raise ValueError("signal cannot be empty.")
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if f1 <= 0 or f2 <= 0 or f1 >= f2:
            raise ValueError("Require 0 < f1 < f2.")

        n = len(signal)
        nfft = n

        fft_signal = np.fft.fft(signal, nfft)
        freq_vec = np.fft.fftfreq(nfft, d=dt)[: nfft // 2]

        band_idx = np.where((freq_vec >= f1) & (freq_vec <= f2))[0]
        if len(band_idx) == 0:
            raise ValueError("No FFT bins found inside the whitening band.")

        low = band_idx[0] - napod
        if low <= 0:
            low = 1

        porte1 = band_idx[0]
        porte2 = band_idx[-1]

        high = band_idx[-1] + napod
        if high > nfft // 2:
            high = nfft // 2

        whitened_fft = fft_signal.copy()

        # Left stop band
        whitened_fft[0:low] = 0

        # Left taper
        if porte1 > low:
            taper = np.cos(np.linspace(np.pi / 2.0, np.pi, porte1 - low)) ** 2
            whitened_fft[low:porte1] = taper * np.exp(1j * np.angle(whitened_fft[low:porte1]))

        # Pass band
        if porte2 > porte1:
            whitened_fft[porte1:porte2] = np.exp(1j * np.angle(whitened_fft[porte1:porte2]))
        else:
            whitened_fft[porte1:porte2 + 1] = np.exp(1j * np.angle(whitened_fft[porte1:porte2 + 1]))

        # Right taper
        if high > porte2:
            taper = np.cos(np.linspace(0.0, np.pi / 2.0, high - porte2)) ** 2
            whitened_fft[porte2:high] = taper * np.exp(1j * np.angle(whitened_fft[porte2:high]))

        # Right stop band
        whitened_fft[high:nfft + 1] = 0

        # Hermitian symmetry for real-valued time-domain reconstruction
        whitened_fft[-(nfft // 2) + 1:] = whitened_fft[1:(nfft // 2)].conjugate()[::-1]

        whitened_signal = np.real(np.fft.ifft(whitened_fft, nfft))

        return whitened_signal

    '''def cc(self, x1, x2, dt, lag0, lagu):
        #x1 = np.asarray(x1)
        #x2 = np.asarray(x2)
        N = len(x1)
        M = len(x2)
        if N != M:
            raise ValueError("x1 and x2 must be same length for this function.")
        # Numerator: cross-correlation
        cc = correlate(x1, x2, mode='full', method='fft')
        # Sliding (windowed) sum of squares (for denominator)
        tnorm = np.sum(x2 ** 2)  # x2 is template, usually fixed
        win = np.ones(M)
        x1_sq_cumsum = fftconvolve(x1 ** 2, win, mode='full')  # same length as cc
        # For 'full' mode, the valid lags are from -(M-1) to +(N-1)
        # So for each lag, denominator is sqrt( sum(x1^2 in window) * sum(x2^2) )
        denom = np.sqrt(x1_sq_cumsum * tnorm)
        # To avoid divide by zero
        eps = np.finfo(float).eps
        mask = denom > eps
        cc_norm = np.zeros_like(cc)
        cc_norm[mask] = cc[mask] / denom[mask]
        cc_norm[~mask] = 0
        # The center of cc_norm is zero lag
        lags = np.arange(-(M-1), N)
        t = lags * dt
        sel = (t >= lag0) & (t <= lagu)
        t_out = t[sel]
        cc_out = cc_norm[sel]
        cc_zero_lag = cc_norm[np.where(lags == 0)[0][0]]
        return t_out, cc_out, cc_zero_lag'''

    def cc(self, x1, x2, dt, lag0, lagu):
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        if x1.ndim != 1 or x2.ndim != 1:
            raise ValueError("x1 and x2 must be 1D arrays.")

        N = len(x1)
        M = len(x2)

        if N != M:
            raise ValueError("x1 and x2 must be same length for this function.")

        Nt = N

        # FFT length
        nfft = Nt

        # FFT of both signals
        X1 = np.fft.fft(x1, n=nfft)
        X2 = np.fft.fft(x2, n=nfft)

        # Frequency-domain cross-correlation
        cc = np.conj(X1) * X2
        cc = np.real(np.fft.ifft(cc, nfft)) / Nt

        # Rearrange to lags from -(Nt-1) to +(Nt-1)
        cc = np.concatenate((cc[-Nt + 1:], cc[:Nt]))

        # Time / lag axis
        lags = np.arange(-Nt + 1, Nt)
        t = lags * dt

        # Global normalization
        # Equivalent to dividing by the product of RMS amplitudes
        rms1 = np.sqrt(np.mean(x1 ** 2))
        rms2 = np.sqrt(np.mean(x2 ** 2))
        E = rms1 * rms2

        if E > np.finfo(float).eps:
            cc = cc / E
        else:
            cc = np.zeros_like(cc)

        # Select requested lag window
        sel = (t >= lag0) & (t <= lagu)
        t_out = t[sel]
        cc_out = cc[sel]

        # Zero-lag value
        zero_idx = np.where(lags == 0)[0]
        cc_zero_lag = cc[zero_idx[0]] if len(zero_idx) else np.nan

        return t_out, cc_out, cc_zero_lag
    
    # PCC2 computation from Ventosa et al. (2019)
    def pcc2(self, x1, x2, dt, lag0, lagu):
        # Function to find the next power of 2 greater than or equal to n
        def next_power_of_2(n):
            return 2**(n-1).bit_length()
        # Get the length of the signals
        N = len(x1)
        # Find the next power of 2 greater than or equal to 2N for zero-padding
        Nz = next_power_of_2(2 * N)
        # Compute the analytic signals using Hilbert transform
        xa1 = hilbert(x1)
        xa2 = hilbert(x2)
        # Normalize the analytic signals to obtain unitary phasors
        xa1 = xa1 / np.abs(xa1)
        xa2 = xa2 / np.abs(xa2)
        # Pad the normalized signals with zeros up to length Nz
        xa1 = np.append(xa1, np.zeros((Nz - N), dtype=np.complex128))
        xa2 = np.append(xa2, np.zeros((Nz - N), dtype=np.complex128))
        # Compute the FFT of the zero-padded signals
        xa1 = np.fft.fft(xa1)
        xa2 = np.fft.fft(xa2)
        # Multiply the FFT of xa1 with the complex conjugate of the FFT of xa2
        amp = xa1 * np.conj(xa2)
        # Compute the inverse FFT to get the cross-correlation in the time domain
        pcc = np.real(np.fft.ifft(amp)) / N
        # Shift the zero-frequency component to the center of the spectrum
        pcc = np.fft.ifftshift(pcc)
        # Get the phase cross-correlation at zero lag
        pcc_zero_lag = pcc[len(pcc) // 2]
        # Create the time vector t ranging from -tt to tt
        tt = Nz // 2 * dt
        t = np.arange(-tt, tt, dt)
        # Return the time vector and the PCC values within the specified lag time range
        return t[(t >= lag0) & (t <= lagu)], pcc[(t >= lag0) & (t <= lagu)], pcc_zero_lag
    
    def correlation(self):
        if self.current_project_path is None:
            messagebox.showwarning("SANBA", "No project path detected. Create or load a project to continue.")
            return

        if self.pairs is None:
            messagebox.showwarning("SANBA", "No pair(s) of station(s) detected. Select stations to continue.")
            return

        data_dir = os.path.join(self.current_project_path, "data")
        out_dir = os.path.join(self.current_project_path, "out")
        corr_root = os.path.join(out_dir, "corr")
        os.makedirs(corr_root, exist_ok=True)

        self.progress["value"] = 0
        self.progress["maximum"] = len(self.pairs)

        window_length_samples = int(self.corr_window_size * self.corr_resample_rate)
        window_step = int(window_length_samples * (1 - self.corr_overlap))

        if window_length_samples <= 1:
            messagebox.showwarning("SANBA", "Correlation window is too short for the current resample rate.")
            return

        if window_step <= 0:
            messagebox.showwarning("SANBA", "Invalid overlap value. The resulting window step must be greater than zero.")
            return

        for station1, station2 in self.pairs:
            try:
                # --------------------------------------------------------------
                # Define channel pairs for THIS station pair only
                # --------------------------------------------------------------
                if self.do_crosscomponent_analysis:
                    station1_dir = os.path.join(data_dir, station1)
                    station2_dir = os.path.join(data_dir, station2)

                    if not os.path.isdir(station1_dir):
                        print(f"Station directory not found: {station1_dir}")
                        self.progress["value"] += 1
                        self.progress.update_idletasks()
                        continue

                    if not os.path.isdir(station2_dir):
                        print(f"Station directory not found: {station2_dir}")
                        self.progress["value"] += 1
                        self.progress.update_idletasks()
                        continue

                    channels1 = [
                        item for item in os.listdir(station1_dir)
                        if os.path.isdir(os.path.join(station1_dir, item))
                    ]
                    channels2 = [
                        item for item in os.listdir(station2_dir)
                        if os.path.isdir(os.path.join(station2_dir, item))
                    ]

                    channel_pairs = [
                        (ch1, ch2)
                        for ch1 in channels1
                        for ch2 in channels2
                        if ch1 <= ch2
                    ]
                else:
                    channel_pairs = [(self.channel_code, self.channel_code)]

                # --------------------------------------------------------------
                # Process each channel pair
                # --------------------------------------------------------------
                for channel1, channel2 in channel_pairs:
                    self.status_var.set(
                        f"Running correlation calculation for {station1} {channel1} and {station2} {channel2}"
                    )
                    print(
                        f"Iniciando o método de correlação para {station1} {channel1} e {station2} {channel2}..."
                    )

                    dir1 = os.path.join(data_dir, station1, channel1)
                    dir2 = os.path.join(data_dir, station2, channel2)

                    if not os.path.isdir(dir1):
                        print(f"Channel directory not found: {dir1}")
                        continue

                    if not os.path.isdir(dir2):
                        print(f"Channel directory not found: {dir2}")
                        continue

                    log_filepath = os.path.join(
                        out_dir,
                        f"log_corr_{station1}_{station2}_{channel1}_{channel2}.txt"
                    )

                    if os.path.isfile(log_filepath):
                        with open(log_filepath, "r") as f:
                            excluded_files = set(line.strip() for line in f if line.strip())
                    else:
                        excluded_files = set()

                    files1_all = [f for f in os.listdir(dir1) if f not in excluded_files]
                    files2_all = [f for f in os.listdir(dir2) if f not in excluded_files]

                    if not files1_all or not files2_all:
                        print(f"No files to process for {station1}.{channel1} - {station2}.{channel2}")
                        continue

                    # ----------------------------------------------------------
                    # Find common dates between both stations/channels
                    # ----------------------------------------------------------
                    dates1 = [f.split(".")[-2] + "." + f.split(".")[-1] for f in files1_all]
                    dates2 = [f.split(".")[-2] + "." + f.split(".")[-1] for f in files2_all]

                    matching_dates = natsorted(list(set(dates1) & set(dates2)))

                    if not matching_dates:
                        print(f"No matching dates for {station1}.{channel1} - {station2}.{channel2}")
                        continue

                    print(f"Dias {matching_dates}")

                    files1 = natsorted(
                        [f for f in files1_all if any(date in f for date in matching_dates)]
                    )
                    files2 = natsorted(
                        [f for f in files2_all if any(date in f for date in matching_dates)]
                    )

                    if not files1 or not files2:
                        print(f"No matching waveform files for {station1}.{channel1} - {station2}.{channel2}")
                        continue

                    # ----------------------------------------------------------
                    # Prepare output stream
                    # ----------------------------------------------------------
                    station_pair_path = os.path.join(
                        corr_root,
                        f"{station1}_{station2}_{channel1}_{channel2}"
                    )
                    os.makedirs(station_pair_path, exist_ok=True)

                    mseed_file_path = os.path.join(
                        station_pair_path,
                        f"{station1}_{station2}_{channel1}_{channel2}_corr.mseed"
                    )

                    if os.path.exists(mseed_file_path):
                        corr_stream = read(mseed_file_path, format="MSEED")
                    else:
                        corr_stream = Stream()

                    # ----------------------------------------------------------
                    # Process daily waveform files
                    # ----------------------------------------------------------
                    for file1, file2 in tqdm(
                        zip(files1, files2),
                        total=min(len(files1), len(files2)),
                        desc="Processing files\n"
                    ):
                        try:
                            st1 = read(os.path.join(dir1, file1))
                            st2 = read(os.path.join(dir2, file2))

                            if len(st1) > 1:
                                st1.merge(method=0, fill_value="interpolate")
                            if len(st2) > 1:
                                st2.merge(method=0, fill_value="interpolate")

                            # Common overlapping time interval
                            common_start = max(st1[0].stats.starttime, st2[0].stats.starttime)
                            common_end = min(st1[0].stats.endtime, st2[0].stats.endtime)

                            if common_start >= common_end:
                                print(f"Skipping {file1} and {file2}: no overlap in time.")
                                del st1, st2
                                gc.collect()
                                continue

                            st1.trim(common_start, common_end)
                            st2.trim(common_start, common_end)

                            # Pre-processing
                            if self.corr_remove_mean:
                                st1.detrend("demean")
                                st2.detrend("demean")

                            if self.corr_remove_trend:
                                st1.detrend("linear")
                                st2.detrend("linear")

                            if self.corr_taper:
                                st1.taper(max_percentage=0.05, type="cosine")
                                st2.taper(max_percentage=0.05, type="cosine")

                            if self.corr_bandpass_filter:
                                st1.filter(
                                    "bandpass",
                                    freqmin=self.corr_min_freq,
                                    freqmax=self.corr_max_freq,
                                    zerophase=True
                                )
                                st2.filter(
                                    "bandpass",
                                    freqmin=self.corr_min_freq,
                                    freqmax=self.corr_max_freq,
                                    zerophase=True
                                )

                            if self.corr_onebit_norm:
                                st1[0].data = np.sign(st1[0].data)
                                st2[0].data = np.sign(st2[0].data)

                            if self.corr_spectral_whitening:
                                st1[0].data = self.spectral_whitening(
                                    st1[0].data,
                                    st1[0].stats.delta,
                                    self.corr_min_freq,
                                    self.corr_max_freq
                                )
                                st2[0].data = self.spectral_whitening(
                                    st2[0].data,
                                    st2[0].stats.delta,
                                    self.corr_min_freq,
                                    self.corr_max_freq
                                )

                            # Resample
                            st1[0].interpolate(
                                sampling_rate=self.corr_resample_rate,
                                method="lanczos",
                                a=1.0
                            )
                            st2[0].interpolate(
                                sampling_rate=self.corr_resample_rate,
                                method="lanczos",
                                a=1.0
                            )

                            if len(st1[0].data) < window_length_samples or len(st2[0].data) < window_length_samples:
                                print(f"Skipping {file1} and {file2}: trace shorter than correlation window.")
                                del st1, st2
                                gc.collect()
                                continue

                            # --------------------------------------------------
                            # Moving-window correlation
                            # --------------------------------------------------
                            for n in trange(
                                0,
                                len(st1[0].data) - window_length_samples + 1,
                                window_step,
                                desc="Processing windows\n"
                            ):
                                try:
                                    window1 = st1[0].data[n:n + window_length_samples]
                                    window2 = st2[0].data[n:n + window_length_samples]

                                    if self.correlation_method == "cc":
                                        _, correlation, _ = self.cc(
                                            window1,
                                            window2,
                                            1 / self.corr_resample_rate,
                                            -self.corr_max_lag,
                                            self.corr_max_lag
                                        )
                                    elif self.correlation_method == "pcc":
                                        _, correlation, _ = self.pcc2(
                                            window1,
                                            window2,
                                            1 / self.corr_resample_rate,
                                            -self.corr_max_lag,
                                            self.corr_max_lag
                                        )
                                    else:
                                        raise ValueError(f"Unknown correlation method: {self.correlation_method}")

                                    if isnan(correlation).any() or isinf(correlation).any():
                                        continue

                                    # --------------------------------------------------
                                    # Midpoint timestamp of the original time window
                                    # --------------------------------------------------
                                    window_start_time = st1[0].stats.starttime + (n / self.corr_resample_rate)
                                    window_mid_time = window_start_time + (self.corr_window_size / 2.0)

                                    if len(corr_stream) > 0 and window_mid_time <= corr_stream[-1].stats.starttime:
                                        continue

                                    corr_trace = Trace(data=np.asarray(correlation, dtype=np.float32))
                                    corr_trace.stats.starttime = window_mid_time
                                    corr_trace.stats.sampling_rate = self.corr_resample_rate

                                    corr_stream.append(corr_trace)
                                    #print(corr_trace.stats.starttime)

                                except Exception as e:
                                    print(f"Error processing window in {file1} / {file2}: {e}")
                                    continue

                            # Log processed files
                            with open(log_filepath, "a") as f:
                                f.write(file1 + "\n")
                                f.write(file2 + "\n")

                            del st1, st2
                            gc.collect()

                        except Exception as e:
                            print(f"Deu ruim no {file1} e {file2}")
                            print(e)
                            continue

                    # ----------------------------------------------------------
                    # Save and plot results
                    # ----------------------------------------------------------
                    if len(corr_stream) > 0:
                        corr_stream.write(mseed_file_path, format="MSEED", dtype="float32")

                        if self.corr_plot:
                            self.ax.clear()

                            n_traces = len(corr_stream)
                            n_samples = len(corr_stream[0].data)
                            data = np.zeros((n_traces, n_samples), dtype=float)

                            for i, tr in enumerate(corr_stream):
                                max_amp = np.max(np.abs(tr.data))
                                if max_amp > 0:
                                    data[i, :] = tr.data / max_amp
                                else:
                                    data[i, :] = tr.data

                            start_times = [tr.stats.starttime.datetime for tr in corr_stream]
                            end_times = [tr.stats.endtime.datetime for tr in corr_stream]
                            lag = np.linspace(-self.corr_max_lag, self.corr_max_lag, n_samples)

                            self.ax.imshow(
                                data,
                                aspect="auto",
                                cmap="seismic",
                                origin="lower",
                                interpolation="bilinear",
                                extent=[
                                    lag[0],
                                    lag[-1],
                                    mdates.date2num(start_times[0]),
                                    mdates.date2num(end_times[-1])
                                ]
                            )

                            self.ax.set_title(
                                f"Correlation functions over time | "
                                f"{station1}.{channel1} - {station2}.{channel2} | "
                                f"{self.corr_min_freq} - {self.corr_max_freq} Hz"
                            )
                            self.ax.yaxis_date()
                            self.ax.yaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y %H:%M"))
                            self.ax.set_xlabel("Time lag (s)")
                            self.ax.set_ylabel("Time (dd/mm/yyyy hh:mm)")

                            self.ax2.set_ylabel("")
                            self.ax2.set_yticks([])
                            self.ax2.tick_params(right=False, labelright=False)

                            self.ax.figure.canvas.draw()

                            self.fig.savefig(
                                os.path.join(
                                    station_pair_path,
                                    f"{station1}_{station2}_{channel1}_{channel2}_corr.png"
                                ),
                                dpi=300
                            )

                    self.status_var.set(
                        f"Correlation calculation for {station1} {channel1} and {station2} {channel2} completed"
                    )

                self.progress["value"] += 1
                self.progress.update_idletasks()

            except Exception as e:
                print(f"Error processing pair {station1} - {station2}: {e}")
                self.progress["value"] += 1
                self.progress.update_idletasks()
                continue

    def stack(self):
        if self.current_project_path is None:
            messagebox.showwarning("SANBA", "No project path detected. Create or load a project to continue.")
            return

        if self.pairs is None:
            messagebox.showwarning("SANBA", "No pair(s) of station(s) detected. Select stations to continue.")
            return

        data_dir = os.path.join(self.current_project_path, "data")
        out_dir = os.path.join(self.current_project_path, "out")
        corr_root = os.path.join(out_dir, "corr")
        stack_root = os.path.join(out_dir, "stack")
        os.makedirs(stack_root, exist_ok=True)

        self.progress["value"] = 0
        self.progress["maximum"] = len(self.pairs)

        for station1, station2 in self.pairs:
            try:
                # --------------------------------------------------------------
                # Define channel pairs for THIS station pair only
                # --------------------------------------------------------------
                if self.do_crosscomponent_analysis:
                    station1_dir = os.path.join(data_dir, station1)
                    station2_dir = os.path.join(data_dir, station2)

                    if not os.path.isdir(station1_dir):
                        print(f"Station directory not found: {station1_dir}")
                        self.progress["value"] += 1
                        self.progress.update_idletasks()
                        continue

                    if not os.path.isdir(station2_dir):
                        print(f"Station directory not found: {station2_dir}")
                        self.progress["value"] += 1
                        self.progress.update_idletasks()
                        continue

                    channels1 = [
                        item for item in os.listdir(station1_dir)
                        if os.path.isdir(os.path.join(station1_dir, item))
                    ]
                    channels2 = [
                        item for item in os.listdir(station2_dir)
                        if os.path.isdir(os.path.join(station2_dir, item))
                    ]

                    channel_pairs = [
                        (ch1, ch2)
                        for ch1 in channels1
                        for ch2 in channels2
                        if ch1 <= ch2
                    ]
                else:
                    channel_pairs = [(self.channel_code, self.channel_code)]

                # --------------------------------------------------------------
                # Process each channel pair
                # --------------------------------------------------------------
                for channel1, channel2 in channel_pairs:
                    self.status_var.set(
                        f"Running stacking for {station1} {channel1} and {station2} {channel2}"
                    )
                    print(f"Iniciando o método stack para {station1} {channel1} e {station2} {channel2}...")

                    pair_name = f"{station1}_{station2}_{channel1}_{channel2}"
                    pair_stack_dir = os.path.join(stack_root, pair_name)
                    os.makedirs(pair_stack_dir, exist_ok=True)

                    corr_path = os.path.join(corr_root, pair_name, f"{pair_name}_corr.mseed")
                    stack_mseed_file_path = os.path.join(pair_stack_dir, f"{pair_name}_stacks.mseed")

                    if not os.path.exists(corr_path):
                        print(f"Correlation file not found: {corr_path}")
                        continue

                    try:
                        corr_stream = read(corr_path, format="MSEED")
                    except Exception as e:
                        print(f"Error reading correlation file {corr_path}: {e}")
                        continue

                    if len(corr_stream) == 0:
                        print(f"No correlation traces found in {corr_path}")
                        continue

                    corr_stream.sort(keys=["starttime"])

                    # ----------------------------------------------------------
                    # Load existing stacks if present
                    # ----------------------------------------------------------
                    if os.path.exists(stack_mseed_file_path):
                        try:
                            stacks_stream = read(stack_mseed_file_path, format="MSEED")
                            stacks_stream.sort(keys=["starttime"])
                        except Exception as e:
                            print(f"Error reading existing stacks file {stack_mseed_file_path}: {e}")
                            stacks_stream = Stream()
                    else:
                        stacks_stream = Stream()

                    # Timestamp of the last stack that already existed BEFORE this run
                    if len(stacks_stream) > 0:
                        existing_last_stack_time = stacks_stream[-1].stats.starttime
                    else:
                        existing_last_stack_time = None

                    # ----------------------------------------------------------
                    # Estimate correlation time step from timestamps
                    # ----------------------------------------------------------
                    if len(corr_stream) >= 2:
                        corr_dt_seconds = float(
                            corr_stream[1].stats.starttime - corr_stream[0].stats.starttime
                        )
                    else:
                        corr_dt_seconds = float(self.corr_window_size * (1.0 - self.corr_overlap))

                    if corr_dt_seconds <= 0:
                        corr_dt_seconds = float(self.corr_window_size * (1.0 - self.corr_overlap))

                    # Approximate number of traces needed to cover the requested stack window
                    approx_window_length = max(
                        2,
                        int(np.ceil(self.stack_window_length_days * 86400.0 / corr_dt_seconds))
                    )

                    new_trace_added = False
                    last_consumed_index = -1

                    # ----------------------------------------------------------
                    # Build moving stacks
                    # ----------------------------------------------------------
                    for i in tqdm(range(len(corr_stream)), desc="Processing windows for stacking\n"):
                        try:
                            window_range = corr_stream[i:min(i + approx_window_length, len(corr_stream))]

                            if len(window_range) < 2:
                                continue

                            # Trim so actual time span does not exceed stack_window_length_days
                            while len(window_range) >= 2:
                                time_difference_days = float(
                                    window_range[-1].stats.starttime - window_range[0].stats.starttime
                                ) / 86400.0

                                if time_difference_days <= self.stack_window_length_days:
                                    break

                                window_range = window_range[:-1]

                            if len(window_range) < 2:
                                continue

                            # Midpoint timestamp of the stacked window
                            first_time = window_range[0].stats.starttime
                            last_time = window_range[-1].stats.starttime
                            midpoint_time = first_time + (last_time - first_time) / 2.0

                            # Skip only stacks that were already generated in previous runs
                            if existing_last_stack_time is not None and midpoint_time <= existing_last_stack_time:
                                continue

                            correlations = [tr.data for tr in window_range]
                            avg_correlation = np.mean(correlations, axis=0)

                            new_trace = Trace(data=np.asarray(avg_correlation, dtype=np.float32))
                            new_trace.stats.starttime = midpoint_time
                            new_trace.stats.sampling_rate = self.corr_resample_rate
                            stacks_stream.append(new_trace)

                            new_trace_added = True

                            # Keep enough tail for future runs:
                            # traces strictly before the last trace used here can be dropped,
                            # but keep the last trace because it may participate in future stacks
                            current_last_index = i + len(window_range) - 1
                            last_consumed_index = max(last_consumed_index, current_last_index - 1)

                            del correlations
                            del window_range
                            gc.collect()

                        except Exception as e:
                            print(f"Error while stacking {pair_name} at window index {i}: {e}")
                            continue

                    # ----------------------------------------------------------
                    # Keep only the residual tail of corr_stream for future runs
                    # ----------------------------------------------------------
                    if last_consumed_index >= 0:
                        residual_start = last_consumed_index + 1
                        corr_stream = corr_stream[residual_start:]

                    # ----------------------------------------------------------
                    # Save results
                    # ----------------------------------------------------------
                    if new_trace_added:
                        stacks_stream.sort(keys=["starttime"])
                        corr_stream.sort(keys=["starttime"])

                        stacks_stream.write(
                            stack_mseed_file_path,
                            format="MSEED",
                            mode="w",
                            dtype="float32"
                        )

                        corr_stream.write(
                            corr_path,
                            format="MSEED",
                            dtype="float32"
                        )

                        if self.stack_plot and len(stacks_stream) > 0:
                            self.ax.clear()
                            self.ax2.clear()

                            n_traces = len(stacks_stream)
                            n_samples = len(stacks_stream[0].data)
                            data = np.zeros((n_traces, n_samples), dtype=float)

                            for j, tr in enumerate(stacks_stream):
                                max_amp = np.max(np.abs(tr.data))
                                if max_amp > 0:
                                    data[j, :] = tr.data / max_amp
                                else:
                                    data[j, :] = tr.data

                            start_times = [tr.stats.starttime.datetime for tr in stacks_stream]
                            end_times = [tr.stats.endtime.datetime for tr in stacks_stream]
                            lag = np.linspace(-self.corr_max_lag, self.corr_max_lag, n_samples)

                            self.ax.imshow(
                                data,
                                aspect="auto",
                                cmap="seismic",
                                origin="lower",
                                interpolation="bilinear",
                                extent=[
                                    lag[0],
                                    lag[-1],
                                    mdates.date2num(start_times[0]),
                                    mdates.date2num(end_times[-1])
                                ]
                            )

                            self.ax.set_title(
                                f"Stacked correlation functions over time | "
                                f"{station1}.{channel1} - {station2}.{channel2} | "
                                f"{self.stack_window_length_days:.1f} day(s)"
                            )

                            self.ax.yaxis_date()
                            self.ax.yaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y %H:%M"))
                            self.ax.set_xlabel("Time lag (s)")
                            self.ax.set_ylabel("Time (dd/mm/yyyy hh:mm)")

                            self.ax2.set_ylabel("")
                            self.ax2.set_yticks([])
                            self.ax2.tick_params(right=False, labelright=False)

                            self.fig.canvas.draw()

                            self.fig.savefig(
                                os.path.join(pair_stack_dir, f"{pair_name}_stack.png"),
                                dpi=300
                            )

                    self.status_var.set(
                        f"Stacking for {station1} {channel1} and {station2} {channel2} completed"
                    )

                self.progress["value"] += 1
                self.progress.update_idletasks()

            except Exception as e:
                print(f"Error processing pair {station1} - {station2}: {e}")
                self.progress["value"] += 1
                self.progress.update_idletasks()
                continue

    def moving_window_crosscorrelation(self, s1, s2, fs, window_length, step_size):
        # Convert window length and step size from seconds to samples
        window_samples = int(window_length * fs)
        step_samples = int(step_size * fs)
        
        # Initialize an empty list to store crosscorrelation results and central lag times
        zero_lag_correlations = []
        central_lags = []
        
        # Calculate the number of windows
        num_windows = int((len(s1) - window_samples) / step_samples) + 1
        
        for i in range(num_windows):
            start_idx = i * step_samples
            end_idx = start_idx + window_samples
            
            # Extract the window segments from both signals
            window_s1 = s1[start_idx:end_idx]
            window_s2 = s2[start_idx:end_idx]

            if self.mwcs_similarity_method == "zero_lag_pcc":
                timevec, correlation, corr_zero_lag = self.pcc2(window_s1, window_s2, 1/self.corr_resample_rate, -self.corr_max_lag, self.corr_max_lag)
            elif self.mwcs_similarity_method == "zero_lag_cc":
                timevec, correlation, corr_zero_lag = self.cc(window_s1, window_s2, 1/self.corr_resample_rate, -self.corr_max_lag, self.corr_max_lag)
                            
            zero_lag_correlations.append(corr_zero_lag)
            
            # Calculate the central lag time of the current window
            central_lag = (start_idx + end_idx) / (2 * fs) - len(s1) / (2 * fs)
            central_lags.append(central_lag)
        
        # Convert lists to numpy arrays for easier manipulation
        zero_lag_correlations = np.array(zero_lag_correlations)
        central_lags = np.array(central_lags)

        # Dictionary to store mean zero lag crosscorrelation values for opposite central lags
        mean_zero_lag_correlations = {}

        # Loop through central lags and find opposite pairs
        for lag in central_lags:
            if lag not in mean_zero_lag_correlations:
                opposite_lag = -lag
                # Find indices for the current lag and its opposite
                indices_lag = np.where(np.isclose(central_lags, lag, atol=0.1))[0]
                indices_opposite_lag = np.where(np.isclose(central_lags, opposite_lag, atol=0.1))[0]
                # Check if both lags have corresponding entries
                if indices_lag.size > 0 and indices_opposite_lag.size > 0:
                    combined_indices = np.concatenate((indices_lag, indices_opposite_lag))
                    mean_zero_lag_correlation = np.mean(zero_lag_correlations[combined_indices], axis=0)
                    #mean_zero_lag_correlations[(lag, opposite_lag)] = mean_zero_lag_correlation
                    if np.round(abs(lag),2) >= self.mwcs_lagtime_ballistic and np.round(abs(lag),2) <= self.mwcs_lagtime_max:
                        mean_zero_lag_correlations[np.round(abs(lag),2)] = mean_zero_lag_correlation
        
        return mean_zero_lag_correlations#np.array(abs(central_lags)), mean_zero_lag_correlations

    # ------------------------------------------------------------------
    # MWCS implementation adapted from MSNoise
    # Original logic based on the MWCS method implementation available in
    # the MSNoise package. Rewritten here in a self-contained form.
    # ------------------------------------------------------------------

    def _nextpow2(self, n):
        """Return p such that 2**p is the next power of two >= n."""
        if n < 1:
            return 0
        return int(np.ceil(np.log2(n)))


    def _mwcs_smooth(self, x, window="boxcar", half_win=3):
        """
        Smooth a 1D array using a symmetric window.

        This helper follows the logic used in the MSNoise MWCS implementation.
        """
        window_len = 2 * half_win + 1

        if window_len < 3:
            return x.copy()

        # Reflect signal at both ends to reduce border effects
        s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]

        if window == "boxcar":
            w = np.ones(window_len, dtype=complex)
        else:
            # Equivalent to scipy.signal.windows.hann(window_len)
            w = np.hanning(window_len).astype(complex)

        y = np.convolve(w / w.sum(), s, mode="valid")
        return y[half_win:len(y) - half_win]


    def _mwcs_get_coherence(self, dcs, ds1, ds2):
        """
        Compute coherence from cross-spectrum amplitude and auto-spectrum amplitudes.

        This helper follows the logic used in the MSNoise MWCS implementation.
        """
        n = len(dcs)
        coh = np.zeros(n, dtype=complex)

        valid = np.argwhere(np.logical_and(np.abs(ds1) > 0, np.abs(ds2) > 0))
        coh[valid] = dcs[valid] / (ds1[valid] * ds2[valid])
        coh[coh > (1.0 + 0j)] = 1.0 + 0j

        return coh


    def mwcs(self, current, reference, freqmin, freqmax, df, tmin, window_length,
             step, smoothing_half_win=5):
        """
        Moving-Window Cross-Spectral (MWCS) analysis.
        This implementation is adapted from the MSNoise package and rewritten
        here in a self-contained form for direct use.

        """
        current = np.asarray(current, dtype=float)
        reference = np.asarray(reference, dtype=float)

        if current.ndim != 1 or reference.ndim != 1:
            raise ValueError("current and reference must be 1D arrays.")
        if len(current) != len(reference):
            raise ValueError("current and reference must have the same length.")
        if df <= 0:
            raise ValueError("df must be positive.")
        if freqmin <= 0 or freqmax <= 0 or freqmin >= freqmax:
            raise ValueError("Require 0 < freqmin < freqmax.")
        if window_length <= 0 or step <= 0:
            raise ValueError("window_length and step must be positive.")

        delta_t = []
        delta_err = []
        delta_mcoh = []
        time_axis = []

        window_length_samples = int(window_length * df)
        step_samples = int(step * df)

        if window_length_samples < 2:
            raise ValueError("window_length is too short for the given sampling rate.")
        if step_samples < 1:
            raise ValueError("step is too short for the given sampling rate.")

        # padd = 2 ** (nextpow2(window_length_samples) + 2)
        padd = int(2 ** (self._nextpow2(window_length_samples) + 2))

        # taper
        tp = cosine_taper(window_length_samples, 0.85)

        minind = 0
        maxind = window_length_samples
        count = 0

        while maxind <= len(current):
            # Slice current and reference windows
            cci = current[minind:minind + window_length_samples].copy()
            cri = reference[minind:minind + window_length_samples].copy()

            # Detrend and taper
            cci = detrend(cci, type="linear")
            cri = detrend(cri, type="linear")
            cci *= tp
            cri *= tp

            # Advance indices for next loop
            minind += step_samples
            maxind += step_samples

            # FFT (positive frequencies only)
            fcur = np.fft.fft(cci, n=padd)[:padd // 2]
            fref = np.fft.fft(cri, n=padd)[:padd // 2]

            # Power spectra
            fcur2 = np.real(fcur) ** 2 + np.imag(fcur) ** 2
            fref2 = np.real(fref) ** 2 + np.imag(fref) ** 2

            # Cross-spectrum
            X = fref * np.conj(fcur)

            # Optional smoothing, following MSNoise logic
            if smoothing_half_win != 0:
                dcur = np.sqrt(self._mwcs_smooth(fcur2, window="hanning",
                                                 half_win=smoothing_half_win))
                dref = np.sqrt(self._mwcs_smooth(fref2, window="hanning",
                                                 half_win=smoothing_half_win))
                X = self._mwcs_smooth(X, window="hanning",
                                      half_win=smoothing_half_win)
            else:
                dcur = np.sqrt(fcur2)
                dref = np.sqrt(fref2)

            dcs = np.abs(X)

            # Frequency vector
            freq_vec = np.fft.fftfreq(len(X) * 2, d=1.0 / df)[:padd // 2]

            # Frequency range of interest
            index_range = np.argwhere(
                np.logical_and(freq_vec >= freqmin, freq_vec <= freqmax)
            )

            if index_range.size == 0:
                continue

            # Coherence and mean coherence
            coh = self._mwcs_get_coherence(dcs, dref, dcur)
            mcoh = np.mean(coh[index_range])

            # Weights from MSNoise / Clarke et al. formulation
            w = 1.0 / (1.0 / (coh[index_range] ** 2) - 1.0)
            w[coh[index_range] >= 0.99] = 1.0 / (1.0 / 0.9801 - 1.0)
            w = np.sqrt(w * np.sqrt(dcs[index_range]))
            w = np.real(w)

            # Angular frequency
            v = np.real(freq_vec[index_range]) * 2.0 * np.pi

            # Unwrapped phase
            phi = np.angle(X)
            phi[0] = 0.0
            phi = np.unwrap(phi)
            phi = phi[index_range]

            # Weighted linear regression
            # Uses the same linear_regression function you already import
            m, em = linear_regression(v.flatten(), phi.flatten(), w.flatten())

            # Delay time
            delta_t.append(m)

            # Error estimate, following MSNoise logic
            e = np.sum((phi - m * v) ** 2) / (np.size(v) - 1)
            s2x2 = np.sum(v ** 2 * w ** 2)
            sx2 = np.sum(w * v ** 2)
            e = np.sqrt(e * s2x2 / sx2 ** 2)

            delta_err.append(e)
            delta_mcoh.append(np.real(mcoh))
            time_axis.append(tmin + window_length / 2.0 + count * step)

            count += 1

        return np.array([time_axis, delta_t, delta_err, delta_mcoh]).T
    
    def compute_dvv(self):
        if self.current_project_path is None:
            messagebox.showwarning("SANBA", "No project path detected. Create or load a project to continue.")
            return

        if self.pairs is None:
            messagebox.showwarning("SANBA", "No pair(s) of station(s) detected. Select stations to continue.")
            return

        data_dir = os.path.join(self.current_project_path, "data")
        out_dir = os.path.join(self.current_project_path, "out")
        stack_root = os.path.join(out_dir, "stack")
        dvv_root = os.path.join(out_dir, "dvv")
        os.makedirs(dvv_root, exist_ok=True)

        self.progress["value"] = 0
        self.progress["maximum"] = len(self.pairs)

        for station1, station2 in self.pairs:
            try:
                # --------------------------------------------------------------
                # Define channel pairs for THIS station pair only
                # --------------------------------------------------------------
                if self.do_crosscomponent_analysis:
                    station1_dir = os.path.join(data_dir, station1)
                    station2_dir = os.path.join(data_dir, station2)

                    if not os.path.isdir(station1_dir):
                        print(f"Station directory not found: {station1_dir}")
                        self.progress["value"] += 1
                        self.progress.update_idletasks()
                        continue

                    if not os.path.isdir(station2_dir):
                        print(f"Station directory not found: {station2_dir}")
                        self.progress["value"] += 1
                        self.progress.update_idletasks()
                        continue

                    channels1 = [
                        item for item in os.listdir(station1_dir)
                        if os.path.isdir(os.path.join(station1_dir, item))
                    ]
                    channels2 = [
                        item for item in os.listdir(station2_dir)
                        if os.path.isdir(os.path.join(station2_dir, item))
                    ]

                    channel_pairs = [
                        (ch1, ch2)
                        for ch1 in channels1
                        for ch2 in channels2
                        if ch1 <= ch2
                    ]
                else:
                    channel_pairs = [(self.channel_code, self.channel_code)]

                # --------------------------------------------------------------
                # Process each channel pair
                # --------------------------------------------------------------
                for channel1, channel2 in channel_pairs:
                    pair_name = f"{station1}_{station2}_{channel1}_{channel2}"

                    self.status_var.set(
                        f"Running the MWCS method for {station1} {channel1} and {station2} {channel2}"
                    )
                    print(
                        f"Iniciando o método mwcs para {station1} {channel1} e {station2} {channel2} "
                        f"({self.mwcs_freq_min}-{self.mwcs_freq_max}Hz)..."
                    )

                    stack_path = os.path.join(stack_root, pair_name)
                    dvv_path = os.path.join(dvv_root, pair_name)
                    os.makedirs(dvv_path, exist_ok=True)

                    stack_file = os.path.join(stack_path, f"{pair_name}_stacks.mseed")
                    log_file = os.path.join(
                        out_dir,
                        f"log_mwcs_{pair_name}_{self.mwcs_freq_min}-{self.mwcs_freq_max}Hz.txt"
                    )
                    csv_file = os.path.join(
                        dvv_path,
                        f"{pair_name}_{self.mwcs_freq_min}-{self.mwcs_freq_max}Hz_dvv.csv"
                    )

                    if not os.path.exists(stack_file):
                        print(f"Stack file not found: {stack_file}")
                        continue

                    try:
                        stack_stream = read(stack_file, format="MSEED")
                    except Exception as e:
                        print(f"Error reading stack file {stack_file}: {e}")
                        continue

                    if len(stack_stream) == 0:
                        print(f"No stack traces found in {stack_file}")
                        continue

                    stack_stream.sort(keys=["starttime"])

                    # ----------------------------------------------------------
                    # Load processed timestamps log
                    # ----------------------------------------------------------
                    if not os.path.isfile(log_file):
                        with open(log_file, "w") as lf:
                            lf.write("")

                    with open(log_file, "r") as lf:
                        processed_timestamps = set(line.strip() for line in lf if line.strip())

                    # ----------------------------------------------------------
                    # Define static / mean reference if needed
                    # ----------------------------------------------------------
                    if self.mwcs_reference == "static":
                        reference_correlation = stack_stream[0].data
                    elif self.mwcs_reference == "mean":
                        reference_correlation = np.mean(
                            [trace.data for trace in stack_stream], axis=0
                        )
                    elif self.mwcs_reference == "following":
                        reference_correlation = None
                    else:
                        print(f"Unknown MWCS reference mode: {self.mwcs_reference}")
                        continue

                    # ----------------------------------------------------------
                    # Accumulate results in memory, then persist safely
                    # ----------------------------------------------------------
                    result_rows = []
                    processed_now = []

                    for i in tqdm(range(len(stack_stream)), desc="Processing traces for mwcs\n"):
                        try:
                            tr = stack_stream[i]
                            trace_timestamp_utc = pd.Timestamp(
                                tr.stats.starttime.datetime, tz="UTC"
                            ).strftime("%Y-%m-%dT%H:%M:%SZ")

                            if trace_timestamp_utc in processed_timestamps:
                                continue

                            current_data = tr.data
                            fs = float(tr.stats.sampling_rate)

                            if self.mwcs_reference == "following":
                                if i == 0:
                                    reference_correlation = stack_stream[0].data
                                else:
                                    reference_correlation = stack_stream[i - 1].data

                            # --------------------------------------------------
                            # Optional single-value similarity
                            # --------------------------------------------------
                            corr_zero_lag = np.nan
                            if self.mwcs_do_similarity_analysis:
                                if self.mwcs_similarity_method == "zero_lag_cc":
                                    _, _, corr_zero_lag = self.cc(
                                        current_data,
                                        reference_correlation,
                                        1 / fs,
                                        -self.corr_max_lag,
                                        self.corr_max_lag
                                    )
                                elif self.mwcs_similarity_method == "zero_lag_pcc":
                                    _, _, corr_zero_lag = self.pcc2(
                                        current_data,
                                        reference_correlation,
                                        1 / fs,
                                        -self.corr_max_lag,
                                        self.corr_max_lag
                                    )
                                else:
                                    print(f"Unknown similarity method: {self.mwcs_similarity_method}")
                                    corr_zero_lag = np.nan

                            # --------------------------------------------------
                            # MWCS
                            # --------------------------------------------------
                            mwcs_data = self.mwcs(
                                current=current_data,
                                reference=reference_correlation,
                                df=fs,
                                freqmin=self.mwcs_freq_min,
                                freqmax=self.mwcs_freq_max,
                                tmin=self.mwcs_moving_start,
                                window_length=self.mwcs_window_length,
                                step=self.mwcs_window_step
                            )

                            if mwcs_data.size == 0:
                                continue

                            time_axis = mwcs_data[:, 0]
                            delay_time = mwcs_data[:, 1]
                            err = mwcs_data[:, 2]
                            coh = mwcs_data[:, 3]

                            # --------------------------------------------------
                            # Filter MWCS points
                            # --------------------------------------------------
                            mask = (
                                (np.abs(time_axis) >= self.mwcs_lagtime_ballistic)
                                & (np.abs(time_axis) <= (self.mwcs_lagtime_max - (self.mwcs_window_length / 2.0)))
                            )
                            mask &= (coh >= self.mwcs_coherency_min)
                            mask &= (err <= self.mwcs_error_max)
                            mask &= (np.abs(delay_time) <= self.mwcs_abs_delay_time_limit)

                            time_axis_filtered = time_axis[mask]
                            delay_time_filtered = delay_time[mask]
                            err_filtered = err[mask]

                            if len(time_axis_filtered) < 2:
                                continue

                            finite_mask = (
                                np.isfinite(time_axis_filtered)
                                & np.isfinite(delay_time_filtered)
                                & np.isfinite(err_filtered)
                            )
                            time_axis_filtered = time_axis_filtered[finite_mask]
                            delay_time_filtered = delay_time_filtered[finite_mask]
                            err_filtered = err_filtered[finite_mask]

                            if len(time_axis_filtered) < 2:
                                continue

                            positive_err_mask = err_filtered > 0
                            time_axis_filtered = time_axis_filtered[positive_err_mask]
                            delay_time_filtered = delay_time_filtered[positive_err_mask]
                            err_filtered = err_filtered[positive_err_mask]

                            if len(time_axis_filtered) < 2:
                                continue

                            # --------------------------------------------------
                            # Weighted linear regression
                            # --------------------------------------------------
                            weights = 1.0 / err_filtered

                            slope, intercept, std, npts = linear_regression(
                                time_axis_filtered,
                                delay_time_filtered,
                                weights=weights,
                                intercept_origin=False
                            )

                            if (
                                np.isnan(slope) or np.isinf(slope)
                                or np.isnan(std) or np.isinf(std)
                            ):
                                print("regressão linear falhou")
                                continue

                            dvv = -100.0 * slope
                            dvv_std = 100.0 * std

                            row = {
                                "timestamp": trace_timestamp_utc,
                                "dvv": dvv,
                                "dvv_std": dvv_std,
                            }

                            if self.mwcs_do_similarity_analysis:
                                row["similarity"] = corr_zero_lag

                            result_rows.append(row)
                            processed_now.append(trace_timestamp_utc)

                            # --------------------------------------------------
                            # Optional MWCS diagnostic plot
                            # --------------------------------------------------
                            if self.mwcs_plot:
                                self.ax.clear()
                                self.ax2.clear()
                                self.ax2.yaxis.set_label_position("right")
                                self.ax2.yaxis.tick_right()
                                self.ax2.spines["right"].set_visible(True)
                                self.ax2.spines["left"].set_visible(False)
                                self.ax2.tick_params(axis="y", right=True, labelright=True, left=False, labelleft=False)

                                plot_dt = 1.0 / fs
                                n_samples = len(current_data)
                                limit = (n_samples - 1) / (2.0 * fs)
                                timevec = np.linspace(-limit, limit, n_samples)

                                plot_date = pd.to_datetime(trace_timestamp_utc, utc=True)
                                if hasattr(self, "output_timezone") and self.output_timezone:
                                    try:
                                        plot_date = plot_date.tz_convert(self.output_timezone)
                                    except Exception:
                                        pass

                                cer = self.ax.plot(
                                    timevec,
                                    reference_correlation,
                                    lw=2,
                                    c="r",
                                    label="Reference correlation"
                                )
                                cem = self.ax.plot(
                                    timevec,
                                    current_data,
                                    lw=1,
                                    c="k",
                                    label="Moving correlation"
                                )

                                ref_max = np.max(np.abs(reference_correlation))
                                cur_max = np.max(np.abs(current_data))
                                amp_max = max(ref_max, cur_max)

                                if amp_max > 0:
                                    self.ax.set_ylim(-1.25 * amp_max, 1.25 * amp_max)

                                self.ax.set_ylabel("Correlation")

                                delay_line = self.ax2.plot(
                                    time_axis_filtered,
                                    delay_time_filtered,
                                    "o",
                                    c="k",
                                    lw=0,
                                    label="dt"
                                )
                                reg_line = self.ax2.plot(
                                    time_axis_filtered,
                                    slope * time_axis_filtered + intercept,
                                    ls="--",
                                    c="k",
                                    label=f"dv/v = {dvv:.2f}% (±{dvv_std:.3f}%)"
                                )

                                self.ax.set_xlabel("Time lag (s)")
                                self.ax2.set_ylabel("dt (s)")
                                self.ax2.set_ylim(
                                    [
                                        -self.mwcs_abs_delay_time_limit * 1.25,
                                        self.mwcs_abs_delay_time_limit * 1.25
                                    ]
                                )

                                self.ax.set_title(
                                    f"MWCS | {station1}.{channel1} - {station2}.{channel2} | "
                                    f"{plot_date.strftime('%d/%m/%Y %H:%M:%S')} | "
                                    f"{self.mwcs_freq_min} - {self.mwcs_freq_max} Hz"
                                )

                                lines = cer + cem + delay_line + reg_line
                                labels = [line.get_label() for line in lines]
                                self.ax.legend(lines, labels, loc="upper right", fontsize=9)
                                self.ax.grid(True, axis="x", alpha=0.5)

                                self.fig.savefig(
                                    os.path.join(
                                        dvv_path,
                                        f"{pair_name}_{self.mwcs_freq_min}-{self.mwcs_freq_max}Hz_"
                                        f"stack{self.stack_window_length_days}d_mwcs_{i}.png"
                                    ),
                                    format="PNG"
                                )

                                self.ax.figure.canvas.draw()
                                self.ax2.figure.canvas.draw()

                        except Exception as e:
                            print(f"Error processing MWCS for {pair_name}, trace {i}: {e}")
                            continue

                    # ----------------------------------------------------------
                    # Persist results safely
                    # ----------------------------------------------------------
                    if result_rows:
                        results_df = pd.DataFrame(result_rows)

                        if os.path.exists(csv_file):
                            results_df.to_csv(csv_file, mode="a", header=False, index=False)
                        else:
                            results_df.to_csv(csv_file, index=False)

                        # Only after successful CSV write do we mark as processed
                        with open(log_file, "a") as lf:
                            for ts in processed_now:
                                lf.write(ts + "\n")

                    self.status_var.set(
                        f"MWCS for {station1} {channel1} and {station2} {channel2} completed"
                    )

                self.progress["value"] += 1
                self.progress.update_idletasks()

            except Exception as e:
                print(f"Error processing pair {station1} - {station2}: {e}")
                self.progress["value"] += 1
                self.progress.update_idletasks()
                continue

    def plot_dvv_mean(self):
        pass
    
    def plot_dvv(self):
        if self.current_project_path is None:
            messagebox.showwarning(
                "SANBA",
                "No project path detected. Create or load a project to continue."
            )
            return

        if not self.pairs:
            messagebox.showwarning(
                "SANBA",
                "No pair(s) of station(s) detected. Select stations to continue."
            )
            return

        plot_similarity = messagebox.askyesno(
            "SANBA",
            "Plot similarity in second y axis?"
        )
        plot_separately = messagebox.askyesno(
            "SANBA",
            "Plot dv/v separately for each pair of stations?"
        )

        data_dir = os.path.join(self.current_project_path, "data")
        out_dir = os.path.join(self.current_project_path, "out")

        # --------------------------------------------------------------
        # Build full list of pair/channel combinations
        # --------------------------------------------------------------
        pair_channel_list = []

        for station1, station2 in self.pairs:
            if self.do_crosscomponent_analysis:
                station1_dir = os.path.join(data_dir, station1)
                station2_dir = os.path.join(data_dir, station2)

                if not os.path.isdir(station1_dir):
                    print(f"Station directory not found: {station1_dir}")
                    continue

                if not os.path.isdir(station2_dir):
                    print(f"Station directory not found: {station2_dir}")
                    continue

                channels1 = [
                    item for item in os.listdir(station1_dir)
                    if os.path.isdir(os.path.join(station1_dir, item))
                ]
                channels2 = [
                    item for item in os.listdir(station2_dir)
                    if os.path.isdir(os.path.join(station2_dir, item))
                ]

                channel_pairs = [
                    (ch1, ch2)
                    for ch1 in channels1
                    for ch2 in channels2
                    if ch1 <= ch2
                ]

                for channel1, channel2 in channel_pairs:
                    pair_channel_list.append((station1, station2, channel1, channel2))
            else:
                pair_channel_list.append(
                    (station1, station2, self.channel_code, self.channel_code)
                )

        if not pair_channel_list:
            messagebox.showwarning(
                "SANBA",
                "No valid station/channel combinations were found to plot."
            )
            return

        self.progress["value"] = 0
        self.progress["maximum"] = len(pair_channel_list)

        if not plot_separately:
            self.ax.clear()
            self.ax2.clear()

        for station1, station2, channel1, channel2 in pair_channel_list:
            self.status_var.set(
                f"Plotting the dv/v series for {station1} {channel1} and {station2} {channel2}"
            )
            print(
                f"Plotting dv/v for {station1} {channel1} and {station2} {channel2} "
                f"({self.mwcs_freq_min}-{self.mwcs_freq_max} Hz)..."
            )

            pair_name = f"{station1}_{station2}_{channel1}_{channel2}"
            dvv_path = os.path.join(out_dir, "dvv", pair_name)
            csv_file = os.path.join(
                dvv_path,
                f"{pair_name}_{self.mwcs_freq_min}-{self.mwcs_freq_max}Hz_dvv.csv"
            )

            if not os.path.exists(csv_file):
                print(f"CSV file not found: {csv_file}")
                self.progress["value"] += 1
                self.progress.update_idletasks()
                continue

            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"Error reading CSV file {csv_file}: {e}")
                messagebox.showerror(
                    "SANBA",
                    f"Error reading CSV file:\n{csv_file}\n\n{e}"
                )
                self.progress["value"] += 1
                self.progress.update_idletasks()
                continue

            if "timestamp" not in df.columns:
                print(f"'timestamp' column not found in {csv_file}")
                self.progress["value"] += 1
                self.progress.update_idletasks()
                continue

            if "dvv" not in df.columns:
                print(f"'dvv' column not found in {csv_file}")
                self.progress["value"] += 1
                self.progress.update_idletasks()
                continue

            # ----------------------------------------------------------
            # Parse timestamps written by the new compute_dvv():
            # UTC ISO strings like 2026-04-09T15:30:00Z
            # ----------------------------------------------------------
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

                if df["timestamp"].isnull().all():
                    raise ValueError("No valid timestamps could be parsed.")

                df = df.dropna(subset=["timestamp"]).copy()

                try:
                    df["timestamp_local"] = df["timestamp"].dt.tz_convert(self.output_timezone)
                except Exception:
                    print(
                        f"Invalid or unsupported timezone '{self.output_timezone}'. Falling back to UTC."
                    )
                    df["timestamp_local"] = df["timestamp"]

            except Exception as e:
                print(f"Error parsing timestamps in {csv_file}: {e}")
                messagebox.showerror(
                    "SANBA",
                    f"Error parsing timestamps in:\n{csv_file}\n\n{e}"
                )
                self.progress["value"] += 1
                self.progress.update_idletasks()
                continue

            # ----------------------------------------------------------
            # Numeric coercion
            # ----------------------------------------------------------
            df["dvv"] = pd.to_numeric(df["dvv"], errors="coerce")

            if "dvv_std" in df.columns:
                df["dvv_std"] = pd.to_numeric(df["dvv_std"], errors="coerce")

            if "similarity" in df.columns:
                df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce")

            df = df.dropna(subset=["dvv"]).copy()

            if df.empty:
                print(f"No valid dv/v data found in {csv_file}")
                self.progress["value"] += 1
                self.progress.update_idletasks()
                continue

            df = df.sort_values("timestamp").reset_index(drop=True)

            # ----------------------------------------------------------
            # Determine similarity usability
            # ----------------------------------------------------------
            has_similarity = (
                "similarity" in df.columns and df["similarity"].notna().any()
            )
            use_similarity = plot_similarity and has_similarity

            if plot_similarity and not has_similarity:
                print(f"'similarity' column missing or empty in {csv_file}. Similarity will not be plotted.")

            # ----------------------------------------------------------
            # Prepare axes
            # ----------------------------------------------------------
            if plot_separately:
                self.ax.clear()
                self.ax2.clear()

            if use_similarity:
                self.ax2.set_visible(True)
                self.ax2.set_ylabel("Similarity")
                self.ax2.tick_params(right=True, labelright=True)
            else:
                self.ax2.set_visible(False)
                self.ax2.set_ylabel("")
                self.ax2.set_yticks([])
                self.ax2.tick_params(right=False, labelright=False)

            # ----------------------------------------------------------
            # Compute plotted dv/v series
            # ----------------------------------------------------------
            if self.mwcs_reference == "following":
                dvv_plot = df["dvv"].cumsum()
            else:
                dvv_plot = df["dvv"]

            # ----------------------------------------------------------
            # Plot
            # ----------------------------------------------------------
            if plot_separately:
                self.ax.plot(
                    df["timestamp_local"],
                    dvv_plot,
                    label="dv/v"
                )

                if use_similarity:
                    self.ax2.plot(
                        df["timestamp_local"],
                        df["similarity"],
                        ls="--",
                        c="k",
                        label="Similarity"
                    )
            else:
                label = f"{station1} {channel1} - {station2} {channel2}"

                self.ax.plot(
                    df["timestamp_local"],
                    dvv_plot,
                    label=label
                )

                if use_similarity:
                    self.ax2.plot(
                        df["timestamp_local"],
                        df["similarity"],
                        ls="--",
                        label=f"Similarity {label}"
                    )

            # Uncertainty band
            if plot_separately and "dvv_std" in df.columns:
                valid_std = df["dvv_std"].notna()
                if valid_std.any():
                    self.ax.fill_between(
                        df.loc[valid_std, "timestamp_local"],
                        dvv_plot.loc[valid_std] - df.loc[valid_std, "dvv_std"],
                        dvv_plot.loc[valid_std] + df.loc[valid_std, "dvv_std"],
                        alpha=0.25
                    )

            self.ax.set_ylabel("dv/v (%)")
            self.ax.grid(True)
            self.ax.spines["right"].set_visible(False)
            self.ax.spines["top"].set_visible(False)
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y\n%H:%M"))

            min_date = df["timestamp_local"].min()
            max_date = df["timestamp_local"].max()
            self.ax.set_xlim(min_date, max_date)

            if plot_separately:
                self.ax.legend(loc="upper right", fontsize="small")

                if use_similarity:
                    self.ax2.legend(loc="lower right", fontsize="small")

                self.ax.set_title(
                    f"{station1} {channel1} - {station2} {channel2} | "
                    f"{df['timestamp_local'].iloc[0].strftime('%d/%m/%Y')} - "
                    f"{df['timestamp_local'].iloc[-1].strftime('%d/%m/%Y')} | "
                    f"{self.mwcs_freq_min}-{self.mwcs_freq_max} Hz"
                )

                self.fig.savefig(
                    os.path.join(
                        dvv_path,
                        f"{pair_name}_{self.mwcs_freq_min}-{self.mwcs_freq_max}Hz_dvv.png"
                    ),
                    dpi=300
                )
            else:
                self.ax.legend(loc="best", fontsize="small")
                if use_similarity:
                    self.ax2.legend(loc="lower right", fontsize="small")

            self.ax.figure.canvas.draw()

            self.status_var.set(
                f"Completed plotting dv/v series for {station1} {channel1} and "
                f"{station2} {channel2} ({self.mwcs_freq_min}-{self.mwcs_freq_max} Hz)"
            )
            self.progress["value"] += 1
            self.progress.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    app = PSVM(root)
    root.mainloop()
    
