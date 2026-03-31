import sys
import tkinter as tk
from tkinter import ttk
#from ttkthemes import ThemedTk
from tkinter import filedialog
from PIL import ImageTk, Image
import os
from natsort import natsorted
from obspy import read, Trace, Stream
from obspy.core.inventory import read_inventory
from obspy.signal.cross_correlation import correlate
from obspy.signal.filter import bandpass
from matplotlib import pyplot as plt
from obspy.signal.invsim import cosine_taper
from obspy.signal.util import _npts2nfft
from obspy import UTCDateTime
from scipy.fftpack import fft, ifft
from scipy.signal import hilbert
import numpy as np
from msnoise.move2obspy import mwcs as msnoise_mwcs #whiten
from numpy import isnan, isinf, savez
import time
import datetime
from obspy.signal.regression import linear_regression
import pandas as pd
from matplotlib.dates import date2num, num2date
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tqdm import tqdm, trange
import warnings
import gc
from scipy.signal import medfilt
warnings.filterwarnings('ignore')
from obspy.io.mseed.headers import InternalMSEEDWarning
warnings.filterwarnings('ignore', category=InternalMSEEDWarning)
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from io import BytesIO
from re import sub
from scipy.interpolate import griddata, interp2d
#import rasterio
from threading import Thread
import glob
from scipy.signal import correlate, fftconvolve

class PSVM(ttk.Frame):
    def __init__(self, parent: tk.Tk):
        super().__init__(parent)
        self.parent = parent

        self.parent.overrideredirect(False)
        
        self.version = "v1.0.0"

        self.parent.geometry("1024x768")
            
        self.parent.title(f"SANBA | Seismic Ambient Noise-Based Analysis {self.version}")

        style = ttk.Style(self.parent)
        style.theme_use("vista")
        default_font = ("Segoe UI", 10)  # looks modern on Windows; ok elsewhere
        self.parent.option_add("*Font", default_font)
        style.configure("Toolbar.TFrame", padding=(8, 6))
        style.configure("Toolbar.TButton", padding=(8, 6))
        style.configure("Status.TLabel", padding=(10, 6))
        style.configure("Tooltip.TFrame", relief="solid", borderwidth=1)
        style.configure("Tooltip.TLabel")

        self.parent.protocol("WM_DELETE_WINDOW", self.on_closing)

        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.window_ico = ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "icons", "ico_sanba.png")))
        self.parent.iconphoto(False, self.window_ico)

        menubar = tk.Menu(self.parent)
        self.parent.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Create new project path", command=self.create_project)
        file_menu.add_command(label="Load a project path", command=self.load_project)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

        processing_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="Processing", menu=processing_menu)
        processing_menu.add_command(label="Get pairs", command=self.get_pairs)
        processing_menu.add_command(label="Run correlation", command=self.correlation)
        processing_menu.add_command(label="Run stacking", command=self.stack)
        processing_menu.add_command(label="Run MWCS", command=self.mwcs)
        processing_menu.add_separator()
        processing_menu.add_command(label="Run all steps", command=self.run_all)

        plotting_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="Plot", menu=plotting_menu)
        plotting_menu.add_command(label="Plot dv/v", command=self.plot_dvv)
        #plotting_menu.add_command(label="Plot spatially averaged dv/v", command=self.spatial_average)
        #plotting_menu.add_command(label="Plot mean dv/v", command=self.plot_dvv_mean)

        options_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="Options", menu=options_menu)
        #options_menu.add_command(label="Correlation and stacking parameters", command=self.set_parameters_corr_stack)
        #options_menu.add_command(label="MWCS parameters", command=self.set_parameters_mwcs)
        #options_menu.add_command(label="Plot parameters", command=self.ploting_options)
        options_menu.add_command(label="Settings", command=self.options)
        #options_menu.add_command(label="Spatial average parameters", command=self.settings_spatial_average)
        #options_menu.add_separator()
        #options_menu.add_command(label="Load parameters from file", command=self.load_settings_file)

        toolbar_frame = ttk.Frame(self.parent, style="Toolbar.TFrame")
        toolbar_frame.pack(fill="x")

        self.create_ico_img = ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "icons", "ico_new.png")))
        self.open_ico_img = ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "icons", "ico_load.png")))
        self.get_pairs_ico_img = ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "icons", "ico_pair.png")))
        self.corr_ico_img = ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "icons", "ico_corr.png")))
        self.stack_ico_img = ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "icons", "ico_stack.png")))
        self.mwcs_ico_img = ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "icons", "ico_mwcs.png")))
        self.plot_dvv_ico_img = ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "icons", "ico_dvv.png")))
        #self.spatAverage_dvv_ico_img = ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "icons", "spatial_average_ico.png")))
        self.options_ico_img = ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "icons", "ico_options.png")))

        self.create_project_button = ttk.Button(toolbar_frame, image=self.create_ico_img, command=self.create_project,style="Toolbar.TButton")
        self.create_project_button.pack(side="left")

        self.load_project_button = ttk.Button(toolbar_frame, image=self.open_ico_img, command=self.load_project,style="Toolbar.TButton")
        self.load_project_button.pack(side="left")

        self.find_pairs_button = ttk.Button(toolbar_frame, image=self.get_pairs_ico_img, command=self.get_pairs,style="Toolbar.TButton")
        self.find_pairs_button.pack(side="left")

        self.corr_button = ttk.Button(toolbar_frame, image=self.corr_ico_img, command=lambda: Thread(target=self.correlation).start(),style="Toolbar.TButton")
        self.corr_button.pack(side="left")

        self.stack_button = ttk.Button(toolbar_frame, image=self.stack_ico_img, command=lambda: Thread(target=self.stack).start(),style="Toolbar.TButton")
        self.stack_button.pack(side="left")

        self.mwcs_button = ttk.Button(toolbar_frame, image=self.mwcs_ico_img, command=lambda: Thread(target=self.mwcs).start(),style="Toolbar.TButton")
        self.mwcs_button.pack(side="left")

        self.plot_dvv_button = ttk.Button(toolbar_frame, image=self.plot_dvv_ico_img, command=lambda: Thread(target=self.plot_dvv).start(),style="Toolbar.TButton")
        self.plot_dvv_button.pack(side="left")

        #self.plot_spatial_average_dvv_button = ttk.Button(toolbar_frame, image=self.spatAverage_dvv_ico_img, command=lambda: Thread(target=self.spatial_average).start())
        #self.plot_spatial_average_dvv_button.pack(side="left")

        self.options_button = ttk.Button(toolbar_frame, image=self.options_ico_img, command=self.options,style="Toolbar.TButton")
        self.options_button.pack(side="left")

        ttk.Label(toolbar_frame, text="Progress: ").pack(side="left")
        
        self.progress = ttk.Progressbar(toolbar_frame, length=220, mode="determinate")
        self.progress.pack(side="left")

        frame_plot = ttk.Frame(self.parent)
        frame_plot.pack(fill="both", expand="True")

        self.fig = plt.figure()
        
        self.ax = self.fig.add_subplot(111)
        self.ax2 = self.ax.twinx()

        canvas = FigureCanvasTkAgg(self.fig, master=frame_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=1)

        toolbar = NavigationToolbar2Tk(canvas, frame_plot)
        toolbar.update()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=1)

        status_frame = ttk.Frame(self.parent)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, relief="groove", style="Status.TLabel")
        status_label.pack(fill="x")

        self.current_project_path = None

        self.pairs = None

        self.network_code = "AM"
        self.channel_code = "EHZ.D"
        self.do_crosscomponent_analysis = False
        self.corr_sorting_type = "both"#"both" #"individual" #"pairs"
        self.correlation_method = "pcc"#"pcc" #cc
        self.corr_remove_response = False
        self.corr_remove_mean = True
        self.corr_remove_trend = True
        self.corr_taper = True
        self.corr_bandpass_filter = True
        self.corr_onebit_norm = False
        self.corr_spectral_whitening = False
        self.corr_window_size = 3600
        self.corr_overlap = 0
        self.corr_min_freq = 3
        self.corr_max_freq = 11
        self.corr_resample_rate = self.corr_max_freq*2
        self.corr_max_lag = 3
        self.corr_snr_threshold = 0
        self.stack_window_length_days = 1 #1 = 24h/1day

        self.mwcs_reference = "mean" #static,following,mean
        self.mwcs_freq_min = 4#self.corr_min_freq
        self.mwcs_freq_max = 10#self.corr_max_freq
        self.mwcs_window_length = 1
        self.mwcs_window_step = self.mwcs_window_length/5
        self.mwcs_moving_start = -self.corr_max_lag
        self.mwcs_coherency_min = 0.5#0.5
        self.mwcs_error_max = 0.2#0.2
        self.mwcs_lagtime_ballistic = 1#self.corr_max_lag/10
        self.mwcs_lagtime_max = self.corr_max_lag
        self.mwcs_abs_delay_time_limit = 0.1#0.1
        self.mwcs_do_similarity_analysis = False
        self.mwcs_similarity_method = "zero_lag_pcc"#"zero_lag_cc","zero_lag_pcc"

        self.corr_plot = True
        self.stack_plot = True
        self.mwcs_plot = True
        #self.plot_dvv_gap_limit = 999999999#60*60*24*self.stack_window_length_days/2

        #self.spatial_average_median_filter = False
        #self.spatial_average_filter_window_size = "4H"
        #self.spatial_average_gap_limit = self.plot_dvv_gap_limit

    def on_closing(self):
        if tk.messagebox.askyesno("SANBA", "Exit?"):
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

        '''title_label = ttk.Label(
            main_frame,
            text="PSVM Settings",
            font=("TkDefaultFont", 12, "bold")
        )
        title_label.pack(pady=(0, 10))'''

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

        '''ttk.Label(corr_scrollable_frame, text="Minimum SNR tolerance (0 = accept all):").grid(row=24, column=0, sticky="w", padx=corr_padx, pady=corr_pady)
        entry_xcorr_snr = ttk.Entry(corr_scrollable_frame, width=25)
        entry_xcorr_snr.grid(row=24, column=1, sticky="ew", padx=corr_padx, pady=corr_pady)
        entry_xcorr_snr.insert(0, self.corr_snr_threshold)'''

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

                self.status_var.set("New settings saved successfully.")
                tk.messagebox.showinfo("SANBA", "Settings saved successfully.")
                self.top_options.destroy()

            except ValueError as e:
                tk.messagebox.showwarning("SANBA", f"Invalid inputs:\n{e}")
                self.top_options.lift()
                self.top_options.focus_force()

        def cancel():
            self.top_options.destroy()

        ttk.Button(button_frame, text="Cancel", command=cancel, width=18).pack(side="right", padx=(6, 0))
        ttk.Button(button_frame, text="Save settings", command=done, width=18).pack(side="right")

    def run_all(self):
        
        if self.current_project_path == None:
            tk.messagebox.showwarning("SANBA", "No project path detected. Create or load a project to continue.")
            return
            
        if self.pairs == None:
            tk.messagebox.showwarning("SANBA", "No pair(s) of station(s) detected. Select stations to continue.")
            return

        self.correlation()
        self.stack()
        self.mwcs()
        self.plot_dvv()

    def create_project(self):

        directory = filedialog.askdirectory()

        if directory:
            project_name = tk.simpledialog.askstring("SANBA", "Enter the name of the new project:")

            if project_name:
                proj_dir = os.path.join(directory, project_name)
                if os.path.exists(proj_dir):
                    messagebox.showwarning("SANBA", "This project already exists, please enter a different name.")
                    return
                else:
                    os.makedirs(proj_dir, exist_ok=True)
                    data_dir = os.path.join(proj_dir, "data")
                    os.makedirs(data_dir, exist_ok=True)
                    out_dir = os.path.join(proj_dir, "out")
                    os.makedirs(out_dir, exist_ok=True)
                    os.makedirs(os.path.join(out_dir, "corr"), exist_ok=True)
                    os.makedirs(os.path.join(out_dir, "stack"), exist_ok=True)
                    os.makedirs(os.path.join(out_dir, "dvv"), exist_ok=True)

                    self.current_project_path = os.path.abspath(f'projects/{project_name}')
                    self.status_var.set("Finished creating project.")
                    tk.messagebox.showinfo("SANBA", "Project created successfully.")

    def load_project(self):

        project_dir = tk.filedialog.askdirectory(initialdir="projects")

        if project_dir:
            if os.path.exists(project_dir+"/out/corr") and os.path.exists(project_dir+"/out/stack") and os.path.exists(project_dir+"/out/dvv"):
                self.current_project_path = project_dir
                self.status_var.set("Finished loading project.")
                tk.messagebox.showinfo("SANBA", "Project loaded successfully.")
            else:
                tk.messagebox.showwarning("SANBA", "The selected directory is not a valid project.")
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
                            tk.messagebox.showwarning("SANBA", "Current setting for sorting of stations is set to 'pairs' or 'both'. Select at least two stations to continue.")
                            return
                    elif self.corr_sorting_type == "individual":
                        if len(stations2use) < 1:
                            tk.messagebox.showwarning("SANBA", "Current setting for sorting of stations is set to 'individual'. Select at least one station to continue.")
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
                    tk.messagebox.showinfo("SANBA", f"A total of {len(pairs)} pair(s) defined for a total of {len(stations2use)} selected station(s)")
                    self.top_get_pairs.destroy()
                
                ttk.Button(self.top_get_pairs, text="Get pairs", command=done, width=35).pack(pady=5)

            else:
                tk.messagebox.showwarning("SANBA", "No stations were found in the 'data' directory. Add these folders and waveform files to continue.")
            
        else:
            tk.messagebox.showwarning("SANBA", "No project path detected. Create or load a project to continue.")

    def spectral_whitening(self, signal, dt, f1, f2):
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
        return whitened_signal
    
    '''def cc(self, x1, x2, dt, lag0, lagu):
        
        N = len(x1)
        # Compute the FFT of the signals
        x1_fft = np.fft.fft(x1)
        x2_fft = np.fft.fft(x2)
        
        # Compute the cross-correlation using the FFT
        amp = x1_fft * np.conj(x2_fft)
        cc = np.real(np.fft.ifft(amp))
        
        # Normalize the cross-correlation
        #cc /= N
        
        # Reorganize the result to have the correct symmetry
        cc = np.fft.ifftshift(cc)

        # Get the cross-correlation at zero lag
        cc_zero_lag = cc[len(cc) // 2]

        # Create the time vector t ranging from -tt to tt
        tt = N // 2 * dt
        t = np.arange(-tt, tt, dt)

        # Return the time vector and the cc values within the specified lag time range
        return t[(t >= lag0) & (t <= lagu)], cc[(t >= lag0) & (t <= lagu)], cc_zero_lag'''

    def cc(self, x1, x2, dt, lag0, lagu):
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
        return t_out, cc_out, cc_zero_lag
    
    '''def cc(self, x1, x2, dt, lag0, lagu):  
        N = len(x1)
        x1_fft = np.fft.fft(x1)
        x2_fft = np.fft.fft(x2)
        amp = x1_fft * np.conj(x2_fft)
        cc = np.real(np.fft.ifft(amp))
        cc = np.fft.ifftshift(cc)
        # Global normalization
        norm_factor = np.linalg.norm(x1) * np.linalg.norm(x2)
        if norm_factor != 0:
            cc /= norm_factor
        cc_zero_lag = cc[len(cc) // 2]
        tt = N // 2 * dt
        t = np.arange(-tt, tt, dt)
        return t[(t >= lag0) & (t <= lagu)], cc[(t >= lag0) & (t <= lagu)], cc_zero_lag'''
    
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

        if self.current_project_path == None:
            tk.messagebox.showwarning("SANBA", "No project path detected. Create or load a project to continue.")
            return
            
        if self.pairs == None:
            tk.messagebox.showwarning("SANBA", "No pair(s) of station(s) detected. Select stations to continue.")
            return

        self.progress["value"] = 0
        self.progress["maximum"] = len(self.pairs)
        
        if self.do_crosscomponent_analysis:
            for pair in self.pairs:
                station1, station2 = pair
                dir1 = os.path.join(os.path.join(self.current_project_path, "data"), station1)
                dir2 = os.path.join(os.path.join(self.current_project_path, "data"), station2)
                channels1 = [item for item in os.listdir(dir1)]
                channels2 = [item for item in os.listdir(dir2)]
                #channel_pairs = [(ch1, ch2) for ch1 in channels1 for ch2 in channels2]
                channel_pairs = [(ch1, ch2) for ch1 in channels1 for ch2 in channels2 if ch1 <= ch2]
        else:
            channel_pairs = [(self.channel_code, self.channel_code)]

        #channel_pairs.remove(('HHE.D', 'HHE.D'))
        
        for pair in self.pairs:

            for channel_pair in channel_pairs:
            
                data_dir = os.path.join(self.current_project_path, "data")
                out_dir = os.path.join(self.current_project_path, "out")
                resp_dir = os.path.join(self.current_project_path, "data/instrument_response")
                
                # split the pair into two stations
                station1, station2 = pair
                channel1, channel2 = channel_pair

                # get the directory paths for the two stations
                dir1 = os.path.join(data_dir, f"{station1}/{channel1}")
                dir2 = os.path.join(data_dir, f"{station2}/{channel2}")

                # Check if log file exists
                log_filename = f"log_corr_{station1}_{station2}_{channel1}_{channel2}.txt"
                log_filepath = os.path.join(out_dir, log_filename)
                if os.path.isfile(log_filepath):
                    with open(log_filepath, 'r') as f:
                        excluded_files = f.read().splitlines()
                else:
                    excluded_files = []

                # get the list of miniseed files in each directory
                files1 = [f for f in os.listdir(dir1) if f not in excluded_files]
                files2 = [f for f in os.listdir(dir2) if f not in excluded_files]

                # extract year and julian day from each filename
                dates1 = [f.split('.')[-2] + '.' + f.split('.')[-1] for f in files1]
                dates2 = [f.split('.')[-2] + '.' + f.split('.')[-1] for f in files2]

                # find the intersection of the two lists (the matching dates)
                matching_dates = list(set(dates1) & set(dates2))

                # sort the matching dates using natsort
                matching_dates = natsorted(matching_dates)

                self.status_var.set(f"Running correlation calculation for {station1} {channel1} and {station2} {channel2}")
                print(f"Iniciando o método xcorr para {station1} {channel1} e {station2} {channel2}...\nDias {matching_dates}")

                # Get waveform files for both stations, excluding the ones already processed
                files1 = natsorted([f for f in os.listdir(dir1) if f not in excluded_files and any(date in f for date in matching_dates)])#[:3] #-------
                files2 = natsorted([f for f in os.listdir(dir2) if f not in excluded_files and any(date in f for date in matching_dates)])#[:3] #-------

                # Create an empty list to collect names of processed files
                new_processed_files = []

                # Check for output directories and create them if they don't exist
                corr_path = os.path.join(out_dir, 'corr')
                os.makedirs(corr_path, exist_ok=True)
                #station_pair_path = os.path.join(corr_path, "_".join(pair))
                station_pair_path = os.path.join(corr_path, f"{station1}_{station2}_{channel1}_{channel2}")
                os.makedirs(station_pair_path, exist_ok=True)

                # Define the path for the output HDF5 file
                mseed_file_path = os.path.join(station_pair_path, f"{station1}_{station2}_{channel1}_{channel2}_corr.mseed")
                #mseed_file_path = os.path.join(station_pair_path,f"corr_{self.correlation_method}_{str(self.corr_min_freq).replace('.', '')}-{str(self.corr_max_freq).replace('.', '')}Hz_{station1}_{station2}_{channel1}_{channel2}.mseed")

                corr_stream = None
                
                if files1 and files2:
                    # If the mseed file exist, load it as a Stream
                    if os.path.exists(mseed_file_path):
                        corr_stream = read(mseed_file_path, format="MSEED")
                    else:  # If the mseed file doesn't exists, create an empty Stream
                        corr_stream = Stream()

                    n0_corr_stream = len(corr_stream)
                
                for file1, file2 in tqdm(zip(files1, files2), total=len(files1), desc="Processing files\n"):
                    try:
                        # Read the data from the files
                        st1 = read(os.path.join(dir1, file1))#, format="MSEED")
                        st2 = read(os.path.join(dir2, file2))#, format="MSEED")

                        window1, window2 = None, None
                        correlation = None
                        
                        # Combine all traces into a single trace if needed
                        if len(st1) > 1:
                            st1.merge(method=0, fill_value='interpolate')
                        if len(st2) > 1:
                            st2.merge(method=0, fill_value='interpolate')

                        # Get the later start time and earlier end time
                        start_time = max(st1[0].stats.starttime, st2[0].stats.starttime)
                        end_time = min(st1[0].stats.endtime, st2[0].stats.endtime)

                        # Check if the start time is after the end time
                        if start_time >= end_time:
                            print(f'Skipping pair {file1}, {file2} as they do not overlap in time.')
                            continue

                        # Trim both streams
                        st1.trim(start_time, end_time)
                        st2.trim(start_time, end_time)

                        '''if self.corr_remove_response:
                            #if response_type == "rshake":
                                
                            xml = BytesIO(edit_xml_content_RS1D(os.path.join(resp_path, "1Dv7.xml"), st1[0], st1[0].stats.station))
                            resp1 = read_inventory(xml)
                            xml = BytesIO(edit_xml_content_RS1D(os.path.join(resp_path, "1Dv7.xml"), st2[0], st2[0].stats.station))
                            resp2 = read_inventory(xml)

                            st1.attach_response(resp1)
                            st2.attach_response(resp2)
                                
                            st1.remove_response(inventory=resp1, pre_filt = [min_freq/2, min_freq, max_freq, max_freq+5], taper = False)
                            st2.remove_response(inventory=resp2, pre_filt = [min_freq/2, min_freq, max_freq, max_freq+5], taper = False)'''
                        
                        # Detrend, demean, taper
                        if self.corr_remove_mean:
                            st1.detrend('demean')
                            st2.detrend('demean')

                        if self.corr_remove_trend:
                            st1.detrend('linear')
                            st2.detrend('linear')

                        if self.corr_taper:
                            st1.taper(max_percentage=0.05, type='cosine')
                            st2.taper(max_percentage=0.05, type='cosine')

                        # Bandpass filter
                        if self.corr_bandpass_filter:
                            st1.filter('bandpass', freqmin=self.corr_min_freq, freqmax=self.corr_max_freq, zerophase=True)
                            st2.filter('bandpass', freqmin=self.corr_min_freq, freqmax=self.corr_max_freq, zerophase=True)
                            #st1.filter('highpass', freq=49, zerophase=True)
                            #st2.filter('highpass', freq=49, zerophase=True)
                     
                        # One-bit normalization
                        if self.corr_onebit_norm:
                            st1[0].data = np.sign(st1[0].data)
                            st2[0].data = np.sign(st2[0].data)
                        
                        # Spectral whitening
                        if self.corr_spectral_whitening:
                            #st1[0].data = ifft(whiten(st1[0].data, len(st1[0].data), st1[0].stats.delta, float(self.corr_min_freq), float(self.corr_max_freq))).real
                            #st2[0].data = ifft(whiten(st2[0].data, len(st2[0].data), st2[0].stats.delta, float(self.corr_min_freq), float(self.corr_max_freq))).real
                            st1[0].data = self.spectral_whitening(st1[0].data,st1[0].stats.delta,self.corr_min_freq,self.corr_max_freq)
                            st2[0].data = self.spectral_whitening(st2[0].data,st2[0].stats.delta,self.corr_min_freq,self.corr_max_freq)

                        # Resample the streams
                        st1[0].interpolate(sampling_rate=self.corr_resample_rate, method='lanczos', a=1.0)
                        st2[0].interpolate(sampling_rate=self.corr_resample_rate, method='lanczos', a=1.0)
                        #st1.interpolate(sampling_rate=100, method='lanczos', a=1.0)
                        #st2.interpolate(sampling_rate=100, method='lanczos', a=1.0)
                                
                        # Iterate over windows with overlap
                        window_length_samples = int(self.corr_window_size * self.corr_resample_rate)
                        window_step = int(window_length_samples * (1 - self.corr_overlap / 100))
                        #max_lag_samples = int(max_lag * resample_rate)

                        for n in trange(0, len(st1[0].data) - window_length_samples, window_step, desc="Processing windows\n"):

                            try:
                                window1 = st1[0].data[n:n+window_length_samples]
                                window2 = st2[0].data[n:n+window_length_samples]

                                # Cross-correlation
                                if self.correlation_method == "cc":
                                    #correlation = correlate(window1, window2, shift =  int(self.corr_max_lag*st1[0].stats.sampling_rate),
                                    #                        demean = False, normalize = None, method = "direct")
                                    timevec, correlation, cc_zero_lag = self.cc(window1, window2, 1/self.corr_resample_rate, -self.corr_max_lag, self.corr_max_lag)
                                    
                                # Phase cross-correlation
                                if self.correlation_method == "pcc":
                                    timevec, correlation, pcc_zero_lag = self.pcc2(window1, window2, 1/self.corr_resample_rate, -self.corr_max_lag, self.corr_max_lag)
                                
                                # Check for nan and inf values
                                if not (isnan(correlation).any() or isinf(correlation).any()):
                                    # Get the start time for the window
                                    start_time = st1[0].stats.starttime + n / self.corr_resample_rate

                                    # Signal-to-noise ratio (SNR) check
                                    signal = np.max(np.abs(correlation))  # the peak of the CCF
                                    noise = np.std(correlation)  # the standard deviation of the CCF
                                    #print(signal / noise)
                                    if signal / noise >= self.corr_snr_threshold:
                                        if len(corr_stream) > 0:
                                            if start_time <= corr_stream[-1].stats.starttime:
                                                continue

                                        # Create a new Trace with the cross-correlation results
                                        corr_trace = Trace(data=correlation)
                                        corr_trace.stats.starttime = start_time
                                        corr_trace.stats.sampling_rate = self.corr_resample_rate
                                        
                                        #corr_trace.filter('bandpass', freqmin=self.corr_min_freq, freqmax=self.corr_max_freq, zerophase=True)

                                        corr_stream.append(corr_trace)
                                else:
                                    print(correlation)
                                    
                            except Exception as e: print(e)#continue

                        # Clear the data which are not needed anymore
                        del st1
                        del st2
                        del window1
                        del window2
                        del correlation
                        gc.collect()
                        
                        # At the end of processing, append newly processed files to the log
                        with open(log_filepath, 'a') as f:
                            f.write(file1 + '\n')
                            f.write(file2 + '\n')

                    except Exception as e:
                        print(f"Deu ruim no {file1} e {file2}")
                        print(e)
                        continue
                    
                if corr_stream:     
                    # Save the mseed as a mseed file
                    corr_stream.write(mseed_file_path, 'MSEED', dtype='float16')
                    
                    if self.corr_plot:

                        self.ax.clear()
                        
                        # Prepare the data for plotting
                        n_traces = len(corr_stream)
                        data = np.zeros((n_traces, len(corr_stream[0].data)))
                        
                        for i, tr in enumerate(corr_stream):
                            data[i, :] = tr.data/max(abs(tr.data))

                        # Prepare time and lag arrays
                        start_times = [tr.stats.starttime.datetime for tr in corr_stream]
                        end_times = [tr.stats.endtime.datetime for tr in corr_stream]
                        n_samples = len(corr_stream[0])
                        dt = corr_stream[0].stats.delta
                        lag = np.linspace(-self.corr_max_lag, self.corr_max_lag, n_samples)

                        # Create an image of the data
                        im = self.ax.imshow(data, aspect='auto', cmap='seismic', origin='lower',interpolation='bilinear',
                                       extent=[lag[0], lag[-1], mdates.date2num(start_times[0]),
                                               mdates.date2num(end_times[-1])])

                        self.ax.set_title(f"Correlation functions over time | {station1}.{channel1} - {station2}.{channel2} | {self.corr_min_freq} - {self.corr_max_freq} Hz")
                        # Format the y-axis to display dates
                        self.ax.yaxis_date()
                        #self.ax.yaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
                        self.ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

                        # Set the x and y labels
                        #self.ax.set_xlabel('Lag-time (s)')
                        #self.ax.set_ylabel('Start time')
                        self.ax.set_xlabel('Time lag (s)')
                        self.ax.set_ylabel('Time (hh:mm)')

                        # Add a colorbar
                        #self.fig.colorbar(im, ax=self.ax, label='Cross-correlation')

                        # Hide everything from the twin axis
                        self.ax2.set_ylabel("")
                        self.ax2.set_yticks([])
                        self.ax2.tick_params(right=False, labelright=False)
                
                        self.ax.figure.canvas.draw()

                        self.fig.savefig(os.path.join(station_pair_path, f'{station1}_{station2}_{channel1}_{channel2}_corr.png'), dpi=300)

                self.status_var.set(f"Correlation calculation for {station1} {channel1} and {station2} {channel2} completed")
                self.progress.update_idletasks()
                self.progress["value"] += 1

    def edit_xml_content_RS1D(xml_original, trace, station_name):

        replacements = {
            '<station publicID="STNNM.Station" code="STNNM">': f'<station publicID="{station_name}.Station" code="{station_name}">',
            '<start>YYYY-MM-DDT00:00:00.00Z</start>': f'<start>{trace.stats.starttime}</start>'
        }

        with open(xml_original, 'r') as in_file:
            content = in_file.read()

        for pattern, replacement in replacements.items():
            content = sub(pattern, replacement, content, count=1)

        content_bytes = content.encode('utf-8')

        return content_bytes

    def stack(self):

        if self.current_project_path == None:
            tk.messagebox.showwarning("SANBA", "No project path detected. Create or load a project to continue.")
            return
            
        if self.pairs == None:
            tk.messagebox.showwarning("SANBA", "No pair(s) of station(s) detected. Select stations to continue.")
            return

        self.progress["value"] = 0
        self.progress["maximum"] = len(self.pairs)

        if self.do_crosscomponent_analysis:
            for pair in self.pairs:
                station1, station2 = pair
                dir1 = os.path.join(os.path.join(self.current_project_path, "data"), station1)
                dir2 = os.path.join(os.path.join(self.current_project_path, "data"), station2)
                channels1 = [item for item in os.listdir(dir1)]
                channels2 = [item for item in os.listdir(dir2)]
                #channel_pairs = [(ch1, ch2) for ch1 in channels1 for ch2 in channels2]
                channel_pairs = [(ch1, ch2) for ch1 in channels1 for ch2 in channels2 if ch1 <= ch2]
        else:
            channel_pairs = [(self.channel_code, self.channel_code)]
        
        for pair in self.pairs:

            for channel_pair in channel_pairs:
            
                data_dir = os.path.join(self.current_project_path, "data")
                out_dir = os.path.join(self.current_project_path, "out")
                
                station1, station2 = pair
                channel1, channel2 = channel_pair

                self.status_var.set(f"Running stacking for {station1} {channel1} and {station2} {channel2}")
                print(f"Iniciando o método stack para {station1} {channel1} e {station2} {channel2}...")
                
                stack_path = os.path.join(self.current_project_path, "out/stack")
                pair_path = os.path.join(stack_path, f'{station1}_{station2}_{channel1}_{channel2}')

                # Check and create necessary directories
                if not os.path.exists(stack_path):
                    os.makedirs(stack_path)
                if not os.path.exists(pair_path):
                    os.makedirs(pair_path)

                # Window length in files
                #window_length = self.stack_window_length_days * int(86400 / self.corr_window_size)
                window_length = int(self.stack_window_length_days * int(86400 / self.corr_window_size))

                # Read the Stream from the mseed file
                corr_path = os.path.join(out_dir, 'corr', f'{station1}_{station2}_{channel1}_{channel2}', f"{station1}_{station2}_{channel1}_{channel2}_corr.mseed")
                corr_stream = read(corr_path)

                # Define the path for the output HDF5 file
                stack_mseed_file_path = os.path.join(pair_path, f"{station1}_{station2}_{channel1}_{channel2}_stacks.mseed")

                # If the mseed file exist, load it as a Stream
                if os.path.exists(stack_mseed_file_path):
                    stacks_stream = read(stack_mseed_file_path, format="MSEED")
                else:  # If the mseed file doesn't exists, create an empty Stream
                    stacks_stream = Stream()

                # Create a flag for new traces
                new_trace_added = False

                # Create a list to hold indices to delete
                indices_to_delete = []

                # Iterate over traces in the Stream
                for i in tqdm(range(0, len(corr_stream)), desc="Processing windows for stacking\n"):
                    try:
                        window_range = corr_stream[i:min(i+window_length, len(corr_stream))]
                        correlations = []
                        central_times = []

                        if len(stacks_stream) > 0:
                            if window_range[0].stats.starttime - stacks_stream[-1].stats.starttime < self.corr_window_size:
                                continue
                        
                        # If window_range is empty, continue to the next iteration
                        if not window_range:
                            continue

                        # Check if the time_difference is not greater than window_length_days
                        time_difference = (window_range[-1].stats.starttime - window_range[0].stats.starttime) / (60 * 60 * 24)  # in days
                        
                        while time_difference > self.stack_window_length_days:
                            if len(window_range) < 2:
                                break
                            # Remove the last trace
                            window_range = window_range[:-1]
                            # Recalculate time_difference
                            time_difference = (window_range[-1].stats.starttime - window_range[0].stats.starttime) / (60 * 60 * 24)  # in days
                        
                        # If window_range has less than 2 traces, skip
                        if len(window_range) < 2:
                            continue
                        
                        for tr in window_range:
                            correlations.append(tr.data)
                            #central_times.append(tr.stats.starttime.timestamp)
                        
                        avg_correlation = np.mean(correlations, axis=0)
                        #central_timestamp = np.mean(central_times)
                        central_timestamp = window_range[0].stats.starttime.timestamp

                        # Delete the first trace in the window_range after processing
                        if i > 0:  # Skip the first iteration because there is no previous trace to delete
                            indices_to_delete.append(i-1)

                        new_trace = Trace(data=avg_correlation)
                        new_trace.stats.starttime = central_timestamp
                        new_trace.stats.sampling_rate = self.corr_resample_rate
                        stacks_stream.append(new_trace)
                        #last_trace_time = central_timestamp  # update the last_trace_time
                        new_trace_added = True
                            
                        # Check if the last trace of stream was included in window_range
                        if corr_stream[-1] in window_range:
                            indices_to_delete.append(i)
                            break

                        del correlations
                        del central_times
                        del window_range
                        gc.collect()
                        
                    except Exception as e: print(e)
                    
                # After the loop, delete the collected indices:
                for index in sorted(indices_to_delete, reverse=True):
                    del corr_stream[index]
                
                if new_trace_added:
                    
                    # Save the stack Stream to a mseed file
                    stacks_stream.write(stack_mseed_file_path, format='MSEED', mode='w', dtype='float16')

                    # Save the xcorr Stream to a mseed file
                    corr_stream.write(corr_path, format='MSEED', dtype='float16')

                    if self.stack_plot:

                        self.ax.clear()
                        self.ax2.clear()

                        # Prepare the data for plotting
                        n_traces = len(stacks_stream)
                        data = np.zeros((n_traces, len(stacks_stream[0].data)))
                        for i, tr in enumerate(stacks_stream):
                            data[i, :] = tr.data/max(abs(tr.data))

                        # Prepare time and lag arrays
                        start_times = [tr.stats.starttime.datetime for tr in stacks_stream]
                        end_times = [tr.stats.endtime.datetime for tr in stacks_stream]
                        n_samples = len(stacks_stream[0])
                        dt = stacks_stream[0].stats.delta
                        lag = np.linspace(-self.corr_max_lag, self.corr_max_lag, n_samples)

                        # Create an image of the data
                        im = self.ax.imshow(data, aspect='auto', cmap='seismic', origin='lower',interpolation='bilinear',
                                       extent=[lag[0], lag[-1], mdates.date2num(start_times[0]),
                                               mdates.date2num(end_times[-1])])

                        self.ax.set_title(f"Stacked correlation functions over time | {station1}.{channel1} - {station2}.{channel2}")
                        
                        # Format the y-axis to display dates
                        self.ax.yaxis_date()
                        #self.ax.yaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
                        self.ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

                        # Set the x and y labels
                        #self.ax.set_xlabel('Lag Time (s)')
                        #self.ax.set_ylabel('Start Time')
                        self.ax.set_xlabel('Time lag (s)')
                        self.ax.set_ylabel('Horário (hh:mm)')


                        # Add a colorbar
                        #self.fig.colorbar(im, ax=self.ax, label='Cross-correlation')

                        # Hide everything from the twin axis
                        self.ax2.set_ylabel("")
                        self.ax2.set_yticks([])
                        self.ax2.tick_params(right=False, labelright=False)
                
                        self.fig.canvas.draw()

                        self.fig.savefig(os.path.join(pair_path, f'{station1}_{station2}_{channel1}_{channel2}_stack.png'), dpi=300)
                        
                self.status_var.set(f"Stacking for {station1} {channel1} and {station2} {channel2} completed")
                self.progress.update_idletasks()
                self.progress["value"] += 1

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
    
    def mwcs(self):
        
        if self.current_project_path == None:
            tk.messagebox.showwarning("SANBA", "No project path detected. Create or load a project to continue.")
            return
            
        if self.pairs == None:
            tk.messagebox.showwarning("SANBA", "No pair(s) of station(s) detected. Select stations to continue.")
            return

        self.progress["value"] = 0
        self.progress["maximum"] = len(self.pairs)

        if self.do_crosscomponent_analysis:
            for pair in self.pairs:
                station1, station2 = pair
                dir1 = os.path.join(os.path.join(self.current_project_path, "data"), station1)
                dir2 = os.path.join(os.path.join(self.current_project_path, "data"), station2)
                channels1 = [item for item in os.listdir(dir1)]
                channels2 = [item for item in os.listdir(dir2)]
                #channel_pairs = [(ch1, ch2) for ch1 in channels1 for ch2 in channels2]
                channel_pairs = [(ch1, ch2) for ch1 in channels1 for ch2 in channels2 if ch1 <= ch2]
        else:
            channel_pairs = [(self.channel_code, self.channel_code)]
                            
        for pair in self.pairs:

            for channel_pair in channel_pairs:

                station1, station2 = pair
                channel1, channel2 = channel_pair
                
                self.status_var.set(f"Running the MWCS method for {station1} {channel1} and {station2} {channel2}")
                print(f"Iniciando o método mwcs para {station1} {channel1} e {station2} {channel2} ({self.mwcs_freq_min}-{self.mwcs_freq_max}Hz)...")

                out_dir = os.path.join(self.current_project_path, "out")
                stack_path = os.path.join(out_dir, 'stack', f'{station1}_{station2}_{channel1}_{channel2}')
                #stack_path = os.path.join(self.stack_path_temp, f'{station1}_{station2}_{channel1}_{channel2}')
                dvv_path = os.path.join(out_dir, 'dvv', f'{station1}_{station2}_{channel1}_{channel2}')
                log_file = os.path.join(out_dir, f'log_mwcs_{station1}_{station2}_{channel1}_{channel2}_{self.mwcs_freq_min}-{self.mwcs_freq_max}Hz.txt')
                csv_file = os.path.join(dvv_path, f'{station1}_{station2}_{channel1}_{channel2}_{self.mwcs_freq_min}-{self.mwcs_freq_max}Hz_dvv.csv')

                if not os.path.exists(dvv_path):
                    os.makedirs(dvv_path)
                
                if not os.path.isfile(log_file):
                    with open(log_file, 'w') as lf:
                        lf.write('')

                # Reading logged traces
                with open(log_file, 'r') as lf:
                    logged_indices = [int(line.strip()) for line in lf]
                    
                # Load stack Stream from the mseed file
                stack_stream = read(os.path.join(stack_path, f"{station1}_{station2}_{channel1}_{channel2}_stacks.mseed"), format="MSEED")
                #stack_stream.filter('bandpass', freqmin=self.mwcs_freq_min, freqmax=self.mwcs_freq_max, zerophase=True)

                # Define reference_correlation
                if self.mwcs_reference == "static":
                    reference_correlation = stack_stream[0].data
                elif self.mwcs_reference == "mean":
                    reference_correlation = np.mean([trace.data for trace in stack_stream], axis=0)
                
                results = pd.DataFrame(columns=['timestamp', 'dvv', 'dvv_std'])

                if self.mwcs_do_similarity_analysis:
                    similarity_csv_file = os.path.join(dvv_path, f'{station1}_{station2}_{channel1}_{channel2}_{self.mwcs_freq_min}-{self.mwcs_freq_max}Hz_similarity.csv')
                    similarity_results = pd.DataFrame(columns=['timestamp', 'central_lag', 'similarity'])
                    
##                STRETCH_PCTS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0,
##                    -0.1, -0.2, -0.3, -0.4, -0.5, -0.4, -0.3, -0.2, -0.1, 0]
                
                for i in tqdm(range(len(stack_stream)), desc="Processing traces for mwcs\n"):
                    try:
                        if i in logged_indices:
                            continue

                        with open(log_file, 'a') as lf:
                            lf.write(str(i) + '\n')

                        current_data = stack_stream[i].data

                        if self.mwcs_reference == "following":
                            if i == 0:
                                reference_correlation = stack_stream[0].data
                            else:
                                reference_correlation = stack_stream[i-1].data

                        if self.mwcs_do_similarity_analysis:
                            #for each dv/v measurement, a zero-lag waveform similarity value
                            if self.mwcs_similarity_method == "zero_lag_cc":
                                _, _, corr_zero_lag = self.cc(current_data, reference_correlation, 1/self.corr_resample_rate, -self.corr_max_lag, self.corr_max_lag)
                                
                            elif self.mwcs_similarity_method == "zero_lag_pcc":
                                _, _, corr_zero_lag = self.pcc2(current_data, reference_correlation, 1/self.corr_resample_rate, -self.corr_max_lag, self.corr_max_lag)

                            #for each dv/v measurement, a zero-lag time-dependent waveform similarity value
                            similarity = self.moving_window_crosscorrelation(current_data, reference_correlation, self.corr_resample_rate,
                                                                            self.mwcs_window_length, self.mwcs_window_step)
                                                
                        mwcs_data = msnoise_mwcs(current=current_data, 
                                                reference=reference_correlation, 
                                                df=self.corr_resample_rate, 
                                                freqmin=self.mwcs_freq_min, 
                                                freqmax=self.mwcs_freq_max, 
                                                tmin=self.mwcs_moving_start,
                                                window_length=self.mwcs_window_length, 
                                                step=self.mwcs_window_step)
                        
                        time_axis = mwcs_data.T[0]
                        delay_time = mwcs_data.T[1]
                        err = mwcs_data.T[2]
                        coh = mwcs_data.T[3]

                        # Filter based on criteria
                        mask = (np.abs(time_axis) >= self.mwcs_lagtime_ballistic) & (np.abs(time_axis) <= self.mwcs_lagtime_max-(self.mwcs_window_length/2))
                        #mask = ~((time_axis >= -3.0) & (time_axis <= -1.0)) & (time_axis <= 4) 
                        mask &= (coh >= self.mwcs_coherency_min)
                        mask &= (err <= self.mwcs_error_max)
                        mask &= (np.abs(delay_time) <= self.mwcs_abs_delay_time_limit)

                        time_axis_filtered = time_axis[mask]
                        delay_time_filtered = delay_time[mask]
                        err_filtered = err[mask]
                        
                        # Perform linear regression
                        slope, intercept, std, n = linear_regression(time_axis_filtered,delay_time_filtered,
                                                weights = 1/err_filtered, intercept_origin = False)

                        # Check if slope and std are valid numbers
                        if np.isnan(slope) or np.isinf(slope) or np.isnan(std) or np.isinf(std):
                            print("regressão linear falhou")
                            continue

                        # Compute dvv and dvv_std
                        dvv = -100 * slope
                        dvv_std = 100 * std

                        # Save to results
                        timestamp = stack_stream[i].stats.starttime.timestamp

                        if self.mwcs_do_similarity_analysis:
                            #results = results.append({'timestamp': timestamp, 'dvv': dvv, 'dvv_std': dvv_std, 'similarity': corr_zero_lag}, ignore_index=True)
                            results = pd.concat([results, pd.DataFrame([{'timestamp': timestamp, 'dvv': dvv, 'dvv_std': dvv_std, 'similarity': corr_zero_lag}])], ignore_index=True)
                            #similarity_results = similarity_results.append({'timestamp': timestamp, 'central_lags': similarity.keys(), 'similarity': similarity}, ignore_index=True)
                            similarity_results = pd.concat([similarity_results, pd.DataFrame([{'timestamp': timestamp, 'central_lags': similarity.keys(), 'similarity': similarity}])], ignore_index=True)
                            
                        else:
                            #results = results.append({'timestamp': timestamp, 'dvv': dvv, 'dvv_std': dvv_std}, ignore_index=True)
                            results = pd.concat([results, pd.DataFrame([{'timestamp': timestamp,'dvv': dvv,'dvv_std': dvv_std}])], ignore_index=True)
                            
                        # Create plot
                        if self.mwcs_plot:

                            self.ax2.set_visible(True)
                            self.ax2.tick_params(right=True, labelright=True)
        
                            timestamp = stack_stream[i].stats.starttime.timestamp
                            date = datetime.datetime.fromtimestamp(timestamp)
                            
                            self.ax.clear()
                            self.ax2.clear()
                            
                            n = len(current_data)
                            limit = (n - 1) / (2 * self.corr_resample_rate)
                            timevec = np.linspace(-limit, limit, n)

                            cer = self.ax.plot(timevec, reference_correlation, lw=2, c="r", label='Reference correlation')
                            cem = self.ax.plot(timevec, current_data, lw=1, c="k", label='Moving correlation')

                            self.ax.set_ylim(-1.25*max(abs(min(reference_correlation)), abs(max(reference_correlation))),
                                             1.25*max(abs(min(reference_correlation)), abs(max(reference_correlation))))
         
                            self.ax.set_ylabel('Correlation')

                            delayTime = self.ax2.plot(time_axis_filtered, delay_time_filtered, 'o-', c="k",lw=0, label = "dt")
                            linReg = self.ax2.plot(time_axis_filtered, slope * time_axis_filtered + intercept, ls='--', c = 'k', label=f'dv/v = {dvv:.2f}% (±{dvv_std:.3f}%)')
                            
                            self.ax.set_xlabel('Time lag (s)')
                            self.ax2.set_ylabel('dt (s)')
                            self.ax2.set_ylim([-self.mwcs_abs_delay_time_limit*1.25,self.mwcs_abs_delay_time_limit*1.25])

                            self.ax.set_title(f'MWCS | {station1}.{channel1} - {station2}.{channel2} | {date.strftime("%d/%m/%Y %H:%M:%S")} | Stack of {self.stack_window_length_days} day(s) | {self.mwcs_freq_min} - {self.mwcs_freq_max} Hz')
                            lns = cer + cem + delayTime + linReg
                            labels = [l.get_label() for l in lns]
                            self.ax.legend(lns, labels, loc="upper right", fontsize=9)
                            self.ax.grid(True,axis="x",alpha=.5)

                            self.fig.savefig(os.path.join(dvv_path, f"{station1}_{station2}_{channel1}_{channel2}_{self.mwcs_freq_min}-{self.mwcs_freq_max}Hz_stack{self.stack_window_length_days}d_mwcs_{i}.png"),format="PNG")

                            self.ax.figure.canvas.draw()
                            self.ax2.figure.canvas.draw()
                            
                    except Exception as e:
                        print(e)
                        continue
                    
                # Save to csv file
                results['timestamp'] = results['timestamp'].astype(float)
                results['timestamp'] = pd.to_datetime(results['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert('America/Argentina/Buenos_Aires').dt.strftime("%d/%m/%Y %H:%M:%S")

                if self.mwcs_do_similarity_analysis:
                    similarity_results['timestamp'] = similarity_results['timestamp'].astype(float)
                    similarity_results['timestamp'] = pd.to_datetime(similarity_results['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert('America/Argentina/Buenos_Aires').dt.strftime("%d/%m/%Y %H:%M:%S")

                # Check if the csv file exists and if the results dataframe is not empty
                if not results.empty:
                    if os.path.exists(csv_file):
                        # The file exists, so append without writing headers
                        results.to_csv(csv_file, mode='a', header=False, index=False)
                    else:
                        # The file does not exist, write with headers
                        results.to_csv(csv_file, index=False)

                if self.mwcs_do_similarity_analysis:
                    
                    if not similarity_results.empty:
                        reformatted_similarity_results = []

                        for idx, row in similarity_results.iterrows():
                            timestamp = row['timestamp']
                            similarity_dict = row['similarity']

                            for key, value in similarity_dict.items():
                                central_lag = key
                                mean_correlation = value
                                reformatted_similarity_results.append([timestamp, central_lag, mean_correlation])

                        # Convert to DataFrame
                        reformatted_similarity_df = pd.DataFrame(reformatted_similarity_results, columns=['timestamp', 'central_lag', 'mean_correlation'])

                        if os.path.exists(similarity_csv_file):
                            # The file exists, so append without writing headers
                            reformatted_similarity_df.to_csv(similarity_csv_file, mode='a', header=False, index=False)
                        else:
                            # The file does not exist, write with headers
                            reformatted_similarity_df.to_csv(similarity_csv_file, index=False)

                        try:
                            # Read the CSV file
                            #csv_file = '/home/vcavalcanti/Surffy/tempProject/out/dvv/AM.R85AF_AM.R85AF/AM.R85AF_AM.R85AF_similarity.csv'
                            data = reformatted_similarity_df

                            # Convert the timestamp column to datetime objects
                            data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d/%m/%Y %H:%M:%S')

                            # Convert timestamps to numerical values
                            data['timestamp_num'] = data['timestamp'].apply(lambda x: x.timestamp())

                            # Extract numerical timestamps, central lags, and mean correlations
                            timestamps_num = data['timestamp_num'].values
                            central_lags = data['central_lag'].values
                            mean_correlations = data['mean_correlation'].values

                            # Create a grid for interpolation
                            unique_timestamps_num = np.unique(timestamps_num)
                            unique_central_lags = np.unique(central_lags)
                            grid_x, grid_y = np.meshgrid(unique_timestamps_num, unique_central_lags)

                            # Perform interpolation
                            mean_corr_grid = griddata((timestamps_num, central_lags), mean_correlations, (grid_x, grid_y), method='cubic')

                            # Convert numerical timestamps back to datetime for plotting
                            time_grid = [datetime.datetime.fromtimestamp(ts) for ts in unique_timestamps_num]

                            # Create the filled contour plot
                            fig = plt.figure(figsize=(24, 4))
                            ax = fig.add_subplot(111)
                            levels = np.linspace(np.nanmin(mean_corr_grid), np.nanmax(mean_corr_grid), 50)
                            contour = ax.contourf(time_grid, unique_central_lags, mean_corr_grid, levels=levels, cmap='jet')

                            # Add color bar
                            cbar = fig.colorbar(contour, label = "Similarity")

                            # Set plot labels and title
                            #plt.xlabel('Timestamp')
                            ax.set_ylabel('Lapse time (s)')
                            #plt.title('Mean Correlation Over Time and Central Lag')
                            ax.invert_yaxis()
                            #ax.set_title(f"{station1} - {station2}")

                            # Format the x-axis for better readability
                            fig.autofmt_xdate()

                            fig.tight_layout()

                            # Show the plot
                            fig.savefig(os.path.join(dvv_path, f'{station1}_{station2}_{channel1}_{channel2}_{self.mwcs_freq_min}-{self.mwcs_freq_max}Hz_time_lapse_similarity.png'), dpi = 300)
                            
                        except Exception as e:
                            print(e)

                self.status_var.set(f"MWCS for {station1} {channel1} and {station2} {channel2} completed")
                self.progress.update_idletasks()
                self.progress["value"] += 1

    def plot_dvv_mean(self):
        pass
    
    def plot_dvv(self):
        if self.current_project_path is None:
            tk.messagebox.showwarning(
                "SANBA",
                "No project path detected. Create or load a project to continue."
            )
            return

        if not self.pairs:
            tk.messagebox.showwarning(
                "SANBA",
                "No pair(s) of station(s) detected. Select stations to continue."
            )
            return

        plot_similarity = tk.messagebox.askyesno(
            "SANBA",
            "Plot similarity in second y axis?"
        )
        plot_separately = tk.messagebox.askyesno(
            "SANBA",
            "Plot dv/v separately for each pair of stations?"
        )

        self.ax.clear()
        self.ax2.clear()

        # Build the list of pair/channel combinations to plot
        pair_channel_list = []

        if self.do_crosscomponent_analysis:
            for pair in self.pairs:
                station1, station2 = pair

                dir1 = os.path.join(self.current_project_path, "data", station1)
                dir2 = os.path.join(self.current_project_path, "data", station2)

                channels1 = os.listdir(dir1)
                channels2 = os.listdir(dir2)

                channel_pairs = [
                    (ch1, ch2)
                    for ch1 in channels1
                    for ch2 in channels2
                    if ch1 <= ch2
                ]

                for channel1, channel2 in channel_pairs:
                    pair_channel_list.append((station1, station2, channel1, channel2))
        else:
            for pair in self.pairs:
                station1, station2 = pair
                pair_channel_list.append(
                    (station1, station2, self.channel_code, self.channel_code)
                )

        self.progress["value"] = 0
        self.progress["maximum"] = len(pair_channel_list)

        for station1, station2, channel1, channel2 in pair_channel_list:

            self.status_var.set(
                f"Plotting the dv/v series for {station1} {channel1} and {station2} {channel2}"
            )
            print(
                f"Plotting dv/v for {station1} {channel1} and {station2} {channel2} "
                f"({self.mwcs_freq_min}-{self.mwcs_freq_max} Hz)..."
            )

            out_dir = os.path.join(self.current_project_path, "out")
            dvv_path = os.path.join(
                out_dir, "dvv", f"{station1}_{station2}_{channel1}_{channel2}"
            )
            csv_file = os.path.join(
                dvv_path,
                f"{station1}_{station2}_{channel1}_{channel2}_"
                f"{self.mwcs_freq_min}-{self.mwcs_freq_max}Hz_dvv.csv"
            )

            if not os.path.exists(csv_file):
                print(f"CSV file not found: {csv_file}")
                tk.messagebox.showwarning(
                    "SANBA",
                    f"CSV file not found:\n{csv_file}"
                )
                self.progress["value"] += 1
                self.progress.update_idletasks()
                continue

            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"Error reading CSV file {csv_file}: {e}")
                tk.messagebox.showerror(
                    "SANBA",
                    f"Error reading CSV file:\n{csv_file}\n\n{e}"
                )
                self.progress["value"] += 1
                self.progress.update_idletasks()
                continue

            if "timestamp" not in df.columns:
                print(f"'timestamp' column not found in {csv_file}")
                tk.messagebox.showerror(
                    "SANBA",
                    f"'timestamp' column not found in:\n{csv_file}"
                )
                self.progress["value"] += 1
                self.progress.update_idletasks()
                continue

            if "dvv" not in df.columns:
                print(f"'dvv' column not found in {csv_file}")
                tk.messagebox.showerror(
                    "SANBA",
                    f"'dvv' column not found in:\n{csv_file}"
                )
                self.progress["value"] += 1
                self.progress.update_idletasks()
                continue

            try:
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"],
                    format="%d/%m/%Y %H:%M:%S",
                    errors="coerce"
                )

                if df["timestamp"].isnull().any():
                    raise ValueError(
                        "Some dates could not be parsed. Check the format in the CSV file."
                    )
            except Exception as e:
                print(f"Error parsing dates in {csv_file}: {e}")
                tk.messagebox.showerror(
                    "Date Parsing Error",
                    f"Error parsing dates in:\n{csv_file}\n\n{e}"
                )
                self.progress["value"] += 1
                self.progress.update_idletasks()
                continue

            # Remove rows with invalid timestamps just in case
            df = df.dropna(subset=["timestamp"]).copy()

            if df.empty:
                print(f"No valid data found in {csv_file}")
                self.progress["value"] += 1
                self.progress.update_idletasks()
                continue

            # Sort by time to ensure proper plotting
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Check whether similarity can actually be plotted
            has_similarity = "similarity" in df.columns
            use_similarity = plot_similarity and has_similarity

            if plot_similarity and not has_similarity:
                print(f"'similarity' column not found in {csv_file}. Similarity will not be plotted.")

            if use_similarity:
                self.ax2.set_visible(True)
                self.ax2.set_ylabel("Similarity")
                self.ax2.tick_params(right=True, labelright=True)
                s#elf.ax2.spines["right"].set_visible(True)
            else:
                # Hide everything from the twin axis
                self.ax2.set_ylabel("")
                self.ax2.set_yticks([])
                self.ax2.tick_params(right=False, labelright=False)
                #self.ax2.spines["right"].set_visible(False)

            # Compute dv/v series
            if self.mwcs_reference == "following":
                dvv_plot = df["dvv"].cumsum()
            else:
                dvv_plot = df["dvv"]

            if plot_separately:
                self.ax.clear()
                self.ax2.clear()

                self.ax.plot(df["timestamp"], dvv_plot, label="dv/v")

                if use_similarity:
                    self.ax2.plot(
                        df["timestamp"],
                        df["similarity"],
                        ls="--",
                        c="k",
                        label="Similarity"
                    )
            else:
                label = f"{station1} {channel1} - {station2} {channel2}"
                self.ax.plot(df["timestamp"], dvv_plot, label=label)

                if use_similarity:
                    self.ax2.plot(
                        df["timestamp"],
                        df["similarity"],
                        ls="--",
                        label=f"Similarity {label}"
                    )

            if "dvv_std" in df.columns:
                self.ax.fill_between(
                    df["timestamp"],
                    dvv_plot - df["dvv_std"],
                    dvv_plot + df["dvv_std"],
                    alpha=0.25
                )

            self.ax.set_ylabel("dv/v (%)")
            self.ax.grid(True)

            self.ax.spines["right"].set_visible(False)
            self.ax.spines["top"].set_visible(False)

            self.ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y\n%H:%M"))

            min_date = df["timestamp"].min()
            max_date = df["timestamp"].max()
            self.ax.set_xlim(min_date, max_date)

            if use_similarity:
                self.ax2.set_ylabel("Similarity")

            if plot_separately:
                if use_similarity:
                    self.ax.legend(loc="upper right", fontsize="small")
                    self.ax2.legend(loc="lower right", fontsize="small")

                self.ax.set_title(
                    f"{station1} {channel1} - {station2} {channel2} | "
                    f"{df['timestamp'].iloc[0].strftime('%d/%m/%Y')} - "
                    f"{df['timestamp'].iloc[-1].strftime('%d/%m/%Y')} | "
                    f"{self.mwcs_freq_min}-{self.mwcs_freq_max} Hz"
                )

                self.fig.savefig(
                    os.path.join(
                        dvv_path,
                        f"{station1}_{station2}_{channel1}_{channel2}_"
                        f"{self.mwcs_freq_min}-{self.mwcs_freq_max}Hz_dvv.png"
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
    
