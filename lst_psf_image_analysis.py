"""
LST PSF Image analysis -- V1.1 -- UCLA Weidong Jin

Single-file GUI for viewing FITS images from a ZWO ASI camera and analyzing a
selected PSF region.

Features
--------
- Open 8-bit or 16-bit FITS images
- Mouse-drag ROI selection for PSF analysis
- Robust centroiding that ignores isolated bright noise
- Radial profile and encircled-energy plots
- Approximate FWHM estimates
- Red centroid marker and red outlier markers drawn on the image after analysis
- FITS header display on the left side of the GUI
- Annulus-based background estimation
- Automatic star-boundary estimate from the radial profile
- 80% encircled-energy radius used as an automatic PSF size estimate

Important note
--------------
The red dashed circle is NOT a hard physical star edge.
It is the radius where the encircled energy reaches 80%,
measured inside an automatically estimated star aperture.

Dependencies
------------
- numpy
- matplotlib
- astropy
- tkinter (must be available in your Python build)

Optional
--------
- scipy (improves median filtering and Gaussian fitting)

Run
---
python lst_psf_image_analysis_gui.py
"""

from __future__ import annotations

import math
import os
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.widgets import EllipseSelector

try:
    from astropy.io import fits
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This application requires astropy. Install it with: pip install astropy"
    ) from exc

try:  # Optional
    from scipy.optimize import curve_fit
except Exception:  # pragma: no cover
    curve_fit = None

try:  # Optional
    from scipy.ndimage import median_filter
except Exception:  # pragma: no cover
    median_filter = None


@dataclass
class PSFResult:
    """
    Container for PSF-analysis outputs.

    cx, cy:
        Final centroid in full-image pixel coordinates.

    background:
        Estimated scalar background level in the selected ROI.

    peak:
        Maximum pixel value in the ROI (not background-subtracted).

    flux:
        Total positive flux inside the automatically estimated star aperture.

    fwhm_x, fwhm_y:
        Approximate FWHM values from second moments along X/Y.

    fwhm_radial:
        Approximate radial FWHM measured from the radial profile.

    gaussian_sigma, gaussian_fwhm:
        Gaussian fit parameters from the radial profile, if fitting succeeds.

    star_boundary_radius:
        Estimated radius where the star profile merges into background noise.

    psf_radius:
        80% encircled-energy radius, measured inside the estimated star aperture.

    radial_r, radial_profile:
        Radial bin centers and mean intensity in each bin.

    encircled_radius, encircled_energy:
        Sorted radius samples and cumulative normalized flux.

    outlier_x, outlier_y:
        Full-image coordinates of pixels rejected as outliers in the PSF core.
    """
    cx: float
    cy: float
    background: float
    peak: float
    flux: float
    fwhm_x: float
    fwhm_y: float
    fwhm_radial: float
    gaussian_sigma: float | None
    gaussian_fwhm: float | None
    star_boundary_radius: float | None
    psf_radius: float | None
    radial_r: np.ndarray
    radial_profile: np.ndarray
    encircled_radius: np.ndarray
    encircled_energy: np.ndarray
    outlier_x: np.ndarray
    outlier_y: np.ndarray


class FITSImageAnalyzer(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("LST PSF Image analysis -- V1.1 -- UCLA Weidong Jin")
        self.geometry("1500x900")
        self.minsize(1200, 760)

        self.data: np.ndarray | None = None
        self.display_data: np.ndarray | None = None
        self.current_path: str | None = None
        self.fits_header = None
        self.psf_result: PSFResult | None = None

        self.selector: EllipseSelector | None = None
        self.selecting_psf = False

        self.annotation_artists: list = []
        self._img_cbar = None

        self._build_ui()
        self._bind_events()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=8)
        left.grid(row=0, column=0, sticky="nsw")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(7, weight=1)

        ttk.Label(left, text="Controls", font=("TkDefaultFont", 11, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )

        ttk.Button(left, text="Open FITS", command=self.open_fits).grid(
            row=1, column=0, sticky="ew", pady=4
        )
        ttk.Button(left, text="Analyze PSF", command=self.toggle_psf_analysis).grid(
            row=2, column=0, sticky="ew", pady=4
        )
        ttk.Button(left, text="Reset View", command=self.reset_view).grid(
            row=3, column=0, sticky="ew", pady=4
        )
        ttk.Button(left, text="Quit", command=self.destroy).grid(
            row=4, column=0, sticky="ew", pady=4
        )

        self.file_label = ttk.Label(left, text="No file loaded", wraplength=260)
        self.file_label.grid(row=5, column=0, sticky="w", pady=(12, 6))

        ttk.Label(left, text="FITS Header", font=("TkDefaultFont", 10, "bold")).grid(
            row=6, column=0, sticky="w", pady=(10, 4)
        )

        header_frame = ttk.Frame(left)
        header_frame.grid(row=7, column=0, sticky="nsew", pady=(0, 8))
        header_frame.columnconfigure(0, weight=1)
        header_frame.rowconfigure(0, weight=1)

        self.header_text = tk.Text(
            header_frame,
            height=18,
            width=34,
            wrap="none",
            font=("Courier", 9),
        )
        self.header_text.grid(row=0, column=0, sticky="nsew")

        v_scroll = ttk.Scrollbar(
            header_frame, orient="vertical", command=self.header_text.yview
        )
        v_scroll.grid(row=0, column=1, sticky="ns")

        h_scroll = ttk.Scrollbar(
            header_frame, orient="horizontal", command=self.header_text.xview
        )
        h_scroll.grid(row=1, column=0, sticky="ew")

        self.header_text.configure(
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set,
        )

        self.header_text.insert("1.0", "Open a FITS file to view header info here.\n")
        self.header_text.configure(state="disabled")

        self.info_var = tk.StringVar(value="Load an 8-bit or 16-bit FITS image.")
        ttk.Label(left, textvariable=self.info_var, wraplength=260, justify="left").grid(
            row=8, column=0, sticky="w", pady=(6, 0)
        )

        ttk.Separator(left).grid(row=9, column=0, sticky="ew", pady=12)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(left, textvariable=self.status_var, wraplength=260, justify="left").grid(
            row=10, column=0, sticky="w"
        )

        main = ttk.Frame(self, padding=(0, 8, 8, 8))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        self.img_frame = ttk.Frame(main)
        self.img_frame.grid(row=0, column=0, sticky="nsew")
        self.img_frame.rowconfigure(0, weight=1)
        self.img_frame.columnconfigure(0, weight=1)

        self.fig_img = Figure(figsize=(7, 6), dpi=100)
        self.ax_img = self.fig_img.add_subplot(111)
        self.ax_img.set_title("FITS Image")
        self.ax_img.set_xlabel("X (pixels)")
        self.ax_img.set_ylabel("Y (pixels)")
        self.im_artist = self.ax_img.imshow(np.zeros((10, 10)), origin="lower", cmap="gray")
        self._img_cbar = self.fig_img.colorbar(
            self.im_artist, ax=self.ax_img, fraction=0.046, pad=0.04
        )

        self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=self.img_frame)
        self.canvas_img_widget = self.canvas_img.get_tk_widget()
        self.canvas_img_widget.pack(side="top", fill="both", expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas_img, self.img_frame)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")

        self.psf_frame = ttk.Frame(main)
        self.psf_frame.grid(row=0, column=1, sticky="nsew")
        self.psf_frame.rowconfigure(0, weight=1)
        self.psf_frame.rowconfigure(1, weight=0)
        self.psf_frame.columnconfigure(0, weight=1)

        self.psf_plot_frame = ttk.Frame(self.psf_frame)
        self.psf_plot_frame.grid(row=0, column=0, sticky="nsew")
        self.psf_plot_frame.rowconfigure(0, weight=1)
        self.psf_plot_frame.columnconfigure(0, weight=1)

        self.fig_psf = Figure(figsize=(5, 7), dpi=100)
        self.ax_radial = self.fig_psf.add_subplot(211)
        self.ax_enc = self.fig_psf.add_subplot(212)

        self.ax_radial.set_title("Radial Profile")
        self.ax_radial.set_xlabel("Radius (pixels)")
        self.ax_radial.set_ylabel("Intensity")

        self.ax_enc.set_title("Encircled Energy")
        self.ax_enc.set_xlabel("Radius (pixels)")
        self.ax_enc.set_ylabel("Normalized flux")

        self.fig_psf.tight_layout()

        self.canvas_psf = FigureCanvasTkAgg(self.fig_psf, master=self.psf_plot_frame)
        self.canvas_psf_widget = self.canvas_psf.get_tk_widget()
        self.canvas_psf_widget.pack(side="top", fill="both", expand=True)

        self.toolbar_psf = NavigationToolbar2Tk(self.canvas_psf, self.psf_plot_frame)
        self.toolbar_psf.update()
        self.toolbar_psf.pack(side="bottom", fill="x")

        # Frame to hold text + scrollbar
        metrics_frame = ttk.Frame(self.psf_frame)
        metrics_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        metrics_frame.columnconfigure(0, weight=1)
        metrics_frame.rowconfigure(0, weight=1)

        # Text widget
        self.metrics_text = tk.Text(metrics_frame, height=9, wrap="word")
        self.metrics_text.grid(row=0, column=0, sticky="nsew")

        # Vertical scrollbar
        metrics_scroll = ttk.Scrollbar(
            metrics_frame,
            orient="vertical",
            command=self.metrics_text.yview
        )
        metrics_scroll.grid(row=0, column=1, sticky="ns")

        # Connect scrollbar to text widget
        self.metrics_text.configure(yscrollcommand=metrics_scroll.set)
        self.metrics_text.insert(
            "1.0",
            "Load a FITS image, then click Analyze PSF and drag a box around the star.\n",
        )
        self.metrics_text.configure(state="disabled")

    def _bind_events(self) -> None:
        self.canvas_img.mpl_connect("draw_event", self._on_draw)
        self.ax_img.callbacks.connect("xlim_changed", self._on_limits_changed)
        self.ax_img.callbacks.connect("ylim_changed", self._on_limits_changed)

    def _on_draw(self, event) -> None:
        pass

    def _on_limits_changed(self, ax) -> None:
        pass

    def open_fits(self) -> None:
        path = filedialog.askopenfilename(
            title="Open FITS file",
            filetypes=[("FITS files", "*.fits *.fit *.fts"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            data, header = self._load_first_image_hdu(path)
        except Exception as exc:
            messagebox.showerror("Error", f"Could not read FITS file:\n{exc}")
            return

        if data.ndim != 2:
            messagebox.showerror(
                "Error",
                f"Expected a 2D FITS image, but got array with shape {data.shape}.",
            )
            return

        self._disable_selector()
        self.selecting_psf = False

        self.current_path = path
        self.data = np.asarray(data)
        self.display_data = self._prepare_display_data(self.data)
        self.fits_header = header
        self.file_label.configure(text=os.path.basename(path))

        self._show_header_info(header)

        bit_depth = self._guess_bit_depth(self.data)
        self.info_var.set(
            f"Shape: {self.data.shape[1]} x {self.data.shape[0]} | Bit depth: {bit_depth}-bit | dtype: {self.data.dtype}"
        )
        self.status_var.set("File loaded.")

        self._show_image()
        self._clear_psf_results()
        self.status_var.set("Ready. Click Analyze PSF and drag a box around the star.")

    def _load_first_image_hdu(self, path: str) -> tuple[np.ndarray, "fits.Header"]:
        """
        Open the FITS file and return the first HDU that contains image data.
        """
        with fits.open(path, memmap=False) as hdul:
            for hdu in hdul:
                if hdu.data is None:
                    continue
                arr = np.asarray(hdu.data)
                if arr.ndim >= 2:
                    while arr.ndim > 2:
                        arr = arr[0]
                    return arr, hdu.header
        raise ValueError("No image HDU found in the FITS file.")

    def _show_header_info(self, header) -> None:
        """
        Render FITS header cards into the left text panel.
        """
        if header is None:
            text = "No FITS header available.\n"
        else:
            lines = []
            for key in header.keys():
                if key in ("COMMENT", "HISTORY"):
                    try:
                        value = header[key]
                        if isinstance(value, list):
                            for item in value:
                                lines.append(f"{key:8s} {item}")
                        else:
                            lines.append(f"{key:8s} {value}")
                    except Exception:
                        continue
                    continue

                try:
                    value = header[key]
                    comment = header.comments[key] if key in header else ""
                    if comment:
                        lines.append(f"{key:8s} = {str(value):<20} / {comment}")
                    else:
                        lines.append(f"{key:8s} = {value}")
                except Exception:
                    continue

            text = "\n".join(lines) if lines else "Header is empty.\n"

        self.header_text.configure(state="normal")
        self.header_text.delete("1.0", "end")
        self.header_text.insert("1.0", text)
        self.header_text.configure(state="disabled")

    def _guess_bit_depth(self, arr: np.ndarray) -> int:
        """
        Infer bit depth from the FITS array dtype.
        """
        dt = np.asarray(arr).dtype
        if np.issubdtype(dt, np.uint8):
            return 8
        if np.issubdtype(dt, np.uint16):
            return 16
        if np.issubdtype(dt, np.integer):
            return int(dt.itemsize * 8)
        return 16

    def _prepare_display_data(self, arr: np.ndarray) -> np.ndarray:
        """
        Convert image data to float32 and replace bad values with zeros.
        """
        data = np.asarray(arr, dtype=np.float32)
        return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    def _robust_limits(self, data: np.ndarray) -> tuple[float, float]:
        """
        Pick display limits using percentiles so hot pixels do not dominate the grayscale stretch.
        """
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return 0.0, 1.0

        lo = float(np.percentile(finite, 1.0))
        hi = float(np.percentile(finite, 99.7))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
            if hi <= lo:
                hi = lo + 1.0
        return lo, hi

    def _show_image(self) -> None:
        """
        Draw the loaded FITS image in grayscale.
        """
        if self.display_data is None:
            return

        data = self.display_data
        lo, hi = self._robust_limits(data)

        self.ax_img.clear()
        self.ax_img.set_title("FITS Image")
        self.ax_img.set_xlabel("X (pixels)")
        self.ax_img.set_ylabel("Y (pixels)")

        self.im_artist = self.ax_img.imshow(
            data,
            origin="lower",
            cmap="gray",
            vmin=lo,
            vmax=hi,
            interpolation="nearest",
        )

        if self._img_cbar is not None:
            try:
                self._img_cbar.remove()
            except Exception:
                pass
        self._img_cbar = self.fig_img.colorbar(
            self.im_artist, ax=self.ax_img, fraction=0.046, pad=0.04
        )

        self.ax_img.set_xlim(-0.5, data.shape[1] - 0.5)
        self.ax_img.set_ylim(-0.5, data.shape[0] - 0.5)
        self.canvas_img.draw_idle()

    def reset_view(self) -> None:
        """
        Remove PSF annotations, remove the selector overlay, and restore the full image view.
        """
        if self.display_data is None:
            return

        self._disable_selector()
        self.selecting_psf = False
        self._clear_annotations()

        self.ax_img.set_xlim(-0.5, self.display_data.shape[1] - 0.5)
        self.ax_img.set_ylim(-0.5, self.display_data.shape[0] - 0.5)
        self.canvas_img.draw_idle()

    def toggle_psf_analysis(self) -> None:
        """
        Turn PSF selection mode on/off.

        When on, the user can drag an ellipse on the image; we convert it to a
        circular analysis region.
        """
        if self.data is None:
            messagebox.showinfo("Analyze PSF", "Load a FITS image first.")
            return

        self.selecting_psf = not self.selecting_psf
        if self.selecting_psf:
            self._clear_annotations()
            self.status_var.set("PSF mode: drag on the image to select a star region.")
            self._enable_selector()
        else:
            self.status_var.set("PSF mode off.")
            self._disable_selector()

    def _enable_selector(self) -> None:
        """
        Activate a draggable ellipse selector.

        The selector itself is an ellipse tool, but the analysis uses the
        center of the dragged region and the smaller of the x/y half-spans as
        a circular radius.
        """
        self._disable_selector()
        self._deactivate_toolbar_modes()

        def onselect(eclick, erelease):
            if self.display_data is None:
                return
            if eclick.xdata is None or erelease.xdata is None:
                return

            # Center of the dragged selection
            cx = 0.5 * (eclick.xdata + erelease.xdata)
            cy = 0.5 * (eclick.ydata + erelease.ydata)

            # Half-width / half-height in pixels
            rx = abs(erelease.xdata - eclick.xdata) * 0.5
            ry = abs(erelease.ydata - eclick.ydata) * 0.5

            # Use a circle with the smaller radius so the mask stays within the drag box
            r = min(rx, ry)

            self._analyze_psf_circle(cx, cy, r)

        self.selector = EllipseSelector(
            self.ax_img,
            onselect,
            useblit=True,
            button=[1],
            interactive=True,
            spancoords="pixels",
            minspanx=5,
            minspany=5,
            props=dict(facecolor="none", edgecolor="yellow", linewidth=2),
        )

        self.canvas_img.draw_idle()

    def _disable_selector(self) -> None:
        """
        Fully disable and remove the selector widget if it exists.

        This is intentionally defensive because Matplotlib stores the selector
        visuals in different internal attributes depending on version.
        """
        if self.selector is None:
            return

        def _remove_obj(obj) -> None:
            if obj is None:
                return
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    _remove_obj(item)
                return
            try:
                obj.remove()
            except Exception:
                try:
                    obj.set_visible(False)
                except Exception:
                    pass

        try:
            self.selector.set_active(False)

            # Try common selector artist containers across matplotlib versions.
            for attr in (
                "artists",
                "_selection_artist",
                "_handles_artists",
                "_corner_handles",
                "_edge_handles",
                "_center_handle",
            ):
                _remove_obj(getattr(self.selector, attr, None))

            self.selector.disconnect_events()
        except Exception:
            pass

        self.selector = None
        self.canvas_img.draw_idle()

    def _deactivate_toolbar_modes(self) -> None:
        """
        Turn off zoom/pan modes in the Matplotlib toolbars, since those can
        interfere with drag-selection.
        """
        for tb in (getattr(self, "toolbar", None), getattr(self, "toolbar_psf", None)):
            if tb is None:
                continue
            try:
                mode = str(getattr(tb, "mode", "")).lower()
                if "zoom" in mode:
                    tb.zoom()
                elif "pan" in mode:
                    tb.pan()
            except Exception:
                pass

    def _analyze_psf_circle(self, cx: float, cy: float, r: float) -> None:
        """
        Analyze a circular ROI centered at (cx, cy) with radius r.
        """
        if self.display_data is None:
            return

        data = self.display_data
        h, w = data.shape

        # Convert circle bounds to a rectangular crop first for efficiency.
        x0 = int(max(0, math.floor(cx - r)))
        x1 = int(min(w, math.ceil(cx + r)))
        y0 = int(max(0, math.floor(cy - r)))
        y1 = int(min(h, math.ceil(cy + r)))

        if x1 <= x0 or y1 <= y0:
            self.status_var.set("Invalid selection. Try again.")
            return

        roi = data[y0:y1, x0:x1]
        if roi.size < 9:
            self.status_var.set("Selected region is too small.")
            return

        yy, xx = np.indices(roi.shape)
        cx_local = cx - x0
        cy_local = cy - y0
        rr = np.sqrt((xx - cx_local) ** 2 + (yy - cy_local) ** 2)

        # Keep only pixels inside the circular selection.
        mask = rr <= r

        result = self._compute_psf_metrics(
            roi,
            x0,
            y0,
            mask=mask,
            cx_local=cx_local,
            cy_local=cy_local,
        )
        self.psf_result = result
        self._show_psf_results(result)
        self._draw_psf_annotations(result)
        self.status_var.set(
            f"PSF analysis updated (circular region, r={r:.1f}px)."
        )

    def _compute_psf_metrics(
        self,
        roi: np.ndarray,
        x_offset: int,
        y_offset: int,
        mask: np.ndarray | None = None,
        cx_local: float | None = None,
        cy_local: float | None = None,
    ) -> PSFResult:
        """
        Compute centroid, background, radial profile, encircled energy, and
        several width metrics from the selected ROI.

        Main idea:
        - estimate background from an annulus near the ROI edge
        - background-subtract the ROI
        - estimate a star boundary from the radial profile
        - compute total flux inside that boundary
        - compute 80% encircled-energy radius inside that boundary
        """
        roi = np.asarray(roi, dtype=np.float64)
        roi = np.nan_to_num(roi, nan=0.0, posinf=0.0, neginf=0.0)

        if mask is None:
            mask = np.ones_like(roi, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != roi.shape:
                raise ValueError("mask shape must match roi shape")

        # Optional median filter helps suppress isolated hot pixels before peak-finding.
        sm = median_filter(roi, size=3) if median_filter is not None else roi

        # Initial peak location from the smoothed image.
        py, px = np.unravel_index(np.argmax(sm), sm.shape)

        yy, xx = np.indices(roi.shape)
        rr0 = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)

        # Annulus-based background estimate:
        # use a ring near the ROI boundary, centered on the initial peak.
        # This tries to capture sky/background pixels rather than star pixels.
        r_bg_max = min(px, py, roi.shape[1] - 1 - px, roi.shape[0] - 1 - py)

        bg_samples = roi.ravel()
        if r_bg_max >= 5:
            bg_inner = 0.70 * r_bg_max
            bg_outer = 0.98 * r_bg_max
            bg_annulus = (rr0 >= bg_inner) & (rr0 <= bg_outer)
            candidate = roi[bg_annulus]
            if candidate.size >= 10:
                bg_samples = candidate
            else:
                # Fallback: use a broader outer ring if the thin annulus is too small.
                bg_annulus = rr0 >= bg_inner
                candidate = roi[bg_annulus]
                if candidate.size >= 10:
                    bg_samples = candidate
                else:
                    # Final fallback: use the ROI border.
                    border = np.concatenate([roi[0, :], roi[-1, :], roi[:, 0], roi[:, -1]])
                    bg_samples = border

        background = float(np.median(bg_samples))

        # Background-subtracted image.
        bg_sub = roi - background

        # For centroiding, use only positive signal so negative noise does not pull the centroid around.
        signal_pos = np.where(bg_sub > 0, bg_sub, 0.0)

        # Use a small core region around the initial peak for robust centroiding.
        r_core = max(5.0, min(15.0, 0.2 * min(roi.shape)))
        core_mask = rr0 <= r_core
        core_vals = signal_pos[core_mask]

        # Robust noise scale estimate using MAD.
        if core_vals.size > 0:
            med = float(np.median(core_vals))
            mad = float(np.median(np.abs(core_vals - med)))
            robust_sigma = 1.4826 * mad if mad > 0 else float(np.std(core_vals))
        else:
            med = 0.0
            robust_sigma = 0.0

        # Reject unusually bright pixels in the core as outliers.
        if robust_sigma > 0:
            keep_mask = core_mask & (signal_pos <= med + 5.0 * robust_sigma)
        else:
            keep_mask = core_mask

        outlier_mask = core_mask & (~keep_mask) & (signal_pos > 0)

        # Flux weights used for centroiding.
        weights = signal_pos * keep_mask
        total_flux_centroid = float(np.sum(weights))

        # Flux-weighted centroid (center of mass).
        if total_flux_centroid > 0:
            cx_local = float(np.sum(xx * weights) / total_flux_centroid)
            cy_local = float(np.sum(yy * weights) / total_flux_centroid)
        else:
            cx_local = float(px) if cx_local is None else float(cx_local)
            cy_local = float(py) if cy_local is None else float(cy_local)

        # Convert ROI coordinates back to full-image coordinates.
        cx = cx_local + x_offset
        cy = cy_local + y_offset

        # Second-moment widths give a simple approximate size estimate.
        if total_flux_centroid > 0:
            var_x = float(np.sum(((xx - cx_local) ** 2) * weights) / total_flux_centroid)
            var_y = float(np.sum(((yy - cy_local) ** 2) * weights) / total_flux_centroid)
        else:
            var_x = 0.0
            var_y = 0.0

        sigma_x = math.sqrt(max(var_x, 0.0))
        sigma_y = math.sqrt(max(var_y, 0.0))
        fwhm_x = 2.354820045 * sigma_x if sigma_x > 0 else 0.0
        fwhm_y = 2.354820045 * sigma_y if sigma_y > 0 else 0.0

        # Radial profile: average background-subtracted intensity in annular bins.
        rr = np.sqrt((xx - cx_local) ** 2 + (yy - cy_local) ** 2)
        r_max = float(np.max(rr))

        # About 2 bins per pixel of radius range, but never fewer than 20 bins.
        nbins = max(20, int(math.ceil(r_max)) * 2)
        bins = np.linspace(0.0, r_max, nbins + 1)

        bin_idx = np.digitize(rr.ravel(), bins) - 1
        bin_idx = np.clip(bin_idx, 0, nbins - 1)

        radial_sum = np.bincount(bin_idx, weights=bg_sub.ravel(), minlength=nbins)
        radial_count = np.bincount(bin_idx, minlength=nbins)

        radial_profile = np.divide(
            radial_sum,
            radial_count,
            out=np.zeros_like(radial_sum, dtype=np.float64),
            where=radial_count > 0,
        )
        radial_r = 0.5 * (bins[:-1] + bins[1:])

        # Approximate radial FWHM from the radial profile.
        # For a star profile centered at r=0, the half-maximum point is a radius.
        # The full width is therefore 2 * r_half.
        peak_profile = float(np.max(radial_profile)) if radial_profile.size else 0.0
        halfmax = 0.5 * peak_profile

        if radial_profile.size >= 2:
            peak_idx = int(np.argmax(radial_profile))

            # Search outward from the peak for the first bin below half maximum.
            below = np.where(radial_profile[peak_idx:] < halfmax)[0]

            if below.size > 0:
                j = peak_idx + int(below[0])  # first index below half max
                i = j - 1

                if i >= 0 and j < radial_profile.size:
                    y1 = float(radial_profile[i])
                    y2 = float(radial_profile[j])
                    r1 = float(radial_r[i])
                    r2 = float(radial_r[j])

                    # Linear interpolation for a smoother crossing estimate.
                    if y2 != y1:
                        r_half = r1 + (halfmax - y1) * (r2 - r1) / (y2 - y1)
                    else:
                        r_half = r1

                    fwhm_radial = 2.0 * float(r_half)
                else:
                    fwhm_radial = 0.0
            else:
                # Never dropped below half max inside the sampled range.
                # Use the last radius as a lower-quality estimate.
                fwhm_radial = 2.0 * float(radial_r[-1])
        else:
            fwhm_radial = 0.0

        # Estimate the noise level from the background samples.
        bg_samples = np.asarray(bg_samples, dtype=np.float64)
        bg_sigma = 0.0
        if bg_samples.size > 0:
            bg_med = float(np.median(bg_samples))
            bg_mad = float(np.median(np.abs(bg_samples - bg_med)))
            bg_sigma = 1.4826 * bg_mad if bg_mad > 0 else float(np.std(bg_samples))

        # Automatic star boundary:
        # find where the smoothed radial profile falls below background noise
        # for several consecutive bins.
        star_boundary_radius = self._find_star_boundary_radius(
            radial_r=radial_r,
            radial_profile=radial_profile,
            bg_sigma=bg_sigma,
            n_consecutive=3,
            smooth_window=5,
        )

        # Use the detected boundary as the star aperture for flux and curve-of-growth.
        if star_boundary_radius is None or not np.isfinite(star_boundary_radius):
            ap_mask = np.ones_like(rr, dtype=bool)
        else:
            ap_mask = rr <= star_boundary_radius

        # Total flux above background, measured only inside the star aperture.
        # We clip negatives to zero so noise below background does not subtract flux.
        aperture_signal = np.clip(bg_sub[ap_mask], 0.0, None)
        total_flux = float(np.sum(aperture_signal))

        # Encircled energy:
        # sort pixels inside the aperture by radius and take cumulative positive flux.
        flat_r = rr[ap_mask].ravel()
        flat_signal = np.clip(bg_sub[ap_mask].ravel(), 0.0, None)

        if flat_r.size == 0 or flat_signal.size == 0 or np.sum(flat_signal) <= 0:
            enc_r = np.array([0.0])
            enc_e = np.array([0.0])
            psf_radius = None
        else:
            sort_idx = np.argsort(flat_r)
            sorted_r = flat_r[sort_idx]
            sorted_signal = flat_signal[sort_idx]
            cum_flux = np.cumsum(sorted_signal)

            enc_r = sorted_r
            enc_e = cum_flux / cum_flux[-1]
            psf_radius = self._radius_at_energy(enc_r, enc_e, 0.8)

        # Optional Gaussian fit to the radial profile.
        gaussian_sigma = None
        gaussian_fwhm = None
        if radial_profile.size >= 5:
            try:
                gaussian_sigma, gaussian_fwhm = self._fit_radial_gaussian(
                    radial_r, radial_profile
                )
            except Exception:
                gaussian_sigma = None
                gaussian_fwhm = None

        # Outlier pixel coordinates back in full-image coordinates.
        out_y, out_x = np.where(outlier_mask)
        out_x = out_x + x_offset
        out_y = out_y + y_offset

        return PSFResult(
            cx=cx,
            cy=cy,
            background=background,
            peak=float(np.max(roi)),
            flux=total_flux,
            fwhm_x=fwhm_x,
            fwhm_y=fwhm_y,
            fwhm_radial=fwhm_radial,
            gaussian_sigma=gaussian_sigma,
            gaussian_fwhm=gaussian_fwhm,
            star_boundary_radius=star_boundary_radius,
            psf_radius=psf_radius,
            radial_r=radial_r,
            radial_profile=radial_profile,
            encircled_radius=enc_r,
            encircled_energy=enc_e,
            outlier_x=out_x.astype(float),
            outlier_y=out_y.astype(float),
        )

    def _find_star_boundary_radius(
        self,
        radial_r: np.ndarray,
        radial_profile: np.ndarray,
        bg_sigma: float,
        n_consecutive: int = 3,
        smooth_window: int = 5,
    ) -> float | None:
        """
        Estimate the star boundary radius from the radial profile.

        The profile is assumed to be background-subtracted.
        We declare the star to end where the smoothed profile stays below
        a small noise threshold for several consecutive bins.

        This is a practical boundary estimate, not a hard physical edge.
        """
        radial_r = np.asarray(radial_r, dtype=np.float64)
        radial_profile = np.asarray(radial_profile, dtype=np.float64)

        if radial_r.size < 5 or radial_profile.size < 5:
            return None

        # Smooth the profile so one noisy bin does not decide the edge.
        if smooth_window > 1 and radial_profile.size >= smooth_window:
            kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
            prof = np.convolve(radial_profile, kernel, mode="same")
        else:
            prof = radial_profile

        # Threshold is based on the estimated background noise level.
        # If bg_sigma is extremely small, keep a tiny positive floor.
        threshold = max(3.0 * max(bg_sigma, 1e-12), 0.0)

        peak_idx = int(np.argmax(prof))

        for i in range(peak_idx, prof.size - n_consecutive):
            window = prof[i : i + n_consecutive]
            if np.all(window < threshold):
                return float(radial_r[i])

        return float(radial_r[-1])

    def _radius_at_energy(
        self,
        radius: np.ndarray,
        energy: np.ndarray,
        target: float,
    ) -> float | None:
        """
        Return the radius where the normalized encircled energy reaches 'target'.

        Linear interpolation between the two nearest samples gives a smoother
        estimate than simply taking the first bin above the threshold.
        """
        radius = np.asarray(radius, dtype=np.float64)
        energy = np.asarray(energy, dtype=np.float64)

        if radius.size == 0 or energy.size == 0 or radius.size != energy.size:
            return None
        if not np.isfinite(target):
            return None

        # Enforce monotonic energy because tiny numerical wiggles can occur.
        energy_mono = np.maximum.accumulate(energy)

        if target <= energy_mono[0]:
            return float(radius[0])
        if target >= energy_mono[-1]:
            return float(radius[-1])

        idx = int(np.searchsorted(energy_mono, target, side="left"))
        if idx <= 0:
            return float(radius[0])
        if idx >= radius.size:
            return float(radius[-1])

        e1 = float(energy_mono[idx - 1])
        e2 = float(energy_mono[idx])
        r1 = float(radius[idx - 1])
        r2 = float(radius[idx])

        if e2 <= e1:
            return float(r2)

        return float(r1 + (target - e1) * (r2 - r1) / (e2 - e1))

    def _fit_radial_gaussian(self, r: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
        """
        Fit a 1D Gaussian + constant background to the radial profile.

        Model:
            y(r) = amp * exp(-r^2 / (2 sigma^2)) + bg

        Returns:
            sigma, FWHM
        """
        r = np.asarray(r, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if r.size < 5:
            return None, None

        bg0 = float(np.median(y[-max(3, y.size // 5):]))
        amp0 = float(max(np.max(y) - bg0, 1.0))
        sigma0 = max(float(np.sum(r * y) / max(np.sum(y), 1e-12)) / 1.5, 1.0)

        def model(rr, amp, sigma, bg):
            return amp * np.exp(-(rr ** 2) / (2.0 * sigma ** 2)) + bg

        if curve_fit is not None:
            popt, _ = curve_fit(
                model,
                r,
                y,
                p0=[amp0, sigma0, bg0],
                bounds=([0.0, 1e-3, -np.inf], [np.inf, np.inf, np.inf]),
                maxfev=20000,
            )
            sigma = float(popt[1])
            return sigma, 2.354820045 * sigma

        # Fallback if scipy is not installed.
        peak = float(np.max(y))
        half = bg0 + 0.5 * (peak - bg0)
        above = np.where(y >= half)[0]
        if above.size >= 2:
            r_half = float(0.5 * (r[above[0]] + r[above[-1]]))
            sigma = r_half / math.sqrt(2.0 * math.log(2.0)) if r_half > 0 else None
            return sigma, 2.354820045 * sigma if sigma is not None else None
        return None, None

    def _show_psf_results(self, result: PSFResult) -> None:
        """
        Plot the radial profile and encircled energy curves, then print metrics.
        """
        self.ax_radial.clear()
        self.ax_radial.set_title("Radial Profile")
        self.ax_radial.set_xlabel("Radius (pixels)")
        self.ax_radial.set_ylabel("Intensity")
        self.ax_radial.plot(
            result.radial_r,
            result.radial_profile,
            marker="o",
            markersize=3,
            linewidth=1,
        )

        if result.gaussian_fwhm is not None and result.gaussian_sigma is not None:
            rr = np.linspace(0, float(np.max(result.radial_r)), 300)
            bg = float(
                np.median(
                    result.radial_profile[-max(3, result.radial_profile.size // 5):]
                )
            )
            amp = float(max(np.max(result.radial_profile) - bg, 0.0))
            fit = amp * np.exp(-(rr ** 2) / (2.0 * result.gaussian_sigma ** 2)) + bg
            self.ax_radial.plot(rr, fit, linewidth=2)
            self.ax_radial.legend(
                ["Profile", f"Gaussian FWHM={result.gaussian_fwhm:.2f} px"],
                loc="best",
            )
        else:
            self.ax_radial.legend(["Profile"], loc="best")

        self.ax_enc.clear()
        self.ax_enc.set_title("Encircled Energy")
        self.ax_enc.set_xlabel("Radius (pixels)")
        self.ax_enc.set_ylabel("Normalized flux")
        self.ax_enc.plot(result.encircled_radius, result.encircled_energy, linewidth=1.5)
        self.ax_enc.set_ylim(0.0, 1.02)
        self.ax_enc.grid(True, alpha=0.3)

        self.fig_psf.tight_layout()
        self.canvas_psf.draw_idle()

        metrics = [
            f"Centroid: x = {result.cx:.2f}, y = {result.cy:.2f}",
            f"Background: {result.background:.3f}",
            f"Peak pixel: {result.peak:.3f}",
            f"Total flux above background: {result.flux:.3f}",
            f"Approx FWHM X: {result.fwhm_x:.3f} px",
            f"Approx FWHM Y: {result.fwhm_y:.3f} px",
            f"Radial FWHM: {result.fwhm_radial:.3f} px",
        ]

        if result.psf_radius is not None:
            metrics.append(f"80% encircled-energy radius: {result.psf_radius:.3f} px")
        else:
            metrics.append("80% encircled-energy radius: not available")

        if result.star_boundary_radius is not None:
            metrics.append(
                f"Estimated star boundary radius: {result.star_boundary_radius:.3f} px"
            )
        else:
            metrics.append("Estimated star boundary radius: not available")

        if result.gaussian_fwhm is not None:
            metrics.append(f"Gaussian-fit FWHM: {result.gaussian_fwhm:.3f} px")
        else:
            metrics.append("Gaussian-fit FWHM: not available")


        self.metrics_text.configure(state="normal")
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", "\n".join(metrics))
        self.metrics_text.configure(state="disabled")

    def _draw_psf_annotations(self, result: PSFResult) -> None:
        """
        Draw the centroid marker, the 80% flux circle, and the rejected outlier pixels.
        """
        self._clear_annotations()

        centroid_artist = self.ax_img.scatter(
            [result.cx],
            [result.cy],
            marker="x",
            s=120,
            c="red",
            linewidths=2.5,
            zorder=10,
        )
        self.annotation_artists.append(centroid_artist)

        # Red dashed circle = automatic PSF size, defined by 80% encircled flux.
        if result.psf_radius is not None and result.psf_radius > 0:
            psf_circle = Circle(
                (result.cx, result.cy),
                result.psf_radius,
                fill=False,
                edgecolor="red",
                linestyle="--",
                linewidth=2.0,
                zorder=8,
            )
            self.ax_img.add_patch(psf_circle)
            self.annotation_artists.append(psf_circle)

        if result.outlier_x.size > 0:
            max_show = 200
            ox = result.outlier_x[:max_show]
            oy = result.outlier_y[:max_show]
            out_artist = self.ax_img.scatter(
                ox,
                oy,
                marker="o",
                s=45,
                facecolors="none",
                edgecolors="red",
                linewidths=1.5,
                zorder=9,
            )
            self.annotation_artists.append(out_artist)

        self.canvas_img.draw_idle()

    def _clear_annotations(self) -> None:
        """
        Remove all annotation artists currently drawn on the image.
        """
        for artist in self.annotation_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.annotation_artists.clear()

    def _clear_psf_results(self) -> None:
        """
        Reset the PSF plot area and remove prior analysis state.
        """
        self.psf_result = None
        self._clear_annotations()

        self.ax_radial.clear()
        self.ax_radial.set_title("Radial Profile")
        self.ax_radial.set_xlabel("Radius (pixels)")
        self.ax_radial.set_ylabel("Intensity")

        self.ax_enc.clear()
        self.ax_enc.set_title("Encircled Energy")
        self.ax_enc.set_xlabel("Radius (pixels)")
        self.ax_enc.set_ylabel("Normalized flux")
        self.ax_enc.set_ylim(0.0, 1.02)

        self.fig_psf.tight_layout()
        self.canvas_psf.draw_idle()


def main() -> int:
    app = FITSImageAnalyzer()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())