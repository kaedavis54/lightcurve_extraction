"""
Jupiter Meteor Impact Model
============================
Based on the methods of Hueso et al. 2018:
  "Small impacts on the giant planet Jupiter"
  A&A 617, A68 (2018)
  https://doi.org/10.1051/0004-6361/201832689

Full pipeline
-------------
  Part A – Light curve extraction from raw video frames
    A1  Load PNG frames and convert to grayscale
    A2  Co-register frames by FFT cross-correlation
    A3  Build median reference image from quiet frames
    A4  Detect flash frame and location via differential imaging
    A5  Aperture photometry on every differential frame
    A6  Sum excess counts → I_star;  sum disk counts → I_Jupiter

  Part B – Impact energy and impactor characterisation  (Eq. numbers
           refer to Hueso et al. 2018)
    B1  Compute conversion factor K  (Eq. 2)
    B2  Detected flash energy L_star  (Eq. 1)
    B3  Bolometric correction L_f(T)  (Eq. 4)
    B4  Kinetic energy via luminous efficiency η
    B5  Impactor mass and diameter
"""

import numpy as np
import os
import re
from pathlib import Path
from PIL import Image
from scipy.ndimage import shift
from scipy.interpolate import interp1d
from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd

# ---------------------------------------------------------------------------
# Physical & Jovian constants
# ---------------------------------------------------------------------------

R_JUP_EQ  = 71492e3       # Jupiter equatorial radius [m]
R_JUP_POL = 66854e3       # Jupiter polar radius      [m]
S_EARTH   = 1361.0         # Solar constant at 1 AU    [W m⁻²]


# ===========================================================================
# PART A — LIGHT CURVE EXTRACTION
# ===========================================================================

# ---------------------------------------------------------------------------
# A1.  Frame loading
# ---------------------------------------------------------------------------

def load_frames(frame_dir: str, pattern: str = "frame*.png") -> tuple:
    """
    Load all PNG frames from *frame_dir* matching *pattern*.

    Frames are converted to float64 grayscale by averaging the three
    colour channels, consistent with how Hueso et al. (2018) treat
    Bayer RGB video.

    Parameters
    ----------
    frame_dir : str
        Directory containing the PNG frames.
    pattern : str
        Glob pattern for frame files.

    Returns
    -------
    frames : ndarray, shape (N, H, W)
        Grayscale frames in filename order, float64 [0–255].
    frame_paths : list of str
        Sorted file paths corresponding to each frame.
    frame_numbers : list of int
        Integer frame indices parsed from filenames.
    """
    paths = sorted(Path(frame_dir).glob(pattern)) #searches for files in a directory with a specific naming pattern
    if not paths:
        raise FileNotFoundError(f"No frames matching '{pattern}' in {frame_dir}")

    frames, numbers = [], []
    for p in paths:
        img = np.array(Image.open(p), dtype=np.float64)
        gray = img.mean(axis=2) if img.ndim == 3 else img #turns the RGB image into grayscale
        frames.append(gray)
        m = re.search(r"(\d+)", p.stem) #pulls the fram number from the filename and turns it into an integer value
        numbers.append(int(m.group(1)) if m else -1)

    #returns stacked frame arrays so you can call frame array from frame[i], paths in string format, and the frame numbers
    return np.stack(frames, axis=0), [str(p) for p in paths], numbers


# ---------------------------------------------------------------------------
# A2.  Co-registration by FFT cross-correlation
# ---------------------------------------------------------------------------

def coregister_frames(frames: np.ndarray, ref_idx: int = 0) -> tuple:
    """
    Shift each frame to align with *frames[ref_idx]* using FFT-based
    image cross-correlation.

    Atmospheric seeing causes the planet to drift between frames.
    Co-registration removes this drift so differential images only
    contain genuine brightness changes, not positional offsets.

    Parameters
    ----------
    frames : ndarray, shape (N, H, W)
        Grayscale frames.
    ref_idx : int
        Index of the frame to use as the alignment reference.

    Returns
    -------
    aligned : ndarray, shape (N, H, W)
        Co-registered frames.
    shifts : ndarray, shape (N, 2)
        Applied (dy, dx) shifts in pixels for each frame.
    """
    #from numpy.fft import fft2, ifft2, fftshift

    reference = frames[ref_idx]
    #calculate the complex conjugate of the FFT of the reference image
    #subtracts mean brightness from reference - removes overall brightness
    #so correlation measures structural similarity rather than brightness matching
    ref_fft   = np.conj(fft2(reference - reference.mean())) 
    aligned   = np.empty_like(frames)
    shifts    = np.zeros((len(frames), 2))

    for i, frame in enumerate(frames):
        if i == ref_idx:
            aligned[i] = frame
            continue
        #slide the reference fram across the frame and compute sum of pixel-wise products at every offset
        #fftshift move zer-frequency point from the corner of the frame to the middle, a peak at the center
        #means that they are already aligned
        corr  = np.abs(fftshift(ifft2(fft2(frame - frame.mean()) * ref_fft)))
        #peak of corr tells the shift between the frame and reference frame
        peak  = np.unravel_index(np.argmax(corr), corr.shape)
        #displacement of peak from center in pixels
        dy    = peak[0] - corr.shape[0] // 2
        dx    = peak[1] - corr.shape[1] // 2
        #move the frame in the negative dy and dx directions to correct
        #fill any pixel values that get shifted outside of frame boundary to black
        # -- okay since background is black
        aligned[i] = shift(frame, shift=(-dy, -dx), mode="constant", cval=0.0)
        shifts[i]  = (dy, dx) #records correction made for diagnostic purposes - large shifts = bad seeing

    return aligned, shifts


# ---------------------------------------------------------------------------
# A3.  Reference image
# ---------------------------------------------------------------------------

def build_reference(frames: np.ndarray, quiet_indices: list) -> np.ndarray:
    """
    Build a reference image by median-stacking a set of quiet
    (non-flash) frames.

    The median stack suppresses cosmic rays, hot pixels, and transient
    brightness variations while preserving the stable Jupiter disk.

    Parameters
    ----------
    frames : ndarray, shape (N, H, W)
        Co-registered grayscale frames.
    quiet_indices : list of int
        Indices of frames assumed to contain no flash.

    Returns
    -------
    reference : ndarray, shape (H, W)
        Median-stacked reference image.
    """
    return np.median(frames[quiet_indices], axis=0)


# ---------------------------------------------------------------------------
# A4.  Flash detection
# ---------------------------------------------------------------------------

def detect_flash(
    frames: np.ndarray,
    reference: np.ndarray,
    n_sigma: float = 5.0,
    aperture_radius: float = 10.0,
) -> tuple:
    """
    Identify all frames containing flash signal and locate the flash
    centroid in each one individually.

    A frame is considered a flash frame if its peak differential value
    exceeds n_sigma times the standard deviation of peak values across
    all quiet frames.  The flash location is centroided separately in
    each flash frame rather than using a single fixed pixel, correcting
    for residual seeing distortion after co-registration.

    Parameters
    ----------
    frames : ndarray, shape (N, H, W)
    reference : ndarray, shape (H, W)
    n_sigma : float
        Detection threshold in units of the noise standard deviation. 
        5sigma is common threshold in astronomy for claiming confident detection.
        3sigma is useful if the flash is faint relative to the frame-to-frame noise
        7-10sigma is useful if the seing-induced noise are triggering flase detections
        choose based on whether there are clear outliers from noise floor where the flash is seen
    aperture_radius : float
        Radius for the mask that encompasses entire flash PSF.
        Should be checked to ensure flash is being captured in its entriety.

    Returns
    -------
    flash_indices : ndarray of int
        Indices of all frames identified as containing flash signal.
    flash_yx_per_frame : dict {int: (row, col)}
        Per-frame centroid position of the flash.
    peak_flash_idx : int
        Index of the single brightest flash frame (for aperture
        placement in frames where the centroid fit is uncertain).
    peak_vals : ndarray, shape (N,)
        Peak differential value for every frame.
    """
    peak_vals = np.array([(f - reference).max() for f in frames])

    # Estimate noise from the standard deviation of peak values —
    # quiet frames cluster near zero, flash frames are outliers
    threshold       = peak_vals.mean() + n_sigma * peak_vals.std()
    flash_indices   = np.where(peak_vals > threshold)[0]
    peak_flash_idx  = int(np.argmax(peak_vals))

    # If threshold catches nothing (very faint flash), fall back to
    # the single brightest frame
    if len(flash_indices) == 0:
        flash_indices = np.array([peak_flash_idx])

    # Centroid the flash location in each flash frame individually
    flash_yx_per_frame = {}
    for idx in flash_indices:
        diff = frames[idx] - reference
        # Mask to a region around the rough peak to avoid noise peaks
        rough_yx = np.unravel_index(np.argmax(diff), diff.shape)
        cy, cx   = rough_yx
        H, W     = diff.shape
        y, x     = np.ogrid[:H, :W] #np,ogrid more memory efficient than np.meshgrid
        #create circular mask with aperture_radius and only if the pixel is brighter than the reference
        region   = (np.sqrt((y - cy)**2 + (x - cx)**2) <= aperture_radius) & (diff > 0)
        if region.any(): #checks for True values in region, i.e., are there any pixels with positive differential within mask area
            total  = diff[region].sum()
            y_full = np.broadcast_to(y, diff.shape)
            x_full = np.broadcast_to(x, diff.shape)
            cy_c   = y_full[region] @ (diff[region] / total)
            cx_c   = x_full[region] @ (diff[region] / total)
            flash_yx_per_frame[int(idx)] = (float(cy_c), float(cx_c))
        else:
            flash_yx_per_frame[int(idx)] = rough_yx

    return flash_indices, flash_yx_per_frame, peak_flash_idx, peak_vals

def save_impact_stack(
    aligned_frames: np.ndarray,
    flash_indices: np.ndarray,
    reference: np.ndarray,
    save_path: str = None,
) -> np.ndarray:
    """
    Create and save a stacked image of all detected flash frames.

    The stack is computed as the mean of all flash frames after
    subtracting the reference, isolating the impact signal.  The
    result is saved as a PNG and also returned as an array.

    Two versions are saved side by side:
      Left  – mean stack of raw flash frames (shows Jupiter + flash)
      Right – mean stack of differential images (flash signal only)

    Parameters
    ----------
    aligned_frames : ndarray, shape (N, H, W)
        Co-registered grayscale frames.
    flash_indices : ndarray of int
        Indices of all detected flash frames.
    reference : ndarray, shape (H, W)
        Median-stacked reference image.
    save_path : str or None
        Path to save the stacked image PNG.  If None, displayed
        interactively.

    Returns
    -------
    stack_diff : ndarray, shape (H, W)
        Mean differential stack of flash frames.
    """

    flash_frames = aligned_frames[flash_indices]

    # Mean stack of raw flash frames
    stack_raw  = flash_frames.mean(axis=0)

    # Mean stack of differential images
    stack_diff = (flash_frames - reference).mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: raw stack
    axes[0].imshow(stack_raw, cmap="gray", origin="upper",
                   vmin=np.percentile(stack_raw, 1),
                   vmax=np.percentile(stack_raw, 99))
    axes[0].set_title(f"Mean stack — {len(flash_indices)} flash frame(s)")
    axes[0].axis("off")

    # Right: differential stack
    vmin = np.percentile(stack_diff, 10)
    vmax = np.percentile(stack_diff, 99.9)
    axes[1].imshow(stack_diff, cmap="gray", origin="upper",
                   vmin=vmin, vmax=vmax)
    axes[1].set_title("Mean differential stack (flash − reference)")
    axes[1].axis("off")

    fig.suptitle(
        f"Impact stack  |  {len(flash_indices)} flash frame(s) combined",
        fontsize=10,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Impact stack saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)

    return stack_diff

# ---------------------------------------------------------------------------
# A5.  Aperture photometry
# ---------------------------------------------------------------------------

def aperture_photometry(
    diff_image: np.ndarray,
    center_yx: tuple,
    aperture_radius: float,
    sky_inner: float,
    sky_outer: float,
) -> tuple:
    """
    Measure the background-subtracted flux within a circular aperture,
    and estimate its uncertainty from the sky background noise.

    A circular aperture of radius *aperture_radius* is placed at
    *center_yx*.  The local sky background is estimated from the median
    of pixels in an annulus between *sky_inner* and *sky_outer* and
    subtracted from the aperture sum.

    The per-frame uncertainty is:
        σ = sqrt(N_pix) × sky_std
    where N_pix is the number of pixels in the aperture and sky_std is the
    standard deviation of pixel values in the sky annulus. This
    captures the diminant noise source - spatial variation of the
    background - without requiring knowledge of the gain or read noise.

    Parameters
    ----------
    diff_image : ndarray, shape (H, W)
        Differential image (frame − reference) in DN.
    center_yx : tuple (row, col)
        Centre of the aperture in pixel coordinates.
    aperture_radius : float
        Radius of the photometric aperture [pixels].
    sky_inner, sky_outer : float
        Inner and outer radii of the sky annulus [pixels].

    Returns
    -------
    flux : float
        Background-subtracted sum of DN within the aperture.
    uncertainty : float
        1-sigma uncertainty on the flux [DN]
    """
    H, W   = diff_image.shape
    cy, cx = center_yx
    y, x   = np.ogrid[:H, :W]
    r      = np.sqrt((y - cy)**2 + (x - cx)**2)

    aperture_mask = r <= aperture_radius
    sky_mask      = (r > sky_inner) & (r <= sky_outer)

    sky_pixels = diff_image[sky_mask] if sky_mask.any() else np.array([0.0])
    sky_median = float(np.median(sky_pixels))
    sky_std    = float(np.std(sky_pixels))
    n_pix      = int(aperture_mask.sum())

    flux        = float((diff_image[aperture_mask] - sky_median).sum())
    uncertainty = float(np.sqrt(n_pix) * sky_std)
    return flux, uncertainty


# ---------------------------------------------------------------------------
# A6.  Jupiter disk counts
# ---------------------------------------------------------------------------

def jupiter_disk_counts(
    reference: np.ndarray,
    threshold_frac: float = 0.05,
) -> float:
    """
    Sum all pixel counts belonging to Jupiter's disk in the reference image.

    The disk is identified as all pixels above *threshold_frac* × peak.

    Parameters
    ----------
    reference : ndarray, shape (H, W)
        Median-stacked reference image in DN.
    threshold_frac : float
        Fraction of peak value used as the disk/sky boundary.

    Returns
    -------
    I_Jupiter : float
        Total DN summed over Jupiter's disk.
    """
    return float(reference[reference > threshold_frac * reference.max()].sum())

def measure_I_jupiter(
    image_path: str,
    threshold_frac: float = 0.05,
) -> float:
    """
    Measure I_Jupiter from a single reference image.

    Loads the image, converts to grayscale by averaging colour channels,
    and sums all pixels belonging to Jupiter's disk.  The disk is
    identified as all pixels above *threshold_frac* times the peak pixel
    value — the same method used internally when processing video frames.

    Parameters
    ----------
    image_path : str
        Path to the reference image (PNG, JPEG, TIFF, etc.).
    threshold_frac : float
        Fraction of the peak pixel value used as the disk/sky boundary.
        Default 0.05 (5 % of peak).  Increase if sky background noise
        is being counted as disk; decrease if the dim limb of Jupiter
        is being excluded.

    Returns
    -------
    I_Jupiter : float
        Total DN summed over Jupiter's disk in the reference image.
    """
    img  = np.array(Image.open(image_path), dtype=np.float64)
    gray = img.mean(axis=2) if img.ndim == 3 else img

    threshold = threshold_frac * gray.max()
    I_Jupiter = float(gray[gray > threshold].sum())

    print(f"  Image loaded : {image_path}")
    print(f"  Peak pixel   : {gray.max():.1f} DN")
    print(f"  Threshold    : {threshold:.1f} DN  ({threshold_frac*100:.1f}% of peak)")
    print(f"  Disk pixels  : {(gray > threshold).sum()}")
    print(f"  I_Jupiter    : {I_Jupiter:.2f} DN")

    return I_Jupiter


# ---------------------------------------------------------------------------
# A – Full light curve extraction pipeline
# ---------------------------------------------------------------------------

def extract_light_curve(
    frame_dir: str,
    pattern: str = "frame*.png",
    aperture_radius: float = 10.0,
    sky_inner: float = 15.0,
    sky_outer: float = 25.0,
    quiet_fraction: float = 0.8,
    disk_threshold_frac: float = 0.05,
    n_sigma: float = 5.0,
    savgol_window: int = 5,
    savgol_polyorder: int = 2,
) -> dict:
    """
    End-to-end light curve extraction.

    Loads raw PNG frames, co-registers them, builds a reference image,
    detects the flash automatically, and returns I_star and I_Jupiter
    ready for use in :func:`jupiter_impact_model`.

    Parameters
    ----------
    frame_dir : str
        Directory containing the PNG frames.
    pattern : str
        Glob pattern for frame files.
    aperture_radius : float
        Photometric aperture radius [pixels].
    sky_inner, sky_outer : float
        Inner and outer radii of the sky annulus [pixels].
    quiet_fraction : float
        Fraction of frames (those with the smallest peak differential)
        used to build the reference image.
    disk_threshold_frac : float
        Threshold fraction of peak for Jupiter disk detection.
    n_sigma : float
        Detection threshold for flash frames in units of the noise
        standard deviation of peak differential values.
    savgol_window : int
        Window length for the Savitzky-Golay smoothing filter (must be odd and greater than 
        savgol_polyorder).
        Smaller values follow the raw curve more closely; larger values smooth more aggressively.
        Set to 0 to skip smoothing.
    savgol_polyorder : int
        Polynomial order for the Savitzky-Golay filter.
        Must be less than savgol_window. Order 2 or 3 works well for flash profiles.

    Returns
    -------
    dict with keys:
        I_star             – integrated flash excess counts [DN]
        I_Jupiter          – total Jupiter disk counts in reference [DN]
        I_star_peak        – peak single-frame excess DN
        light_curve        – per-frame aperture flux array [DN]
        uncertainties      – 1-sigma per-frame flux uncertainty [DN]
        light_curve_smooth – Savitzky-Golay smoother light cirve [DN]
        frame_numbers      – integer frame indices
        flash_indices      – indices of all detected flash frames
        flash_yx_per_frame – per-frame centroid positions
        peak_flash_idx     – index of the brightest flash frame
        reference          – 2-D reference image array
        aligned_frames     – co-registered frame stack (N, H, W)
        shifts             – (N, 2) co-registration shifts applied
    """
    # A1: load
    frames, paths, frame_numbers = load_frames(frame_dir, pattern)
    N = len(frames)
    print(f"  Loaded {N} frames")

    # A2: co-register
    aligned, shifts = coregister_frames(frames, ref_idx=0)
    print(f"  Co-registered {N} frames")

    # A2b: normalise each frame to the same total flux so that
    # atmospheric transparency variations don't leave a nonzero
    # background in the differential images
    disk_mask_norm = aligned[0] > 0.05 * aligned[0].max()
    ref_flux       = aligned[0][disk_mask_norm].sum()
    normalised     = np.empty_like(aligned)
    for i, frame in enumerate(aligned):
        frame_flux      = frame[disk_mask_norm].sum()
        scale           = ref_flux / frame_flux if frame_flux > 0 else 1.0
        normalised[i]   = frame * scale
    aligned = normalised
    print(f"  Flux-normalised {N} frames")

    # A3: build reference from quietest frames
    rough_ref    = np.median(aligned, axis=0)
    peak_vals    = np.array([(f - rough_ref).max() for f in aligned])
    n_quiet      = max(2, int(N * quiet_fraction))
    quiet_idx    = np.argsort(peak_vals)[:n_quiet].tolist()
    reference    = build_reference(aligned, quiet_idx)
    print(f"  Reference built from {n_quiet} quiet frames")

    # A4: detect all flash frames with per-frame centroiding
    flash_indices, flash_yx_per_frame, peak_flash_idx, peak_vals = detect_flash(
        aligned, reference, n_sigma=n_sigma, aperture_radius=aperture_radius,
    )
    print(f"  Flash detected in {len(flash_indices)} frame(s): "
          f"{[frame_numbers[i] for i in flash_indices]}")
    for idx in flash_indices:
        yx = flash_yx_per_frame[idx]
        print(f"    Frame {frame_numbers[idx]}: centroid "
              f"(row={yx[0]:.1f}, col={yx[1]:.1f})")

    # A5: aperture photometry on every frame using per-frame centroid
    # where available, falling back to the peak frame location elsewhere
    fallback_yx = flash_yx_per_frame[peak_flash_idx]
    light_curve = np.zeros(N)
    uncertainties = np.zeros(N)
    for i, frame in enumerate(aligned):
        diff           = frame - reference
        center         = flash_yx_per_frame.get(i, fallback_yx)
        light_curve[i], uncertainties[i] = aperture_photometry(
            diff, center, aperture_radius, sky_inner, sky_outer
        )

    # A6: integrate
    I_star      = float(light_curve[light_curve > 0].sum())
    I_star_peak = float(light_curve.max())
    I_Jupiter   = jupiter_disk_counts(reference, disk_threshold_frac)

    # A7: Savitzky-Golay smoothing
    # window_length must be odd, >= polyorder+1, and <= N
    # If N is too small or savgol_window=0, smoothing is skipped
    if savgol_window > 0 and N >= savgol_window and savgol_window > savgol_polyorder:
        # Ensure window is odd
        window = savgol_window if savgol_window % 2 == 1 else savgol_window + 1
        light_curve_smooth = savgol_filter(light_curve, window, savgol_polyorder)
    else:
        light_curve_smooth = light_curve.copy()
        if savgol_window > 0:
            print(f"  Warning: savgol_window={savgol_window} too large for "
                  f"{N} frames — smoothing skipped")

    print(f"  I_star       = {I_star:.2f} DN")
    print(f"  I_star_peak  = {I_star_peak:.2f} DN")
    print(f"  I_Jupiter    = {I_Jupiter:.2f} DN")
    print(f"  Ratio        = {I_star / I_Jupiter:.4e}")

    return {
        "I_star":             I_star,
        "I_Jupiter":          I_Jupiter,
        "I_star_peak":        I_star_peak,
        "light_curve":        light_curve,
        "uncertainties":      uncertainties,
        "light_curve_smooth": light_curve_smooth,
        "frame_numbers":      frame_numbers,
        "flash_indices":      flash_indices,
        "flash_yx_per_frame": flash_yx_per_frame,
        "peak_flash_idx":     peak_flash_idx,
        "reference":          reference,
        "aligned_frames":     aligned,
        "shifts":             shifts,
    }


# ===========================================================================
# PART B — IMPACT ENERGY AND IMPACTOR CHARACTERISATION
# ===========================================================================

# ---------------------------------------------------------------------------
# B – Spectral data loaders
# ---------------------------------------------------------------------------

def load_solar_spectrum(csv_path: str) -> tuple:
    """
    Load the Colina et al. (1996) solar spectrum from CSV.

    CSV format : index | Wavelength [Å] | log10(Flux) [erg s⁻¹ cm⁻² Å⁻¹]

    Conversions applied
    -------------------
    Wavelength : Å → nm  (÷ 10)
    Flux       : 10^x  then  erg s⁻¹ cm⁻² Å⁻¹ → W m⁻² nm⁻¹  (× 1e-3)
                 (1 erg = 1e-7 J; 1 cm² = 1e-4 m²; 1 Å = 0.1 nm
                  → 1e-7 / 1e-4 / 0.1 = 1e-3)
    """
    df       = pd.read_csv(csv_path)
    wav_nm   = df.iloc[:, 1].values.astype(float) / 10.0
    flux_W   = 10.0 ** df.iloc[:, 2].values.astype(float) * 1e-3
    order    = np.argsort(wav_nm)
    return wav_nm[order], flux_W[order]


def load_jupiter_reflectivity(csv_path: str) -> tuple:
    """
    Load the Karkoschka (1994) Jupiter geometric albedo spectrum from CSV.

    CSV format : index | Wavelength [nm] | Albedo (dimensionless)

    No unit conversion needed — wavelength is already in nm.
    """
    df     = pd.read_csv(csv_path, index_col=0)
    wav_nm = df.iloc[:, 0].values.astype(float)
    albedo = df.iloc[:, 1].values.astype(float)
    order  = np.argsort(wav_nm)
    return wav_nm[order], albedo[order]


def solar_spectrum(wavelength_nm: np.ndarray, csv_path: str = None) -> np.ndarray:
    """
    Solar spectral irradiance at 1 AU interpolated onto *wavelength_nm*
    [W m⁻² nm⁻¹].  Uses tabulated CSV if supplied, else blackbody fallback.
    """
    lam = np.asarray(wavelength_nm, dtype=float)
    if csv_path is not None:
        wav, flux = load_solar_spectrum(csv_path)
        return interp1d(wav, flux, kind="linear",
                        bounds_error=False, fill_value=0.0)(lam)
    # Fallback: Planck at T_sun normalised to S_EARTH
    h, c, k = 6.626e-34, 2.998e8, 1.381e-23
    lam_m   = lam * 1e-9
    B       = (2*h*c**2 / lam_m**5) / (np.exp(h*c / (lam_m*k*5778.0)) - 1.0)
    spect_solar = (B / np.trapezoid(B, lam_m)) * S_EARTH * 1e-9
    return spect_solar


def jupiter_reflectivity(wavelength_nm: np.ndarray, csv_path: str = None) -> np.ndarray:
    """
    Jupiter geometric albedo I/F interpolated onto *wavelength_nm*.
    Uses tabulated CSV if supplied, else analytic fallback.
    """
    lam = np.asarray(wavelength_nm, dtype=float)
    if csv_path is not None:
        wav, alb = load_jupiter_reflectivity(csv_path)
        return np.clip(interp1d(wav, alb, kind="linear",
                                bounds_error=False, fill_value=0.0)(lam), 0.0, 1.0)
    base  = 0.52 * np.ones_like(lam)
    base -= 0.30 * np.exp(-0.5 * ((lam - 890.0) / 15.0)**2) #simulates methane absorption band at 890 nm
    base -= 0.15 * np.exp(-0.5 * ((lam - 728.0) / 12.0)**2) #methane absorption band at 728 nm
    base -= 0.08 * np.exp(-0.5 * ((lam - 619.0) / 10.0)**2) #methane absorption band at 619 nm
    jup_ref = np.clip(base, 0.0, 1.0)
    return jup_ref


# ---------------------------------------------------------------------------
# B – Camera / filter spectral response  C(λ)
# ---------------------------------------------------------------------------

def camera_filter_response(
    wavelength_nm: np.ndarray,
    qe_csvs: list = None,
    # qe_b_csv: str = None,
    # qe_v_csv: str = None,
    # qe_r_csv: str = None,
    transmission_csv: str = None,
) -> np.ndarray:
    """
    Combined camera + filter spectral response C(λ).

    CSV format for QE files   : index | Wavelength [nm] | QE [%]
    CSV format for filter file : index | Wavelength [nm] | Transmission [%]

    Accepts any number of QE channel CSVs (e.g. one for monochrome,
    three for RGB).  Channels are combined as an integral-weighted average:

        C(λ) = Σ [w_i · QE_i(λ)] / Σ w_i  ×  T(λ)

    where w_i = ∫ QE_i(λ) dλ.  This ensures broader or more sensitive
    channels contribute proportionally rather than equally.


    Falls back to a generic Gaussian Bayer-RGB approximation if CSVs
    are not supplied.
    """
    lam = np.asarray(wavelength_nm, dtype=float)

    # have_csvs = all(p is not None for p in
    #                 [qe_b_csv, qe_v_csv, qe_r_csv, transmission_csv])

    have_csvs = qe_csvs is not None and len(qe_csvs) > 0

    if have_csvs:
        def _load(path):
            df  = pd.read_csv(path, index_col=0)
            wav = df.iloc[:, 0].values.astype(float)
            val = df.iloc[:, 1].values.astype(float) / 100.0   # % → fraction
            order = np.argsort(wav)
            return interp1d(wav[order], val[order], kind="linear",
                            bounds_error=False, fill_value=0.0)(lam)

        # qe_b_i = _load(qe_b_csv)
        # qe_v_i = _load(qe_v_csv)
        # qe_r_i = _load(qe_r_csv)

        # w_b = np.trapezoid(qe_b_i, lam)
        # w_v = np.trapezoid(qe_v_i, lam)
        # w_r = np.trapezoid(qe_r_i, lam)
        # w_total = w_b + w_v + w_r

        # qe_mean = ((w_b * qe_b_i) + (w_v * qe_v_i) + (w_r * qe_r_i)) / w_total #weighted average of quantum efficiency
        # T       = _load(transmission_csv)
        # C       = qe_mean * T #camera response

        # Load all QE channels and compute integral-weighted average
        qe_channels = [_load(p) for p in qe_csvs]
        weights     = [np.trapezoid(qe, lam) for qe in qe_channels]
        w_total     = sum(weights)
        qe_mean     = sum(w * qe for w, qe in zip(weights, qe_channels)) / w_total

        # Apply filter transmission if supplied
        T = _load(transmission_csv) if transmission_csv is not None else np.ones_like(lam)
        C = qe_mean * T

    else:
        # Fallnack: generic Gaussian Baer-RGB, no filter
        #simulate guassian curves centered at channel's peak wavelength
        qe_r = np.exp(-0.5 * ((lam - 620.0) / 80.0)**2)
        qe_g = np.exp(-0.5 * ((lam - 520.0) / 60.0)**2)
        qe_b = np.exp(-0.5 * ((lam - 450.0) / 50.0)**2)
        C    = (qe_r + qe_g + qe_b) / 3.0 #unweighted

    #normalize spectral (camera) response = 1 with no filter
    norm = np.trapezoid(C, lam)
    if norm > 0:
        C = C / norm * (lam[-1] - lam[0])
    return C


# ---------------------------------------------------------------------------
# B1.  Conversion factor K  (Eq. 2)
# ---------------------------------------------------------------------------

def compute_conversion_factor(
    d_jupiter_au: float,
    wavelength_nm: np.ndarray,
    C: np.ndarray,
    dt_frame: float,
    solar_csv: str = None,
    albedo_csv: str = None,
) -> float:
    """
    Compute the energy conversion factor K [J]  (Eq. 2, Hueso et al. 2018).

        K = [S_J · π · R_eq · R_p]
            · [∫ F(λ) · (I/F)(λ) · C(λ) dλ  /  ∫ F(λ) dλ]
            · Δt

    Parameters
    ----------
    d_jupiter_au : float
        Jupiter–Sun distance [AU].
    wavelength_nm : ndarray
        Wavelength grid [nm].
    C : ndarray
        Camera+filter spectral response on the same grid.
    dt_frame : float
        Exposure time per frame [s].
    solar_csv, albedo_csv : str or None
        Paths to tabulated spectral data files.

    Returns
    -------
    K : float
        Conversion factor [J].
    """
    lam    = wavelength_nm
    S_J    = S_EARTH / d_jupiter_au**2
    P_J    = S_J * np.pi * R_JUP_EQ * R_JUP_POL
    F_lam  = solar_spectrum(lam, csv_path=solar_csv)
    IoverF = jupiter_reflectivity(lam, csv_path=albedo_csv)
    num    = np.trapezoid(F_lam * IoverF * C, lam)
    denom  = np.trapezoid(F_lam, lam)
    return P_J * (num / denom if denom > 0 else 0.0) * dt_frame


# ---------------------------------------------------------------------------
# B2.  Detected flash energy L_star  (Eq. 1)
# ---------------------------------------------------------------------------

def detected_flash_energy(I_star: float, I_Jupiter: float, K: float) -> float:
    """
    Detected luminous energy of the flash  (Eq. 1, Hueso et al. 2018).

        L_star = K · (I_star / I_Jupiter)
    """
    return K * (I_star / I_Jupiter)


# ---------------------------------------------------------------------------
# B3.  Bolometric correction  (Eq. 4)
# ---------------------------------------------------------------------------

def bolometric_correction(
    L_star: float,
    T_flash_K: float,
    wavelength_nm: np.ndarray,
    C: np.ndarray,
) -> float:
    """
    Convert detector-band flash energy to total emitted luminous energy
    assuming a blackbody at *T_flash_K*  (Eq. 4, Hueso et al. 2018).

        L_f(T) = L_star · [∫ F_BB(T,λ) dλ  /  ∫ F_BB(T,λ) · C(λ) dλ]

    Hueso et al. (2018) use a range of 3500–8500 K based on terrestrial
    fireball data, SL9 Galileo observations, and the 2010 Jovian bolide.
    """
    lam_m    = wavelength_nm * 1e-9
    h, c, k  = 6.626e-34, 2.998e8, 1.381e-23
    B        = (2*h*c**2 / lam_m**5) / (
                   np.exp(np.clip(h*c / (lam_m*k*T_flash_K), 0, 700)) - 1.0)
    B_nm     = B * 1e-9
    denom    = np.trapezoid(B_nm * C, wavelength_nm)
    numer    = np.trapezoid(B_nm,     wavelength_nm)
    return L_star * (numer / denom if denom > 0 else 1.0)


# ---------------------------------------------------------------------------
# B4.  Kinetic energy
# ---------------------------------------------------------------------------

def kinetic_energy_from_luminous(L_f: float) -> float:
    """
    Convert total luminous energy to impactor kinetic energy using the
    empirical power-law relation:
 
        E_kin = 8.2508 · E0^0.885
 
    where E0 is the total luminous energy in kt TNT and E_kin is
    returned in kt TNT, then converted to Joules.
 
    This power-law accounts for the nonlinear relationship between
    luminous and kinetic energy more accurately than a fixed luminous
 
    1 kt TNT = 4.184e12 J
    """
    KT_TO_J  = 4.184e12
    E0_kt    = L_f / KT_TO_J          # luminous energy in kt TNT
    E_kin_kt = 8.2508 * (E0_kt ** 0.885)
    return E_kin_kt * KT_TO_J         # back to Joules


# ---------------------------------------------------------------------------
# B5.  Impactor mass and diameter
# ---------------------------------------------------------------------------

def impactor_mass(E_kin: float, v_impact_kms: float = 60.0) -> float:
    """
    Derive impactor mass from kinetic energy.

        m = 2 · E_kin / v²

    Default impact velocity 60 km s⁻¹ (cometary; Hueso et al. 2018).
    """
    return 2.0 * E_kin / (v_impact_kms * 1e3)**2


def impactor_diameter(mass: float, density_kg_m3: float = 1000.0) -> float:
    """
    Sphere-equivalent diameter from mass and bulk density.

        d = 2 · (3m / 4πρ)^(1/3)
    """
    return 2.0 * ((3.0 * mass / (4.0 * np.pi * density_kg_m3)) ** (1.0 / 3.0))


# ---------------------------------------------------------------------------
# B – Flash apparent magnitude
# ---------------------------------------------------------------------------

def flash_magnitude(
    I_star_peak: float,
    I_Jupiter: float,
    V_jupiter: float = -2.5,
) -> float:
    """
    Apparent magnitude of the flash at peak brightness.

        m_flash = V_Jupiter - 2.5 · log10(I_star_peak / I_Jupiter)
    """
    if I_Jupiter <= 0 or I_star_peak <= 0:
        return np.nan
    return V_jupiter - 2.5 * np.log10(I_star_peak / I_Jupiter)


# ===========================================================================
# COMBINED END-TO-END PIPELINE
# ===========================================================================

def jupiter_impact_model(
    # --- Light-curve photometry (from extract_light_curve or supplied directly) ---
    I_star: float,
    I_Jupiter: float,
    I_star_peak: float = None,
    # --- Observing setup ---
    d_jupiter_au: float = 5.2,
    dt_frame_s: float = 25e-3,
    # --- Camera QE and filter transmission CSVs ---
    # qe_b_csv: str = None,
    # qe_v_csv: str = None,
    # qe_r_csv: str = None,
    qe_csvs: list = None,
    transmission_csv: str = None,
    # --- Solar spectrum and Jupiter albedo CSVs ---
    solar_csv: str = None,
    albedo_csv: str = None,
    # --- Cloud reflection ---
    fcr: float = 0.3,
    # --- Physical assumptions ---
    T_flash_range_K: tuple = (3500.0, 8500.0),
    v_kms_values: list = None, # defaults to [60, 65, 70] if None
    density_values: list = None, # defaults to [500, 2500, 5000] if None
    # --- Apparent magnitude ---
    V_jupiter: float = -2.5,
    # --- Wavelength grid ---
    lam_min_nm: float = 400.0,
    lam_max_nm: float = 1000.0,
    n_lam: int = 1000,
) -> dict:
    """
    Characterise a Jupiter meteor impact from photometric quantities.

    Can be called directly with *I_star* / *I_Jupiter* values obtained from
    :func:`extract_light_curve`, or with manually supplied values.

    Parameters
    ----------
    I_star : float
        Integrated flash excess counts [DN].
    I_Jupiter : float
        Total Jupiter disk counts in the reference image [DN].
    I_star_peak : float or None
        Peak single-frame excess DN (for magnitude estimate).
        Defaults to I_star if not provided.
    d_jupiter_au : float
        Jupiter–Sun distance [AU].
    dt_frame_s : float
        Exposure time per frame [s]  (= 1 / fps).
    qe_csvs : list of str or None
        Paths to QE CSV files [nm | %], one per camera channel.
        Pass a single-element list for monochrome cameras, or three
        elements for RGB.  If None, a generic fallback is used.
    transmission_csv : str or None
        Path to filter transmission CSV [nm | %].
    solar_csv : str or None
        Path to solar spectrum CSV (Colina et al. 1996).
    albedo_csv : str or None
        Path to Jupiter albedo CSV (Karkoschka 1994).
    fcr : float
        Cloud reflection ratio defined by cloud_reflection.py (Arimatsu et al. 2023)
        0.3 is the default estimate (Hueso et al. 2018)
    T_flash_range_K : tuple (T_low, T_high)
        Blackbody temperature range [K] for bolometric correction.
    v_kms_values : float
        Impact velocity [km s⁻¹].
    density_values : float
        Impactor bulk density [kg m⁻³].
    V_jupiter : float
        Apparent visual magnitude of Jupiter at observation.
    lam_min_nm, lam_max_nm : float
        Wavelength integration range [nm].
    n_lam : int
        Number of wavelength grid points.

    Returns
    -------
    dict
        K_J                     – conversion factor [J]
        L_star_J                – detected flash energy [J]
        L_f_mid/low/high_J      – bolometric flash energy [J]
        E_kin_mid/low/high_J    – kinetic energy [J]
        E_kin_mid/low/high_kt   – kinetic energy [kt TNT]
        mass_mid_kg             – impactor mass [kg]
        diameter_mid/low/high_m – impactor diameter [m]
        flash_magnitude         – apparent magnitude at peak
        T_low/mid/high_K        – temperature nodes used
    """
    if density_values is None:
        density_values = [500.0, 2500.0, 5000.0]
    if v_kms_values is None:
        v_kms_values = [60.0, 65.0, 70.0]

    lam = np.linspace(lam_min_nm, lam_max_nm, n_lam)
    C   = camera_filter_response(lam, qe_csvs, transmission_csv)
    K   = compute_conversion_factor(d_jupiter_au, lam, C, dt_frame_s,
                                    solar_csv, albedo_csv)

    # B2: detected flash energy
    L_star = detected_flash_energy(I_star, I_Jupiter, K)

    # B2b: cloud reflection correction from cloud_reflection.py based on Arimatsu et al. 2023
    L_star_corr = L_star*(1/(1+fcr))

    # B3 & B4: sample the full temperature range uniformly, compute
    # bolometric correction and kinetic energy at each temperature, then
    # take the mean of the resulting E_kin distribution.
    #
    # This follows Hueso et al. (2018): due to the nonlinearity of the
    # power-law conversion from luminous to kinetic energy, the mean
    # energy cannot be taken as the average of the range endpoints.
    # Instead, all blackbody temperatures in [T_low, T_high] are assumed
    # equally probable, giving a distribution of E_kin values whose mean
    # is the correct average energy estimate.
    T_low, T_high = T_flash_range_K
    n_samples     = int(T_high - T_low)
    T_samples     = np.linspace(T_low, T_high, n_samples)

    L_f_samples    = np.array([bolometric_correction(L_star_corr, T, lam, C)
                            for T in T_samples])
    E_kin_samples  = np.array([kinetic_energy_from_luminous(L_f)
                            for L_f in L_f_samples])

    # Boundary values for the reported range
    E_kin_low  = E_kin_samples.min()
    E_kin_high = E_kin_samples.max()

    # Mean of the distribution — the correct average given nonlinearity
    E_kin_mean = E_kin_samples.mean()

    #Luminous efficiency η = L_f / E_kin at each temperature sample
    eta_samples = L_f_samples / E_kin_samples
    eta_mean    = float(eta_samples.mean())
    eta_low     = float(eta_samples.min())
    eta_high    = float(eta_samples.max())

    # B5: impactor properties at mean, low, and high energies
    masses      = {v: impactor_mass(E_kin_mean, v) for v in v_kms_values}
    diameters   = {
        v: {rho: impactor_diameter(masses[v], rho) for rho in density_values}
        for v in v_kms_values
    }

    KT = 4.184e12   # 1 kt TNT in Joules

    peak = I_star_peak if I_star_peak is not None else I_star

    return {
        "K_J":              K,
        "L_star_J":         L_star_corr,
        "L_f_mean_J":       L_f_samples.mean(),
        "L_f_low_J":        L_f_samples.min(),
        "L_f_high_J":       L_f_samples.max(),
        "E_kin_mean_J":     E_kin_mean,
        "E_kin_low_J":      E_kin_low,
        "E_kin_high_J":     E_kin_high,
        "E_kin_mean_kt":    E_kin_mean  / KT,
        "E_kin_low_kt":     E_kin_low  / KT,
        "E_kin_high_kt":    E_kin_high / KT,
        "E_kin_samples":    E_kin_samples,
        "eta_mean":         eta_mean,
        "eta_low":          eta_low,
        "eta_high":         eta_high,
        "T_samples":        T_samples,
        "masses_kg":        masses,
        "diameters_m":      diameters,
        "v_kms_values":     v_kms_values,
        "density_values":   density_values,
        "flash_magnitude":  flash_magnitude(peak, I_Jupiter, V_jupiter),
        "T_low_K":          T_low,
        "T_high_K":         T_high,
    }

def jupiter_impact_model_from_watts(
    # --- Light curve CSV ---
    light_curve_csv: str,
    time_col: str = "time_s",
    flux_col: str = "optical_energy_J",
    dt_frame_s: float = None,
    # --- Camera QE and filter transmission CSVs ---
    qe_csvs: list = None,
    transmission_csv: str = None,
    # --- Physical assumptions ---
    T_flash_range_K: tuple = (3500.0, 8500.0),
    v_impact_kms: list = [60.0, 65.0, 70.0],
    density_values: list = [500.0, 2500.0, 5000.0],
    # --- Wavelength grid ---
    lam_min_nm: float = 400.0,
    lam_max_nm: float = 1000.0,
    n_lam: int = 1000,
) -> dict:
    """
    Characterise a Jupiter meteor impact from a light curve CSV
    already expressed in Watts (or J per frame) per row.

    Picks up the pipeline at the bolometric correction step, skipping
    the K conversion and I_star/I_Jupiter ratio entirely.

    Parameters
    ----------
    light_curve_csv : str
        Path to CSV file containing the light curve.
    time_col : str
        Name of the time column [s].  Default 'time_s'.
    flux_col : str
        Name of the flux column [W or J/frame].  Default 'optical_energy_J'.
        If the values are per-frame energies in Joules (as saved by the
        video pipeline) they are used directly; if they are power in Watts
        they are multiplied by dt_frame_s to get per-frame energy.
    dt_frame_s : float or None
        Exposure time per frame [s].  If None, derived automatically
        from the median time step in the time column.
    qe_csvs : list of str or None
        Paths to QE CSV files [nm | %], one per camera channel.
    transmission_csv : str or None
        Path to filter transmission CSV [nm | %].
    T_flash_range_K : tuple (T_low, T_high)
        Blackbody temperature range [K] for bolometric correction.
    v_impact_kms : list of float
        Impact velocities to evaluate [km s⁻¹].
    density_values : list of float
        Impactor bulk densities to evaluate [kg m⁻³].
    lam_min_nm, lam_max_nm : float
        Wavelength integration range [nm].
    n_lam : int
        Number of wavelength grid points.

    Returns
    -------
    Same dict as jupiter_impact_model minus K_J and flash_magnitude.
    """
    # Load CSV
    df            = pd.read_csv(light_curve_csv)
    times_s       = df[time_col].values.astype(float)
    light_curve_W = df[flux_col].values.astype(float)

    # Derive dt from the time column if not supplied
    if dt_frame_s is None:
        dt_frame_s = float(np.median(np.diff(times_s)))
        print(f"  dt_frame_s derived from time column: {dt_frame_s:.4f} s")

    print(f"  Loaded {len(light_curve_W)} frames from {light_curve_csv}")

    lam = np.linspace(lam_min_nm, lam_max_nm, n_lam)
    C   = camera_filter_response(lam, qe_csvs, transmission_csv)

    # Integrate positive frames only → total detected luminous energy [J]
    flash_mask  = light_curve_W > 0
    L_star      = float(light_curve_W[flash_mask].sum() * dt_frame_s)
    L_star_peak = float(light_curve_W.max() * dt_frame_s)

    print(f"  L_star (integrated) = {L_star:.3e} J")
    print(f"  L_star (peak frame) = {L_star_peak:.3e} J")

    # Bolometric correction sampled over temperature range
    T_low, T_high = T_flash_range_K
    T_samples     = np.linspace(T_low, T_high, 1000)

    L_f_samples   = np.array([bolometric_correction(L_star, T, lam, C)
                               for T in T_samples])
    E_kin_samples = np.array([kinetic_energy_from_luminous(L_f)
                               for L_f in L_f_samples])

    E_kin_low  = E_kin_samples.min()
    E_kin_high = E_kin_samples.max()
    E_kin_mean = E_kin_samples.mean()

    # Luminous efficiency η = L_f / E_kin at each temperature sample
    eta_samples = L_f_samples / E_kin_samples
    eta_mean    = float(eta_samples.mean())
    eta_low     = float(eta_samples.min())
    eta_high    = float(eta_samples.max())

    KT        = 4.184e12
    masses    = {v: impactor_mass(E_kin_mean, v) for v in v_impact_kms}
    diameters = {
        v: {rho: impactor_diameter(masses[v], rho) for rho in density_values}
        for v in v_impact_kms
    }

    return {
        "L_star_J":          L_star,
        "L_star_peak_J":     L_star_peak,
        "L_f_low_J":         L_f_samples.min(),
        "L_f_high_J":        L_f_samples.max(),
        "L_f_mean_J":        L_f_samples.mean(),
        "E_kin_mean_J":      E_kin_mean,
        "E_kin_low_J":       E_kin_low,
        "E_kin_high_J":      E_kin_high,
        "E_kin_mean_kt":     E_kin_mean / KT,
        "E_kin_low_kt":      E_kin_low  / KT,
        "E_kin_high_kt":     E_kin_high / KT,
        "E_kin_samples":     E_kin_samples,
        "eta_mean":          eta_mean,
        "eta_low":           eta_low,
        "eta_high":          eta_high,
        "T_samples":         T_samples,
        "masses_kg":         masses,
        "diameters_m":       diameters,
        "v_kms_values":      v_impact_kms,
        "density_values":    density_values,
        "T_low_K":           T_low,
        "T_high_K":          T_high,
        "times_s":           times_s,
        "light_curve_W":     light_curve_W,
        "dt_frame_s":        dt_frame_s,
    }

def plot_aperture_diagnostic(
    peak_idx : int,
    flash_frame: np.ndarray,
    reference: np.ndarray,
    flash_yx: tuple,
    aperture_radius: float,
    sky_inner: float,
    sky_outer: float,
    save_path: str = None,
) -> None:
    """
    Plot an aperture placement diagnostic to verify that the photometric
    aperture is correctly sized over the flash and the sky annulus is
    not contaminated by Jupiter's disk.
 
    Three panels are shown side by side:
      Left   – the flash frame itself, showing the raw brightness
      Center – the differential image (flash frame − reference),
               isolating the excess signal
      Right  – a zoomed-in view of the differential image around the
               flash location, with the aperture circle and sky annulus
               overlaid so their placement can be judged visually
 
    Parameters
    ----------
    flash_frame : ndarray, shape (H, W)
        The co-registered grayscale flash frame.
    reference : ndarray, shape (H, W)
        The median-stacked reference image.
    flash_yx : tuple (row, col)
        Centroid position of the flash.
    aperture_radius : float
        Radius of the photometric aperture [pixels].
    sky_inner, sky_outer : float
        Inner and outer radii of the sky annulus [pixels].
    save_path : str or None
        If provided, the figure is saved to this path.  Otherwise it
        is displayed interactively.
    """
    diff  = flash_frame - reference
    cy, cx = flash_yx
 
    # Zoom region: sky_outer + 20 px margin around the flash centroid
    margin = int(sky_outer) + 20
    H, W   = diff.shape
    y0 = max(0, int(cy) - margin)
    y1 = min(H, int(cy) + margin)
    x0 = max(0, int(cx) - margin)
    x1 = min(W, int(cx) + margin)
 
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
 
    # Left: flash frame
    axes[0].imshow(flash_frame, cmap="gray", origin="upper",
                   vmin=np.percentile(flash_frame, 1),
                   vmax=np.percentile(flash_frame, 99))
    axes[0].plot(cx, cy, "+", color="red", markersize=12, markeredgewidth=1.5)
    axes[0].set_title(f"Flash frame ({peak_idx})")
    axes[0].axis("off")
 
    # Centre: full differential image
    vmax = np.percentile(diff, 99.9)
    axes[1].imshow(diff, cmap="gray", origin="upper",
                   vmin=0, vmax=vmax)
    axes[1].plot(cx, cy, "+", color="red", markersize=12, markeredgewidth=1.5)
    axes[1].set_title("Differential image")
    axes[1].axis("off")
 
    # Right: zoomed differential with aperture and annulus overlaid
    diff_zoom = diff[y0:y1, x0:x1]
    axes[2].imshow(diff_zoom, cmap="gray", origin="upper",
                   vmin=0, vmax=vmax,
                   extent=[x0, x1, y1, y0])
 
    # Aperture circle
    aperture_circle = Circle((cx, cy), aperture_radius,
                              edgecolor="lime", facecolor="none",
                              linewidth=1.5, label=f"Aperture (r={aperture_radius:.0f} px)")
    # Sky annulus — inner and outer edges
    sky_inner_circle = Circle((cx, cy), sky_inner,
                               edgecolor="orange", facecolor="none",
                               linewidth=1.5, linestyle="--",
                               label=f"Sky inner (r={sky_inner:.0f} px)")
    sky_outer_circle = Circle((cx, cy), sky_outer,
                               edgecolor="orange", facecolor="none",
                               linewidth=1.5, linestyle="-",
                               label=f"Sky outer (r={sky_outer:.0f} px)")
 
    for patch in [aperture_circle, sky_inner_circle, sky_outer_circle]:
        axes[2].add_patch(patch)
 
    axes[2].plot(cx, cy, "+", color="black", markersize=12, markeredgewidth=1.5)
    axes[2].set_xlim(x0, x1)
    axes[2].set_ylim(y1, y0)
    axes[2].set_title("Zoomed differential + apertures")
    axes[2].legend(loc="lower left", fontsize=7, framealpha=0.7)
    axes[2].axis("off")
 
    fig.suptitle(
        f"Aperture diagnostic  |  centroid ({cy:.1f}, {cx:.1f})  |  "
        f"aperture r={aperture_radius:.0f} px  |  "
        f"sky {sky_inner:.0f}–{sky_outer:.0f} px",
        fontsize=10,
    )
    fig.tight_layout()
 
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Aperture diagnostic saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)

def corrected_light_curve(
    light_curve_DN: np.ndarray,
    I_Jupiter: float,
    K: float,
    wavelength_nm: np.ndarray,
    C: np.ndarray,
    T_flash_range_K: tuple = (3500.0, 8500.0),
    fcr: float = 0.3,
    n_T_samples: int = 1000,
) -> dict:
    """
    Apply the full Hueso et al. (2018) correction chain to a per-frame
    DN light curve, returning corrected luminous energy per frame.

    For each frame the pipeline is:
      1. DN → detected luminous energy via K and I_Jupiter  (Eq. 1)
      2. Cloud reflection correction: L_detected / (1 + fcr)
      3. Bolometric correction averaged over the temperature range
         (Eq. 4), giving total emitted luminous energy per frame

    The bolometric correction factor is computed as the mean over
    n_T_samples uniformly sampled temperatures, consistent with the
    treatment of the integrated energy in jupiter_impact_model.

    Parameters
    ----------
    light_curve_DN : ndarray, shape (N,)
        Per-frame aperture excess flux [DN].
    I_Jupiter : float
        Total Jupiter disk counts in reference image [DN].
    K : float
        Conversion factor [J]  (from compute_conversion_factor).
    wavelength_nm : ndarray
        Wavelength grid [nm].
    C : ndarray
        Camera + filter spectral response.
    T_flash_range_K : tuple (T_low, T_high)
        Blackbody temperature range [K].
    fcr : float
        Cloud reflection ratio.
    n_T_samples : int
        Number of temperature samples for bolometric factor averaging.

    Returns
    -------
    dict with keys:
        L_detected_J        – per-frame detected luminous energy [J]
        L_cloud_corrected_J – per-frame cloud-reflection-corrected energy [J]
        L_bolometric_J      – per-frame fully corrected luminous energy [J]
        bolometric_factor   – mean bolometric correction factor (scalar)
        cloud_factor        – cloud reflection correction factor (scalar)
    """
    T_low, T_high = T_flash_range_K
    T_samples     = np.linspace(T_low, T_high, n_T_samples)

    # Mean bolometric correction factor over temperature range
    # bolometric_correction(1.0, T, ...) returns just the factor
    # since L_star=1 → factor = numer/denom
    factors = np.array([
        bolometric_correction(1.0, T, wavelength_nm, C)
        for T in T_samples
    ])
    bolometric_factor = float(factors.mean())
    cloud_factor      = 1.0 / (1.0 + fcr)

    # Per-frame detected luminous energy (Eq. 1)
    L_detected_J        = K * (light_curve_DN / I_Jupiter)

    # Per-frame cloud reflection correction
    L_cloud_corrected_J = L_detected_J * cloud_factor

    # Per-frame bolometric correction
    L_bolometric_J      = L_cloud_corrected_J * bolometric_factor

    return {
        "L_detected_J":        L_detected_J,
        "L_cloud_corrected_J": L_cloud_corrected_J,
        "L_bolometric_J":      L_bolometric_J,
        "bolometric_factor":   bolometric_factor,
        "cloud_factor":        cloud_factor,
    }

# ===========================================================================
# DEMO  –  full end-to-end run
# ===========================================================================

if __name__ == "__main__":

    # ---- paths ------------------------------------------------------------
    #REDEFINE if file names are different
    #QE_CSVS: list for multiple channels on camera - single value (within list []) for monochrome camera
    FRAME_DIR        = "./frames"
    SOLAR_CSV        = "./solar_spectrum.csv"
    ALBEDO_CSV       = "./albedo.csv" #"./li2018_pg.csv"
    QE_CSVS          = ["./B.csv",
                        "./V.csv",
                        "./R.csv"]
    TRANSMISSION_CSV = "./transmission.csv"

    using_spectra = (os.path.exists(SOLAR_CSV) and os.path.exists(ALBEDO_CSV))
    using_camera  = all(os.path.exists(p) for p in QE_CSVS + [TRANSMISSION_CSV])

    # ---- Part A: extract light curve --------------------------------------
    print("=" * 65)
    print("Part A – Light curve extraction")
    print("=" * 65)

    #CHECK VALUES WITH ./aperture_diagnostic.png
    aperture_radius = 11.5 #should encompass entire impact PSF
    sky_inner = 18.0
    sky_outer = 30.0
    fps       = 40 #REDEFINE
    fcr       = 1.83 #REDFINE from cloud_reflection.py
    KT        = 4.184e12

    #Comment out function call if you already have a light curve in DNs
    #frames must be named with the same pattern "frame%03d.png"
    lc = extract_light_curve(
        frame_dir=FRAME_DIR,
        pattern="frame*.png",
        aperture_radius=aperture_radius,
        sky_inner=sky_inner,
        sky_outer=sky_outer,
        quiet_fraction=0.6,
        savgol_window = 10,
        savgol_polyorder = 4,
        n_sigma=1.,
    )

    # Aperture diagnostic — check placement on flash frame and sky annulus
    peak_idx    = lc["peak_flash_idx"]
    flash_frame = lc["aligned_frames"][peak_idx]
    plot_aperture_diagnostic(
        peak_idx      = peak_idx,
        flash_frame   = flash_frame,
        reference     = lc["reference"],
        flash_yx      = lc["flash_yx_per_frame"][peak_idx],
        aperture_radius = aperture_radius,
        sky_inner       = sky_inner,
        sky_outer       = sky_outer,
        save_path       = "./aperture_diagnostic.png",
    )

    # Plot light curve DNs- raw and smoothed
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(lc["frame_numbers"], lc["light_curve"], yerr=lc["uncertainties"],
            fmt="o", color="steelblue", alpha=0.5, markersize=5, 
            ecolor="steelblue", elinewidth=1.0, capsize=3, label="Raw")
    ax.plot(lc["frame_numbers"], lc["light_curve_smooth"],
            "-", color="steelblue", linewidth=2.0, label="Savitzky-Golay")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    flash_num = lc["frame_numbers"][lc["peak_flash_idx"]]
    ax.axvline(flash_num, color="red", linewidth=1.0,
               linestyle="--", label=f"Flash (frame {flash_num})")
    ax.set_xlabel("Frame number")
    ax.set_ylabel("Aperture excess flux [DN]")
    ax.set_title("Jupiter impact — light curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig("./light_curve_DNs.png", dpi=150)
    print("  Light curve saved → light_curve.png")


    # ---- Part B: impact model --------------------------------------------
    print("\n" + "=" * 65)
    print("Part B – Impact characterisation")
    print("=" * 65)
    print(f"  Solar/albedo spectra : {'tabulated CSVs' if using_spectra else 'analytic fallback'}")
    print(f"  Camera + filter      : {'tabulated QE + transmission' if using_camera else 'generic fallback'}")

    #impact velocity values
    v_kms_values = [60.0, 65.0, 70.0]
    density_values = [500.0, 2500.0, 5000.0]

    # #UNCOMMENT WHEN WORKING FROM CSV IN DNS
    # df = pd.read_csv("light_curve_data.csv") #change file name for specific case
    # flux_DN = df["flux_DN"].values.astype(float) #column name "flux_DN" may need to be redefined depending on how you have it named
    # I_star  = float(flux_DN[flux_DN > 0].sum())
    # I_star_peak = float(flux_DN.max())
    # #NEXT FEW LINES ONLY USABLE IF CSV CAME FROM THIS CODE ORIGINALLY
    # meta = pd.read_csv("light_curve_metadata.csv") #file with I_Jupiter, K_J and dt_frame_s
    # I_Jupiter = float(meta["I_Jupiter_DN"][0])
    # #CALCULATE I_JUPITER FROM A SINGLE REFERENCE PHOTO
    # I_Jupiter = measure_I_Jupiter("reference_frame.png") #change file name for reference photo - must be photo from camera used for lc

    #UNCOMMENT WHEN WORKING FROM VIDEO OR CSV IN DNS
    #FUNCTION UNTESTED FOR CSV IN DNS
    '''
    TO REDEFINE FOR DIFFERENT EVENTS:
    I_star: when working fom csv in DNS = I_star (see above)
            lc["I_star"] for start from video
    I_Jupiter: found multiple ways for csv in DNs (see above)
               lc["I_Jupiter"] for start from video
    I_star_peak: when working from csv in DNS = I_star_peak (see above)
                 lc["I_star_peak"] for start from video
    d_jupiter_au: Jupiter's semi-major axis at the time of observation from JPL_horizons.py
    dt_frame_s: exposure time for a single frame (fps defined above)
    fcr: cloud reflectivity ratio from cloud_reflection.py
    V_jupiter: apparent visual magnitude of Jupiter from JPL_horizons.py
    '''
    results = jupiter_impact_model(
        I_star       = lc["I_star"],
        I_Jupiter    = lc["I_Jupiter"],
        I_star_peak  = lc["I_star_peak"],
        d_jupiter_au = 4.9766,
        dt_frame_s   = 1/fps,
        qe_csvs          = QE_CSVS           if using_camera else None,
        transmission_csv = TRANSMISSION_CSV  if using_camera  else None,
        solar_csv        = SOLAR_CSV         if using_spectra else None,
        albedo_csv       = ALBEDO_CSV        if using_spectra else None,
        fcr              = fcr,
        T_flash_range_K  = (6000.0, 14000.0),
        v_kms_values     = v_kms_values,
        density_values   = density_values,
        V_jupiter        = -2.89,
        lam_min_nm       = 400.0,
        lam_max_nm       = 1000.0,
    )

    #UNCOMMENT WHEN WORKING FROM CSV IN WATTS
    #FUNCTION IS UNTESTED
    '''
    TO REDEFINE FOR DIFFERENT EVENTS:
    light_curve_data.csv: filename for light curve data in Watts within cwd
    time_col: column name for time data from light_curve_data.csv
    flux_col: column name for flux data from light_curve_data.csv
    T_flash_range_K: temperature range for flash, may be identified by source of light curve
                     default = (3500.0, 10000.0)
    '''
    # results = jupiter_impact_model_from_watts(
    #      light_curve_csv = "light_curve_data.csv",
    #      time_col        = "time_s", 
    #      flux_col        = "optical_energy_J",
    #      qe_csvs         = QE_CSVS            if using_camera else None,
    #      transmission_csv = TRANSMISSION_CSV  if using_camera else None, 
    #      T_flash_range_K  = (3500.0, 10000.0),
    #      v_impact_kms     = v_kms_values,
    #      density_values   = density_values,
    #      lam_min_nm       = 400.0,
    #      lam_max_nm       = 1000.0,
    # )


    #FOLOWING CODE MAY BE AFFECTED DEPENDING ON METHOD (i.e., start from video or start from extracted lc)
    # Save stacked impact image
    save_impact_stack(
        aligned_frames = lc["aligned_frames"],
        flash_indices  = lc["flash_indices"],
        reference      = lc["reference"],
        save_path      = "./impact_stack.png",
    )

    # Plot optical energy as a function of time
    # Each frame's luminous energy = K * (flux / I_Jupiter)
    K          = results["K_J"]
    frame_nums = np.array(lc["frame_numbers"])
    time_s     = (frame_nums - frame_nums[0]) / fps

    lam_csv = np.linspace(400.0, 1000.0, 1000)
    C_csv = camera_filter_response(lam_csv, 
                                   QE_CSVS          if using_camera else None,
                                   TRANSMISSION_CSV if using_camera else None)

    lc_corr = corrected_light_curve(
        light_curve_DN  = lc["light_curve"],
        I_Jupiter       = lc["I_Jupiter"],
        K               = K,
        wavelength_nm   = lam_csv,
        C               = C_csv,
        T_flash_range_K = (results["T_low_K"], results["T_high_K"]),
        fcr             = fcr,
    ) 
    lc_corr_smooth = corrected_light_curve(
        light_curve_DN  = lc["light_curve_smooth"],
        I_Jupiter       = lc["I_Jupiter"],
        K               = K,
        wavelength_nm   = lam_csv,
        C               = C_csv,
        T_flash_range_K = (results["T_low_K"], results["T_high_K"]),
        fcr             = fcr,
    )
    lc_corr_unc = corrected_light_curve(
        light_curve_DN  = lc["uncertainties"],
        I_Jupiter       = lc["I_Jupiter"],
        K               = K,
        wavelength_nm   = lam_csv,
        C               = C_csv,
        T_flash_range_K = (results["T_low_K"], results["T_high_K"]),
        fcr             = fcr,
    )

    # fig2, ax2 = plt.subplots(figsize=(8, 4))
    # ax2.errorbar(time_s, L_frames,
    #              yerr=L_uncertainties,
    #              fmt="o", color="darkorange", markersize=5, alpha=0.5,
    #              ecolor="darkorange", linewidth=1.0, capsize=3, label="Raw")
    # ax2.plot(time_s, L_frames_smooth,
    #          "-", color="darkorange", linewidth=2.0, label="Savitzky-Golay")
    # ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    flash_time = time_s[lc["peak_flash_idx"]]
    # ax2.axvline(flash_time, color="red", linewidth=1.0,
    #             linestyle="--", label=f"Flash ({flash_time:.2f} s)")
    # ax2.set_xlabel("Time [s]")
    # ax2.set_ylabel("Optical energy per frame [J]")
    # ax2.set_title("Jupiter impact — optical energy vs time")
    # ax2.legend()
    # fig2.tight_layout()
    # fig2.savefig("./optical_energy.png", dpi=150)
    # print("  Optical energy plot saved → optical_energy.png")

    # Plot corrected light curve (cloud + bolometric) with error bars
    fig2, axes2 = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Top panel: cloud reflection corrected
    axes2[0].errorbar(time_s, lc_corr["L_cloud_corrected_J"],
                      yerr=lc_corr_unc["L_cloud_corrected_J"],
                      fmt="o", color="steelblue", markersize=5, alpha=0.5,
                      ecolor="steelblue", elinewidth=1.0, capsize=3, label="Raw")
    axes2[0].plot(time_s, lc_corr_smooth["L_cloud_corrected_J"],
                  "-", color="steelblue", linewidth=2.0, label="Savitzky-Golay")
    axes2[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes2[0].axvline(flash_time, color="red", linewidth=1.0,
                     linestyle="--", label=f"Flash ({flash_time:.2f} s)")
    axes2[0].set_ylabel("Cloud-corrected energy [J]")
    axes2[0].set_title(f"Cloud reflection corrected  (fcr={fcr:.2f}, "
                       f"factor={lc_corr['cloud_factor']:.4f})")
    axes2[0].legend(fontsize=8)

    # Bottom panel: fully corrected (cloud + bolometric)
    axes2[1].errorbar(time_s, lc_corr["L_bolometric_J"],
                      yerr=lc_corr_unc["L_bolometric_J"],
                      fmt="o", color="darkorange", markersize=5, alpha=0.5,
                      ecolor="darkorange", elinewidth=1.0, capsize=3, label="Raw")
    axes2[1].plot(time_s, lc_corr_smooth["L_bolometric_J"],
                  "-", color="darkorange", linewidth=2.0, label="Savitzky-Golay")
    axes2[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes2[1].axvline(flash_time, color="red", linewidth=1.0,
                     linestyle="--", label=f"Flash ({flash_time:.2f} s)")
    axes2[1].set_xlabel("Time [s]")
    axes2[1].set_ylabel("Bolometric energy [J]")
    axes2[1].set_title(f"Cloud + bolometric corrected  "
                       f"(bolometric factor={lc_corr['bolometric_factor']:.3f})")
    axes2[1].legend(fontsize=8)

    fig2.suptitle("Jupiter impact — corrected light curves", fontsize=11)
    fig2.tight_layout()
    fig2.savefig("./corrected_light_curve.png", dpi=150)
    print("  Corrected light curve saved → corrected_light_curve.png")

    print(f"  Cloud correction factor  : {lc_corr['cloud_factor']:.4f}")
    print(f"  Bolometric factor (mean) : {lc_corr['bolometric_factor']:.3f}")

    df_out = pd.DataFrame({
        "frame_number":                      frame_nums,
        "time_s":                            time_s,
        "flux_DN":                           lc["light_curve"],
        "flux_DN_smooth":                    lc["light_curve_smooth"],
        "flux_DN_uncertainty":               lc["uncertainties"],
        "L_detected_J":                      lc_corr["L_detected_J"],
        "L_detected_J_smooth":               lc_corr_smooth["L_detected_J"],
        "L_detected_J_uncertainty":          lc_corr_unc["L_detected_J"],
        "L_cloud_corrected_J":               lc_corr["L_cloud_corrected_J"],
        "L_cloud_corrected_J_smooth":        lc_corr_smooth["L_cloud_corrected_J"],
        "L_cloud_corrected_J_uncertainty":   lc_corr_unc["L_cloud_corrected_J"],
        "L_bolometric_J":                    lc_corr["L_bolometric_J"],
        "L_bolometric_J_smooth":             lc_corr_smooth["L_bolometric_J"],
        "L_bolometric_J_uncertainty":        lc_corr_unc["L_bolometric_J"],
    })

    I_Jupiter  = lc["I_Jupiter"]
    dt_frame_s = float(1 / fps)
    df_meta = pd.DataFrame({
        "I_Jupiter_DN":      [I_Jupiter],
        "K_J":               [K],
        "dt_frame_s":        [dt_frame_s],
        "cloud_factor":      [lc_corr["cloud_factor"]],
        "bolometric_factor": [lc_corr["bolometric_factor"]],
    })

    csv_path      = "./light_curve_data.csv"
    meta_csv_path = "./light_curve_metadata.csv"
    df_out.to_csv(csv_path, index=False)
    df_meta.to_csv(meta_csv_path, index=False)
    print(f"\n  Light curve data saved     → light_curve_data.csv")
    print(f"  Light curve metadata saved → light_curve_metadata.csv")
    print(f"  I_Jupiter = {I_Jupiter:.2f} DN | K = {K:.3e} J | dt = {dt_frame_s:.4f} s")

    fmt = "  {:<30s}  {:>15}"
    print(f"\n  {'Parameter':<30s}  {'Value':>15}")
    print("  " + "-" * 48)
    print(fmt.format("Conversion factor K",       f"{results['K_J']:.3e} J"))
    print(fmt.format("Detected energy L*",         f"{results['L_star_J']:.3e} J"))
    print(fmt.format("Flash temperature range",
                     f"{results['T_low_K']:.0f}–{results['T_high_K']:.0f} K"))
    print(fmt.format("Luminous energy (mean T)",    f"{results['L_f_mean_J']:.3e} J"))
    print(fmt.format("Luminous energy (kt, mean)",   f"{results['L_f_mean_J'] / KT:.3f} kt"))
    print(fmt.format("Luminous energy range [J]",
                     f"{results['L_f_low_J']:.2e}–{results['L_f_high_J']:.2e} J"))
    print(fmt.format ("Kinetic energy (mean T)",     f"{results['E_kin_mean_J']:.3e} J"))
    print(fmt.format("Kinetic energy (kt, mean)",   f"{results['E_kin_mean_kt']:.1f} kt"))
    print(fmt.format("Kinetic energy range [kt]",
                     f"{results['E_kin_low_kt']:.1f}–{results['E_kin_high_kt']:.1f} kt"))
    print(fmt.format("Luminous efficiency η (mean)", f"{results['eta_mean']:.4f}"))
    print(fmt.format("Luminous efficiency η (range)",
                     f"{results['eta_low']:.4f}–{results['eta_high']:.4f}"))
    for v in v_kms_values:
        print(fmt.format(f"Mass at {v:.0f} km/s",
                         f"{results['masses_kg'][v]:.2e} kg"))
        for rho in density_values:
            print(fmt.format(f"Diameter at {v:.0f} km/s, {rho:.0f} kg/m³",
                             f"{results['diameters_m'][v][rho]:.3f} m"))
    print(fmt.format("Flash apparent magnitude",   f"{results['flash_magnitude']:.1f}"))
