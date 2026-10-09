"""
Jupiter Meteor Impact Model
============================
Based on the methods of Hueso et al. 2018:
  "Small impacts on the giant planet Jupiter"
  A&A 617, A68 (2018)
  https://doi.org/10.1051/0004-6361/201832689

Cloud reflection correction based on Arimatsu et al. 2023:
  A&A 677, A165 (2023)

Full pipeline
-------------
  Part 0 – Observing geometry and cloud reflection
    01  Query JPL Horizons for Jupiter's distance, magnitude and
        sub-observer point
    02  Emission angle cosine μ at the impact site
    03  Cloud reflection correction factor f_CR  (Arimatsu et al. 2023)

  Part A – Light curve extraction from raw video frames
    A1  Load PNG frames and convert to grayscale
    A2  Co-register frames by FFT cross-correlation
    A3  Build median reference image from quiet frames
    A4  Detect flash frame and location via differential imaging
    A5  Aperture photometry on every differential frame
    A6  Sum excess counts over the flash frames → I_star;  sum disk counts → I_Jupiter
    A7  Savitzky-Golay smoothing of the light curve

  Part B – Impact energy and impactor characterisation  (Eq. numbers
           refer to Hueso et al. 2018)
    B1  Compute conversion factor K  (Eq. 2)
    B2  Detected flash energy L_star  (Eq. 1)
    B3  Bolometric correction L_f(T)  (Eq. 4)
    B4  Kinetic energy from luminous energy (power law)
    B5  Impactor mass and diameter

  Part C – Diagnostic plots

File layout
-----------
  Constants and small shared helpers
  Part 0 functions             → get_jupiter_geometry(), compute_mu(),
                                 compute_fCR_combined()
  Part A functions            → extract_light_curve() runs all of Part A
  Part B functions             → jupiter_impact_model() runs all of Part B
                                 jupiter_impact_model_from_watts() runs Part B
                                 from a light curve already in Watts
                                 corrected_light_curve() applies Part B
                                 corrections frame-by-frame
  Part C plotting functions
  jupiter_impact_pipeline.ipynb runs the full pipeline step by step;
  edit the settings cell there per event
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from numpy.fft import fft2, fftshift, ifft2
from PIL import Image
from scipy.interpolate import interp1d
from scipy.ndimage import shift
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# Physical & Jovian constants
# ---------------------------------------------------------------------------

R_JUP_EQ  = 71492e3       # Jupiter equatorial radius [m]
R_JUP_POL = 66854e3       # Jupiter polar radius      [m]
S_EARTH   = 1361.0        # Solar constant at 1 AU    [W m⁻²]
KT_TO_J   = 4.184e12      # 1 kt TNT                  [J]

H_PLANCK  = 6.626e-34     # Planck constant           [J s]
C_LIGHT   = 2.998e8       # Speed of light            [m s⁻¹]
K_BOLTZ   = 1.381e-23     # Boltzmann constant        [J K⁻¹]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Average the colour channels of an RGB image; pass grayscale through."""
    return img.mean(axis=2) if img.ndim == 3 else img


def _circular_radius_map(shape: tuple, center_yx: tuple):
    """
    Return (y, x, r) where y and x are open pixel-coordinate grids and r is
    each pixel's distance from *center_yx*.  np.ogrid is used rather than
    np.meshgrid because it is more memory efficient.
    """
    H, W   = shape
    cy, cx = center_yx
    y, x   = np.ogrid[:H, :W]
    r      = np.sqrt((y - cy)**2 + (x - cx)**2)
    return y, x, r


def _planck(lam_m: np.ndarray, T: float, clip_exponent: bool = False) -> np.ndarray:
    """
    Planck spectral radiance B(λ, T) [W m⁻² sr⁻¹ m⁻¹].

    If *clip_exponent* is True the exponent is clipped to [0, 700] to
    avoid overflow at short wavelengths / low temperatures.
    """
    exponent = H_PLANCK * C_LIGHT / (lam_m * K_BOLTZ * T)
    if clip_exponent:
        exponent = np.clip(exponent, 0, 700)
    return (2 * H_PLANCK * C_LIGHT**2 / lam_m**5) / (np.exp(exponent) - 1.0)


# ===========================================================================
# PART 0 — OBSERVING GEOMETRY AND CLOUD REFLECTION
# ===========================================================================

# ---------------------------------------------------------------------------
# 01.  JPL Horizons query
# ---------------------------------------------------------------------------

def get_jupiter_geometry(obs_time_utc: str, observer_location="500") -> dict:
    """
    Query JPL Horizons for Jupiter's apparent magnitude and observing
    geometry at a given UTC time.  Requires astroquery and an internet
    connection.

    Parameters
    ----------
    obs_time_utc : str
        Observation time in ISO format, e.g. '2023-11-15 12:41:17'.
    observer_location : str or dict
        MPC observatory code
        (https://www.minorplanetcenter.net/iau/lists/ObsCodes.html);
        '500' = geocentre (default).  For a custom location pass a dict
        {"lon": 180.5, "lat": 20.7, "elevation": 4.2}, elevation in km.

    Returns
    -------
    dict with keys:
        V_jupiter       – apparent visual magnitude
        subobs_lat      – sub-observer latitude [deg]
        subobs_lon      – sub-observer longitude (System III) [deg]
        d_jupiter_au    – Jupiter–Sun distance [AU]
        delta_au        – Jupiter–observer distance [AU]
        phase_angle_deg – solar phase angle Sun-Jupiter-Observer [deg]
    """
    # Imported here so the rest of the module works without astroquery
    from astropy.time import Time
    from astroquery.jplhorizons import Horizons

    t   = Time(obs_time_utc, format="iso", scale="utc")
    obj = Horizons(id="599", location=observer_location, epochs=t.jd)   # 599 = Jupiter

    # quantities:  9 = visual magnitude
    #             14 = sub-observer latitude
    #             15 = sub-observer longitude (System III)
    #             19 = heliocentric range (Jupiter–Sun distance)
    #             20 = observer range
    #             24 = phase angle
    eph = obj.ephemerides(quantities="9,14,15,19,20,24")

    return {
        "V_jupiter":       float(eph["V"][0]),
        "subobs_lat":      float(eph["PDObsLat"][0]),
        "subobs_lon":      float(eph["PDObsLon"][0]),
        "d_jupiter_au":    float(eph["r"][0]),
        "delta_au":        float(eph["delta"][0]),
        "phase_angle_deg": float(eph["alpha"][0]),
    }


# ---------------------------------------------------------------------------
# 02.  Emission angle at the impact site
# ---------------------------------------------------------------------------

def compute_mu(impact_lat: float, impact_lon: float,
               subobs_lat: float, subobs_lon: float) -> float:
    """
    Cosine of the emission angle (μ) at the impact site, from the
    planetographic coordinates of the impact site and the sub-observer
    point.  Jupiter is treated as a sphere, which is a good approximation
    for this purpose.

    Parameters
    ----------
    impact_lat, impact_lon : float
        Planetographic latitude and longitude of the impact site [deg].
    subobs_lat, subobs_lon : float
        Planetographic latitude and longitude of the sub-observer point [deg].

    Returns
    -------
    mu : float
        Cosine of the emission angle, clipped to [0, 1]
        (1 = disk centre, 0 = limb).
    """
    def _unit_normal(lat_deg, lon_deg):
        lat, lon = np.radians(lat_deg), np.radians(lon_deg)
        return np.array([np.cos(lat) * np.cos(lon),
                         np.cos(lat) * np.sin(lon),
                         np.sin(lat)])

    mu = np.dot(_unit_normal(impact_lat, impact_lon),
                _unit_normal(subobs_lat, subobs_lon))
    mu = np.clip(mu, 0.0, 1.0)

    print(f"\n--- Emission angle calculation ---")
    print(f"  Impact site   : lat={impact_lat:.2f}°, lon={impact_lon:.2f}°")
    print(f"  Sub-observer  : lat={subobs_lat:.2f}°, lon={subobs_lon:.2f}°")
    print(f"  Emission angle: {np.degrees(np.arccos(mu)):.2f}°")
    print(f"  mu            : {mu:.4f}")

    return mu


# ---------------------------------------------------------------------------
# 03a.  Cloud reflection — simple Lambertian model  (Eqs. 1–5, Arimatsu 2023)
# ---------------------------------------------------------------------------
# f_CR is the ratio of flash light reflected off the clouds below to the
# light received directly.  The corrected flux is F_obs / (1 + f_CR).

def fCR_lambertian(mu: float, pL: float) -> float:
    """
    Cloud reflection correction factor for an isotropic Lambertian cloud
    deck with Lambertian albedo *pL*  (Eq. 4).

        f_CR = 2 μ p_L
    """
    return 2.0 * mu * pL


def fCR_lambertian_geometric(mu: float, pg: float) -> float:
    """
    Same as :func:`fCR_lambertian` but in terms of geometric albedo *pg*.
    Since p_g = (2/3) p_L  (Eq. 5):

        f_CR = 3 μ p_g
    """
    return 3.0 * mu * pg


def fcor_from_fobs(Fobs: float, fCR: float) -> float:
    """
    Corrected flux from observed flux  (Eq. 1).

        F_cor = F_obs / (1 + f_CR)
    """
    return Fobs / (1.0 + fCR)


# ---------------------------------------------------------------------------
# 03b.  Cloud reflection — full Double Henyey-Greenstein model  (Eqs. 6–9)
# ---------------------------------------------------------------------------

def chandrasekhar_H_array(mu_arr: np.ndarray, omega: float, n_iter: int = 30) -> np.ndarray:
    """
    Chandrasekhar H-function on the grid *mu_arr*  (Eq. 7), solved by
    successive substitution over the whole grid:

        H(μ) = 1 + ½ ω μ H(μ) ∫₀¹ H(μ') / (μ + μ') dμ'

    Parameters
    ----------
    mu_arr : ndarray
        Grid of μ values on [0, 1] (also used as the integration grid).
    omega : float
        Single-scattering albedo.
    n_iter : int
        Number of iterations.
    """
    H = np.ones_like(mu_arr, dtype=float)
    for _ in range(n_iter):
        H_new = np.empty_like(H)
        for i, mu in enumerate(mu_arr):
            integrand = H / (mu + mu_arr + 1e-100)
            H_new[i]  = 1.0 + 0.5 * omega * mu * np.trapezoid(integrand, mu_arr)
        H = H_new
    return H


def DHG_phase(cos_alpha, g1: float, g2: float, f: float):
    """
    Double Henyey-Greenstein scattering phase function  (Eq. 8).

        P(α) = f (1-g1²) / (1 + g1² + 2 g1 cos α)^1.5
             + (1-f) (1-g2²) / (1 + g2² + 2 g2 cos α)^1.5

    Parameters
    ----------
    cos_alpha : float or ndarray
        Cosine of the scattering phase angle.
    g1, g2 : float
        Asymmetry factors of the two lobes.
    f : float
        Fractional weight of the first lobe.
    """
    term1 = f * (1 - g1**2) / (1 + g1**2 + 2*g1*cos_alpha)**1.5
    term2 = ((1 - f) * (1 - g2**2)) / (1 + g2**2 + 2*g2*cos_alpha)**1.5
    return term1 + term2


def delta_ICR_DHG(mu: float, omega: float, g1: float, g2: float, f: float,
                  h: float = 50.0, r_max: float = 5000.0, n_r: int = 500,
                  n_phi: int = 360) -> float:
    """
    ΔI_CR / I₀ at a single wavelength: the cloud-reflected flash light
    relative to the direct light  (Eq. 6).

    Integrates over a flat cloud plane in polar coordinates (r, φ)
    centred below a flash source at height *h*.  The observer azimuth is
    fixed at φ = 0 (the result is φ-symmetric for uniform clouds).

    Parameters
    ----------
    mu : float
        Cosine of the emission angle.
    omega : float
        Single-scattering albedo.
    g1, g2, f : float
        DHG phase function parameters.
    h : float
        Flash height above the clouds [km].
    r_max : float
        Maximum radial integration limit [km].
    n_r, n_phi : int
        Radial and azimuthal integration resolution.

    Returns
    -------
    float
        ΔI_CR / I₀ (dimensionless).
    """
    # H-function on a μ grid, interpolated at the angles needed below
    mu_grid = np.linspace(0.0, 1.0, 100)
    H_grid  = chandrasekhar_H_array(mu_grid, omega, n_iter=20)

    def H_interp(m):
        return np.interp(np.clip(m, 0.0, 1.0), mu_grid, H_grid)

    r_arr   = np.linspace(0.0, r_max, n_r + 1)[1:]   # skip the r = 0 singularity
    phi_arr = np.linspace(0.0, 2*np.pi, n_phi, endpoint=False)
    dr      = r_arr[1] - r_arr[0]
    dphi    = phi_arr[1] - phi_arr[0]

    # These depend only on r (or nothing), not on φ
    dist2     = r_arr**2 + h**2
    dist      = np.sqrt(dist2)
    mu0_arr   = h / dist                       # cosine of incidence angle
    sin_theta = np.sqrt(max(1 - mu**2, 0.0))   # sine of emission angle
    H_obs     = H_interp(mu)
    H_inc_arr = H_interp(mu0_arr)

    total = 0.0
    for phi in phi_arr:
        x = r_arr * np.cos(phi)

        # Phase angle α between the incoming ray (source → surface,
        # (x/d, y/d, -μ0)) and the outgoing ray to the observer
        # (sin θ, 0, μ):  cos α = dot product of the two
        cos_alpha = (x / dist) * sin_theta + (-mu0_arr) * mu
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)

        P_arr = DHG_phase(cos_alpha, g1, g2, f)

        # Reflectance from Eq. 6:
        #   ω/4 · μ0/(μ0+μ) · [P - 1 + H(μ) H(μ0)]
        RADF = (omega / 4.0) * (mu0_arr / (mu0_arr + mu + 1e-100)) * \
               (P_arr - 1.0 + H_obs * H_inc_arr)

        integrand = (mu / dist2) * RADF * r_arr   # includes the r dr dφ area element
        total += np.sum(integrand) * dr * dphi

    return total


def compute_fCR_integrated(wavelengths: np.ndarray,
                           response: np.ndarray,
                           delta_ICR_over_I0: np.ndarray) -> float:
    """
    Band-integrated f_CR for instrument response R(λ)  (Eq. 9).

        f_CR = ∫ R(λ) · (ΔI_CR/I₀)(λ) dλ  /  ∫ R(λ) dλ

    All three arrays must be on the same wavelength grid [nm].
    """
    numerator   = np.trapezoid(response * delta_ICR_over_I0, wavelengths)
    denominator = np.trapezoid(response, wavelengths)
    return numerator / denominator


# ---------------------------------------------------------------------------
# 03c.  Cloud reflection — data loaders
# ---------------------------------------------------------------------------

def load_DHG_parameters(omega_path: str, g1_path: str, g2_path: str, f_path: str) -> tuple:
    """
    Load the wavelength-dependent DHG parameters (Heng & Li 2021) from CSV
    and interpolate them onto a common 1 nm grid covering the wavelength
    range shared by all four files.

    CSV format : index | Wavelength [nm] | value

    Returns
    -------
    wavelengths, omega, g1, g2, f : ndarray
    """
    names = ["omega", "g1", "g2", "f"]
    dfs   = [pd.read_csv(path, index_col=0, header=0, names=["Wavelength", name])
             for path, name in zip([omega_path, g1_path, g2_path, f_path], names)]

    wl_min = max(df["Wavelength"].min() for df in dfs)
    wl_max = min(df["Wavelength"].max() for df in dfs)
    wavelengths = np.arange(wl_min, wl_max, 1.0)

    omega, g1, g2, f = [np.interp(wavelengths, df["Wavelength"].values, df[name].values)
                        for df, name in zip(dfs, names)]

    print(f"Common wavelength grid: {wl_min:.1f} - {wl_max:.1f} nm ({len(wavelengths)} points)")
    return wavelengths, omega, g1, g2, f


def load_camera_response(transmission_path, qe_paths: list) -> tuple:
    """
    Per-channel camera + filter response R(λ) = QE(λ) · T(λ) for the cloud
    reflection calculation, on a 1 nm grid covering the wavelength range
    shared by all files.

    Parameters
    ----------
    transmission_path : str or None
        Filter transmission CSV  [index | Wavelength [nm] | T].
        None if there is no filter, in which case R(λ) = QE(λ).
    qe_paths : list of str
        QE CSVs, one per camera channel  [index | Wavelength [nm] | QE].
        Use a one-item list for a monochrome camera.  Each channel is
        labelled by its file name (e.g. "R.csv" → "R").

    Returns
    -------
    wavelengths : ndarray
        Common wavelength grid [nm].
    responses : dict {channel: ndarray}
        Response of each channel on that grid.
    """
    df_qes = {Path(p).stem: pd.read_csv(p, index_col=0, header=0, names=["Wavelength", "QE"])
              for p in qe_paths}
    all_dfs = list(df_qes.values())
    if transmission_path is not None:
        df_t = pd.read_csv(transmission_path, index_col=0, header=0, names=["Wavelength", "T"])
        all_dfs.append(df_t)

    wl_min  = max(df["Wavelength"].min() for df in all_dfs)
    wl_max  = min(df["Wavelength"].max() for df in all_dfs)
    wavelengths = np.arange(wl_min, wl_max, 1.0)

    responses = {
        channel: np.interp(wavelengths, df["Wavelength"].values, df["QE"].values)
        for channel, df in df_qes.items()
    }
    if transmission_path is not None:
        T = np.interp(wavelengths, df_t["Wavelength"].values, df_t["T"].values)
        responses = {channel: qe * T for channel, qe in responses.items()}

    print(f"Camera response wavelength grid: {wl_min:.1f} - {wl_max:.1f} nm ({len(wavelengths)} points)")
    print(f"Camera type: {'monochrome' if len(qe_paths) == 1 else 'colour'} — channels: {list(responses.keys())}"
          f" — filter: {'yes' if transmission_path is not None else 'none'}")
    return wavelengths, responses


# ---------------------------------------------------------------------------
# 03.  Cloud reflection — end-to-end f_CR for the camera
# ---------------------------------------------------------------------------

def compute_fCR_combined(cam_wavelengths: np.ndarray, responses: dict,
                         wavelengths_DHG: np.ndarray,
                         omega: np.ndarray, g1: np.ndarray,
                         g2: np.ndarray, f: np.ndarray,
                         mu: float = 1.0, h: float = 50.0,
                         r_max: float = 3000.0, n_r: int = 300,
                         n_phi: int = 180) -> tuple:
    """
    f_CR for each camera channel and combined over all channels, using the
    full DHG model (Eq. 6) for ΔI_CR/I₀ at every wavelength.  Takes
    roughly 25 s for a 600-point wavelength grid.

    The combined value is Σ ∫ R_i ΔI_CR/I₀ dλ / Σ ∫ R_i dλ, i.e. channels are
    weighted by their integrated response.

    Parameters
    ----------
    cam_wavelengths : ndarray
        Camera wavelength grid [nm] (from load_camera_response).
    responses : dict {channel: ndarray}
        Response of each channel on cam_wavelengths.
    wavelengths_DHG : ndarray
        Wavelength grid of the DHG parameters [nm].
    omega, g1, g2, f : ndarray
        DHG parameters on wavelengths_DHG (from load_DHG_parameters).
    mu : float
        Cosine of the emission angle (from compute_mu).
    h : float
        Flash height above the clouds [km].
    r_max, n_r, n_phi
        Integration limit [km] and resolution, see delta_ICR_DHG.

    Returns
    -------
    fCR_combined : float
        Combined cloud reflection correction factor.
    delta_ratio : ndarray
        ΔI_CR/I₀ on wavelengths_DHG.
    """
    print("Computing delta_ICR/I0 over wavelength grid (this may take a while)...")
    delta_ratio = np.empty_like(wavelengths_DHG)
    for i in range(len(wavelengths_DHG)):
        delta_ratio[i] = delta_ICR_DHG(mu, omega[i], g1[i], g2[i], f[i],
                                       h=h, r_max=r_max, n_r=n_r, n_phi=n_phi)
        if i % 50 == 0:
            print(f"  Wavelength {wavelengths_DHG[i]:.1f} nm ({i}/{len(wavelengths_DHG)})")

    delta_ratio_cam = np.interp(cam_wavelengths, wavelengths_DHG, delta_ratio)

    numerators   = {}
    denominators = {}
    print(f"\n--- fCR by channel (mu={mu}) ---")
    for channel, response in responses.items():
        numerators[channel]   = np.trapezoid(response * delta_ratio_cam, cam_wavelengths)
        denominators[channel] = np.trapezoid(response, cam_wavelengths)
        print(f"  fCR_{channel} = {numerators[channel] / denominators[channel]:.4f}")

    fCR_combined = sum(numerators.values()) / sum(denominators.values())
    print(f"  fCR_combined = {fCR_combined:.4f}, h = {h} km")
    print(f"  (Previous constant assumption: fCR = 0.3)")

    return fCR_combined, delta_ratio


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
        Index a single frame with frames[i].
    frame_paths : list of str
        Sorted file paths corresponding to each frame.
    frame_numbers : list of int
        Integer frame indices parsed from filenames (-1 if none found).
    """
    paths = sorted(Path(frame_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No frames matching '{pattern}' in {frame_dir}")

    frames, numbers = [], []
    for p in paths:
        img = np.array(Image.open(p), dtype=np.float64)
        frames.append(_to_grayscale(img))

        # Pull the first run of digits out of the filename as the frame number
        m = re.search(r"(\d+)", p.stem)
        numbers.append(int(m.group(1)) if m else -1)

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
        Large shifts indicate bad seeing.
    """
    reference = frames[ref_idx]

    # Complex conjugate of the reference FFT.  The mean is subtracted so the
    # correlation measures structural similarity rather than overall brightness.
    ref_fft = np.conj(fft2(reference - reference.mean()))
    aligned = np.empty_like(frames)
    shifts  = np.zeros((len(frames), 2))

    for i, frame in enumerate(frames):
        if i == ref_idx:
            aligned[i] = frame
            continue

        # Cross-correlation at every offset.  fftshift moves zero offset from
        # the array corner to the centre, so a peak at the centre means the
        # frames are already aligned.
        corr = np.abs(fftshift(ifft2(fft2(frame - frame.mean()) * ref_fft)))

        # Displacement of the correlation peak from the centre = frame offset
        peak = np.unravel_index(np.argmax(corr), corr.shape)
        dy   = peak[0] - corr.shape[0] // 2
        dx   = peak[1] - corr.shape[1] // 2

        # Shift back by (-dy, -dx).  Pixels shifted in from outside the frame
        # are filled with 0, which is fine because the sky background is black.
        aligned[i] = shift(frame, shift=(-dy, -dx), mode="constant", cval=0.0)
        shifts[i]  = (dy, dx)

    return aligned, shifts


def normalise_frame_flux(aligned: np.ndarray, disk_threshold_frac: float = 0.05) -> np.ndarray:
    """
    Scale every frame so the flux on Jupiter's disk matches frame 0.

    This removes atmospheric transparency variations that would otherwise
    leave a nonzero background in the differential images.  The disk is
    defined once, from frame 0, as all pixels above
    *disk_threshold_frac* × peak.
    """
    disk_mask  = aligned[0] > disk_threshold_frac * aligned[0].max()
    ref_flux   = aligned[0][disk_mask].mean()
    normalised = np.empty_like(aligned)
    for i, frame in enumerate(aligned):
        frame_flux    = frame[disk_mask].mean()
        scale         = ref_flux / frame_flux if frame_flux > 0 else 1.0
        normalised[i] = frame * scale
    return normalised


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


def select_quiet_frames(frames: np.ndarray, quiet_fraction: float,
                        exclude=None) -> list:
    """
    Pick the *quiet_fraction* of frames least likely to contain the flash.

    Each frame is compared with a rough median of all frames; the frames
    with the smallest peak differential value are returned (at least 2).
    Frame indices in *exclude* (e.g. known flash frames) are never chosen,
    and *quiet_fraction* then applies to the remaining frames.
    """
    rough_ref  = np.median(frames, axis=0)
    peak_vals  = np.array([(f - rough_ref).max() for f in frames])
    candidates = np.setdiff1d(np.arange(len(frames)),
                              [] if exclude is None else exclude)
    n_quiet    = max(2, int(len(candidates) * quiet_fraction))
    return candidates[np.argsort(peak_vals[candidates])][:n_quiet].tolist()


# ---------------------------------------------------------------------------
# A4.  Flash detection
# ---------------------------------------------------------------------------

def _flash_centroid(diff: np.ndarray, aperture_radius: float,
                    near_yx: tuple = None) -> tuple:
    """
    Intensity-weighted centroid of the flash in a single differential image.

    Only pixels within *aperture_radius* of the brightest pixel and brighter
    than the reference (diff > 0) are used, so unrelated noise peaks are
    ignored.  Falls back to the brightest pixel if no such pixels exist.
    If *near_yx* is given, the brightest pixel is searched for only within
    2 × *aperture_radius* of it, so faint flash frames are not pulled onto
    noise elsewhere in the image.
    """
    if near_yx is not None:
        _, _, r_near = _circular_radius_map(diff.shape, near_yx)
        diff_search  = np.where(r_near <= 2 * aperture_radius, diff, -np.inf)
        rough_yx     = np.unravel_index(np.argmax(diff_search), diff.shape)
    else:
        rough_yx = np.unravel_index(np.argmax(diff), diff.shape)
    y, x, r  = _circular_radius_map(diff.shape, rough_yx)
    region   = (r <= aperture_radius) & (diff > 0)

    if not region.any():
        return rough_yx

    weights = diff[region] / diff[region].sum()
    cy = np.broadcast_to(y, diff.shape)[region] @ weights
    cx = np.broadcast_to(x, diff.shape)[region] @ weights
    return (float(cy), float(cx))


def detect_flash(
    frames: np.ndarray,
    reference: np.ndarray,
    n_sigma: float = 5.0,
    aperture_radius: float = 10.0,
    flash_indices=None,
) -> tuple:
    """
    Identify all frames containing flash signal and locate the flash
    centroid in each one individually.

    If *flash_indices* is given, those frames are used as the flash frames
    and the n_sigma threshold is skipped.  The flash is then centroided in
    the brightest of them, and every other frame is centroided near that
    position.

    A frame is considered a flash frame if its peak differential value
    exceeds the mean + n_sigma × standard deviation of peak values across
    all frames.  The flash location is centroided separately in
    each flash frame rather than using a single fixed pixel, correcting
    for residual seeing distortion after co-registration.

    Parameters
    ----------
    frames : ndarray, shape (N, H, W)
    reference : ndarray, shape (H, W)
    n_sigma : float
        Detection threshold in units of the noise standard deviation.
        5σ is the common threshold in astronomy for a confident detection.
        3σ is useful if the flash is faint relative to the frame-to-frame noise.
        7–10σ is useful if seeing-induced noise is triggering false detections.
        Choose based on whether the flash frames stand out clearly from the
        noise floor.
    aperture_radius : float
        Radius of the mask that encompasses the entire flash PSF.
        Should be checked to ensure the flash is captured in its entirety.
    flash_indices : array-like of int, optional
        Frame indices to treat as flash frames instead of detecting them.

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

    if flash_indices is not None:
        # Flash frames chosen by hand: locate the flash in the brightest one,
        # then centroid every other frame near that position
        flash_indices  = np.asarray(flash_indices, dtype=int)
        peak_flash_idx = int(flash_indices[np.argmax(peak_vals[flash_indices])])
        peak_yx = _flash_centroid(frames[peak_flash_idx] - reference, aperture_radius)
        flash_yx_per_frame = {
            int(idx): _flash_centroid(frames[idx] - reference, aperture_radius,
                                      near_yx=peak_yx)
            for idx in flash_indices
        }
        return flash_indices, flash_yx_per_frame, peak_flash_idx, peak_vals

    # Quiet frames cluster near zero; flash frames are outliers
    threshold      = peak_vals.mean() + n_sigma * peak_vals.std()
    flash_indices  = np.where(peak_vals > threshold)[0]
    peak_flash_idx = int(np.argmax(peak_vals))

    # If the threshold catches nothing (very faint flash), fall back to
    # the single brightest frame
    if len(flash_indices) == 0:
        flash_indices = np.array([peak_flash_idx])

    flash_yx_per_frame = {
        int(idx): _flash_centroid(frames[idx] - reference, aperture_radius)
        for idx in flash_indices
    }

    return flash_indices, flash_yx_per_frame, peak_flash_idx, peak_vals


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
    captures the dominant noise source - spatial variation of the
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
    _, _, r = _circular_radius_map(diff_image.shape, center_yx)

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
    and sums all pixels belonging to Jupiter's disk using
    :func:`jupiter_disk_counts` — the same method used internally when
    processing video frames.

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
    gray      = _to_grayscale(np.array(Image.open(image_path), dtype=np.float64))
    threshold = threshold_frac * gray.max()
    I_Jupiter = jupiter_disk_counts(gray, threshold_frac)

    print(f"  Image loaded : {image_path}")
    print(f"  Peak pixel   : {gray.max():.1f} DN")
    print(f"  Threshold    : {threshold:.1f} DN  ({threshold_frac*100:.1f}% of peak)")
    print(f"  Disk pixels  : {(gray > threshold).sum()}")
    print(f"  I_Jupiter    : {I_Jupiter:.2f} DN")

    return I_Jupiter


# ---------------------------------------------------------------------------
# A7.  Smoothing
# ---------------------------------------------------------------------------

def smooth_light_curve(light_curve: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """
    Savitzky-Golay smoothing of a light curve.

    An even *window* is bumped up by one to make it odd.  Smoothing is
    skipped (a copy is returned) if window == 0, if there are fewer points
    than the window, or if window <= polyorder.
    """
    N = len(light_curve)
    if window > 0 and N >= window and window > polyorder:
        odd_window = window if window % 2 == 1 else window + 1
        return savgol_filter(light_curve, odd_window, polyorder)

    if window > 0:
        print(f"  Warning: savgol_window={window} too large for "
              f"{N} frames — smoothing skipped")
    return light_curve.copy()


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
    flash_frames: tuple = None,
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
        Threshold fraction of peak for Jupiter disk detection when
        computing I_Jupiter.  (Flux normalisation always uses 0.05.)
    n_sigma : float
        Detection threshold for flash frames in units of the noise
        standard deviation of peak differential values.
    savgol_window : int
        Window length for the Savitzky-Golay smoothing filter (must be
        greater than savgol_polyorder; even values are bumped up by one).
        Smaller values follow the raw curve more closely; larger values
        smooth more aggressively.  Set to 0 to skip smoothing.
    savgol_polyorder : int
        Polynomial order for the Savitzky-Golay filter.
        Must be less than savgol_window. Order 2 or 3 works well for flash profiles.
    flash_frames : tuple (first, last), optional
        First and last flash frame numbers (as in the filenames, inclusive),
        to set the flash frames by hand.  These frames are kept out of the
        reference image and *n_sigma* is ignored.  Use this when a faint
        rise or tail is visible by eye but missed by automatic detection.

    Returns
    -------
    dict with keys:
        I_star             – flash excess counts summed from the first to the
                             last flash frame [DN]
        I_Jupiter          – total Jupiter disk counts in reference [DN]
        I_star_peak        – peak single-frame excess DN
        light_curve        – per-frame aperture flux array [DN]
        uncertainties      – 1-sigma per-frame flux uncertainty [DN]
        light_curve_smooth – Savitzky-Golay smoothed light curve [DN]
        frame_numbers      – integer frame indices
        flash_indices      – indices of all detected flash frames
        flash_yx_per_frame – per-frame centroid positions
        peak_flash_idx     – index of the brightest flash frame
        reference          – 2-D reference image array
        aligned_frames     – co-registered, flux-normalised frame stack (N, H, W)
        shifts             – (N, 2) co-registration shifts applied
    """
    # A1: load
    frames, _, frame_numbers = load_frames(frame_dir, pattern)
    N = len(frames)
    print(f"  Loaded {N} frames")

    # A2: co-register, then normalise transparency variations
    aligned, shifts = coregister_frames(frames, ref_idx=0)
    print(f"  Co-registered {N} frames")
    aligned = normalise_frame_flux(aligned, disk_threshold_frac=0.05)
    print(f"  Flux-normalised {N} frames")

    # Convert hand-picked flash frame numbers to array indices
    manual_idx = None
    if flash_frames is not None:
        first, last = flash_frames
        manual_idx  = [i for i, n in enumerate(frame_numbers) if first <= n <= last]
        if not manual_idx:
            raise ValueError(f"No frames numbered {first}–{last} in {frame_dir}")

    # A3: build reference from quietest frames
    quiet_idx = select_quiet_frames(aligned, quiet_fraction, exclude=manual_idx)
    reference = build_reference(aligned, quiet_idx)
    print(f"  Reference built from {len(quiet_idx)} quiet frames")

    # A4: detect all flash frames with per-frame centroiding
    flash_indices, flash_yx_per_frame, peak_flash_idx, _ = detect_flash(
        aligned, reference, n_sigma=n_sigma, aperture_radius=aperture_radius,
        flash_indices=manual_idx,
    )
    print(f"  Flash detected in {len(flash_indices)} frame(s): "
          f"{[frame_numbers[i] for i in flash_indices]}")
    for idx in flash_indices:
        yx = flash_yx_per_frame[idx]
        print(f"    Frame {frame_numbers[idx]}: centroid "
              f"(row={yx[0]:.1f}, col={yx[1]:.1f})")

    # A5: aperture photometry on every frame, using the per-frame centroid
    # where available and the peak frame's location elsewhere
    fallback_yx   = flash_yx_per_frame[peak_flash_idx]
    light_curve   = np.zeros(N)
    uncertainties = np.zeros(N)
    for i, frame in enumerate(aligned):
        center = flash_yx_per_frame.get(i, fallback_yx)
        light_curve[i], uncertainties[i] = aperture_photometry(
            frame - reference, center, aperture_radius, sky_inner, sky_outer
        )

    # A6: integrate over the flash window only (first to last flash frame,
    # including any frames in between that missed the threshold).  Negative
    # values inside the window are kept so noise averages out rather than
    # only adding positive fluctuations.
    flash_window = slice(int(flash_indices.min()), int(flash_indices.max()) + 1)
    I_star       = float(light_curve[flash_window].sum())
    I_star_peak = float(light_curve.max())
    I_Jupiter   = jupiter_disk_counts(reference, disk_threshold_frac)

    # A7: smooth
    light_curve_smooth = smooth_light_curve(light_curve, savgol_window, savgol_polyorder)

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

def _interp_onto(wav: np.ndarray, values: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Linearly interpolate (wav, values) onto *lam*; zero outside the data range."""
    return interp1d(wav, values, kind="linear",
                    bounds_error=False, fill_value=0.0)(lam)


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
    df     = pd.read_csv(csv_path)
    wav_nm = df.iloc[:, 1].values.astype(float) / 10.0
    flux_W = 10.0 ** df.iloc[:, 2].values.astype(float) * 1e-3
    order  = np.argsort(wav_nm)
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
        return _interp_onto(wav, flux, lam)

    # Fallback: Planck at T_sun normalised to S_EARTH
    lam_m = lam * 1e-9
    B     = _planck(lam_m, 5778.0)
    return (B / np.trapezoid(B, lam_m)) * S_EARTH * 1e-9


def jupiter_reflectivity(wavelength_nm: np.ndarray, csv_path: str = None) -> np.ndarray:
    """
    Jupiter geometric albedo I/F interpolated onto *wavelength_nm*.
    Uses tabulated CSV if supplied, else analytic fallback.
    """
    lam = np.asarray(wavelength_nm, dtype=float)
    if csv_path is not None:
        wav, alb = load_jupiter_reflectivity(csv_path)
        return np.clip(_interp_onto(wav, alb, lam), 0.0, 1.0)

    # Fallback: flat albedo with Gaussian dips for the methane absorption bands
    base  = 0.52 * np.ones_like(lam)
    base -= 0.30 * np.exp(-0.5 * ((lam - 890.0) / 15.0)**2)   # CH₄ 890 nm
    base -= 0.15 * np.exp(-0.5 * ((lam - 728.0) / 12.0)**2)   # CH₄ 728 nm
    base -= 0.08 * np.exp(-0.5 * ((lam - 619.0) / 10.0)**2)   # CH₄ 619 nm
    return np.clip(base, 0.0, 1.0)


# ---------------------------------------------------------------------------
# B – Camera / filter spectral response  C(λ)
# ---------------------------------------------------------------------------

def camera_filter_response(
    wavelength_nm: np.ndarray,
    qe_csvs: list = None,
    transmission_csv: str = None,
) -> np.ndarray:
    """
    Combined camera + filter spectral response C(λ).

    CSV format for QE files    : index | Wavelength [nm] | QE [%]
    CSV format for filter file : index | Wavelength [nm] | Transmission [%]

    Accepts any number of QE channel CSVs (e.g. one for monochrome,
    three for RGB).  Channels are combined as an integral-weighted average:

        C(λ) = Σ [w_i · QE_i(λ)] / Σ w_i  ×  T(λ)

    where w_i = ∫ QE_i(λ) dλ.  This ensures broader or more sensitive
    channels contribute proportionally rather than equally.

    Falls back to a generic Gaussian Bayer-RGB approximation (no filter)
    if no QE CSVs are supplied.

    The result is normalised so that a flat response of 1 over the
    wavelength range is unchanged.
    """
    lam = np.asarray(wavelength_nm, dtype=float)

    def _load_percent_curve(path):
        df    = pd.read_csv(path, index_col=0)
        wav   = df.iloc[:, 0].values.astype(float)
        val   = df.iloc[:, 1].values.astype(float) / 100.0   # % → fraction
        order = np.argsort(wav)
        return _interp_onto(wav[order], val[order], lam)

    if qe_csvs is not None and len(qe_csvs) > 0:
        # Integral-weighted average of all QE channels
        qe_channels = [_load_percent_curve(p) for p in qe_csvs]
        weights     = [np.trapezoid(qe, lam) for qe in qe_channels]
        w_total     = sum(weights)
        qe_mean     = sum(w * qe for w, qe in zip(weights, qe_channels)) / w_total

        # Apply filter transmission if supplied
        if transmission_csv is not None:
            T = _load_percent_curve(transmission_csv)
        else:
            T = np.ones_like(lam)
        C = qe_mean * T

    else:
        # Fallback: Gaussian curves centred on each Bayer channel's peak,
        # averaged without weighting
        qe_r = np.exp(-0.5 * ((lam - 620.0) / 80.0)**2)
        qe_g = np.exp(-0.5 * ((lam - 520.0) / 60.0)**2)
        qe_b = np.exp(-0.5 * ((lam - 450.0) / 50.0)**2)
        C    = (qe_r + qe_g + qe_b) / 3.0

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
    S_J    = S_EARTH / d_jupiter_au**2                 # solar constant at Jupiter
    P_J    = S_J * np.pi * R_JUP_EQ * R_JUP_POL        # power intercepted by disk
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
    B_nm  = _planck(wavelength_nm * 1e-9, T_flash_K, clip_exponent=True) * 1e-9
    denom = np.trapezoid(B_nm * C, wavelength_nm)
    numer = np.trapezoid(B_nm,     wavelength_nm)
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

    This power law accounts for the nonlinear relationship between
    luminous and kinetic energy more accurately than a fixed luminous
    efficiency.

    1 kt TNT = 4.184e12 J
    """
    E0_kt    = L_f / KT_TO_J
    E_kin_kt = 8.2508 * (E0_kt ** 0.885)
    return E_kin_kt * KT_TO_J


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


# ---------------------------------------------------------------------------
# B3–B5 combined: luminous energy → kinetic energy → impactor properties
# ---------------------------------------------------------------------------

def _energy_and_impactor_properties(
    L_star: float,
    T_flash_range_K: tuple,
    n_T_samples: int,
    wavelength_nm: np.ndarray,
    C: np.ndarray,
    v_kms_values: list,
    density_values: list,
) -> dict:
    """
    Run steps B3–B5 for a detected luminous energy *L_star*.

    The full temperature range is sampled uniformly, and the bolometric
    correction and kinetic energy are computed at each temperature.  The
    reported energy is the mean of the resulting E_kin distribution.

    This follows Hueso et al. (2018): due to the nonlinearity of the
    power-law conversion from luminous to kinetic energy, the mean
    energy cannot be taken as the average of the range endpoints.
    Instead, all blackbody temperatures in [T_low, T_high] are assumed
    equally probable, giving a distribution of E_kin values whose mean
    is the correct average energy estimate.

    Impactor masses are computed at each velocity, and diameters at each
    (velocity, density) pair, all from the mean kinetic energy.
    """
    T_low, T_high = T_flash_range_K
    T_samples     = np.linspace(T_low, T_high, n_T_samples)

    # B3: bolometric correction at each temperature
    L_f_samples   = np.array([bolometric_correction(L_star, T, wavelength_nm, C)
                              for T in T_samples])
    # B4: kinetic energy at each temperature
    E_kin_samples = np.array([kinetic_energy_from_luminous(L_f)
                              for L_f in L_f_samples])

    E_kin_mean = E_kin_samples.mean()
    E_kin_low  = E_kin_samples.min()
    E_kin_high = E_kin_samples.max()

    # Luminous efficiency η = L_f / E_kin at each temperature sample
    eta_samples = L_f_samples / E_kin_samples

    # B5: impactor mass per velocity, diameter per (velocity, density)
    masses    = {v: impactor_mass(E_kin_mean, v) for v in v_kms_values}
    diameters = {
        v: {rho: impactor_diameter(masses[v], rho) for rho in density_values}
        for v in v_kms_values
    }

    return {
        "L_f_mean_J":     L_f_samples.mean(),
        "L_f_low_J":      L_f_samples.min(),
        "L_f_high_J":     L_f_samples.max(),
        "E_kin_mean_J":   E_kin_mean,
        "E_kin_low_J":    E_kin_low,
        "E_kin_high_J":   E_kin_high,
        "E_kin_mean_kt":  E_kin_mean / KT_TO_J,
        "E_kin_low_kt":   E_kin_low  / KT_TO_J,
        "E_kin_high_kt":  E_kin_high / KT_TO_J,
        "E_kin_samples":  E_kin_samples,
        "eta_mean":       float(eta_samples.mean()),
        "eta_low":        float(eta_samples.min()),
        "eta_high":       float(eta_samples.max()),
        "T_samples":      T_samples,
        "masses_kg":      masses,
        "diameters_m":    diameters,
        "v_kms_values":   v_kms_values,
        "density_values": density_values,
        "T_low_K":        T_low,
        "T_high_K":       T_high,
    }


# ===========================================================================
# PART B — END-TO-END ENTRY POINTS
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
    qe_csvs: list = None,
    transmission_csv: str = None,
    # --- Solar spectrum and Jupiter albedo CSVs ---
    solar_csv: str = None,
    albedo_csv: str = None,
    # --- Cloud reflection ---
    fcr: float = 0.3,
    # --- Physical assumptions ---
    T_flash_range_K: tuple = (3500.0, 8500.0),
    v_kms_values: list = None,      # defaults to [60, 65, 70] if None
    density_values: list = None,    # defaults to [500, 2500, 5000] if None
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
        Cloud reflection ratio from cloud_reflection.py (Arimatsu et al. 2023).
        0.3 is the default estimate (Hueso et al. 2018).
    T_flash_range_K : tuple (T_low, T_high)
        Blackbody temperature range [K] for bolometric correction.
        Sampled at 1 K spacing (int(T_high - T_low) samples).
    v_kms_values : list of float or None
        Impact velocities to evaluate [km s⁻¹].
    density_values : list of float or None
        Impactor bulk densities to evaluate [kg m⁻³].
    V_jupiter : float
        Apparent visual magnitude of Jupiter at observation.
    lam_min_nm, lam_max_nm : float
        Wavelength integration range [nm].
    n_lam : int
        Number of wavelength grid points.

    Returns
    -------
    dict
        K_J                       – conversion factor [J]
        L_star_J                  – detected flash energy after cloud correction [J]
        L_f_mean/low/high_J       – bolometric flash energy [J]
        E_kin_mean/low/high_J     – kinetic energy [J]
        E_kin_mean/low/high_kt    – kinetic energy [kt TNT]
        E_kin_samples             – E_kin at every temperature sample [J]
        eta_mean/low/high         – luminous efficiency L_f / E_kin
        T_samples                 – temperatures sampled [K]
        masses_kg                 – {v: mass}
        diameters_m               – {v: {rho: diameter}}
        v_kms_values, density_values
        flash_magnitude           – apparent magnitude at peak
        T_low_K, T_high_K         – temperature range used
    """
    if density_values is None:
        density_values = [500.0, 2500.0, 5000.0]
    if v_kms_values is None:
        v_kms_values = [60.0, 65.0, 70.0]

    # B1: conversion factor
    lam = np.linspace(lam_min_nm, lam_max_nm, n_lam)
    C   = camera_filter_response(lam, qe_csvs, transmission_csv)
    K   = compute_conversion_factor(d_jupiter_au, lam, C, dt_frame_s,
                                    solar_csv, albedo_csv)

    # B2: detected flash energy, then cloud reflection correction
    # (fcr from cloud_reflection.py, based on Arimatsu et al. 2023)
    L_star      = detected_flash_energy(I_star, I_Jupiter, K)
    L_star_corr = L_star * (1 / (1 + fcr))

    # B3–B5
    T_low, T_high = T_flash_range_K
    energy = _energy_and_impactor_properties(
        L_star_corr, T_flash_range_K, int(T_high - T_low),
        lam, C, v_kms_values, density_values,
    )

    peak = I_star_peak if I_star_peak is not None else I_star

    return {
        "K_J":             K,
        "L_star_J":        L_star_corr,
        **energy,
        "flash_magnitude": flash_magnitude(peak, I_Jupiter, V_jupiter),
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
    v_impact_kms: list = None,      # defaults to [60, 65, 70] if None
    density_values: list = None,    # defaults to [500, 2500, 5000] if None
    # --- Wavelength grid ---
    lam_min_nm: float = 400.0,
    lam_max_nm: float = 1000.0,
    n_lam: int = 1000,
) -> dict:
    """
    Characterise a Jupiter meteor impact from a light curve CSV
    already expressed in Watts per row.

    Picks up the pipeline at the bolometric correction step, skipping
    the K conversion and I_star/I_Jupiter ratio entirely.  No cloud
    reflection correction is applied.

    Parameters
    ----------
    light_curve_csv : str
        Path to CSV file containing the light curve.
    time_col : str
        Name of the time column [s].  Default 'time_s'.
    flux_col : str
        Name of the flux column [W].  Default 'optical_energy_J'.
        Values are multiplied by dt_frame_s to get per-frame energy.
    dt_frame_s : float or None
        Exposure time per frame [s].  If None, derived automatically
        from the median time step in the time column.
    qe_csvs : list of str or None
        Paths to QE CSV files [nm | %], one per camera channel.
    transmission_csv : str or None
        Path to filter transmission CSV [nm | %].
    T_flash_range_K : tuple (T_low, T_high)
        Blackbody temperature range [K] for bolometric correction.
        Sampled at 1000 points.
    v_impact_kms : list of float or None
        Impact velocities to evaluate [km s⁻¹].
    density_values : list of float or None
        Impactor bulk densities to evaluate [kg m⁻³].
    lam_min_nm, lam_max_nm : float
        Wavelength integration range [nm].
    n_lam : int
        Number of wavelength grid points.

    Returns
    -------
    Same dict as jupiter_impact_model minus K_J and flash_magnitude, plus
    L_star_peak_J, times_s, light_curve_W and dt_frame_s.
    """
    if v_impact_kms is None:
        v_impact_kms = [60.0, 65.0, 70.0]
    if density_values is None:
        density_values = [500.0, 2500.0, 5000.0]

    df            = pd.read_csv(light_curve_csv)
    times_s       = df[time_col].values.astype(float)
    light_curve_W = df[flux_col].values.astype(float)

    if dt_frame_s is None:
        dt_frame_s = float(np.median(np.diff(times_s)))
        print(f"  dt_frame_s derived from time column: {dt_frame_s:.4f} s")

    print(f"  Loaded {len(light_curve_W)} frames from {light_curve_csv}")

    lam = np.linspace(lam_min_nm, lam_max_nm, n_lam)
    C   = camera_filter_response(lam, qe_csvs, transmission_csv)

    # Integrate positive frames only → total detected luminous energy [J]
    L_star      = float(light_curve_W[light_curve_W > 0].sum() * dt_frame_s)
    L_star_peak = float(light_curve_W.max() * dt_frame_s)

    print(f"  L_star (integrated) = {L_star:.3e} J")
    print(f"  L_star (peak frame) = {L_star_peak:.3e} J")

    # B3–B5
    energy = _energy_and_impactor_properties(
        L_star, T_flash_range_K, 1000,
        lam, C, v_impact_kms, density_values,
    )

    return {
        "L_star_J":      L_star,
        "L_star_peak_J": L_star_peak,
        **energy,
        "times_s":       times_s,
        "light_curve_W": light_curve_W,
        "dt_frame_s":    dt_frame_s,
    }


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

    # With L_star = 1, bolometric_correction returns just the correction factor
    factors = np.array([
        bolometric_correction(1.0, T, wavelength_nm, C)
        for T in T_samples
    ])
    bolometric_factor = float(factors.mean())
    cloud_factor      = 1.0 / (1.0 + fcr)

    L_detected_J        = K * (light_curve_DN / I_Jupiter)       # Eq. 1
    L_cloud_corrected_J = L_detected_J * cloud_factor
    L_bolometric_J      = L_cloud_corrected_J * bolometric_factor

    return {
        "L_detected_J":        L_detected_J,
        "L_cloud_corrected_J": L_cloud_corrected_J,
        "L_bolometric_J":      L_bolometric_J,
        "bolometric_factor":   bolometric_factor,
        "cloud_factor":        cloud_factor,
    }


# ===========================================================================
# PART C — DIAGNOSTIC PLOTS
# ===========================================================================

def _save_or_show(fig, save_path: str, label: str) -> None:
    """Save *fig* to *save_path* (or show it if None), then close it."""
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  {label} saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def save_impact_stack(
    aligned_frames: np.ndarray,
    flash_indices: np.ndarray,
    reference: np.ndarray,
    save_path: str = None,
) -> np.ndarray:
    """
    Create and save a stacked image of all detected flash frames.

    Two versions are shown side by side:
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
    stack_raw    = flash_frames.mean(axis=0)
    stack_diff   = (flash_frames - reference).mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(stack_raw, cmap="gray", origin="upper",
                   vmin=np.percentile(stack_raw, 1),
                   vmax=np.percentile(stack_raw, 99))
    axes[0].set_title(f"Mean stack — {len(flash_indices)} flash frame(s)")
    axes[0].axis("off")

    axes[1].imshow(stack_diff, cmap="gray", origin="upper",
                   vmin=np.percentile(stack_diff, 10),
                   vmax=np.percentile(stack_diff, 99.9))
    axes[1].set_title("Mean differential stack (flash − reference)")
    axes[1].axis("off")

    fig.suptitle(
        f"Impact stack  |  {len(flash_indices)} flash frame(s) combined",
        fontsize=10,
    )
    fig.tight_layout()
    _save_or_show(fig, save_path, "Impact stack")

    return stack_diff


def plot_aperture_diagnostic(
    peak_idx: int,
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
    peak_idx : int
        Index of the flash frame (used in the panel title).
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
    diff   = flash_frame - reference
    cy, cx = flash_yx
    vmax   = np.percentile(diff, 99.9)

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
    axes[1].imshow(diff, cmap="gray", origin="upper", vmin=0, vmax=vmax)
    axes[1].plot(cx, cy, "+", color="red", markersize=12, markeredgewidth=1.5)
    axes[1].set_title("Differential image")
    axes[1].axis("off")

    # Right: zoomed differential with aperture and sky annulus overlaid
    axes[2].imshow(diff[y0:y1, x0:x1], cmap="gray", origin="upper",
                   vmin=0, vmax=vmax, extent=[x0, x1, y1, y0])

    circles = [
        (aperture_radius, "lime",   "-",  f"Aperture (r={aperture_radius:.0f} px)"),
        (sky_inner,       "orange", "--", f"Sky inner (r={sky_inner:.0f} px)"),
        (sky_outer,       "orange", "-",  f"Sky outer (r={sky_outer:.0f} px)"),
    ]
    for radius, colour, linestyle, label in circles:
        axes[2].add_patch(Circle((cx, cy), radius,
                                 edgecolor=colour, facecolor="none",
                                 linewidth=1.5, linestyle=linestyle, label=label))

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
    _save_or_show(fig, save_path, "Aperture diagnostic")


def _crop_bounds(shape: tuple, flash_yx: tuple, half_width: int) -> tuple:
    """
    (y0, y1, x0, x1) of a ± *half_width* px box around *flash_yx* (or the
    image centre if None), clipped to the image.  The full image if
    *half_width* is None.
    """
    H, W = shape
    if half_width is None:
        return 0, H, 0, W
    cy, cx = flash_yx if flash_yx is not None else (H / 2, W / 2)
    return (max(0, int(cy) - half_width), min(H, int(cy) + half_width),
            max(0, int(cx) - half_width), min(W, int(cx) + half_width))


def plot_flash_difference(
    flash_frame: np.ndarray,
    background: np.ndarray,
    flash_yx: tuple = None,
    half_width: int = None,
    mark_flash: bool = True,
    save_path: str = None,
    dpi: int = 300,
) -> None:
    """
    Publication-style image of the flash: the raw flash frame next to the
    background-subtracted frame (flash frame − *background*).

    Parameters
    ----------
    flash_frame : ndarray, shape (H, W)
        Co-registered grayscale frame containing the flash.
    background : ndarray, shape (H, W)
        Image to subtract.  Normally a median of quiet frames, e.g.
        lc["reference"] or np.median(lc["aligned_frames"][i0:i1], axis=0)
        for frames just before the impact; the median averages out
        seeing, so the disk subtracts much more cleanly than with a single
        frame.  A single pre-impact frame also works.
    flash_yx : tuple (row, col) or None
        Flash position, used for the marker and as the zoom centre.
    half_width : int or None
        Crop both panels to ± *half_width* px around *flash_yx*, or around
        the image centre if *flash_yx* is None.  None shows the full frame.
    mark_flash : bool
        Draw four crosshair ticks around the flash.
    save_path : str or None
        Output file; the extension sets the format (.png, .pdf, .eps …).
        If None, the figure is displayed.
    dpi : int
        Resolution for raster formats.
    """
    diff = flash_frame - background

    # Crop region (same for both panels)
    y0, y1, x0, x1 = _crop_bounds(diff.shape, flash_yx, half_width)
    raw_crop  = flash_frame[y0:y1, x0:x1]
    diff_crop = diff[y0:y1, x0:x1]

    # Symmetric stretch so the unchanged background is mid-grey and the
    # flash stands out as white
    dlim = np.percentile(np.abs(diff_crop), 99.9)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
    extent = [x0, x1, y1, y0]
    axes[0].imshow(raw_crop, cmap="gray", origin="upper", extent=extent,
                   vmin=np.percentile(raw_crop, 1), vmax=np.percentile(raw_crop, 99.5))
    axes[1].imshow(diff_crop, cmap="gray", origin="upper", extent=extent,
                   vmin=0, vmax=np.percentile(diff_crop, 99.9)) #vmin=-dlim, vmax=dlim)

    for ax, label in zip(axes, ["(a)", "(b)"]):
        if mark_flash and flash_yx is not None:
            fy, fx = flash_yx
            gap, length = 0.04 * (x1 - x0), 0.08 * (x1 - x0)
            # Four crosshair ticks (left, right, above, below) with a gap
            # so the flash itself is not covered
            for sign in (-1, 1):
                ax.plot([fx + sign * gap, fx + sign * (gap + length)], [fy, fy],
                        color="yellow", lw=1.2)
                ax.plot([fx, fx], [fy + sign * gap, fy + sign * (gap + length)],
                        color="yellow", lw=1.2)
        ax.text(0.03, 0.97, label, transform=ax.transAxes, color="white",
                fontsize=11, va="top", ha="left")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.axis("off")

    fig.tight_layout(pad=0.3)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        print(f"  Flash difference image saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_difference_histogram(
    flash_frame: np.ndarray,
    background: np.ndarray,
    flash_yx: tuple = None,
    half_width: int = None,
    aperture_radius: float = None,
    bins: int = 200,
    save_path: str = None,
) -> None:
    """
    Histogram of pixel values in the differential image
    (flash frame − *background*), over the same region that
    :func:`plot_flash_difference` shows.

    The background pixels should form a narrow peak around 0; the flash
    appears as a tail of bright pixels.  Dashed lines mark the ± stretch
    limits used for panel (b) of :func:`plot_flash_difference` (0 to the
    99.9th percentile), so pixels beyond them are shown there as pure black
    or white.

    Parameters
    ----------
    flash_frame, background, flash_yx, half_width
        As for :func:`plot_flash_difference`.
    aperture_radius : float or None
        If given (with *flash_yx*), pixels inside the photometric aperture
        are histogrammed separately from the rest of the region.
    bins : int
        Number of histogram bins.
    save_path : str or None
        If provided, the figure is saved to this path.  Otherwise it
        is displayed interactively.
    """
    diff = flash_frame - background
    y0, y1, x0, x1 = _crop_bounds(diff.shape, flash_yx, half_width)
    diff_crop = diff[y0:y1, x0:x1]
    vmin, vmax = 0.0, np.percentile(diff_crop, 99.9)   # panel (b) stretch

    # Split the region into aperture and background pixels
    if aperture_radius is not None and flash_yx is not None:
        _, _, r = _circular_radius_map(diff.shape, flash_yx)
        in_ap   = (r <= aperture_radius)[y0:y1, x0:x1]
    else:
        in_ap = np.zeros(diff_crop.shape, dtype=bool)
    bkg_pix = diff_crop[~in_ap]
    ap_pix  = diff_crop[in_ap]

    med = np.median(bkg_pix)
    std = 1.4826 * np.median(np.abs(bkg_pix - med))   # robust σ from the MAD

    edges = np.linspace(diff_crop.min(), diff_crop.max(), bins + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(bkg_pix, bins=edges, color="gray", alpha=0.8,
            label=f"Background pixels (median {med:.1f}, σ {std:.1f} DN)")
    if ap_pix.size:
        ax.hist(ap_pix, bins=edges, color="red", alpha=0.7,
                label=f"Aperture pixels (r={aperture_radius:g} px, max {ap_pix.max():.1f} DN)")
    ax.axvline(vmin, color="black", linestyle="--", linewidth=1.0,
               label=f"Display stretch {vmin:.0f}–{vmax:.1f} DN")
    ax.axvline(vmax, color="black", linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Differential value [DN]")
    ax.set_ylabel("Number of pixels")
    region = "full frame" if half_width is None else f"±{half_width} px around flash"
    ax.set_title(f"Differential image histogram ({region})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_or_show(fig, save_path, "Differential histogram")


def plot_light_curve_DN(lc: dict, save_path: str) -> None:
    """Plot the raw (with error bars) and smoothed light curve in DN."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(lc["frame_numbers"], lc["light_curve"], yerr=lc["uncertainties"],
                fmt="--", color="gray", alpha=0.5, markersize=5,
                ecolor="gray", elinewidth=1.0, capsize=3, label="Raw")
    ax.plot(lc["frame_numbers"], lc["light_curve_smooth"],
            "-", color="r", linewidth=2.0, label="Smoothed")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", label="Noise Level")
    # flash_num = lc["frame_numbers"][lc["peak_flash_idx"]]
    # ax.axvline(flash_num, color="red", linewidth=1.0,
    #            linestyle="--", label=f"Flash (frame {flash_num})")
    ax.set_xlabel("Frame number")
    ax.set_ylabel("Integrated Counts [DN]")
    # ax.set_title("Jupiter impact — light curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  Light curve saved → {save_path}")


def plot_corrected_light_curves(
    time_s: np.ndarray,
    flash_time: float,
    lc_corr: dict,
    lc_corr_smooth: dict,
    lc_corr_unc: dict,
    fcr: float,
    save_path: str,
) -> None:
    """
    Two stacked panels of energy vs time:
      Top    – cloud-reflection corrected
      Bottom – cloud + bolometric corrected
    Each shows the raw per-frame values with error bars and the smoothed curve.
    """
    panels = [
        ("L_cloud_corrected_J", "steelblue", "Cloud-corrected energy [J]",
         f"Cloud reflection corrected  (fcr={fcr:.2f}, "
         f"factor={lc_corr['cloud_factor']:.4f})"),
        ("L_bolometric_J", "darkorange", "Bolometric energy [J]",
         f"Cloud + bolometric corrected  "
         f"(bolometric factor={lc_corr['bolometric_factor']:.3f})"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for ax, (key, colour, ylabel, title) in zip(axes, panels):
        ax.errorbar(time_s, lc_corr[key], yerr=lc_corr_unc[key],
                    fmt="o", color=colour, markersize=5, alpha=0.5,
                    ecolor=colour, elinewidth=1.0, capsize=3, label="Raw")
        ax.plot(time_s, lc_corr_smooth[key],
                "-", color=colour, linewidth=2.0, label="Savitzky-Golay")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(flash_time, color="red", linewidth=1.0,
                   linestyle="--", label=f"Flash ({flash_time:.2f} s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Time [s]")

    fig.suptitle("Jupiter impact — corrected light curves", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  Corrected light curve saved → {save_path}")


def print_results_table(results: dict, v_kms_values: list, density_values: list) -> None:
    """Print a summary table of the jupiter_impact_model results."""
    fmt = "  {:<30s}  {:>15}"
    print(f"\n  {'Parameter':<30s}  {'Value':>15}")
    print("  " + "-" * 48)
    print(fmt.format("Conversion factor K",          f"{results['K_J']:.3e} J"))
    print(fmt.format("Detected energy L*",           f"{results['L_star_J']:.3e} J"))
    print(fmt.format("Flash temperature range",
                     f"{results['T_low_K']:.0f}–{results['T_high_K']:.0f} K"))
    print(fmt.format("Luminous energy (mean T)",     f"{results['L_f_mean_J']:.3e} J"))
    print(fmt.format("Luminous energy (kt, mean)",   f"{results['L_f_mean_J'] / KT_TO_J:.3f} kt"))
    print(fmt.format("Luminous energy range [J]",
                     f"{results['L_f_low_J']:.2e}–{results['L_f_high_J']:.2e} J"))
    print(fmt.format("Kinetic energy (mean T)",      f"{results['E_kin_mean_J']:.3e} J"))
    print(fmt.format("Kinetic energy (kt, mean)",    f"{results['E_kin_mean_kt']:.1f} kt"))
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
    print(fmt.format("Flash apparent magnitude",     f"{results['flash_magnitude']:.1f}"))


# ---------------------------------------------------------------------------
# Cloud reflection plots (Part 0)
# ---------------------------------------------------------------------------

def plot_DHG_parameters(wavelengths, omega, g1, g2, f,
                        save_path: str = "DHG_parameters.png") -> None:
    """Plot the DHG parameters ω, g1, g2 and f against wavelength."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wavelengths, omega, label=r"$\omega$")
    ax.plot(wavelengths, g1, label="g1")
    ax.plot(wavelengths, g2, label="g2")
    ax.plot(wavelengths, f, label="f")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_xlim(400, 1000)
    ax.set_title("DHG parameters")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save_path, "DHG parameters")


def plot_fCR_lambertian(pg_values: list = (0.3, 0.5, 0.65),
                        save_path: str = "fCR_lambertian.png") -> None:
    """
    f_CR against emission angle for several geometric albedos, using the
    Lambertian model.  0.65 is Jupiter's maximum p_g at 500–700 nm.
    """
    mu_arr = np.linspace(0.0, 1.0, 200)

    fig, ax = plt.subplots(figsize=(7, 4))
    for pg in pg_values:
        ax.plot(np.degrees(np.arccos(mu_arr)), fCR_lambertian_geometric(mu_arr, pg),
                label=f"$p_g$ = {pg}")
    ax.axhline(0.3, color="k", linestyle="--", label="Previous constant ($f_{CR}$ = 0.3)")
    ax.set_xlabel("Emission angle (°)")
    ax.set_ylabel("$f_{CR}$")
    ax.set_title("Lambertian cloud-reflectivity correction factor")
    ax.legend()
    ax.invert_xaxis()
    fig.tight_layout()
    _save_or_show(fig, save_path, "Lambertian f_CR")


def plot_delta_ICR(pg_csv_path: str, emission_angles: list = (0, 20, 40, 60, 80),
                   save_path: str = "delta_ICR_wavelength_Pg.png") -> None:
    """
    ΔI_CR/I₀ against wavelength for several emission angles with the
    Lambertian model and a tabulated p_g(λ) (columns "Wavelength", "Pg").
    Replicates Fig. 3 of Arimatsu et al. (2023).
    """
    df          = pd.read_csv(pg_csv_path, index_col=0)
    wavelengths = df["Wavelength"].values
    pg_lambda   = df["Pg"].values

    fig, ax = plt.subplots(figsize=(8, 5))
    for angle_deg in emission_angles:
        delta_ratio = fCR_lambertian_geometric(np.cos(np.radians(angle_deg)), pg_lambda)
        ax.plot(wavelengths, savgol_filter(delta_ratio, window_length=7, polyorder=3),
                label=f"cos-1(mu) = {angle_deg}°")
        ax.plot(wavelengths, delta_ratio, alpha=0.2, color="gray")
    ax.axhline(0.3, color="k", linestyle="--", label="Previous constant ($f_{CR}$ = 0.3)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("$\\Delta I_{CR} / I_0$")
    ax.set_xlim(400, 1000)
    ax.set_title("Cloud-reflectivity correction factor vs wavelength")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save_path, "Lambertian delta_ICR")


def plot_delta_ICR_DHG(wavelengths_DHG, omega, g1, g2, f,
                       emission_angles: list = (0, 20, 40, 60, 80),
                       h: float = 50.0, r_max: float = 3000.0,
                       n_r: int = 300, n_phi: int = 180,
                       save_path: str = "delta_ICR_wavelength_DHG.png") -> None:
    """
    ΔI_CR/I₀ against wavelength for several emission angles with the full
    DHG model, replicating Fig. 3 of Arimatsu et al. (2023).

    Slow: about 25 s per emission angle on a 600-point grid.  Pass every
    10th point (e.g. wavelengths_DHG[::10]) with lower n_r / n_phi for a
    quick look.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for angle_deg in emission_angles:
        mu = np.cos(np.radians(angle_deg))
        delta_ratio = np.array([
            delta_ICR_DHG(mu, omega[i], g1[i], g2[i], f[i],
                          h=h, r_max=r_max, n_r=n_r, n_phi=n_phi)
            for i in range(len(wavelengths_DHG))
        ])
        ax.plot(wavelengths_DHG, delta_ratio, alpha=0.5, color="gray")
        ax.plot(wavelengths_DHG, savgol_filter(delta_ratio, window_length=11, polyorder=3),
                label=f"cos-1(mu) = {angle_deg}°")
        print(f"  Completed emission angle {angle_deg}°")
    ax.axhline(0.3, color="k", linestyle="--", label="Previous constant ($f_{CR}$ = 0.3)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("$\\Delta I_{CR} / I_0$")
    ax.set_xlim(400, 1000)
    ax.set_title("Cloud-reflectivity correction factor vs wavelength (DHG model)")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save_path, "DHG delta_ICR")
