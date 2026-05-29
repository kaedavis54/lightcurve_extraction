"""
Jupiter Impact Flash Cloud-Reflectivity Correction Model
=========================================================
Based on: Arimatsu et al. (2023), A&A, 677, A165

This script models the cloud-reflectivity correction factor (fCR) for observed
Jupiter impact flashes. It accounts for the reflected component of the flash
from Jovian upper clouds using Lambertian and Double Henyey-Greenstein (DHG)
scattering phase functions.

Two models are implemented:
  1. Simple isotropic (Lambertian) model: fCR = 2 * mu * pL = 3 * mu * pg
  2. Full DHG model: integrates wavelength-dependent Jovian phase functions
     from Heng & Li (2021) over the illuminated cloud surface.
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter

plt.rc("text",usetex=True)
plt.rc("text.latex",preamble=r"\usepackage{amsmath} \usepackage{gensymb} \usepackage{siunitx}")
plt.rc("font",size="14")

# ---------------------------------------------------------------------------
# 1. SIMPLE LAMBERTIAN MODEL (Eqs. 1–5 of the paper)
# ---------------------------------------------------------------------------

def fCR_lambertian(mu: float, pL: float) -> float:
    """
    Cloud-reflectivity correction factor using an isotropic Lambertian surface.

    Parameters
    ----------
    mu  : cosine of the emission angle (1 = nadir, 0 = limb)
    pL  : Lambertian albedo of the Jovian cloud deck

    Returns
    -------
    fCR : ratio of reflected flux to direct source flux (Eq. 4)
    """
    return 2.0 * mu * pL


def fCR_lambertian_geometric(mu: float, pg: float) -> float:
    """
    Same as fCR_lambertian but expressed via geometric albedo pg.
    Relation: pg = (2/3) * pL  ->  fCR = 3 * mu * pg  (Eq. 5)
    """
    return 3.0 * mu * pg


def fcor_from_fobs(Fobs: float, fCR: float) -> float:
    """
    Corrected flux from observed flux and correction factor (Eq. 1).

    Fcor = Fobs / (1 + fCR)
    """
    return Fobs / (1.0 + fCR)


# ---------------------------------------------------------------------------
# 2. CHANDRASEKHAR H-FUNCTION (Eq. 7)
# ---------------------------------------------------------------------------

def chandrasekhar_H(mu: float, omega: float, n_iter: int = 50) -> float:
    """
    Numerically solve for the Chandrasekhar H-function via successive
    substitution (Eq. 7).

    H(mu) = 1 + 0.5 * omega * mu * H(mu) * integral_0^1 [ H(mu') / (mu + mu') ] dmu'

    Parameters
    ----------
    mu     : cosine of the angle
    omega  : single-scattering albedo
    n_iter : number of iterations for convergence

    Returns
    -------
    H(mu)
    """
    mu_grid = np.linspace(0.0, 1.0, 200)

    # Initial guess: H = 1 everywhere
    H_grid = np.ones_like(mu_grid)

    for _ in range(n_iter):
        # Build interpolated H over mu_grid for the integral
        integrand = H_grid / (mu + mu_grid + 1e-100)
        integral_val = np.trapezoid(integrand, mu_grid)
        H_new = 1.0 + 0.5 * omega * mu * integral_val

        # Update H at this mu; rebuild full grid for next iteration
        # (for full consistency we rebuild the whole grid each iteration)
        H_grid_new = np.empty_like(mu_grid)
        for i, m in enumerate(mu_grid):
            intgd = H_grid / (m + mu_grid + 1e-100)
            H_grid_new[i] = 1.0 + 0.5 * omega * m * np.trapezoid(intgd, mu_grid)
        H_grid = H_grid_new

    return H_new


def chandrasekhar_H_array(mu_arr: np.ndarray, omega: float, n_iter: int = 30) -> np.ndarray:
    """
    Vectorised version of chandrasekhar_H over an array of mu values.
    Uses the self-consistent iteration on the full grid simultaneously.
    """
    H = np.ones_like(mu_arr, dtype=float)
    for _ in range(n_iter):
        H_new = np.empty_like(H)
        for i, mu in enumerate(mu_arr):
            intgd = H / (mu + mu_arr + 1e-100)
            H_new[i] = 1.0 + 0.5 * omega * mu * np.trapezoid(intgd, mu_arr)
        H = H_new
    return H


# ---------------------------------------------------------------------------
# 3. DOUBLE HENYEY-GREENSTEIN PHASE FUNCTION (Eq. 8)
# ---------------------------------------------------------------------------

def DHG_phase(cos_alpha: float | np.ndarray,
              g1: float, g2: float, f: float) -> float | np.ndarray:
    """
    Double Henyey-Greenstein scattering phase function (Eq. 8).

    P(alpha) = f * (1-g1^2)/(1+g1^2+2*g1*cos_alpha)^1.5
             + (1-f) * (1-g2^2)/(1+g2^2+2*g2*cos_alpha)^1.5

    Parameters
    ----------
    cos_alpha : cosine of the scattering phase angle
    g1, g2    : asymmetry factors for the two HG lobes
    f         : fractional weight for the first lobe

    Returns
    -------
    P(alpha)
    """
    term1 = f * (1 - g1**2) / (1 + g1**2 + 2*g1*cos_alpha)**1.5
    term2 = ((1 - f) * (1 - g2**2)) / (1 + g2**2 + 2*g2*cos_alpha)**1.5
    return term1 + term2


# ---------------------------------------------------------------------------
# 4. FULL DHG CLOUD-REFLECTION INTEGRAL (Eq. 6)
# ---------------------------------------------------------------------------

def delta_ICR_DHG(mu: float, omega: float, g1: float, g2: float, f: float,
                  h: float = 50.0, r_max: float = 5000.0, n_r: int = 500,
                  n_phi: int = 360) -> float:
    """
    Compute delta_ICR / I0 by numerically integrating Eq. 6 over the cloud
    surface using polar coordinates (r, phi) centred below the flash source.

    Geometry
    --------
    Source at height h above a flat cloud plane.
    Observer (emission) direction has cosine mu (= cos of emission angle).
    The observer azimuth is fixed at phi=0 (result is phi-symmetric for
    uniform clouds).

    Parameters
    ----------
    mu     : cosine of emission angle
    omega  : single-scattering albedo
    g1, g2 : DHG asymmetry parameters
    f      : DHG fractional weight
    h      : flash height above clouds [km]  (default 50 km)
    r_max  : maximum radial integration limit [km]
    n_r    : number of radial sample points
    n_phi  : number of azimuthal sample points

    Returns
    -------
    delta_ICR / I0  (dimensionless)
    """
    # Pre-compute H-function on a fine mu grid for interpolation
    mu_grid = np.linspace(0.0, 1.0, 100)
    H_grid = chandrasekhar_H_array(mu_grid, omega, n_iter=20)

    def H_interp(m):
        return np.interp(np.clip(m, 0.0, 1.0), mu_grid, H_grid)

    r_arr = np.linspace(0.0, r_max, n_r + 1)[1:]   # avoid r=0 singularity
    phi_arr = np.linspace(0.0, 2*np.pi, n_phi, endpoint=False)

    dr = r_arr[1] - r_arr[0]
    dphi = phi_arr[1] - phi_arr[0]

    total = 0.0
    for phi in phi_arr:
        # x, y coordinates on cloud plane
        x = r_arr * np.cos(phi)
        # y not needed explicitly; geometry is axially symmetric about nadir
        # mu0: cosine of incidence angle at each surface point
        dist2 = r_arr**2 + h**2
        dist = np.sqrt(dist2)
        mu0_arr = h / dist                        # cos(incidence angle)

        # Phase angle alpha between incident ray and observer direction.
        # Incident ray direction: unit vector from source to surface element
        #   = (x/dist, y/dist, -h/dist)  [downward]
        # Observer direction (emission): (sin_theta * cos_phi_obs, sin_theta * sin_phi_obs, mu)
        # For simplicity set observer azimuth = 0.
        sin_theta = np.sqrt(max(1 - mu**2, 0.0))
        # cos(alpha) = dot product of incoming and outgoing rays
        # incoming to surface: (r*cos(phi)/dist, r*sin(phi)/dist, -mu0)
        # outgoing to observer: (sin_theta, 0, mu)  [phi_obs=0]
        cos_alpha = (x / dist) * sin_theta + (-mu0_arr) * mu
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)

        P_arr = DHG_phase(cos_alpha, g1, g2, f)
        H_obs = H_interp(mu)
        H_inc_arr = H_interp(mu0_arr)

        # Hapke-like RADF from Eq. 6:
        # integrand = I0(lambda) / (r^2 + h^2) * omega/4 * mu0/(mu0+mu) * [P - 1 + H(mu)*H(mu0)]
        RADF = (omega / 4.0) * (mu0_arr / (mu0_arr + mu + 1e-100)) * \
               (P_arr - 1.0 + H_obs * H_inc_arr)

        integrand = (mu / dist2) * RADF * r_arr   # r dr dphi factor

        total += np.sum(integrand) * dr * dphi

    return total


# ---------------------------------------------------------------------------
# 5. WAVELENGTH-INTEGRATED fCR (Eq. 9)
# ---------------------------------------------------------------------------

def compute_fCR_integrated(wavelengths: np.ndarray,
                           response: np.ndarray,
                           delta_ICR_over_I0: np.ndarray) -> float:
    """
    Compute the band-integrated fCR given the instrument response R(lambda)
    and delta_ICR(lambda)/I0(lambda) evaluated at each wavelength (Eq. 9).

    fCR = integral[ R(lambda) * delta_ICR(lambda) dlambda ]
        / integral[ R(lambda) * I0(lambda) dlambda ]

    Since delta_ICR_over_I0 = delta_ICR / I0, this simplifies to:

    fCR = integral[ R(lambda) * (delta_ICR/I0)(lambda) dlambda ]
        / integral[ R(lambda) dlambda ]

    Parameters
    ----------
    wavelengths       : wavelength array [nm]
    response          : instrument response R(lambda) [normalised]
    delta_ICR_over_I0 : delta_ICR(lambda)/I0(lambda) array

    Returns
    -------
    fCR : band-integrated correction factor
    """
    numerator   = np.trapezoid(response * delta_ICR_over_I0, wavelengths)
    denominator = np.trapezoid(response, wavelengths)
    return numerator / denominator


# ---------------------------------------------------------------------------
# 6. DEMONSTRATION / EXAMPLE USAGE
# ---------------------------------------------------------------------------

def demo_lambertian():
    """Plot fCR vs emission angle for several geometric albedos (Lambertian model)."""
    mu_arr = np.linspace(0.0, 1.0, 200)
    pg_values = [0.3, 0.5, 0.65]   # 0.65 is max for Jupiter at 500-700 nm

    fig, ax = plt.subplots(figsize=(7, 4))
    for pg in pg_values:
        fCR_arr = fCR_lambertian_geometric(mu_arr, pg)
        ax.plot(np.degrees(np.arccos(mu_arr)), fCR_arr,
                label=f"$p_g$ = {pg}")

    ax.axhline(0.3, color="k", linestyle="--", label="Previous constant ($f_{CR}$ = 0.3)")
    ax.set_xlabel("Emission angle (°)")
    ax.set_ylabel("$f_{CR}$")
    ax.set_title("Lambertian cloud-reflectivity correction factor")
    ax.legend()
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig("fCR_lambertian.png", dpi=150)
    plt.close()
    print("Saved: fCR_lambertian.png")


def demo_flux_correction():
    """
    Show how observed flux is corrected for several emission angles
    using the Lambertian model with Jupiter's peak geometric albedo.
    """
    print("\n--- Flux correction example (Lambertian, pg=0.65) ---")
    pg = 0.65
    Fobs = 1.0   # normalised observed flux

    for angle_deg in [0, 20, 40, 60, 80]:
        mu = np.cos(np.radians(angle_deg))
        fCR = fCR_lambertian_geometric(mu, pg)
        Fcor = fcor_from_fobs(Fobs, fCR)
        print(f"  Emission angle = {angle_deg:2d}°  |  fCR = {fCR:.3f}  |  "
              f"Fcor/Fobs = {Fcor:.3f}  (correction = {(1-Fcor)*100:.1f}%)")

def load_DHG_parameters(omega_path, g1_path, g2_path, f_path):
    """
    Load DHG parameters from CSV files and interpolate onto a common
    wavelength grid covering the overlapping wavelength range.
    """
    df_omega = pd.read_csv(omega_path, index_col=0, header=0, names=["Wavelength", "omega"])
    df_g1    = pd.read_csv(g1_path,    index_col=0, header=0, names=["Wavelength", "g1"])
    df_g2    = pd.read_csv(g2_path,    index_col=0, header=0, names=["Wavelength", "g2"])
    df_f     = pd.read_csv(f_path,     index_col=0, header=0, names=["Wavelength", "f"])

    # Find the overlapping wavelength range across all four files
    wl_min = max(df_omega["Wavelength"].min(), df_g1["Wavelength"].min(),
                 df_g2["Wavelength"].min(), df_f["Wavelength"].min())
    wl_max = min(df_omega["Wavelength"].max(), df_g1["Wavelength"].max(),
                 df_g2["Wavelength"].max(), df_f["Wavelength"].max())

    # Common grid at 1 nm spacing within the overlapping range
    wavelengths = np.arange(wl_min, wl_max, 1.0)

    # Interpolate each parameter onto the common grid
    omega = np.interp(wavelengths, df_omega["Wavelength"].values, df_omega["omega"].values)
    g1    = np.interp(wavelengths, df_g1["Wavelength"].values,    df_g1["g1"].values)
    g2    = np.interp(wavelengths, df_g2["Wavelength"].values,    df_g2["g2"].values)
    f     = np.interp(wavelengths, df_f["Wavelength"].values,     df_f["f"].values)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wavelengths, omega, label=r"$\omega$")
    ax.plot(wavelengths, g1, label='g1')
    ax.plot(wavelengths, g2, label='g2')
    ax.plot(wavelengths, f, label='f')
    ax.set_xlabel("Wavelength (nm)")
    ax.set_xlim(400,1000)
    ax.set_title("DHG parameters")
    ax.legend()
    plt.tight_layout()
    plt.savefig("DHG_parameters.png", dpi=150)
    plt.close()
    print("Saved: DHG_parameters.png")


    print(f"Common wavelength grid: {wl_min:.1f} - {wl_max:.1f} nm ({len(wavelengths)} points)")

    return wavelengths, omega, g1, g2, f

def demo_DHG_single(wavelengths: np.ndarray, omega: np.ndarray, g1: np.ndarray,
                    g2: np.ndarray, f: np.ndarray, target_wavelength: float = 550.0):
    """
    Compute delta_ICR/I0 for a single wavelength using real DHG parameters
    from Heng & Li (2021).

    Parameters
    ----------
    wavelengths       : wavelength array in nm
    omega, g1, g2, f  : DHG parameter arrays on the same wavelength grid
    target_wavelength : wavelength in nm to evaluate (default 550 nm)
    """
    # Find the index closest to the target wavelength
    idx = np.argmin(np.abs(wavelengths - target_wavelength))
    actual_wavelength = wavelengths[idx]

    # Extract DHG parameters at that wavelength
    omega_val = omega[idx]
    g1_val    = g1[idx]
    g2_val    = g2[idx]
    f_val     = f[idx]

    print(f"\n--- DHG model at {actual_wavelength:.1f} nm ---")
    print(f"  omega={omega_val:.4f}, g1={g1_val:.4f}, g2={g2_val:.4f}, f={f_val:.4f}")

    for angle_deg in [0, 30, 60]:
        mu = np.cos(np.radians(angle_deg))
        ratio = delta_ICR_DHG(mu, omega_val, g1_val, g2_val, f_val,
                               h=50.0, r_max=3000.0, n_r=300, n_phi=180)
        print(f"  cos⁻¹(mu) = {angle_deg}°  |  delta_ICR/I0 = {ratio:.4f}")


def demo_integrated_fCR(pg_csv_path: str, cam_wavelengths: np.ndarray,
                        R_response: np.ndarray, G_response: np.ndarray,
                        B_response: np.ndarray, mu: float = 1.0):
    """
    Compute band-integrated fCR for each camera channel using real pg(lambda)
    and camera system response R(lambda) = QE(lambda) * T(lambda).
    """
    df = pd.read_csv(pg_csv_path, index_col=0, header=0, names=["Wavelength", "pg"])

    # Interpolate pg onto the camera wavelength grid
    pg_lambda = np.interp(cam_wavelengths, df["Wavelength"].values, df["pg"].values)

    delta_ratio = 3.0 * mu * pg_lambda

    fCR_R = compute_fCR_integrated(cam_wavelengths, R_response, delta_ratio)
    fCR_G = compute_fCR_integrated(cam_wavelengths, G_response, delta_ratio)
    fCR_B = compute_fCR_integrated(cam_wavelengths, B_response, delta_ratio)

    print(f"\n--- Band-integrated fCR by channel (mu={mu}) ---")
    print(f"  R channel: fCR = {fCR_R:.4f}")
    print(f"  G channel: fCR = {fCR_G:.4f}")
    print(f"  B channel: fCR = {fCR_B:.4f}")
    print(f"  (Previous constant assumption: fCR = 0.3)")

#     return fCR_R, fCR_G, fCR_B

def plot_delta_ICR(csv_path: str, emission_angles: list = [0, 20, 40, 60, 80]):
    """
    Plot delta_ICR/I0 vs wavelength for several emission angles,
    using real pg(lambda) data from CSV. Replicates Fig. 3 of the paper.

    Parameters
    ----------
    csv_path        : path to CSV file with wavelength and pg columns
    emission_angles : list of emission angles in degrees to plot
    """
    df = pd.read_csv(csv_path, index_col=0)

    wavelengths = df["Wavelength"].values
    pg_lambda   = df["Pg"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    for angle_deg in emission_angles:
        mu = np.cos(np.radians(angle_deg))
        delta_ratio = 3.0 * mu * pg_lambda
        delta_ratio_smooth = savgol_filter(delta_ratio, window_length=7, polyorder=3)

        ax.plot(wavelengths, delta_ratio_smooth, label=f"cos-1(mu) = {angle_deg}°")
        #ax.plot(wavelengths, delta_ratio, label=f"cos-1(mu) = {angle_deg}°")
        ax.plot(wavelengths, delta_ratio, alpha=0.2, color='gray')

    ax.axhline(0.3, color="k", linestyle="--", label="Previous constant ($f_{CR}$ = 0.3)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("$\\Delta I_{CR} / I_0$")
    ax.set_xlim(400,1000)
    ax.set_title("Cloud-reflectivity correction factor vs wavelength")
    ax.legend()
    plt.tight_layout()
    plt.savefig("delta_ICR_wavelength_Pg.png", dpi=150)
    plt.close()
    print("Saved: delta_ICR_wavelength_Pg.png")

def plot_delta_ICR_DHG(wavelengths_DHG: np.ndarray, omega: np.ndarray, g1: np.ndarray,
                    g2: np.ndarray, f: np.ndarray,
                    emission_angles: list = [0, 20, 40, 60, 80],
                    h: float = 50.0, r_max: float = 3000.0,
                    n_r: int = 300, n_phi: int = 180):
    """
    Plot delta_ICR/I0 vs wavelength for several emission angles using the
    full DHG model (Eq. 6), replicating Fig. 3 of Arimatsu et al. (2023).

    Parameters
    ----------
    wavelengths_DHG   : wavelength array in nm
    omega, g1, g2, f  : DHG parameter arrays on the same wavelength grid
    emission_angles   : list of emission angles in degrees
    h                 : flash height above clouds in km
    r_max             : maximum radial integration limit in km
    n_r, n_phi        : radial and azimuthal integration resolution
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for angle_deg in emission_angles:
        mu = np.cos(np.radians(angle_deg))
        delta_ratio = np.empty_like(wavelengths_DHG)

        for i in range(len(wavelengths_DHG)):
            delta_ratio[i] = delta_ICR_DHG(mu, omega[i], g1[i], g2[i], f[i],
                                            h=h, r_max=r_max, n_r=n_r, n_phi=n_phi)

        delta_ratio_smooth = savgol_filter(delta_ratio, window_length=11, polyorder=3)

        ax.plot(wavelengths_DHG, delta_ratio, alpha=0.5, color='gray')
        ax.plot(wavelengths_DHG, delta_ratio_smooth, label=f"cos-1(mu) = {angle_deg}°")
        print(f"  Completed emission angle {angle_deg}°")

    ax.axhline(0.3, color="k", linestyle="--", label="Previous constant ($f_{CR}$ = 0.3)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("$\\Delta I_{CR} / I_0$")
    ax.set_xlim(400, 1000)
    ax.set_title("Cloud-reflectivity correction factor vs wavelength (DHG model)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("delta_ICR_wavelength_DHG.png", dpi=150)
    plt.close()
    print("Saved: delta_ICR_wavelength_DHG.png")

def load_camera_response(transmission_path: str, r_path: str = None, 
                         g_path: str = None, b_path: str = None, 
                         mono_path: str = None):
    """
    Load camera quantum efficiency and filter transmission curves.
    Supports both monochrome (single QE channel) and colour (R, G, B) cameras.

    Parameters
    ----------
    transmission_path : path to filter transmission CSV file
    r_path, g_path, b_path : paths to QE CSV files for RGB channels (colour camera)
    mono_path         : path to QE CSV file for monochrome camera

    Returns
    -------
    wavelengths  : common wavelength grid in nm
    responses    : dict with keys 'R', 'G', 'B' or 'mono' mapping to response arrays
    """
    if mono_path is None and (r_path is None or g_path is None or b_path is None):
        raise ValueError("Provide either mono_path or all of r_path, g_path, b_path.")

    df_t = pd.read_csv(transmission_path, index_col=0, header=0, names=["Wavelength", "T"])

    if mono_path is not None:
        # Monochrome camera
        df_mono = pd.read_csv(mono_path, index_col=0, header=0, names=["Wavelength", "QE"])

        wl_min = max(df_mono["Wavelength"].min(), df_t["Wavelength"].min())
        wl_max = min(df_mono["Wavelength"].max(), df_t["Wavelength"].max())
        wavelengths = np.arange(wl_min, wl_max, 1.0)

        QE_mono = np.interp(wavelengths, df_mono["Wavelength"].values, df_mono["QE"].values)
        T       = np.interp(wavelengths, df_t["Wavelength"].values, df_t["T"].values)

        responses = {"mono": QE_mono * T}

    else:
        # Colour camera
        df_r = pd.read_csv(r_path, index_col=0, header=0, names=["Wavelength", "QE"])
        df_g = pd.read_csv(g_path, index_col=0, header=0, names=["Wavelength", "QE"])
        df_b = pd.read_csv(b_path, index_col=0, header=0, names=["Wavelength", "QE"])

        wl_min = max(df_r["Wavelength"].min(), df_g["Wavelength"].min(),
                     df_b["Wavelength"].min(), df_t["Wavelength"].min())
        wl_max = min(df_r["Wavelength"].max(), df_g["Wavelength"].max(),
                     df_b["Wavelength"].max(), df_t["Wavelength"].max())
        wavelengths = np.arange(wl_min, wl_max, 1.0)

        QE_r = np.interp(wavelengths, df_r["Wavelength"].values, df_r["QE"].values)
        QE_g = np.interp(wavelengths, df_g["Wavelength"].values, df_g["QE"].values)
        QE_b = np.interp(wavelengths, df_b["Wavelength"].values, df_b["QE"].values)
        T    = np.interp(wavelengths, df_t["Wavelength"].values, df_t["T"].values)

        responses = {"R": QE_r * T, "G": QE_g * T, "B": QE_b * T}

    print(f"Camera response wavelength grid: {wl_min:.1f} - {wl_max:.1f} nm ({len(wavelengths)} points)")
    print(f"Camera type: {'monochrome' if mono_path else 'colour'} — channels: {list(responses.keys())}")

    return wavelengths, responses

def compute_fCR_combined(pg_csv_path: str, cam_wavelengths: np.ndarray,
                         responses: dict, wavelengths_DHG: np.ndarray,
                         omega: np.ndarray, g1: np.ndarray,
                         g2: np.ndarray, f: np.ndarray,
                         mu: float = 1.0, h: float = 50.0,
                         r_max: float = 3000.0, n_r: int = 300,
                         n_phi: int = 180):
    """
    Compute fCR for each camera channel and a combined fCR using the full
    DHG model (Eq. 6) for delta_ICR/I0, rather than the Lambertian approximation.

    Parameters
    ----------
    pg_csv_path     : path to pg(lambda) CSV file (not used in DHG model,
                      kept for consistency)
    cam_wavelengths : common wavelength grid in nm (from load_camera_response)
    responses       : dict of channel name -> response array
    wavelengths_DHG : wavelength grid for DHG parameters in nm
    omega, g1, g2, f: DHG parameter arrays on wavelengths_DHG grid
    mu              : cosine of emission angle
    h               : flash height above clouds in km
    r_max           : maximum radial integration limit in km
    n_r, n_phi      : radial and azimuthal integration resolution

    Returns
    -------
    fCR_combined : single combined correction factor
    """
    # Calculate delta_ratio with basic method
    # df = pd.read_csv(pg_csv_path, index_col=0, header=0, names=["Wavelength", "pg"])
    # pg_lambda   = np.interp(cam_wavelengths, df["Wavelength"].values, df["pg"].values)
    # delta_ratio = 3.0 * mu * pg_lambda

    # Compute delta_ICR/I0 at each wavelength in the DHG grid
    print("Computing delta_ICR/I0 over wavelength grid (this may take a while)...")
    delta_ratio = np.empty_like(wavelengths_DHG)
    for i in range(len(wavelengths_DHG)):
        delta_ratio[i] = delta_ICR_DHG(mu, omega[i], g1[i], g2[i], f[i],
                                            h=h, r_max=r_max, n_r=n_r, n_phi=n_phi)
        if i % 50 == 0:
            print(f"  Wavelength {wavelengths_DHG[i]:.1f} nm ({i}/{len(wavelengths_DHG)})")

    # Interpolate delta_ICR/I0 onto the camera wavelength grid
    delta_ratio_cam = np.interp(cam_wavelengths, wavelengths_DHG, delta_ratio)

    numerators   = {}
    denominators = {}

    print(f"\n--- fCR by channel (mu={mu}) ---")
    for channel, response in responses.items():
        num = np.trapezoid(response * delta_ratio, cam_wavelengths)
        den = np.trapezoid(response, cam_wavelengths)
        numerators[channel]   = num
        denominators[channel] = den
        print(f"  fCR_{channel} = {num/den:.4f}")

    fCR_combined = sum(numerators.values()) / sum(denominators.values())

    print(f"  fCR_combined = {fCR_combined:.4f}, h = {h} km")
    print(f"  (Previous constant assumption: fCR = 0.3)")

    return fCR_combined, delta_ratio

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Jupiter Impact Flash Cloud-Reflectivity Correction Model")
    print("Arimatsu et al. (2023), A&A, 677, A165")
    print("=" * 60)

    wavelengths_DHG, omega, g1, g2, f = load_DHG_parameters("omega.csv", "g1.csv", "g2.csv", "f.csv")
    demo_lambertian()
    demo_flux_correction()
    #demo_integrated_fCR("li2018_pg.csv", mu=0.866) # 30 degree emission angle
    plot_delta_ICR("li2018_pg.csv")
    # Test with every 10th wavelength point and coarser integration
    #plot_delta_ICR_DHG(wavelengths_DHG[::10], omega[::10], g1[::10], g2[::10], f[::10],
                   #n_r=100, n_phi=60)
    # Full resolution - will be very slow
    plot_delta_ICR_DHG(wavelengths_DHG, omega, g1, g2, f)

    # DHG single-wavelength demo is slow (nested numerical integrals).
    # Uncomment to run:
    demo_DHG_single(wavelengths_DHG, omega, g1, g2, f)

    #UNCOMMENT FOR COLOR CAMERA
    cam_wavelengths, responses = load_camera_response("transmission.csv", r_path="R.csv", g_path="V.csv", b_path="B.csv")
    #UNCOMMENT FOR MONOCHROME CAMERA
    #cam_wavelengths, responses = load_camera_response("transmission.csv", mono_path="qe_mono.csv")

    #fCR_R, fCR_G, fCR_B = demo_integrated_fCR("li2018_pg.csv", cam_wavelengths, R_response, G_response, B_response, mu=1.0)
    fCR_combined, delta_ratio_DHG = compute_fCR_combined("li2018_pg.csv", cam_wavelengths, 
                                                         responses, wavelengths_DHG, omega, 
                                                         g1, g2, f, mu=0.6468)
    fCR_combined, delta_ratio_DHG = compute_fCR_combined("li2018_pg.csv", cam_wavelengths, 
                                                         responses, wavelengths_DHG, omega, 
                                                         g1, g2, f, mu=0.6468, h=20.0)
    fCR_combined, delta_ratio_DHG = compute_fCR_combined("li2018_pg.csv", cam_wavelengths, 
                                                         responses, wavelengths_DHG, omega, 
                                                         g1, g2, f, mu=0.6468, h=30.0)
    fCR_combined, delta_ratio_DHG = compute_fCR_combined("li2018_pg.csv", cam_wavelengths, 
                                                         responses, wavelengths_DHG, omega, 
                                                         g1, g2, f, mu=0.6468, h=150.0)

    print("\nDone.")