import numpy as np

def compute_mu(impact_lat: float, impact_lon: float,
               subobs_lat: float, subobs_lon: float) -> float:
    """
    Compute the cosine of the emission angle (mu) for a Jupiter impact flash
    given the planetographic coordinates of the impact site and the
    sub-observer point.

    Parameters
    ----------
    impact_lat  : planetographic latitude of impact site in degrees
    impact_lon  : planetographic longitude of impact site in degrees
    subobs_lat  : planetographic latitude of sub-observer point in degrees
    subobs_lon  : planetographic longitude of sub-observer point in degrees

    Returns
    -------
    mu          : cosine of emission angle
    """
    # Convert to radians
    impact_lat_r  = np.radians(impact_lat)
    impact_lon_r  = np.radians(impact_lon)
    subobs_lat_r  = np.radians(subobs_lat)
    subobs_lon_r  = np.radians(subobs_lon)

    # Compute unit normal vectors to the surface at each point
    # assuming a spherical Jupiter (good approximation for this purpose)
    n_impact = np.array([
        np.cos(impact_lat_r) * np.cos(impact_lon_r),
        np.cos(impact_lat_r) * np.sin(impact_lon_r),
        np.sin(impact_lat_r)
    ])

    n_subobs = np.array([
        np.cos(subobs_lat_r) * np.cos(subobs_lon_r),
        np.cos(subobs_lat_r) * np.sin(subobs_lon_r),
        np.sin(subobs_lat_r)
    ])

    # mu = dot product of the two unit vectors (both already unit length)
    mu = np.dot(n_impact, n_subobs)
    mu = np.clip(mu, 0.0, 1.0)

    emission_angle = np.degrees(np.arccos(mu))

    print(f"\n--- Emission angle calculation ---")
    print(f"  Impact site   : lat={impact_lat:.2f}°, lon={impact_lon:.2f}°")
    print(f"  Sub-observer  : lat={subobs_lat:.2f}°, lon={subobs_lon:.2f}°")
    print(f"  Emission angle: {emission_angle:.2f}°")
    print(f"  mu            : {mu:.4f}")

    return mu

if __name__ == "__main__":
    mu = compute_mu(
    impact_lat  = 10.7,   # replace with your impact site latitude
    impact_lon  = 343.9,   # replace with your impact site longitude
    subobs_lat  = 3.7961,   # replace with Jupiter's sub-observer latitude
    subobs_lon  = 33.5696,    # replace with Jupiter's sub-observer longitude
    )
