from astroquery.jplhorizons import Horizons
from astropy.time import Time

def get_jupiter_magnitude(obs_time_utc: str, observer_location: str = "500") -> dict:
    """
    Query JPL Horizons for Jupiter's apparent magnitude and other
    observing geometry at a given UTC time.

    Parameters
    ----------
    obs_time_utc : str
        Observation time in ISO format, e.g. '2023-11-15 12:41:17'.
        observer_location : MPC observatory code (https://www.minorplanetcenter.net/iau/lists/ObsCodes.html).  
                            '500' = geocentre (default).
                            For a custom location, pass a dict:
                            {"lon": 180.5, "lat": 20.7, "elevation": 4.2} where elevation is in km

    Returns
    -------
    dict with keys:
        V_jupiter       – apparent visual magnitude
        d_jupiter_au    – Jupiter–Sun distance [AU]
        delta_au        – Jupiter–observer distance [AU]
        phase_angle_deg – solar phase angle [degrees] (Sun-Jupiter-Observer)
        subobs_lat      – sub-observer latitude in degrees
        subobs_lon      – sub-observer longitude (System III) in degrees
        
    """
    t = Time(obs_time_utc, format="iso", scale="utc")

    obj = Horizons(
        id="599",                    # Jupiter barycentre
        location=observer_location,
        epochs=t.jd,
    )

    eph = obj.ephemerides(quantities="9,14,15,19,20,24")
    # quantities:
    #   9  = visual magnitude
    #  14  = sub-observer latitude
    #  15  = sub-observer longitude (System III)
    #  19  = heliocentric range (Jupiter–Sun distance)
    #  20  = heliocentric range rate
    #  24  = solar elongation / phase angle

    return {
        "V_jupiter":       float(eph["V"][0]),
        "subobs_lat":      float(eph["PDObsLat"][0]),
        "subobs_lon":      float(eph["PDObsLon"][0]),
        "d_jupiter_au":    float(eph["r"][0]),
        "delta_au":        float(eph["delta"][0]),
        "phase_angle_deg": float(eph["alpha"][0]),
    }


if __name__ == "__main__":
    #REDEFINE data and time (UTC) for different events
    #result = get_jupiter_magnitude("2023-11-15 12:41:17", {"lon": 139.6380, "lat": 35.4437, "elevation": 0.013})
    #result = get_jupiter_magnitude("2010-08-20 18:21:56", {"lon": 130.69, "lat": 32.81, "elevation": 9})
    #result = get_jupiter_magnitude("2010-06-03 20:31:20", {"lon": 149.0299, "lat": -34.9701, "elevation": 0.588})
    #result = get_jupiter_magnitude("2012-09-10 11:35:30", {"lon": -87.7828, "lat": 42.7261, "elevation": 0.190}) #{"lon": -96.8067, "lat": 32.7831, "elevation": 0.147})
    result = get_jupiter_magnitude("2021-10-15 13:24:13", {"lon": 135.7936, "lat": 34.9936, "elevation": 0.221})

    print(f"V_jupiter    = {result['V_jupiter']:.3f}")
    print(f"Sub-observer lat = {result['subobs_lat']:.4f}°")
    print(f"sub-observer lon = {result['subobs_lon']:.4f}°")
    print(f"d_jupiter_au = {result['d_jupiter_au']:.4f} AU")
    print(f"delta_au     = {result['delta_au']:.4f} AU")
    print(f"phase_angle  = {result['phase_angle_deg']:.2f}°")