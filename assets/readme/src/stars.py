"""Azure Dragon (苍龙七宿) determinative stars, J2000 RA (h, m) / Dec (deg), projected stereographically."""
import numpy as np

STARS = {  # name: (RA h, RA m, Dec, magnitude)
    "alpVir": (13, 25.2, -11.16, 0.97), "zetVir": (13, 34.7, -0.60, 3.37),                       # 角 horn
    "kapVir": (14, 12.9, -10.27, 4.2), "iotVir": (14, 16.0, -6.00, 4.1), "phiVir": (14, 28.2, -2.23, 4.8),
    "lamVir": (14, 19.1, -13.37, 4.5),                                                             # 亢 neck
    "alpLib": (14, 50.9, -16.04, 2.75), "iotLib": (15, 12.2, -19.79, 4.5), "gamLib": (15, 35.5, -14.79, 3.9),
    "betLib": (15, 17.0, -9.38, 2.6),                                                              # 氐 root
    "piSco": (15, 58.9, -26.11, 2.9), "rhoSco": (15, 56.9, -29.21, 3.9), "delSco": (16, 0.3, -22.62, 2.3),
    "betSco": (16, 5.4, -19.81, 2.6),                                                              # 房 room
    "sigSco": (16, 21.2, -25.59, 2.9), "alpSco": (16, 29.4, -26.43, 1.0), "tauSco": (16, 35.9, -28.22, 2.8),  # 心 heart
    "epsSco": (16, 50.2, -34.29, 2.3), "muSco": (16, 51.9, -38.05, 3.0), "zetSco": (16, 54.6, -42.36, 3.6),
    "etaSco": (17, 12.2, -43.24, 3.3), "theSco": (17, 37.3, -43.00, 1.9), "iotSco": (17, 47.6, -40.13, 3.0),
    "kapSco": (17, 42.5, -39.03, 2.4), "lamSco": (17, 33.6, -37.10, 1.6), "upsSco": (17, 30.8, -37.30, 2.7),  # 尾 tail
    "gamSgr": (18, 5.8, -30.42, 3.0), "delSgr": (18, 21.0, -29.83, 2.7), "epsSgr": (18, 24.2, -34.38, 1.8),
    "etaSgr": (18, 17.6, -36.76, 3.1),                                                             # 箕 basket
}
MANSIONS = {  # Chinese name: asterism stick figure
    "角": ["alpVir", "zetVir"], "亢": ["lamVir", "kapVir", "iotVir", "phiVir"],
    "氐": ["alpLib", "iotLib", "gamLib", "betLib", "alpLib"], "房": ["rhoSco", "piSco", "delSco", "betSco"],
    "心": ["sigSco", "alpSco", "tauSco"],
    "尾": ["epsSco", "muSco", "zetSco", "etaSco", "theSco", "iotSco", "kapSco", "lamSco", "upsSco"],
    "箕": ["gamSgr", "delSgr", "epsSgr", "etaSgr", "gamSgr"],
}
RA0, DEC0 = 15.9 * 15, -22.0
ROT = np.radians(12)   # turn the chart clockwise so the dragon lies flatter


def _rot(x, y):
    return x * np.cos(ROT) + y * np.sin(ROT), -x * np.sin(ROT) + y * np.cos(ROT)


def project_radec(ra_h, dec_deg):
    """Stereographic projection about (RA0, DEC0) in degrees, east to the left as seen on the sky."""
    ra, dec = np.radians(ra_h * 15), np.radians(dec_deg)
    ra0, dec0 = np.radians(RA0), np.radians(DEC0)
    k = 2 / (1 + np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0))
    x = k * np.cos(dec) * np.sin(ra - ra0)
    y = k * (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0))
    return _rot(-np.degrees(x), np.degrees(y))


def project(name):
    h, m, dec, mag = STARS[name]
    return (*project_radec(h + m / 60, dec), mag)


def project_ecliptic(lam_deg, eps=23.4393):
    lam, e = np.radians(lam_deg), np.radians(eps)
    ra = np.degrees(np.arctan2(np.cos(e) * np.sin(lam), np.cos(lam))) % 360
    dec = np.degrees(np.arcsin(np.sin(e) * np.sin(lam)))
    return project_radec(ra / 15, dec)
