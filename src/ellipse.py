# coding: utf-8
# Original author: @HYRY
# https://stackoverflow.com/questions/15445546/finding-intersection-points-of-two-ellipses-python#15446492

import numpy as np
from shapely.geometry.polygon import LinearRing


def ellipse_polyline(x0, y0, a, b, angle=0, n=100):
    """
        a: horizzontal semi-axis (aka hradius)
        b: verital semi-axis (aka vradius)
    """
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    st = np.sin(t)
    ct = np.cos(t)

    angle = np.deg2rad(angle)
    sa = np.sin(angle)
    ca = np.cos(angle)
    p = np.empty((n, 2))
    p[:, 0] = x0 + a * ca * ct - b * sa * st
    p[:, 1] = y0 + a * sa * ct + b * ca * st
    return p


def ellipse_intersect(a, b, ret_points=False):
    ea = LinearRing(a)
    eb = LinearRing(b)
    mp = ea.intersection(eb)

    if ret_points:
        x = [p.x for p in mp]
        y = [p.y for p in mp]
        return x, y
    return bool(mp)
