# coding: utf-8
# Density Based [k-means] Bootstrap

from __future__ import print_function

import logging
from itertools import groupby
from math import fabs, sqrt
from operator import itemgetter as iget

import numpy as np
from ellipse import ellipse_intersect, ellipse_polyline
from matplotlib.mlab import normpdf
from scipy.stats import norm


# Magic numbers :: Here be dragons
SF = 5
STDW = .35
DENSW = .7


def compute_density(serie, nbuckets=None):
    nbuckets = nbuckets or int(sqrt(len(serie)))
    hist_y, bins = np.histogram(serie, nbuckets, density=True)
    # take mean as point instead of boundary
    hist_x = [(bins[i] + bins[i+1])/2 for i in xrange(0, len(bins)-1)]
    return (hist_x, hist_y, nbuckets)


def get_EIA(centroid):
    """ Compute the Expected Influence Area for the given centroid """
    (_, ((_, xstd, wx), (_, ystd, wy))) = centroid
    return (
        ((xstd * 2 * STDW) + (wx * DENSW)) * SF,
        ((ystd * 2 * STDW) + (wy * DENSW)) * SF,
    )


def find_ellipses(centroids, clusters):
    """
        Returns:
            [(centroid_id,
                ((x_mean, x_std, x_density_normalized),
                 (y_mean, y_std, y_density_normalized))
            )]
    """

    c_density = dict()
    dmx, dmy = list(), list()
    for (c, members) in groupby(sorted(clusters, key=iget(2)), iget(2)):
        xs, ys, _ = zip(*members)

        # ignore outliers
        if len(xs) == 1:
            continue

        # fitting data
        ((xmean, xvar), (ymean, yvar)) = (norm.fit(xs), norm.fit(ys))

        # compute density value (y) in mean point
        dmx.append(normpdf([xmean], xmean, xvar))
        dmy.append(normpdf([ymean], ymean, yvar))

        # Save clusters mean and std
        c_density[c] = ((xmean, xvar), (ymean, yvar))

    # Compute dataset mean and std in mean points
    xm = (np.nanmean(dmx), np.nanstd(dmx))
    ym = (np.nanmean(dmy), np.nanstd(dmy))

    # Inject normalized density
    return list((c, (
        (xmean, xvar, fabs(normpdf([xmean], xmean, xvar) - xm[0]) / xm[1]),
        (ymean, yvar, fabs(normpdf([ymean], ymean, yvar) - ym[0]) / ym[1])
    )) for (c, ((xmean, xvar), (ymean, yvar))) in c_density.iteritems())


def find_merges(ellipses):
    merges = list()
    for i in xrange(len(ellipses)):
        (ic, ((ixmean, _, _), (iymean, _, _))) = ellipses[i]
        iw, ih = get_EIA(ellipses[i], SF)
        ie = ellipse_polyline(ixmean, iymean, iw/2.0, ih/2.0)

        for j in xrange(i):
            (jc, ((jxmean, _, _), (jymean, _, _))) = ellipses[j]
            jw, jh = get_EIA(ellipses[j], SF)
            je = ellipse_polyline(jxmean, jymean, jw/2.0, jh/2.0)

            if ellipse_intersect(ie, je):
                merges.append((ic, jc,))
    return merges


def merge(cstats, merges):
    """
    Arguments:
        cstats: {c: (xsum, ysum, n)}
        merges: [(c1, c2)]
    Returns:
        centroids, cstats
    """

    logging.info("merges: ", merges)

    def find_current_group(c):
        while not cstats[c]:
            c = merged[c]
        return c

    merged = dict()
    # Apply merges
    for (c1, c2) in merges:
        c1 = find_current_group(c1)
        c2 = find_current_group(c2)

        # Already merged
        if c1 == c2:
            continue

        c1_xsum, c1_ysum, c1_n = cstats[c1]
        c2_xsum, c2_ysum, c2_n = cstats[c2]

        cstats[c1] = (c1_xsum + c2_xsum, c1_ysum + c2_ysum, c1_n + c2_n)
        merged[c2] = c1
        cstats[c2] = None

    # Recompute centroids
    return [(x/n, y/n) for (c, (x, y, n)) in
            filter(lambda (c, z): z is not None, cstats.iteritems()) if n > 0]
