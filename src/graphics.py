# coding: utf-8
from __future__ import print_function

import numpy.random as rnd

from dbb import get_EIA
from matplotlib import pyplot as plt, transforms
from matplotlib.patches import Ellipse
from pylab import figure
from scipy.interpolate import interp1d

FIGSIZE = (20, 14)
EXT = "png"


def extract_dname(dname):
    return dname.split('/')[-1].split('.')[0]


def plot_cool_figure(xs, ys, hx, hy, centroids, px, py, dname, picbound):
    # Whole figure
    plt.figure(figsize=FIGSIZE, frameon=False)

    # Extract boundaries
    minx, maxx, miny, maxy = picbound

    # Top bar
    x_density = plt.subplot2grid((4, 4), (0, 1), colspan=3)
    plot_density(minx, maxx, hx[0], hx[1], px, x_density)
    x_density.tick_params(axis='x', which='both',
                          bottom='off', top='on',
                          labelbottom='off', labeltop='on')
    x_density.tick_params(axis='y', which='both',
                          left='off', right='on',
                          labelleft='off', labelright='on')
    plt.grid(which='major', axis='x')

    # Left Bar
    y_density = plt.subplot2grid((4, 4), (1, 0), rowspan=3)
    plot_density(miny, maxy, hy[0], hy[1], py, y_density, rotation=90)
    y_density.tick_params(axis='x', which='both',
                          bottom='on', top='off',
                          labelbottom='on', labeltop='off')
    plt.xticks(rotation=90)
    plt.grid(which='major', axis='y')

    # Actual data
    data = plt.subplot2grid((4, 4), (1, 1), rowspan=3, colspan=3)
    data.scatter(xs, ys)
    data.scatter(*zip(*centroids))
    data.tick_params(axis='y', which='both',
                     left='off', right='on',
                     labelleft='off', labelright='on')
    data.set_ylim([miny, maxy])
    data.set_xlim([minx, maxx])
    plt.grid()

    plt.tight_layout()
    plt.savefig('img/%s_coolfig.%s' % (extract_dname(dname), EXT),
                transparent=True, bbox_inches='tight',  pad_inches=0)


def plot_density(mins, maxs, hist_x, hist_y, peaks, ax, rotation=0):
    # Rotation
    base = ax.transData
    rot = transforms.Affine2D().rotate_deg(rotation)

    # Density interpolation
    f = interp1d(hist_x, hist_y, kind=3, assume_sorted=False)
    ax.plot(hist_x, f(hist_x), 'g--', transform=rot + base)

    if rotation in [0, 180]:
        ax.set_xlim([mins, maxs])
    else:
        ax.set_ylim([mins, maxs])

    # peaks
    peaks_x, peaks_y = zip(*[(hist_x[z], hist_y[z]) for z in peaks])
    ax.plot(peaks_x, peaks_y, 'kD', transform=rot + base)


def plot_density_ellipses(xs, ys, ellipses, dname, i, picbound):
    fig = figure(figsize=FIGSIZE, frameon=False)
    ax = fig.add_subplot(111, aspect='equal')

    # The points
    ax.scatter(xs, ys)

    # The ellipses
    for (c, ((xmean, xstd, wx), (ymean, ystd, wy))) in ellipses:
        loc = (xmean, ymean)
        w, h = get_EIA((c, ((xmean, xstd, wx), (ymean, ystd, wy))))
        ellipse = Ellipse(xy=loc, width=w, height=h, color='black')
        ellipse.set_alpha(0.45)
        ellipse.set_facecolor(rnd.rand(3))
        ellipse.set_clip_box(ax.bbox)
        ax.add_patch(ellipse)
        ax.scatter(*loc, color='r')

    ax.set_ylim(picbound[2:])
    ax.set_xlim(picbound[:2])
    plt.grid()
    plt.savefig('img/%s_density_%d.%s' % (extract_dname(dname), i, EXT),
                transparent=True, bbox_inches='tight',  pad_inches=0)
