# coding: utf-8
from __future__ import print_function

from dbb import get_EIA
from matplotlib import pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Ellipse
from pylab import figure
from scipy.interpolate import interp1d

FIGSIZE = (20, 14)


def cool_figure(xs, ys, hx, hy, centroids):
    # Whole figure
    plt.figure(figsize=FIGSIZE)

    # Compute one for all
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    # Top bar
    x_density = plt.subplot2grid((4, 4), (0, 1), colspan=3)
    plot_density(minx, maxx, hx[0], hx[1], x_density)
    x_density.tick_params(axis='x', which='both',
                          bottom='off', top='on',
                          labelbottom='off', labeltop='on')
    x_density.tick_params(axis='y', which='both',
                          left='off', right='on',
                          labelleft='off', labelright='on')
    plt.grid(which='major', axis='x')

    # Left Bar
    y_density = plt.subplot2grid((4, 4), (1, 0), rowspan=3)
    plot_density(miny, maxy, y_density, rotation=90)
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
    plt.show()


def plot_density(mins, maxs, hist_x, hist_y, ax, rotation=0):
    # Rotation
    base = ax.transData
    rot = transforms.Affine2D().rotate_deg(rotation)

    ax.plot(hist_x, hist_y, '*', transform=rot + base)

    # Density interpolation
    f = interp1d(hist_x, hist_y, kind=3, assume_sorted=False)
    ax.plot(hist_x, f(hist_x), 'g--', transform=rot + base)

    if rotation in [0, 180]:
        ax.set_xlim([mins, maxs])
    else:
        ax.set_ylim([mins, maxs])


def plot_density_ellipses(xs, ys, ellipses):
    fig = figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, aspect='equal')

    # The points
    ax.scatter(xs, ys)

    # The ellipsis
    for (c, ((xmean, xvar, wx), (ymean, yvar, wy))) in ellipses:
        loc = (xmean, ymean)
        w, h = get_EIA((c, ((xmean, xvar, wx), (ymean, yvar, wy))))
        ellipse = Ellipse(xy=loc, width=w, height=h, color='black')
        ellipse.set_facecolor('none')
        ellipse.set_clip_box(ax.bbox)
        ax.add_patch(ellipse)
        ax.scatter(*loc, color='r')
    plt.show()
