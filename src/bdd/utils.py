# coding: utf-8
from __future__ import print_function

import matplotlib


COLOR_SET = matplotlib.colors.cnames.values()


def color(c):
    return colors[c%len(colors)]


def plot2d(data, *args, **kwargs):
    x = np.asarray([z[0] for z in data])
    y = np.asarray([z[1] for z in data])
    plt.plot(x, y,  ls='', *args, **kwargs)

"""
        # Plot the situation
        if show:
            print("[%d] Plotting" % z)
            plt.figure(figsize=(12,5))
            for q in xrange(k):
                plot2d(filter(lambda (x,y,c): c == q, clusters), color(q), marker='o')
            for (ci,point) in centroids.iteritems():
                plot2d([point], 'g', marker='^')
            plt.title("Iteration {}".format(z))
            plt.grid()
            plt.show()
"""

