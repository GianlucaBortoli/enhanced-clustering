# coding: utf-8
from __future__ import print_function

import logging
import math

import numpy as np


def generate_points(n, min_x=-5, min_y=-5, max_x=5, max_y=5):
    return [(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))
            for x in range(n)]


def dist(a, b):
    """ euclidean distance """
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def kmeans(dataset, k=None, centroids=None, e=0.01, max_iter=20, sbs=False):
    if not (k or centroids):
        raise Exception('k or centroids required')

    if centroids:
        k = len(centroids)
    else:
        minx = maxx = dataset[0][0]
        miny = maxy = dataset[0][1]
        for x, y in dataset[1:]:
            minx, maxx, miny, maxy = \
                    min(minx, x), max(maxx, x), min(miny, y), max(maxy, y)
        centroids = generate_points(k, minx, miny, maxx, maxy)

    # Add id to centroids
    centroids = dict(enumerate(centroids))
    # centroids: {cid -> (x,y)}

    for z in xrange(max_iter):
        logging.debug("[%d] Assigning points to centroids" % z)

        # points -> (x,y,cid)
        clusters = map(lambda (x, y): (x, y, min(
                map(lambda (ci, (cx, cy)): (ci, dist((cx, cy), (x, y))),
                    centroids.iteritems()),
                key=lambda (ci, distance): distance)[0]), dataset)

        logging.debug("[%d] Recomputing centroids" % z)
        cstats = dict([(cid, (0, 0, 0)) for cid in centroids.keys()])
        # reduce by key (sufficient statistics)
        for (x, y, c) in clusters:
            (cx, cy, n) = cstats.get(c)
            cstats[c] = (cx+x, cy+y, n+1)

        logging.debug("[%d] Updating centroids" % z)
        new_centroids = dict((c, (x/n, y/n))
                             for (c, (x, y, n)) in cstats.iteritems() if n > 0)

        logging.debug("[%d] Checking stop condition" % z)
        stop = all(map(lambda (c, (x, y)): dist(centroids.get(c), (x, y)) < e,
                   new_centroids.iteritems()))

        # Actually adopt new centroids
        centroids = new_centroids

        # If sbs (aka step-by-step) use generators to retrieve
        # clusters & centroids at each iteration
        if sbs:
            logging.debug("[%d] Step completed (sbs mode on)" % z)
            yield (clusters, centroids, cstats)
        if stop:
            break
    yield (clusters, centroids, cstats)
