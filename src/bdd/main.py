# coding: utf-8
from __future__ import print_function

import argparse
import logging as log
import sys

from bdd import compute_density, find_ellipses, find_merges, merge
from detect_peaks import detect_peaks
from kmeans import kmeans


def main():
    parser = argparse.ArgumentParser(
            description='Density Based [k-means] Bootstrap method demo')
    parser.add_argument('-d', '--dataset', default='dataset.txt',
                        help='Dataset to analyze')
    parser.add_argument('-k', '--kmeans_maxiter', default=20,
                        help='k-means max iterations')
    parser.add_argument('-b', '--dbb_maxiter', default=5,
                        help='DBB max iterations')
    parser.add_argument('-s', '--show', default=5,
                        help='Produces plots')
    args = parser.parse_args(sys.argv[1:])

    with open(args.dataset, 'rb') as dataset:
        points = [map(float, row.strip().split()) for row in dataset]

    # Maybe we shall wrap with `np.asarray`
    xs, ys = zip(*points)
    hx = compute_density(xs)
    hy = compute_density(ys)

    px = detect_peaks(hx[1])
    py = detect_peaks(hy[1])

    centroids = [(hx[0][x], hy[0][y]) for x in px for y in py]

    # TODO: plot initial status (cool figure)

    for i in xrange(args.dbb_maxiter):
        for (clusters, centroids, cstats) in kmeans(
                points, centroids=centroids,
                max_iteration=args.kmeans_maxiter, sbs=True):
            log.info(".", end="")

            ellipses = find_ellipses(centroids, clusters)
#            plot_density_ellipses(ellipses)
            merges = find_merges(ellipses)
            if not merges:
                break
            centroids = merge(cstats, merges)


if __name__ == '__main__':
    main()
