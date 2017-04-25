import sys


with open(sys.argv[1]) as dataset:
    for row in dataset:
        x, y = row.strip().split()
        x, y = int(x), int(y)

        if x > 400000:
            x -= 150000

        print '%d %d' % (x, y)
