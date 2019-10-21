from collections import namedtuple
from enum import Enum
import numpy as np
import os
from scipy import spatial
import time

import context
import tinyland


FILE_FMT = "images/%s.%s"


def find_image_file(element):
    for ext in ['jpg', 'png']:
        if os.path.exists(FILE_FMT % (element, ext)):
            return FILE_FMT % (element, ext)

PADDING = 40
LIB_ICON_SIZE = 50

ALCHEMY_RAD = 350
ALCHEMY_IMG = FILE_FMT % ("PENT", "png")

VOID_SIZE = 150
VOID_COLOR1 = (0, 0, 153)
VOID_COLOR2 = context.BLACK

CONTEXT_WIDTH = 1366 # This stuff should be on the context object,
CONTEXT_HEIGHT = 768
MIDPOINT = (CONTEXT_WIDTH / 2, CONTEXT_HEIGHT / 2)
VOID_CENTER = (CONTEXT_WIDTH / 2 + 500, CONTEXT_HEIGHT / 2)

DIST_THRESHOLD = 200


class Element(Enum):
    VOID = 0
    WATER = 393
    FIRE = 666
    AIR = 19
    EARTH = 400
    ALCOHOL = 930
    PRESSURE = 1000
    SEA = 1001
    MUD = 1002
    ENERGY = 1003
    LAVA = 1004
    RAIN = 1005
    DUST = 1006


RECIPES = {
    (Element.FIRE, Element.WATER): Element.ALCOHOL,
    (Element.AIR, Element.FIRE): Element.ENERGY,
    (Element.EARTH, Element.FIRE): Element.LAVA,
    (Element.AIR, Element.AIR): Element.PRESSURE,
    (Element.AIR, Element.EARTH): Element.DUST,
    (Element.AIR, Element.WATER): Element.RAIN,
    (Element.WATER, Element.WATER): Element.SEA,
    (Element.EARTH, Element.WATER): Element.MUD,
    (Element.EARTH, Element.EARTH): Element.PRESSURE,
}


class Vessel:

    def __init__(self, coord, aruco_id, charge, element):
        self.coord = coord  # (x, y) coordinate pair tuple
        self.aruco_id = aruco_id  # ID of corresponding Aruco marker
        self.charge = charge  # Percentage 0=VOID 1=READY
        self.element = element  # Element enum


class Animator:

    def __init__(self):
        self.state1 = 0
        self.state2 = 1
        self.animation_stop = 0

    def animate(self, dur):
        self.animation_stop = time.time() + dur

    @property
    def state(self):
        if time.time() < self.animation_stop:
            self.state1, self.state2 = self.state2, self.state1
            return self.state1
        else:
            return 0


def get_library_kdtree():
    return spatial.KDTree([tup[1] for tup in list(library.items())])


def add_element(element, x, y, l):
    l.append([(x, y), element])


def update_element(index, x, y, l):
    l[index][0] = (x, y)


def nearest_neighbor(x, y, l, k=1):
    tree = spatial.KDTree(l)
    dist, data_index = tree.query((x, y), k)
    return dist, data_index


def dist(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))


image_files = {e: find_image_file(e.name) for e in Element}
workbench = {}
void_animator = Animator()
table_animator = Animator()
HALF = int(LIB_ICON_SIZE / 2)
next_lib_coord = (PADDING + HALF, PADDING)
library = {}
for e in [Element.FIRE, Element.AIR, Element.WATER, Element.EARTH]:
    library[e] = next_lib_coord
    next_lib_coord = (next_lib_coord[0] + LIB_ICON_SIZE + PADDING, next_lib_coord[1])
library_kdtree = get_library_kdtree()


def update_vessel(marker):
    global library_kdtree
    global next_lib_coord
    global table_animator

    x, y = marker.center.x, marker.center.y
    if marker.id not in workbench:
        workbench[marker.id] = Vessel((x, y), marker.id, 0, Element.VOID)
    v = workbench[marker.id]
    v.coord = (x, y)

    # Library instantiations
    library_list = list(library.items())
    library_dist, library_index = library_kdtree.query(
        v.coord,
        distance_upper_bound=DIST_THRESHOLD)
    if library_dist < np.inf and v.element in [Element.VOID, library_list[library_index][0]]:
        element = library_list[library_index][0]
        v.element = element
        v.charge = min(v.charge + 0.1, 1.0)
    else:
        if v.charge < .999:
            v.charge = max(v.charge - 0.1, 0.0)
        if v.charge < .1:
            v.element = Element.VOID

    # Workbench combinations
    workbench_list = list(workbench.items())
    workbench_kdtree = spatial.KDTree([t[1].coord for t in workbench_list])
    nn = workbench_kdtree.query(
        v.coord,
        k=2,
        distance_upper_bound=DIST_THRESHOLD
    )
    if nn[0][1] != np.inf:
        workbench_dist, workbench_index = nn[0][1], nn[1][1]
        neighbor = workbench_list[workbench_index][1]
        candidate = tuple(sorted([v.element, neighbor.element], key=lambda e: e.name))
        if (candidate in RECIPES and
                v.charge > .999 and neighbor.charge > .999 and
                dist(v.coord, MIDPOINT) < ALCHEMY_RAD and
                dist(neighbor.coord, MIDPOINT) < ALCHEMY_RAD):
            table_animator.animate(1.5)
            new_element = RECIPES[candidate]
            v.element = new_element
            neighbor.element = Element.VOID
            neighbor.charge = 0

            if new_element not in library:
                library[new_element] = next_lib_coord
                library_kdtree = get_library_kdtree()
                next_lib_coord = (next_lib_coord[0] + LIB_ICON_SIZE + PADDING, next_lib_coord[1])
    if dist(v.coord, VOID_CENTER) < VOID_SIZE and v.element != Element.VOID:
        void_animator.animate(1.5)
        v.element = Element.VOID
        v.charge = 0.0


def render_vessels(ctx):
    for aruco_id, v in workbench.items():
        r = 40
        diff = np.array(v.coord) - [CONTEXT_WIDTH / 2, CONTEXT_HEIGHT / 2]
        offset = diff / np.linalg.norm(diff) * [r + 40, r + 40]
        icon_coord = tuple(np.array(v.coord) - offset)
        if 0.999 > v.charge > 0.0001:
            ctx.circle(*icon_coord, r, (50, 50, 50))
            ctx.circle(*icon_coord, r*v.charge, context.CYAN)
            ctx.circle(*icon_coord, max(0, r*v.charge - 7), (50, 50, 50))
        if v.element != Element.VOID:
            ctx.image(image_files[v.element], *icon_coord, 40, 40)


def app(snap, ctx):

    # Render alchemy circle
    ctx.circle(*MIDPOINT, ALCHEMY_RAD, context.MAGENTA)
    ctx.circle(*MIDPOINT, ALCHEMY_RAD - 30, context.BLACK)

    if table_animator.state:
        ctx.image(ALCHEMY_IMG, *MIDPOINT, 600, 600)

    # Render void area
    col = VOID_COLOR1
    if void_animator.state:
        col = VOID_COLOR2
    ctx.rect(*VOID_CENTER, 30, VOID_SIZE, 45, col)
    ctx.rect(*VOID_CENTER, 30, VOID_SIZE, -45, col)

    # Rendering new elements by marker
    for aruco_id, markers in snap.markers.items():
        for marker in markers:
            update_vessel(marker)

    render_vessels(ctx)

    for element, coord in library.items():
        ctx.image(image_files[element], *coord, 50, 50)



if __name__ == "__main__":
    tinyland.run(app)

