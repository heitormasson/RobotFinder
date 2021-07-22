"""
Python PRM
@Author: Kristian Gibson
TODO: Comments + Sphynx Docs Structured Text
TODO: Bug-fix, testing

Not ready for use yet.
"""
from numpy import disp
from scipy import integrate
from spatialmath.base.animate import Animate
from spatialmath.base.transforms2d import *
from spatialmath.base.vectors import *
from spatialmath.pose2d import SE2
from spatialmath.base import animate
from scipy.ndimage import *
from matplotlib import cm
from roboticstoolbox.mobile.navigation import Navigation


class PRM(Navigation):
    def __init__(self, occ_grid=None, goal=np.array([]), inflate=0,
                 private=False, reset=False, verbose=None, seed=np.array([]),
                 transform=SE2(), n_points=100, dist_thresh=None):
        if dist_thresh is None:
            self._dist_thresh = 0.3 * self._occ_grid_nav

        self._n_points = n_points
        self._n_points0 = n_points
        self._dist_thresh = dist_thresh
        self._dist_thresh0 = dist_thresh
        self._graph = None
        self._v_goal = None
        self._v_start = None
        self._local_goal = None
        self._local_path = None
        self._v_path = None
        self._g_path = None

    @property
    def n_points(self):
        return self._n_points

    @property
    def n_points_0(self):
        return self._n_points0

    @property
    def dist_thresh(self):
        return self._dist_thresh

    @property
    def dist_thresh0(self):
        return self._dist_thresh0

    @property
    def graph(self):
        return self._graph

    @property
    def v_goal(self):
        return self._v_goal

    @property
    def v_start(self):
        return self._v_start

    @property
    def local_goal(self):
        return self._local_goal

    @property
    def local_path(self):
        return self._local_path

    @property
    def v_path(self):
        return self._v_path

    def plan(self, n_points=None, dist_thresh=None, animate=False, movie=np.array([])):
        self.message('create the graph')

        if n_points is None:
            n_points = self.npoints0
        if dist_thresh is None:
            dist_thresh = self.distthresh0;

        self._n_points = n_points
        self._dist_thresh = dist_thresh

        self._graph.clear()
        self._v_path = np.array([])
        self.create_roadmap(self, n_points, dist_thresh, movie)

    def query(self, start, goal):
        if self._graph.n == 0:
            Error('RTB:PRM:noplan:query: no plan: run the planner')

        self.check_query(self, start, goal)

        self._v_goal = self.closest(self.goal)
        if np.empty(self._v_goal):
            Error('RTB:PRM:nopath', 'plan: no path roadmap -> goal: rerun the planner')

        self._v_start = self.closest(self.start)
        if np.empty(self._v_start):
            Error('RTB:PRM:nopath', 'plan: no path start -> roadmap: rerun the planner')

        self._v_path = self.graph.Astar(self._v_start, self._v_goal)

        self._g_path = self._v_path
        self._g_path = self._g_path[1:len(self._g_path)]

        pp = np.transpose(np.array([self.start, self.graph.coord[self._v_path], self.goal]))
        return pp

    def closest(self, vertex, v_component):
        component = None
        if v_component is not None:
            component = self.graph.component[v_component]
        d, v = self.graph.distances(vertex)
        c = np.array([])

        for i in range(0, len(d)):
            if v_component is not None:
                if self._graph.component(v[i]) != component:
                    continue
            if not self.test_path(vertex, self._graph.coord(v(i))):
                continue
            c = v[i]
            break

        return c

    def next(self, p):
        if all(p[:] == self.goal):
            n = np.array([])
            return n

        if len(self._local_path) == 0:
            if np.empty(self._g_path):
                self._local_path = self.bresenham(p, self._goal)
                self._local_path = self._local_path[1:len(self._local_path), :]
                self._local_goal = np.array([])
            else:
                self._local_goal = self._g_path[0]
                self._g_path = self._g_path[1:len(self._g_path)]

                self._local_path = bresenham(p, self.graph.coord(self._local_goal))
                self._local_path = self._local_path[1:len(self._local_path), :]
                self.graph.highlight_node(self._local_goal)

        n = np.transpose(self._local_path[0, :])
        self._local_pathh = self._local_path[1:len(self._local_path), :]
        return n

    def char(self):
        s = "\ngraph size: " + self._n_points
        s = s + "\nndist thresh: " + self._dist_thresh
        s = s + "\nGraph: " + self.graph
        return s

    def create_roadmap(self, movie=np.array([]), animate=False):
        a = Animate(movie, 'fps', 5)
        x = None
        y = None
        for j in range(0, self._n_points):
            while True:
                x = self.randi(len(self._occ_grid[0]))
                y = self.randi(len(self._occ_grid))
                if not self.is_occupied(np.array([x, y])):
                    break
            new = np.array([[x], [y]])

            vnew = self.graph.add_node(new)

            [d,v] = self.graph.distances(new)

            for i in range(1, len(d)):
                if d(i) > self._dist_thresh:
                    continue
                if not self.test_path(new, self.graph.coord(v[i])):
                    continue

                self.graph.add_edge(v(i), vnew)

            if animate or not np.empty(movie):
                self.plot()
                if not np.empty(movie):
                    a.add()

    def test_path(self, p1, p2):
        p = bresenham(p1, p2)
        c = None
        for pp in np.transpose(p):
            if self.isoccupied(pp):
                c = False
                return c
        c = True
        return c


# Implementation fo Peter Corke's matlab bresenham function from:
# https://github.com/petercorke/toolbox-common-matlab/blob/master/bresenham.m
def bresenham(x1=None, y1=None, x2=None, y2=None):
    p1 = None
    p2 = None

    if x2 is None and y2 is None:
        p1 = x1
        p2 = y1
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]

    elif x1 is None and y1 is None:
        Error('expecting 2 or 4 arguments')

    x1 = np.round(x1)
    x2 = np.round(x2)
    y1 = np.round(y1)
    y2 = np.round(y2)

    dx = np.abs(x2 - x1)
    dy = np.abs(y2 - y1)
    steep = np.abs(dy) > np.abs(dx)

    t = None
    if steep:
        t = dx
        dx = dy
        dy = t

    q = None
    if dy == 0:
        q = np.zeros(dx + 1, 1)
    else:
        q = np.array([
            [0],
            [np.diff(np.mod(np.transpose(np.arrange(np.floor(dx / 2),
                                                    -dy * dx + np.floor(dx / 2),
                                                    -dy)), dx)) >= 0]]
                     )

    if steep:
        if y1 <= y2:
            y = np.transpose(np.arrange([y1, y2]))
        else:
            y = np.transpose(np.arrange(y1, y2, -1))
        if x1 <= x2:
            x = x1 + np.cumsum(q)
        else:
            x = x1 - np.cumsum(q)
    else:
        if x1 <= x2:
            x = np.transpose(np.arrange(x1, x2))
        else:
            x = np.transpose(np.arrange(x1, x2, -1))
        if y1 <= y2:
            y = y1 + np.cumsum(q)
        else:
            y = y1 - np.cumsum(q)

    p = np.array[x, y]

    return p


# Implementation of Peter Corke's matlab homline function from:
# https://github.com/petercorke/spatialmath-matlab/blob/master/homline.m
def hom_line(x1, y1, x2, y2):
    line = np.cross(np.array([x1, y1, 1]), np.array([x2, y2, 1]))

    # normalize so that the result of x*l' is the pixel distance
    # from the line
    line = line / np.linalg.norm(line[0:2])
    return line


# Sourced from: https://stackoverflow.com/questions/28995146/matlab-ind2sub-equivalent-in-python/28995315#28995315
def sub2ind(array_shape, rows, cols):
    ind = rows * array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0] * array_shape[1]] = -1
    return ind


def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0] * array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return rows, cols


def col_norm(x):
    y = np.array([])
    if x.ndim > 1:
        x = np.column_stack(x)
        for vector in x:
            y = np.append(y, np.linalg.norm(vector))
    else:
        y = np.linalg.norm(x)
    return y


class Error(Exception):
    """Base class for other exceptions"""
    pass
