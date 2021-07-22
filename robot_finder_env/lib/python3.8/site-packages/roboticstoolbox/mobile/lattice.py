"""
Python lattice Planner
@Author: Kristian Gibson
TODO: Comments + Sphynx Docs Structured Text
TODO: Bug-fix, testing

Not ready for use yet.
"""
from numpy import disp
from scipy import integrate
from spatialmath.base.transforms2d import *
from spatialmath.base.vectors import *
from spatialmath.pose2d import SE2
from spatialmath.base import *
from spatialmath.base.animate import *
from spatialmath.base.vectors import angdiff
from scipy.ndimage import *
from matplotlib import cm
from roboticstoolbox.mobile.navigation import Navigation


class Lattice(Navigation):
    def __init__(self, occ_grid=None, goal=np.array([]), inflate=0,
                 private=False, reset=False, verbose=None, seed=np.array([]),
                 transform=SE2(), grid=1, root=np.transpose(np.array([0, 0, 0])), iterations=np.Inf,
                 cost=np.array([1, 1, 1])):
        self._iterations = iterations
        self._cost = cost
        self._graph = None
        self._v_goal = None
        self._v_start = None
        self._local_path = None
        self._v_path = None
        self._grid = None  # TODO: Implement PGraph
        self._root = None

    def iterations(self):
        return self._iterations

    def cost(self):
        return self._cost

    def graph(self):
        return self._graph

    def v_goal(self):
        return self._v_goal

    def v_start(self):
        return self._v_start

    def local_path(self):
        return self._local_path

    def v_path(self):
        return self._v_path

    def grid(self):
        return self._grid

    def root(self):
        return self._root

    def plan(self, iterations=None, cost=None):
        iterations = self._iterations
        cost = self._cost

        if np.empty(self.occ_grid_nav) and np.isinf(iterations):
            Error('RTB:Lattice:badarg', 'If no occupancy grid given then iterations must be finite')

        self._iterations = iterations
        self._cost = cost

        if np.empty(self._root):
            Error('no root node specified')

        if self._root == 2:
            self._root = np.array([[self._root[:]], [0]])
        elif self._root == 3:
            self._root = self._root[:]
        else:
            Error('root must be 2- or 3-vector')

        if self.is_occupied(self._root[0:1]):
            Error('root node cell is occupied')

        self.message('create the graph')
        self._graph.clear()
        self.create_lattice(self)
        disp(self._graph.n + ' nodes created\n')

    # Implementing abstract method to make compiler happy.
    def next(self):
        pass

    def query(self, start=None, goal=None):
        if goal is None:
            Error('must specify start and goal')

        self._goal = goal
        self._start = start

        start[2] = round(start[2]*2/np.pi)
        goal[2] = round(goal[2]*2/np.pi)

        self._v_start = self._graph.closest(start, 0.5)
        self._v_goal = self._graph.closest(goal, 0.5)

        if np.empty(self._v_start):
            Error('Lattice:badarg', 'start configuration not in lattice')

        if np.empty(self._v_goal):
            Error('Lattice:badarg', 'goal configuration not in lattice')

        self._v_path, cost = self._graph.Astar(self._v_start,  self._v_goal, 'directed')
        disp('A* path cost ' + cost + "\n")

        p = self._graph.coord(self._v_path)

        pp = np.transpose(p)
        pp[:,2] = angdiff( pp[:,2] * np.pi/2 )

        return pp

    def char(self):
        s = "" + self
        s = s + "\n  grid spacing: " + self._grid
        s = s + "\n  costs [%d,%d,%d]" + self._cost
        s = s + "\n  iterations %d" + self._iterations
        s = s + "\n Graph:" + self._graph

        return s

    def plot(self, goal=None, no_overlay=False):
        if not no_overlay:
            self.showlattice(goal)

        if not np.empty(self._v_path):
            self.highlight(goal)

    def create_lattice(self):
        root = self._graph.add_node(self._root )
        d = self._grid

        destinations = np.array([
            d,  d,  d,
            0, d, -d,
            0,  1,  3,
            ])

        iteration = 1
        while iteration <= self._iterations:
            additions = 0

            for node in np.argwhere(self._graph.connectivity_out == 0):
                pose = self._graph.coord(node)
                xys = pose[0:2]
                heading = pose(2)

                xy = np.add(xys, homtrans(rot2(heading*np.pi/2), destinations[0:2,:]))
                theta = np.mod(heading+destinations[2,:], 4)
                new_destinations = np.array([[xy], [theta]])

                for i in range(1, len(destinations[0])):
                    v = self._graph.closest(new_destinations[:,i], 0.5)
                    if np.empty(v):
                        if not self.is_occupied(new_destinations[0:2,i]):
                            nv = self._graph.add_node( new_destinations[:, i], node,  self._cost[i])
                            self._graph.add_edge(node, nv,  self._cost[i])
                            additions = additions + 1
                    else:
                        self._graph.add_edge(node, v,  self._cost[i])
                        additions = additions + 1

            iteration = iteration + 1
            if additions == 0:
                break

    def showlattice(self,):
        p = self._graph.coord()
        th = p[3,:]
        th[th == 3] = -1

        # TODO: Add plotting

        for e in range(1, self._graph.ne):
            v = self._graph.vertices(e)
            # draw arc

    def highlight(self, p=None):
        vpath = None
        if p is not None:
            assert(len(p[0])==3, 'path must have 3 columns')
            for i in range(0, len(p)):
                vpath[i] = self._graph.closest(p[i,:])
        else:
            vpath = self._v_path

        v1 = None
        v2 = None
        for k in range(1, len(vpath)-1):
            v1 = vpath(k)
            v2 = vpath(k+1)
            # draw arc

    def drawarc(self, v):
        g = self._graph
        narc = None
        if  self._iterations < 4:
            narc = 20
        elif  self._iterations < 10:
            narc = 10
        else:
            narc = 5


        v1 = v[0]
        v2 = v(2)
        p1 = g.coord(v1)
        p2 = g.coord(v2)

        theta = p1[2]*np.pi/2
        t_0n = SE2(p1[0:2], theta)

        dest = np.round(t_0n.inv * p2[0:2])
        if dest[1] == 0:
            th = np.array([p1[2], p2[2]])
            th[th == 3] = -1
            # plot3
        else:
            c = t_0n * np.transpose([0, dest[1]])

            th = (np.linspace(-dest(2)/ self._grid, 0, narc) + p1(3) )*np.pi/2

            x = self._grid*np.cos(th) + c(1)
            y = self._grid*np.sin(th) + c(2)

            th0 = p1[2]
            th0[th0 == 3] = -1
            thf = p2[2]
            thf[thf == 3] = -1
            # plot3

# Sourced from: https://stackoverflow.com/questions/28995146/matlab-ind2sub-equivalent-in-python/28995315#28995315
def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind


def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return rows, cols


class Error(Exception):
    """Base class for other exceptions"""
    pass