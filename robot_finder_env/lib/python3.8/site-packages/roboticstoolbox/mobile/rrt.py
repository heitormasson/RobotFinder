"""
Python RRT
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
from scipy.ndimage import *
from matplotlib import cm
from roboticstoolbox.mobile.navigation import Navigation


class RRT(Navigation):
    def __init__(self, occ_grid=None, goal=np.array([]), inflate=0,
                 private=False, reset=False, verbose=None, seed=np.array([]),
                 transform=SE2(), n_points=500, sim_time=0.5, speed=1, root=np.array([0, 0, 0]),
                 rev_cost=1, range=None):
        self._rand = None
        self._graph = None
        self.vehicle = None
        self._n_points = n_points
        self._sim_time = sim_time
        self._speed = speed
        self._root = root
        self._rev_cost = rev_cost
        self._range = range

        if np.empty(self._occ_grid):
            opt = []
            opt.range = 5

            if range is not None:
                if range == 1:
                    self._x_range = np.array([-range, range])
                    self._y_range = np.array([-range, range])
                elif range == 2:
                    self._x_range = np.array([-opt.range[0], opt.range[0]])
                    self._y_range = np.array([-opt.range[1], opt.range[1]])
                elif range ==3:
                    self._x_range = np.array([opt.range[0], opt.range[1]])
                    self._y_range = np.array([opt.range[2], opt.range[3]])
                else:
                    Error('bad range specified')
        else:
            self._x_range = [0, len(self._occ_grid[0])]
            self._y_range = [0, len(self._occ_grid)]

        # TODO: Add plots

    def plan(self, progress=True, samples=False, goal=np.array([]), ntrials=50, data=None):
        if not np.empty(goal):
            self._goal = goal

        self.message('create the graph')
        self._graph.clear()

        if np.empty(self._root):
            Error('no root node specified')

        if not isvector(self._root, 3):
            Error('root must be 3-vector')

        assert(not self.isoccunp.pied(self._root[1:2]), 'root node cell is occunp.pied')

        v_root = self._graph.add_node(self._root)
        data.vel = 0
        data.path = []
        self._graph.setvdata(v_root, data)

        if progress:
            h = Navigation.progress_init('RRT planning...')

        n_points = 0
        while n_points < self._n_points:
            while True:
                xy = self.rand_xy()

                if np.empty(self._occ_grid):
                    break
                else:
                    xy = round( xy )
                    try:
                        if not self.isoccunp.pied(xy):
                            break
                    except:
                        continue

            theta = self._rand*2*np.pi
            xrand = np.transpose(np.array([xy, theta]))
            if samples:
                # TODO: Add plots
                pass

            vnear = self._graph.closest(xrand)
            xnear = self._graph.coord(vnear)

            self.message('xrand (%g, %g) node %d', xy, vnear)

            best = self.best_path(xnear, xrand, ntrials)

            xnew = best.path[:, best.k]
            if samples:
                # TODO: Add plots
                pass


            vnew = self._graph.add_node(xnew)


            if self._graph.vdata(vnear).vel * best.vel < 0:
                cost = self.revcost
            else:
                cost = 1

            self._graph.add_edge(vnear, vnew, cost)

            self._graph.setvdata(vnew, best)

            n_points = n_points + 1
            if progress:
                self.progress(h, n_points / self._n_points)

        if progress:
            self.progress_delete(h)

        self.message('graph create done')

    def query(self, x_start, x_goal):
        assert(self._graph.n > 0, 'RTB:RRT: there is no plan')
        self.checkquery(x_start, x_goal)

        g = self._graph
        vstart = g.closest(x_start)
        vgoal = g.closest(x_goal)

        [path,cost] = g.Astar(vstart, vgoal)

        disp('A* path cost %g' + cost + '\n')

        cpath = []
        p = None
        for i in range(1, len(path)):
            p = path[i]
            data = g.vdata(p)
            if not np.empty(data):
                if i >= len(path) or g.edgedir(p, path(i+1)) > 0:
                    cpath = np.array([cpath, data.path])
                else:
                    cpath = np.array([cpath, data.path[:, len(data.path):-1:1]])

        if p is None:
            # TODO: Plotting
            for i in range(0, len(path)):
                p = path(i)
                b = g.vdata(p)
                if not np.empty(b):
                    if i >= len(path) or g.edgedir(p, path(i+1)) > 0:
                        seg = np.array([g.coord(path[i-1]), b.path])
                    else:
                        seg = np.transpose(np.array([b.path[:,len(b.path):-1:1]. g.coord[path(i+1)]]))
        else:
            p_ = np.transpose(cpath)
            return p_  # Only return if p cas a value

    # "Implemented" to make the compiler happy
    def plot(self):
        pass

    # "Implemented" to make the compiler happy
    def next(self):
        pass

    def char(self):
        s = "" + str(self)
        s = s + "\nregion: X: " +  str(self._x_range) + " Y:" + str(self._y_range)
        s = s + "\nsim time: " + str(self._sim_time)
        s = s + "\nspeed: " + str(self._speed)
        s = s + "\nGraph:"
        s = s + "\n" + str(self._graph)
        if not np.empty(self.vehicle):
            s = s + "\n Vehicle: " + self.vehicle

        return s

    def best_path(self, x0=None, xg=None, N=None):

        x0 = x0[:]
        xg = xg[:]

        best = None
        best.d = np.Inf
        for i in range(0, N):
            if rand > 0.5:
                vel = self._speed
            else:
                vel = -self._speed

            steer = (2*self._rand - 1) * self.vehicle.steermax

            x = np.transpose(self.vehicle.run2(self._sim_time, x0, vel, steer))

            d = col_norm( np.array([[np.minus(x[0:2, :], xg[0:2])], np.array[angdiff(x[2,:], xg[2])]]) )
            dmin, k = np.minimum(d)

            if dmin < best.d:
                best.d = dmin
                best.path = x
                best.steer = steer
                best.vel = vel
                best.k = k
        return best

    def rand_xy(self):
        xy = np.multiply(self._rand(1,2),
                         np.array([self.xrange(2)-self.xrange(1), self.yrange(2)-self.yrange(1)]) +
                         np.array(self.xrange(1), self.yrange(1)))
        return xy


    def clear_path(self, xy):
        if np.empty(self._occ_grid):
            c = True
            return c

        xy = np.round(xy)

        try:
            for pp in np.transpose(xy):
                if self.isoccunp.pied(pp) > 0:
                    c = False
                    return c
            c = True
        except:
            c = False
            return c


# Sourced from: https://stackoverflow.com/questions/28995146/matlab-ind2sub-equivalent-in-python/28995315#28995315
def sub2ind(array_shape, rows, cols):
    ind = rows * array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0] * array_shape[1]] = -1
    return ind


def col_norm(x):
    y = np.array([])
    if x.ndim > 1:
        x = np.column_stack(x)
        for vector in x:
            y = np.append(y, np.linalg.norm(vector))
    else:
        y = np.linalg.norm(x)
    return y


def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0] * array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return rows, cols


class Error(Exception):
    """Base class for other exceptions"""
    pass