"""
Python EKF Planner
@Author: Kristian Gibson
TODO: Comments + Sphynx Docs Structured Text
TODO: Bug-fix, testing, change remaining matlab placeholders
TODO: Replace vargin with parameters

Not ready for use yet.
"""

import numpy as np
from math import pi
from scipy import integrate, randn
from scipy.linalg import sqrtm
from spatialmath.base.animate import Animate
from spatialmath import base


class EKF:
    def __init__(self, robot, v_est, p0, x_est=None, p_est=None, landmarks=None, w_est=None, sensor=None,
                 est_vehicle=None, est_ekf_map=None, joseph=True, verbose=False, keep_history=False,
                 ekf_map=None, history=True, dim=None):
        self._x_est = x_est           #  estimated state
        self._P_est = p_est           #  estimated covariance
        self._landmarks = landmarks           #  ekf_map state
        self._v_est = v_est           #  estimate of covariance V
        self._w_est = w_est           #  estimate of covariance W
        self._robot = robot           #  reference to the robot vehicle
        self._sensor = sensor          #  reference to the sensor
        self._est_vehicle = est_vehicle      #  flag: estimating vehicle location
        self._est_ekf_map = est_ekf_map          #  flag: estimating ekf_map
        self._joseph = joseph          #  flag: use Joseph form to compute p
        self._verbose = verbose
        self._keep_history = keep_history     #  keep history
        self._p0 = p0              #  passed initial covariance
        self._ekf_map = ekf_map             #  passed ekf_map
        self._history = history
        self._dim = dim          #  robot workspace dimensions

        self._robot.init()

        #  clear the history
        self._history = []

        if not np.any(self._v_est):
            #  perfect vehicle case
            self._est_vehicle = False
            self._x_est = []
            self._p_est = []
        else:
            #  noisy odometry case
            self._x_est = self._robot.x   #  column vector
            self._p_est = self._p0
            self._est_vehicle = True

        if np.any(self._sensor):
            self._landmarks = None*np.zeros(2, self._sensor.ekf_map.nlandmarks)

        #  check types for passed objects
        if np.any(self._map) and not isinstance(self._map, 'LandmarkMap'):
            Error('RTB:EKF:badarg\nexpecting LandmarkMap object')

        if np.any(sensor) and not isinstance(sensor, 'Sensor'):
            Error('RTB:EKF:badarg\nexpecting Sensor object')

        if not isinstance(robot, "Vehicle"):
            Error('RTB:EKF:badarg\nexpecting Vehicle object')

        self._init()

    
    @property
    def x_est(self):
        return self._x_est
    
    @property
    def p_est(self):
        return self._p_est
    
    @property
    def landmarks(self):
        return self._landmarks
    
    @property
    def v_est(self):
        return self._v_est

    @property
    def w_est(self):
        return self._w_est
    
    @property
    def robot(self):
        return self._robot
    
    @property
    def sensor(self):
        return self._sensor
    
    @property
    def est_vehicle(self):
        return self._est_vehicle
    
    @property
    def est_ekf_map(self):
        return self._est_ekf_map
    
    @property
    def joseph(self):
        return self._joseph
    
    @property
    def verbose(self):
        return self._verbose
    
    @property
    def keep_history(self):
        return self._keep_history
    
    @property
    def p0(self):
        return self._p0
    
    @property
    def ekf_map(self):
        return self._ekf_map
    
    @property
    def history(self):
        return self._history
    
    @property
    def dim(self):
        return self._dim
    
    def run(self, n, plot=True, movie=np.array([])):
        self._init()
        if plot:
            if np.any(self._sensor):
                self._sensor.ekf_map.plt.plot()
            elif np.any(self._dim):
                if len(self._dim) == 1:
                    d = self._dim
                    plt.axis([-d, d, -d, d])
                if len(self._dim) == 2:
                    w = self._dim[1], h = self._dim(2)
                    plt.axis([-w, w, -h, h])
                if len(self._dim) == 4:
                    plt.axis(self._dim)
            else:
                plot = False

            plt.xlabel('X')
        plt.ylabel('Y')

        anim = Animate(movie)
        for k in range(n):
            if plot:
                self._robot.plt.plot()
            self._step(n)
            anim.add()

        anim.close()

    def get_xy(self):
        if self._est_vehicle:
            xyt = np.zeros(len(self._history), 3)
            for i in range(len(self._history)):
                h = self._history[i]
                xyt[i,:] = np.invert(h.x_est[1:3])
        else:
            xyt = []
        return xyt

    # TODO: Add plotting functions

    def get_ekf_map(self, varargin):
        xy = []
        for i in range(np.numcols(self._landmarks)):
            n = self._landmarks(1,i)
            if n is None:
                #  this landmark never observed
                xy = [xy, [None, None]]
                continue

            #  n is an index into the *landmark* part of the state
            #  vector, we need to offset it to account for the vehicle
            #  state if we are estimating vehicle as well
            if self._est_vehicle:
                n = n + 3

            xf = self._x_est[n][n+1]
            xy = [xy, xf]
        return xy

    def get_P(self, k):
        p = np.zeros(len(self._history),1)
        for i in range(len(self._history)):
            p[i] = np.sqrt(np.det(self._history[i].p))
        return p

    def show_P(self, k=None):
        if k is None:
            k = len(self._history)

        z = np.log10(abs(self._history(k).P))
        mn = min(z(not np.isinf(z)))
        z[np.isinf(z)] = mn

    def char(self):
        s = "EKF object: # " + len(self._x_est) + " states"
        e = ''
        if self._est_vehicle:
            e = np.array([e, 'Vehicle '])

        if self._est_ekf_map:
            e = [e, 'Map ']

        s = s + '\nestimating: ' + e
        if np.any(self._robot):
            s = s + "\nrobot: "  + self._robot

        if np.any(self._sensor):
            s = s + "\nsensor: " + self._sensor

        s = s + '\nw_est:  ' + self._w_est
        s =s + '\nv_est:  ' + self._v_est

        return s

    # TODO: Make the following methods private.
    def step(self, opt, z_pred=None):
        odo = self._robot.step()
        xv_est = None
        xm_est = None
        Pvv_est = None
        Pmm_est = None
        Pvm_est = None
        innov = None
        xm_pred = None
        Hx_k = None
        Hx = None
        Hw = None
        K = None
        S = None
        hist = None

        if self._est_vehicle:
            xv_est = self._x_est[0:3]
            xm_est = self._x_est[3:]
            Pvv_est = self._p_est[0:3,0:3]
            Pmm_est = self._p_est[3:,3:]
            Pvm_est = self._p_est[0:3,3:]
        else:
            xm_est = self._x_est
            Pmm_est = self._p_est


        if self._est_vehicle:
            xv_pred = np.transpose(self._robot.f(np.transpose(xv_est), odo))

            Fx = self._robot.Fx[xv_est, odo]
            Fv = self._robot.Fv[xv_est, odo]
            Pvv_pred = Fx*Pvv_est*np.transpose(Fx) + Fv*self._v_est*np.transpose(Fv)
        else:
            xv_pred = self._robot.x

        if self._est_ekf_map:
            if self._est_vehicle:
                Pvm_pred = Fx*Pvm_est

            Pmm_pred = Pmm_est
            xm_pred = xm_est

        if self._est_vehicle and not self._est_ekf_map:
            x_pred = xv_pred
            P_pred =  Pvv_pred
        elif not self._est_vehicle and self._est_ekf_map:
            x_pred = xm_pred
            P_pred = Pmm_pred
        elif self._est_vehicle and self._est_ekf_map:
            x_pred = np.array([xv_pred, xm_pred])
            P_pred = np.array([ Pvv_pred, Pvm_pred, np.inverse(Pvm_pred), Pmm_pred])

        doUpdatePhase = False

        # disp('x_pred:') x_pred'
            # sensorReading = False
        if np.any(self._sensor):
            #  read the sensor
            [z,js] = self._sensor.reading()

            #  test if the sensor has returned a reading at this time interval
            sensorReading = js > 0


        if sensorReading:
            #  here for MBL, MM, SLAM

            #  compute the innovation
            z_pred = np.transpose(self._sensor.h(np.transpose(xv_pred), js))
            innov[0] = z[0] - z_pred[0]
            innov[1] = angdiff(z[2], z_pred[2])

            if self._est_ekf_map:
                #  the ekf_map is estimated MM or SLAM case
                if self._seenBefore(js):

                    #  get previous estimate of its state
                    jx = self._landmarks(1,js)
                    xf = xm_pred[jx:jx+1]

                    #  compute Jacobian for this particular landmark
                    Hx_k = self._sensor.Hp(np.transpose(xv_pred), xf)
                    #  create the Jacobian for all landmarks
                    Hx = np.zeros(2, len(xm_pred))
                    Hx[:,jx:jx+1] = Hx_k

                    Hw = self._sensor.Hw(xv_pred, xf)

                    if self._est_vehicle:
                        #  concatenate Hx for for vehicle and ekf_map
                        Hxv = self._sensor.Hx(np.transpose(xv_pred), xf)
                        Hx = np.array([Hxv, Hx])

                    doUpdatePhase = True
            else:
                    #  get the exted state
                    [x_pred, P_pred] = self._extMap(P_pred, xv_pred, xm_pred, z, js)
                    doUpdatePhase = False

        else:
                Hx = self._sensor.Hx(np.transpose(xv_pred), js)
                Hw = self._sensor.Hw(np.transpose(xv_pred), js)
                doUpdatePhase = True

        if doUpdatePhase:
            # disp('do update\n')
            # #  we have innovation, update state and covariance
            #  compute x_est and p_est

            #  compute innovation covariance
            S = Hx*P_pred*np.array(Hx) + Hw*self._w_est*np.array(Hw)

            #  compute the Kalman gain
            K = P_pred*np.transpose(Hx) / S

            #  update the state vector
            x_est = x_pred + K*np.transpose(innov)

            if self._est_vehicle:
                #  wrap heading state for a vehicle
                x_est[2] = angdiff(x_est[2])


            #  update the covariance
            if self._joseph:
                #  we use the Joseph form
                I = np.eye(np.shape(P_pred))
                p_est = (I-K*Hx)*P_pred*np.inverse(I-K*Hx) + K*self._w_est*np.inverse(K)
            else:
                p_est = P_pred - K*S*np.inverse(K)
                p_est = 0.5*(p_est+np.inverse(p_est))
        else:
            x_est = x_pred
            p_est = P_pred
            innov = []
            S = []
            K = []

        self._x_est = x_est
        self._p_est = p_est

        if self._keep_history:
            hist.x_est = x_est
            hist.odo = odo
            hist.P = p_est
            hist.innov = innov
            hist.S = S
            hist.K = K
            self._history = np.array([self._history, hist])

    def blkdiag(self, p,_w_est):
        # TODO: implement
        return -1

    def seenBefore(self, jf):
        if not np.isnan(self._landmarks[0,jf]):
            if self._verbose:
                disp('landmark # d seen # d times before, state_idx=# d\n', jf, self._landmarks[0,jf], self._landmarks[0,jf])
            self._landmarks[1,jf] = self._landmarks(2,jf)+1
            s = True
        else:
            s = False
        return s

    def extMap(self, P, xv, xm, z, jf):
        M = None
        if self._verbose:
            disp('landmark # d first sighted\n', jf)

        xf = self._sensor.g(xv, z)

        if self._est_vehicle:
            x_ext = np.array([xv, xm, xf])
        else:
            x_ext = np.array([xm, xf])

        Gz = self._sensor.Gz(xv, z)

        if self._est_vehicle:
            Gx = self._sensor.Gx(xv, z)
            n = len(self._x_est)
            M = np.array([np.eye(n), np.zeros(n,2), Gx, np.zeros(2,n-3), Gz])
            P_ext = M * self.blkdiag(P, self._w_est) * np.transpose(M)
        else:
            P_ext = self.blkdiag(P, np.transpose(Gz*self._w_est*Gz))

        self._landmarks[0, jf] = len(xm)+1
        self._landmarks[1, jf] = 1

        if self._verbose:
            disp('exted state vector\n')
        return x_ext, P_ext

def chi2inv_rtb(confidence, n):
    if n != 2:
        raise ValueError('chi2inv_rtb: only valid for 2DOF')

    c = np.linspace(0,1,101)
    x = [0.000000, 0.020101, 0.040405, 0.060918, 0.081644, 0.102587, 0.123751, 
         0.145141 ,0.166763, 0.188621, 0.210721, 0.233068, 0.255667, 0.278524, 
         0.301646, 0.325038, 0.348707, 0.372659, 0.396902, 0.421442, 0.446287, 
         0.471445, 0.496923, 0.522730, 0.548874, 0.575364, 0.602210, 0.629421,
         0.657008, 0.684981, 0.713350, 0.742127, 0.771325, 0.800955,  0.831031,
         0.861566, 0.892574, 0.924071, 0.956072, 0.988593, 1.021651, 1.055265,
         1.089454, 1.124238, 1.159637, 1.195674,  1.232372, 1.269757, 1.307853,
         1.346689, 1.386294, 1.426700, 1.467938, 1.510045, 1.553058, 1.597015,
         1.641961, 1.687940,  1.735001, 1.783196, 1.832581, 1.883217, 1.935168,
         1.988505, 2.043302, 2.099644, 2.157619, 2.217325, 2.278869, 2.342366,
         2.407946, 2.475749, 2.545931, 2.618667, 2.694147, 2.772589, 2.854233,
         2.939352, 3.028255, 3.121295, 3.218876, 3.321462, 3.429597, 3.543914,
         3.665163, 3.794240, 3.932226, 4.080442, 4.240527, 4.414550, 4.605170,
         4.815891, 5.051457, 5.318520, 5.626821, 5.991465, 6.437752, 7.013116,
         7.824046, 9.210340, np.Inf]
    f = np.interp(c, x, confidence)

    return f


# function chi2inv_test(tc)
#     tc.verifyEqual( chi2inv_rtb(0,2), 0);
#     tc.verifyEqual( chi2inv_rtb(1,2), Inf);
#     tc.verifyEqual( chi2inv_rtb(3,2), NaN);
    
#     tc.verifyError( @() chi2inv_rtb(1,1), 'RTB:chi2inv_rtb:badarg');

if __name__ == "__main__":

    from roboticstoolbox import Bicycle

    ### RVC2: Chapter 6


    ##  6.1 Dead reckoning

    ## 6.1.1 Modeling the vehicle
    V = np.diag(np.r_[0.02, 0.5*pi/180] ** 2);

    veh = Bicycle(covar=V)

    odo = veh.step(1, 0.3)

    print(veh.x)

    veh.f([0, 0, 0], odo)

    # veh.add_driver( RandomPath(10) )

    # veh.run()

    ###  6.1.2  Estimating pose
    # veh.Fx( [0,0,0], [0.5, 0.1] )

    P0 = np.diag(np.r_[0.005, 0.005, 0.001]**2);

    ekf = EKF(veh, V, P0);

    ekf.run(1000);

    veh.plot_xy()

    ekf.plot_xy('r')

    P700 = ekf.history(700).P

    sqrt(P700(1,1))

    ekf.plot_ellipse('g')

    # %%  6.2 Map-based localization
    # randinit
    # map = LandmarkMap(20, 10)

    # map.plot()

    # W = diag([0.1, 1*pi/180].^2);

    # sensor = RangeBearingSensor(veh, map, 'covar', W)


    # [z,i] = sensor.reading()

    # map.landmark(17)

    # randinit
    # map = LandmarkMap(20);
    # veh = Bicycle('covar', V);
    # veh.add_driver( RandomPath(map.dim) );
    # sensor = RangeBearingSensor(veh, map, 'covar', W, 'angle', ...
    # [-pi/2 pi/2], 'range', 4, 'animate');
    # ekf = EKF(veh, V, P0, sensor, W, map);

    # ekf.run(1000);
    # map.plot()
    # veh.plot_xy();
    # ekf.plot_xy('r');
    # ekf.plot_ellipse('k')

    # %%  6.3  Creating a map
    # randinit
    # map = LandmarkMap(20);
    # veh = Bicycle(); % error free vehicle
    # veh.add_driver( RandomPath(map.dim) );
    # W = diag([0.1, 1*pi/180].^2);
    # sensor = RangeBearingSensor(veh, map, 'covar', W);
    # ekf = EKF(veh, [], [], sensor, W, []);

    # ekf.run(1000);

    # map.plot();
    # ekf.plot_map('g');
    # veh.plot_xy('b');


    # ekf.landmarks(:,6)

    # ekf.x_est(19:20)'

    # ekf.P_est(19:20,19:20)

    # %%  6.4  EKF SLAM
    # randinit
    # P0 = diag([.01, .01, 0.005].^2);
    # map = LandmarkMap(20);
    # veh = Bicycle('covar', V);
    # veh.add_driver( RandomPath(map.dim) );
    # sensor = RangeBearingSensor(veh, map, 'covar', W);
    # ekf = EKF(veh, V, P0, sensor, W, []);

    # ekf.run(1000);

    # map.plot();
    # ekf.plot_map('g');
    # ekf.plot_xy('r');
    # veh.plot_xy('b');

    # %%  6.6 Pose-graph SLAM
    # syms x_i y_i theta_i x_j y_j theta_j x_m y_m theta_m assume real
    # xi_e = inv( SE2(x_m, y_m, theta_m) ) * inv( SE2(x_i, y_i, theta_i) ) * SE2(x_j, y_j, theta_j);
    # fk = simplify(xi_e.xyt);

    # jacobian ( fk, [x_i y_i theta_i] );
    # Ai = simplify (ans)

    # pg = PoseGraph('pg1.g2o')

    # clf
    # pg.plot()

    # pg.optimize('animate')

    # pg = PoseGraph('killian-small.toro');

    # pg.plot()

    # pg.optimize()

    # %% 6.7  Particle filter
    # randinit
    # map = LandmarkMap(20);
    # W = diag([0.1, 1*pi/180].^2);
    # veh = Bicycle('covar', V);
    # veh.add_driver( RandomPath(10) );

    # V = diag([0.005, 0.5*pi/180].^2);
    # sensor = RangeBearingSensor(veh, map, 'covar', W);

    # Q = diag([0.1, 0.1, 1*pi/180]).^2;

    # L = diag([0.1 0.1]);

    # pf = ParticleFilter(veh, sensor, Q, L, 1000);

    # pf.run(1000);

    # map.plot();
    # veh.plot_xy('b');

    # clf
    # pf.plot_xy('r');

    # clf
    # plot(pf.std(1:100,:))

    # clf
    # pf.plot_pdf()


    # %% 6.8  Application: Scanning laser rangefinder

    # %% Laser odometry
    # pg = PoseGraph('killian.g2o', 'laser');

    # [r, theta] = pg.scan(2580);
    # about r theta

    # clf
    # polar(theta, r)

    # [x,y] = pol2cart (theta, r);
    # plot (x, y, '.')

    # p2580 = pg.scanxy(2580);
    # p2581 = pg.scanxy(2581);
    # about p2580

    # T = icp( p2581, p2580, 'verbose' , 'T0', transl2(0.5, 0), 'distthresh', 3)

    # pg.time(2581)-pg.time(2580)


    # %% Laser-based map building
    # map = pg.scanmap();
    # pg.plot_occgrid(map)