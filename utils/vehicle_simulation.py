"""
Inspirations from: Atsushi Sakai (@Atsushi_twi)
"""

import numpy as np
import cvxpy
from tqdm.auto import tqdm
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def plot_car(ax, x, y, yaw, steer=0.0, truckcolor="-k"):
    LENGTH = 2.25 
    WIDTH = 1.0 
    BACKTOWHEEL = 0.5 
    WHEEL_LEN = 0.15
    WHEEL_WIDTH = 0.1 
    TREAD = 0.35
    WB = 1.25

    outline = np.matrix([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                         [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.matrix([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                          [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.matrix([[np.cos(yaw), np.sin(yaw)],
                      [-np.sin(yaw), np.cos(yaw)]])
    Rot2 = np.matrix([[np.cos(steer), np.sin(steer)],
                      [-np.sin(steer), np.cos(steer)]])

    fr_wheel = (fr_wheel.T * Rot2).T
    fl_wheel = (fl_wheel.T * Rot2).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T * Rot1).T
    fl_wheel = (fl_wheel.T * Rot1).T

    outline = (outline.T * Rot1).T
    rr_wheel = (rr_wheel.T * Rot1).T
    rl_wheel = (rl_wheel.T * Rot1).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    ax.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    ax.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    ax.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    ax.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    ax.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    ax.plot(x, y, "*")

class Vehicle:
    MAX_STEER = np.radians(45.0)  # maximum steering angle [rad]
    MAX_DSTEER = np.radians(30.0)  #np.radians(30.0)  # maximum steering speed [rad/s]
    MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
    MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
    MAX_ACCEL = 1.0  # maximum accel [m/ss]
    WB = 2.5  # wheel base [m]
    
    def __init__(self, x = 0.0, y = 0.0, yaw = 0.0, v = 0.0, dt = 0.1, store_states = False):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.store_states = store_states
        
        self.dt = dt
        
        if self.store_states:
            self.xs = [x]
            self.ys = [y]
            self.yaws = [yaw]
            self.vs = [v]
        
    def update(self, a, delta):
        # input check
        if delta >= self.MAX_STEER:
            delta = self.MAX_STEER
        elif delta <= -self.MAX_STEER:
            delta = -self.MAX_STEER

        self.x = self.x + self.v * np.cos(self.yaw) * self.dt
        self.y = self.y + self.v * np.sin(self.yaw) * self.dt
        self.yaw = self.yaw + self.v / self.WB * np.tan(delta) * self.dt
        self.v = self.v + a * self.dt

        if self. v > self.MAX_SPEED:
            self.v = self.MAX_SPEED
        elif self. v < self.MIN_SPEED:
            self.v = self.MIN_SPEED
        
        if self.store_states:
            self.xs.append(self.x)
            self.ys.append(self.y)
            self.yaws.append(self.yaw)
            self.vs.append(self.v)
            
        return self
    
    def get_state(self):
        return np.array([self.x, self.y, self.v, self.yaw])
    
    def copy(self):
        return Vehicle(self.x, self.y, self.yaw, self.v, self.dt)
    
    def linear_model(self, v, phi, delta):
        A = np.array([[1.0, 0, self.dt * np.cos(phi), - self.dt * v * np.sin(phi)],
                      [0, 1.0, self.dt * np.sin(phi), self.dt * v * np.cos(phi)],
                      [0, 0, 1.0, 0],
                      [0, 0, self.dt * np.tan(delta) / self.WB, 1.0]])
        
        B = np.array([[0, 0],
                     [0, 0],
                     [self.dt, 0],
                     [0, self.dt * v / (self.WB * np.cos(delta) ** 2)]])
        
        C = np.array([self.dt * v * np.sin(phi) * phi,
                      - self.dt * v * np.cos(phi) * phi,
                      0,
                      v * delta / (self.WB * np.cos(delta) ** 2)])
        
        return A, B, C
    
class MPC:
    MAX_ITER = 5  # Max iteration
    DU_TH = 0.1  # iteration finish param
    R = np.diag([0.01, 0.01])  # input cost matrix
    Rd = np.diag([0.01, 1.0])  # input difference cost matrix
    Q = np.diag([1.0, 1.0, 0.5, 0.5])  # state cost matrix
    Qf = Q  # state final matrix

    def __init__(self, state, T):
        self.state = state
        self.T = T
    
    def predict_motion(self, oa, od, xref):
        tmp_state = self.state.copy()
        xbar = xref * 0.0
        xbar[:, 0] = tmp_state.get_state()

        for (ai, di, i) in zip(oa, od, range(1, self.T + 1)):
            tmp_state = tmp_state.update(ai, di)
            xbar[:, i] = tmp_state.get_state()

        return xbar
    
    def add_cost(self, cost, u, x, xref, dref):
        for t in range(self.T):
            cost += cvxpy.quad_form(u[:, t], self.R)

            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], self.Q)

            if t < (self.T - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], self.Rd)

        cost += cvxpy.quad_form(xref[:, self.T] - x[:, self.T], self.Qf)

        return cost

    def add_constraints(self, constraints, x, u, xbar, dref):
        for t in range(self.T):
            A, B, C = self.state.linear_model(xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A * x[:, t] + B * u[:, t] + C]

            if t < (self.T - 1):
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t])
                                <= self.state.MAX_DSTEER * self.state.dt]

        constraints += [x[:, 0] == self.state.get_state()]
        constraints += [x[2, :] <= self.state.MAX_SPEED]
        constraints += [x[2, :] >= self.state.MIN_SPEED]
        constraints += [cvxpy.abs(u[0, :]) <= self.state.MAX_ACCEL]
        constraints += [cvxpy.abs(u[1, :]) <= self.state.MAX_STEER]

        return constraints

    def linear_mpc_control(self, xref, xbar, dref):
        x = cvxpy.Variable((4, self.T + 1))
        u = cvxpy.Variable((2, self.T))

        cost = 0.0
        constraints = []

        cost = self.add_cost(cost, u, x, xref, dref)
        constraints = self.add_constraints(constraints, x, u, xbar, dref)

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox = np.array(x.value[0, :]).flatten()
            oy = np.array(x.value[1, :]).flatten()
            ov = np.array(x.value[2, :]).flatten()
            oyaw = np.array(x.value[3, :]).flatten()
            oa = np.array(u.value[0, :]).flatten()
            odelta = np.array(u.value[1, :]).flatten()

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov
    
    def iterative_linear_mpc_control(self, xref, dref, oa, od):
        """
        MPC control with updating operational point iteratively.

        If no previous controls are given (oa, od are None), initialize them to zero.
        """

        if oa is None or od is None:
            oa = [0.0] * self.T
            od = [0.0] * self.T

        for i in range(self.MAX_ITER):
            xbar = self.predict_motion(oa, od, xref)
            poa, pod = oa[:], od[:]
            oa, od, ox, oy, oyaw, ov = self.linear_mpc_control(xref, xbar, dref)
            du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
            if du <= self.DU_TH:
                break

        return oa, od, ox, oy, oyaw, ov
    
class Simulation:
    N_IND_SEARCH = 10  # Search index number
    T = 5  # horizon length
        
    def __init__(self, initial_state, target_speed = 10.0 / 3.6, goal_speed = 0.01, goal_dist = 1.5, max_time = 100.0):
        self.goal_speed = goal_speed
        self.goal_dist = goal_dist
        self.max_time = max_time
        self.target_speed = target_speed
        
        self.state = initial_state
        self.mpc = MPC(self.state, self.T)
        
    @staticmethod
    def pi_2_pi(angle):
        angle = np.fmod(angle, 2.0 * np.pi)
        if angle > np.pi:
            angle -= 2.0 * np.pi
        elif angle < -np.pi:
            angle += 2.0 * np.pi

        return angle
        
    def simulate(self, cx, cy, cyaw, ck, dl, sp = None):
        if sp is None:
            sp = self.calc_speed_profile(cx, cy, cyaw)
        goal = [cx[-1], cy[-1]]

        # initial yaw compensation
        self.state.yaw = self.pi_2_pi(self.state.yaw - cyaw[0])

        max_iter = int(self.max_time / self.state.dt) + 1

        time = np.zeros(max_iter, dtype=float)
        x = np.zeros(max_iter, dtype=float)
        y = np.zeros(max_iter, dtype=float)
        yaw = np.zeros(max_iter, dtype=float)
        v = np.zeros(max_iter, dtype=float)
        d = np.zeros(max_iter, dtype=float)
        a = np.zeros(max_iter, dtype=float)
        target_inds = np.zeros(max_iter, dtype=int)
        xrefs = [None] * max_iter  # because xref is a complex object, keep it as list

        x[0], y[0], yaw[0], v[0] = self.state.x, self.state.y, self.state.yaw, self.state.v

        target_ind, _ = self.calc_nearest_index(self.state, cx, cy, cyaw, 0)
        target_inds[0] = target_ind
        cyaw = self.smooth_yaw(cyaw)

        odelta, oa = None, None
        i = 0

        with tqdm(desc='MPC') as pbar:
            while self.max_time >= time[i]:
                xref, target_ind, dref = self.calc_ref_trajectory(
                    self.state, cx, cy, cyaw, ck, sp, dl, target_ind)
                xrefs[i] = xref

                oa, odelta, ox, oy, oyaw, ov = self.mpc.iterative_linear_mpc_control(
                    xref, dref, oa, odelta)

                if odelta is not None:
                    di, ai = odelta[0], oa[0]

                self.state = self.state.update(ai, di)
                time[i+1] = time[i] + self.state.dt

                x[i+1] = self.state.x
                y[i+1] = self.state.y
                yaw[i+1] = self.state.yaw
                v[i+1] = self.state.v
                d[i+1] = di
                a[i+1] = ai
                target_inds[i+1] = target_ind

                if self.check_goal(goal, target_ind, len(cx)):
                    print("Goal")
                    break

                dx = self.state.x - goal[0]
                dy = self.state.y - goal[1]
                goal_distance = np.sqrt(dx ** 2 + dy ** 2)
                
                pbar.update(1)
                pbar.set_postfix({'x': self.state.x, 'y': self.state.y, 'd': goal_distance, 'v': abs(self.state.v)})

                i += 1

        # Trim the arrays to the appropriate size
        i += 1  # because we need to include the last element
        time = time[:i]
        x = x[:i]
        y = y[:i]
        yaw = yaw[:i]
        v = v[:i]
        d = d[:i]
        a = a[:i]
        target_inds = target_inds[:i]
        xrefs = np.array(xrefs[:i])
        
        state_history = dict(t = time, x = x, y = y, v = v, yaw = yaw)
        control_history = dict(a = a, d = d, target_inds = target_inds, xrefs = xrefs)

        return state_history, control_history
        
    def calc_nearest_index(self, state, cx, cy, cyaw, pind):
        dx = np.array(cx[pind:(pind + self.N_IND_SEARCH)]) - state.x
        dy = np.array(cy[pind:(pind + self.N_IND_SEARCH)]) - state.y

        d = dx ** 2 + dy ** 2

        ind = np.argmin(d) + pind
        mind = np.sqrt(d[ind - pind])

        dxl = cx[ind] - state.x
        dyl = cy[ind] - state.y

        angle = self.pi_2_pi(cyaw[ind] - np.arctan2(dyl, dxl))
        if angle < 0:
            mind *= -1

        return ind, mind
    
    def calc_ref_trajectory(self, state, cx, cy, cyaw, ck, sp, dl, pind):
        ncourse = len(cx)

        ind, _ = self.calc_nearest_index(state, cx, cy, cyaw, pind)
        if pind >= ind:
            ind = pind

        xref = np.zeros((4, self.T + 1))
        dref = np.zeros((1, self.T + 1))  # steer operational point should be 0

        travel = np.abs(state.v) * state.dt * np.arange(self.T + 1)
        dind = np.round(travel / dl).astype(int)
        inds = np.clip(ind + dind, a_min=None, a_max=ncourse - 1)
        
        xref[0, :] = np.take(cx, inds)
        xref[1, :] = np.take(cy, inds)
        xref[2, :] = np.take(sp, inds)
        xref[3, :] = np.take(cyaw, inds)

        return xref, ind, dref
        
    def check_goal(self, goal, tind, nind):
        dx = self.state.x - goal[0]
        dy = self.state.y - goal[1]
        d = np.sqrt(dx ** 2 + dy ** 2)

        isgoal = d <= self.goal_dist and abs(tind - nind) < 5
        isstop = abs(self.state.v) <= self.goal_speed

        return isgoal and isstop

    def calc_speed_profile(self, cx, cy, cyaw):
        speed_profile = [self.target_speed] * len(cx)

        # Set stop point
        for i in range(len(cx) - 1):
            dx = cx[i + 1] - cx[i]
            dy = cy[i + 1] - cy[i]

            if dx != 0.0 and dy != 0.0:
                dangle = np.abs(self.pi_2_pi(np.arctan2(dy, dx) - cyaw[i]))
                if dangle >= np.pi / 4.0:
                    speed_profile[i] = -self.target_speed
                else:
                    speed_profile[i] = self.target_speed

        speed_profile[-1] = 0.0

        return speed_profile
    
    def smooth_yaw(self, yaw):
        for i in range(len(yaw) - 1):
            dyaw = self.pi_2_pi(yaw[i + 1] - yaw[i])

            if dyaw >= np.pi / 2.0:
                yaw[i + 1] -= np.pi * 2.0
            elif dyaw <= -np.pi / 2.0:
                yaw[i + 1] += np.pi * 2.0

        return yaw
    
if __name__ == '__main__':
    import cubic_spline_planner
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    x1 = np.linspace(5, 43, 6)
    x2 = np.repeat(42.5, 3) + np.random.normal(0, 0.75, 3)
    x2 = np.clip(x2, 41, 44)

    y1 = np.repeat(3, 6) + np.random.normal(0, 0.75, 6)
    y1 = np.clip(y1, 2, 4)

    y2 = np.linspace(7.5, 17.5, 3)

    x_point = np.r_[1, x1, x2, 42.5]
    y_point  = np.r_[3, y1, y2, 20]
        
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(x_point, y_point, ds=0.1)

    # Simulation
    initial_state = Vehicle(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

    dl = 2.0
    simulation = Simulation(initial_state)

    t, x, y, yaw, v, d, a, target_inds, xrefs = simulation.simulate(cx, cy, cyaw, ck, dl)
    xrefs = np.r_[xrefs[0:1], xrefs]