import numpy as np
from utils.vehicle_simulation import Vehicle, Simulation, plot_car
import utils.cubic_spline_planner as cubic_spline_planner
from scipy.stats import multivariate_normal as mvn
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from matplotlib import animation
from os.path import exists

class CarTrajectoryLinear:
    def __init__(self, N = 100, dt = 0.1, q1 = 1, q2 = 1, s1 = 0.5, s2 = 0.5):
        self.N = N
        self.dt = dt
        self.q1 = q1
        self.q2 = q2
        self.s1 = s1
        self.s2 = s2
        
        self.Q = np.array([[(q1 * dt**3) / 3, 0, (q1 * dt**2) / 2, 0],
                        [ 0, (q2 * dt**3) / 3, 0, (q2 * dt**2) / 2],
                        [(q1 * dt**2) / 2, 0, q1 * dt, 0],
                        [0, (q2 * dt**2) / 2, 0, q2 * dt]])
        
        self.R = np.array([[s1, 0],
                        [0, s2]])
        
        self.A = np.array([[1, 0, dt, 0],
                        [0, 1, 0, dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        
        self.H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])
        
        self.dim_m = 4
        self.dim_y = 2
    
    def get_data(self):
        x = np.zeros((self.N, 4))
        y = np.zeros((self.N, 2))
        
        x[0] = mvn(mean = np.array([0, 0, 0, 0]), cov = self.Q).rvs(1)
        y[0] = mvn(mean = self.H @ x[0], cov = self.R).rvs(1)
        
        for i in range(1, self.N):
            x[i] = mvn(mean = self.A @ x[i - 1], cov = self.Q).rvs(1)
            y[i] = mvn(mean = self.H @ x[i], cov = self.R).rvs(1)
        
        return x, y
    
class CarTrajectoryNonLinear:
    def __init__(self, N = 100, dt = 0.1, q1 = 1, q2 = 1, s1 = 0.5, s2 = 0.5):
        self.N = N
        self.dt = dt
        self.q1 = q1
        self.q2 = q2
        self.s1 = s1
        self.s2 = s2
        
        self.Q = np.array([[(q1 * dt**3) / 3, 0, (q1 * dt**2) / 2, 0],
                        [ 0, (q2 * dt**3) / 3, 0, (q2 * dt**2) / 2],
                        [(q1 * dt**2) / 2, 0, q1 * dt, 0],
                        [0, (q2 * dt**2) / 2, 0, q2 * dt]])
        
        self.R = np.array([[s1, 0],
                        [0, s2]])
        
        self.A = np.array([[1, 0, dt, 0],
                        [0, 1, 0, dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        
        self.H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])
        
        self.dim_m = 4
        self.dim_y = 2
        
    def f_linear(self, x):
        return self.A @ x
    
    def f_nonlinear(self, x):
        return np.array([0.1 * np.sin(x[0]), 0.1 * np.sin(x[1]), 0, 0])
    
    def f(self, x):
        return self.f_linear(x) + self.f_nonlinear(x)
    
    def h_linear(self, x):
        return self.H @ x
    
    def h_nonlinear(self, x):
        return np.array([1.0 * np.sin(x[1]), -1.0 * np.cos(x[0])])

    def h(self, x):
        return self.h_linear(x) + self.h_nonlinear(x)
    
    def F_jacobian(self, x):
        return np.array([[1 + 0.1 * np.cos(x[0]), 0, self.dt, 0],
                        [0, 1 + 0.1 * np.cos(x[1]), 0, self.dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    def H_jacobian(self, x):
        return np.array([[1, 1.0 * np.cos(x[1]), 0, 0],
                        [1.0 * np.sin(x[0]), 1, 0, 0]])
    
    def get_data(self):
        self.x = np.zeros((self.N, 4))
        self.y = np.zeros((self.N, 2))
        
        self.x[0] = np.array([0, 0, 0, 0])
        self.y[0] = self.h(self.x[0]) + mvn(np.zeros(2), self.R).rvs(1)
        
        for i in range(1, self.N):
            self.x[i] = self.f(self.x[i - 1]) + mvn(np.zeros(4), self.Q).rvs(1)
            self.y[i] = self.h(self.x[i]) + mvn(np.zeros(2), self.R).rvs(1)
        
        return self.x, self.y
    
    
class MPCTrajectory:
    def __init__(self, savepath=None):

        x1 = np.linspace(5, 43, 6)
        x2 = np.repeat(42.5, 3) + np.random.normal(0, 0.75, 3)
        x2 = np.clip(x2, 41, 44)

        y1 = np.repeat(3, 6) + np.random.normal(0, 0.75, 6)
        y1 = np.clip(y1, 2, 4)

        y2 = np.linspace(7.5, 17.5, 3)

        self.x_points = np.r_[1, x1, x2, 42.5]
        self.y_points = np.r_[3, y1, y2, 20]

        self._states = None
        self._measurements = None

        self.cx = None
        self.cy = None
        self.states_hist = None
        self.controls_hist = None
        
        s1 = 2
        s2 = 2
                
        self.R = np.array([[s1, 0],
                    [0, s2]])
        
        self.savepath = savepath

    @property
    def states(self):
        if not self._states:
            if self.savepath and exists(self.savepath):
                self._states = np.load(self.savepath)
            else:
                self._states = self._calculate_states()
                if self.savepath:
                    np.save(self.savepath, self._states)
                    
        return self._states

    @property
    def measurements(self):
        if self._measurements is None:
            self._calculate_measurements()
        return self._measurements
    
    # Based on gut feelings
    def speed_reduction(self, cyaw, sp):
        speed_reduction = np.diff(cyaw)
        speed_reduction = np.concatenate([speed_reduction[int(len(speed_reduction)*0.025):],
                                        np.linspace(speed_reduction[-1], 0, int(len(speed_reduction)*0.025))])
        speed_reduction = np.convolve(speed_reduction, np.ones(30)/30, mode='same')
        speed_reduction = ((1.5*np.max(np.abs(speed_reduction)) - np.abs(speed_reduction)) / (1.5*np.max(np.abs(speed_reduction))))
        speed_reduction = np.clip(speed_reduction, 0, 1)
        speed_reduction *= np.concatenate([np.ones(int(len(speed_reduction)*0.95)+1), np.linspace(1, 0, int(len(speed_reduction)*0.05))])
        
        sp_new = sp
        sp_new[:-1] *= speed_reduction
        
        return sp_new

    def _calculate_states(self):
        self.cx, self.cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(self.x_points, self.y_points, ds=0.1)
        initial_state = Vehicle(x=self.cx[0], y=self.cy[0], yaw=cyaw[0], v=0.0)
        dl = 1.0
                
        simulation = Simulation(initial_state, goal_speed=0.5, target_speed=3)
        
        # Speed profile
        sp = np.array(simulation.calc_speed_profile(self.cx, self.cy, cyaw))
        sp = self.speed_reduction(cyaw, sp)
        
        # Simulation
        self.states_hist, self.controls_hist = simulation.simulate(self.cx, self.cy, cyaw, ck, dl, sp=sp)

        # Get simulation states
        x, y, v, yaw = self.states_hist['x'], self.states_hist['y'], self.states_hist['v'], self.states_hist['yaw']        

        # Calculate dx and dy
        dx = np.empty_like(x)
        dy = np.empty_like(y)
        for i in range(len(x)):
            dx[i] = v[i] * np.cos(yaw[i]) * initial_state.dt
            dy[i] = v[i] * np.sin(yaw[i]) * initial_state.dt
            
        # Downsample to 2Hz
        # skip = int((1 / initial_state.dt) / 5)
        # x, y, dx, dy = x[::skip], y[::skip], dx[::skip], dy[::skip]            

        return np.c_[x, y, dx, dy]

    def h(self, x):
        H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])
        
        if x.ndim == 1:
            x = x.reshape(1, -1)     
            out = np.dot(x, H.T) + np.array([1.0 * np.sin(x[:, 1]), -1.0 * np.cos(x[:, 0])]).T
            out = out.flatten()
            
        else:
            out = np.dot(x, H.T) + np.array([1.0 * np.sin(x[:, 1]), -1.0 * np.cos(x[:, 0])]).T
        
        return out
    
    def _calculate_measurements(self):
        measurements = self.h(self._states) + mvn([0, 0], self.R).rvs(len(self._states))        
        self._measurements = measurements

    def get_data(self):
        return self.states.copy(), self.measurements.copy()
    
    def plot(self, states, measurements):

        fig, ax = plt.subplots(1, 1, figsize=(16, 6))

        ax.plot(measurements[:, 0], measurements[:, 1], 'o', label='Measurements', markersize=3)
        ax.plot(self.x_points, self.y_points, 'x', label='Waypoints', markersize=10)
        ax.plot(states[:, 0], states[:, 1], label='Trajectory', linewidth=2)
        

        ax.hlines(1, 1, 45, color='k', linestyle='solid', linewidth=1)
        ax.hlines(5, 1, 40, color='k', linestyle='solid', linewidth=1)
        ax.vlines(45, 1, 20, color='k', linestyle='solid', linewidth=1)
        ax.vlines(40, 5, 20, color='k', linestyle='solid', linewidth=1)

        ax.set_xlim(0, 46)
        ax.set_ylim(0, 21)
        ax.set_aspect('equal')

        ax.legend()

        plt.show()

    def animate(self, filename='animation'):
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        
        target_inds = self.controls_hist['target_inds'] 
        t = self.states_hist['t']
        v = self.states_hist['v']
        yaw = self.states_hist['yaw']
        d = self.controls_hist['d']
        states = self.states
        measurements = self.measurements        

        def aux_animate(i):
            ax.cla()
            ax.plot(self.x_points, self.y_points, "kx", markersize=10)
            ax.plot(states[:i, 0], states[:i, 1], "-r", label="trajectory")
            ax.plot(measurements[:i, 0], measurements[:i, 1], 'bx', markersize=3, label="measurements")
            ax.plot(self.cx[target_inds[i]], self.cy[target_inds[i]], "xg", label="target")
            plot_car(ax, states[i, 0], states[i, 1], yaw[i], steer=d[i])#, cabcolor="k", truckcolor="k")
            ax.axis("equal")
            ax.grid(True)
            ax.set_title("Time [s]:" + str(round(t[i], 2)) + ", speed [km/h]:" + str(round(v[i] * 3.6, 2)))
            ax.hlines(1, 1, 45, color='k', linestyle='solid', linewidth=1)
            ax.hlines(5, 1, 40, color='k', linestyle='solid', linewidth=1)
            ax.vlines(45, 1, 20, color='k', linestyle='solid', linewidth=1)
            ax.vlines(40, 5, 20, color='k', linestyle='solid', linewidth=1)

            ax.set_xlim(0, 46)
            ax.set_ylim(0, 21)
            ax.set_aspect('equal')

        ani = animation.FuncAnimation(fig, aux_animate, frames=len(t), repeat=False)

        # Save animation with progress bar
        with tqdm(total=len(t)) as pbar:
            ani.save(f'{filename}.gif', writer='Pillow', fps=25, progress_callback=lambda i, n: pbar.update())
            
        plt.close()
