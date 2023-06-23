import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from IPython.display import display, HTML, Image

from tqdm.auto import tqdm
from scipy.linalg import eigh
from scipy.stats import chi2, norm, multivariate_normal

cornflowerblue_alpha = (0.39215686274509803, 0.5843137254901961, 0.9294117647058824, 0.3)

def rgba_to_rgb(rgba, bg=(1, 1, 1)):
    r, g, b, a = rgba
    bg_r, bg_g, bg_b = bg
    
    # Blend the RGBA color with the background color
    r = r * a + bg_r * (1 - a)
    g = g * a + bg_g * (1 - a)
    b = b * a + bg_b * (1 - a)

    return (r, g, b)

def conf_ellipse(ax, center, covariance, alpha=0.95):
    chi2_quantile = chi2.ppf(alpha, 2)
    eigvals, eigvecs = eigh(covariance)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(eigvals[0] * chi2_quantile), 2 * np.sqrt(eigvals[1] * chi2_quantile)
    ell = Ellipse(xy=center, width=width, height=height, angle=angle, fill=True, facecolor=rgba_to_rgb(cornflowerblue_alpha))
    ax.add_artist(ell)
    
def plot_trajectory(ax, states, cov_estimates, label, color='cornflowerblue', alpha=0.95):
    ax.plot(states[:, 0], states[:, 1], label=label, color=color)
    for i in range(len(states)):
        conf_ellipse(ax, states[i, :2,], cov_estimates[i, :2, :2], alpha=alpha)


def visualize_filter(states, measurements, state_estimates, cov_estimates, particle_history=None, title="Title"):

        fig, ax = plt.subplots(1, 1, figsize=(16, 4), sharey=True)

        if particle_history is not None:
            n = len(particle_history)
            blues = plt.get_cmap('Blues')(np.linspace(0.2, 1.0, n))
            for k in range(n):
                ax.scatter(particle_history[k, :, 0], particle_history[k, :, 1], s=1, color=blues[k])

        ax.plot(states[0, 0], states[0, 1], 'x', color='k', label="Start")
        ax.plot(states[:, 0], states[:, 1], '--', color='r', label="True trajectory")
        ax.plot(measurements[:, 0], measurements[:, 1], '.', color='orange', label="Noisy observations")

        plot_trajectory(ax, state_estimates, cov_estimates, label=title)

        # Show the MSE on the plot in upper right corner
        ax.text(1.00, 1.05, "MSE: {:.2f}".format(np.mean((states[:, :2] - state_estimates[:, :2])**2)),
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

        ax.hlines(1, 1, 45, color='k', linestyle='solid', linewidth=1)
        ax.hlines(5, 1, 40, color='k', linestyle='solid', linewidth=1)
        ax.vlines(45, 1, 20, color='k', linestyle='solid', linewidth=1)
        ax.vlines(40, 5, 20, color='k', linestyle='solid', linewidth=1)

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.legend()

        ax.set_title(title)

        plt.tight_layout()
        plt.show()


def visualize_filter_and_smoother(states, measurements, state_estimates, cov_estimates, state_estimates_smoothed, cov_estimates_smoothed, particle_history=None, variant=""):

        fig, ax = plt.subplots(1, 2, figsize=(16, 4), sharey=True)

        if particle_history is not None:
            n = len(particle_history)
            blues = plt.get_cmap('Blues')(np.linspace(0.2, 1.0, n))
            for k in range(n):
                ax[0].scatter(particle_history[k, :, 0], particle_history[k, :, 1], s=1, color=blues[k])
                ax[1].scatter(particle_history[k, :, 0], particle_history[k, :, 1], s=1, color=blues[k])

        for i in range(2):
                ax[i].plot(states[0, 0], states[0, 1], 'x', color='k', label="Start")
                ax[i].plot(states[:, 0], states[:, 1], '--', color='r', label="True trajectory")
                ax[i].plot(measurements[:, 0], measurements[:, 1], '.', color='orange', label="Noisy observations")

        plot_trajectory(ax[0], state_estimates, cov_estimates, label=f"{variant} Kalman Filter x")
        plot_trajectory(ax[1], state_estimates_smoothed, cov_estimates_smoothed, label=f"{variant} RTS Smoother")

        # Show the MSE on the plot in upper right corner
        ax[0].text(1.00, 1.05, "MSE: {:.2f}".format(np.mean((states[:, :2] - state_estimates[:, :2])**2)),
                horizontalalignment='right', verticalalignment='top', transform=ax[0].transAxes)
        ax[1].text(1.00, 1.05, "MSE: {:.2f}".format(np.mean((states[:, :2] - state_estimates_smoothed[:, :2])**2)),
                horizontalalignment='right', verticalalignment='top', transform=ax[1].transAxes)
        
        for k in range(2):
                ax[k].hlines(1, 1, 45, color='k', linestyle='solid', linewidth=1)
                ax[k].hlines(5, 1, 40, color='k', linestyle='solid', linewidth=1)
                ax[k].vlines(45, 1, 20, color='k', linestyle='solid', linewidth=1)
                ax[k].vlines(40, 5, 20, color='k', linestyle='solid', linewidth=1)

                ax[k].set_xlabel('x')
                ax[k].set_ylabel('y')

                ax[k].legend()

        ax[0].set_title(f"{variant} Kalman Filter")
        ax[1].set_title(f"{variant} RTS Smoother")

        plt.tight_layout()
        plt.show()


class PlotAnimation:

    def __init__(self, states, measurements, state_estimates, cov_estimates, state_estimates_smoothed, cov_estimates_smoothed, name="animation"):
        self.states = states
        self.measurements = measurements
        self.state_estimates = state_estimates.copy()
        self.cov_estimates = cov_estimates.copy()
        self.state_estimates_smoothed = state_estimates_smoothed
        self.cov_estimates_smoothed = cov_estimates_smoothed
        self.name = name

        self.fig = plt.figure(figsize=(16, 6))
        self.gs = self.fig.add_gridspec(
            2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
            left=0.1, right=0.9, bottom=0.1, top=0.9,
            wspace=0.05, hspace=0.05
        )

        self.ax = self.fig.add_subplot(self.gs[1, 0])
        self.ax.plot(states[0, 0], states[0, 1], 'x', color='k', label="Start")
        self.ax.plot(states[:, 0], states[:, 1], '--', color='r', label="True trajectory")

        self.filter_points, = self.ax.plot([], [], '.', color='orange',label="Noisy observations (original)")
        self.smoother_points, = self.ax.plot([], [], '.', color='red', label="Noisy observations (after smoothing)")

        self.ax.hlines(1, 1, 45, color='k', linestyle='solid', linewidth=1)
        self.ax.hlines(5, 1, 40, color='k', linestyle='solid', linewidth=1)
        self.ax.vlines(45, 1, 20, color='k', linestyle='solid', linewidth=1)
        self.ax.vlines(40, 5, 20, color='k', linestyle='solid', linewidth=1)

        self.ax_histx = self.fig.add_subplot(self.gs[0, 0], sharex=self.ax)
        self.ax_histy = self.fig.add_subplot(self.gs[1, 1], sharey=self.ax)
        self.line, = self.ax.plot([], [], '-', color='black', label="Estimated trajectory")
        self.mse_text = self.fig.text(0.75, 0.75, '', transform=self.fig.transFigure, fontsize=12)

        self.ax_histx.tick_params(axis="x", labelbottom=False)
        self.ax_histy.tick_params(axis="y", labelleft=False)

    def update_trajectory(self, frame):
        self.line.set_xdata(self.state_estimates[:frame, 0])
        self.line.set_ydata(self.state_estimates[:frame, 1])

        if frame > 0:
            mse = np.mean((self.states[:frame, :2] - self.state_estimates[:frame, :2]) ** 2)
            self.mse_text.set_text(f'MSE: {mse:.2f}')

    def update_histogram(self, frame):
        self.ax_histx.clear()
        self.ax_histy.clear()
        self.ax_histx.set_ylim(0, 1.5)
        self.ax_histy.set_xlim(0, 1.5)

        tmp_x = np.linspace(
            self.state_estimates[frame, 0] - 3 * np.sqrt(self.cov_estimates[frame, 0, 0]),
            self.state_estimates[frame, 0] + 3 * np.sqrt(self.cov_estimates[frame, 0, 0]), 100
        )

        self.ax_histx.plot(
            tmp_x, norm.pdf(tmp_x, self.state_estimates[frame, 0], np.sqrt(self.cov_estimates[frame, 0, 0])), color='black')

        tmp_y = np.linspace(
            self.state_estimates[frame, 1] - 3 * np.sqrt(self.cov_estimates[frame, 1, 1]),
            self.state_estimates[frame, 1] + 3 * np.sqrt(self.cov_estimates[frame, 1, 1]), 100
        )

        self.ax_histy.plot(
            norm.pdf(tmp_y, self.state_estimates[frame, 1], np.sqrt(self.cov_estimates[frame, 1, 1])), tmp_y, color='black')

    def init(self):
        self.ax.set_xlim(int(np.min(self.measurements[:, 0]) - 5), int(np.max(self.measurements[:, 0] + 5)))
        self.ax.set_ylim(int(np.min(self.measurements[:, 1]) - 5), int(np.max(self.measurements[:, 1] + 5)))
        self.ax_histx.set_ylim(0, 1.0)
        self.ax_histy.set_xlim(0, 1.0)
        return self.ax,

    def update_filter(self, frame):
                
        conf_ellipse(self.ax, self.state_estimates[frame, :2], self.cov_estimates[frame, :2, :2])

        self.update_trajectory(frame)
        self.update_histogram(frame)

        #self.ax.plot(self.measurements[frame, 0], self.measurements[frame, 1], '.', color='orange',label="Noisy observations")
        self.filter_points.set_data(self.measurements[:frame, 0], self.measurements[:frame, 1])
        
        return self.ax,

    def update_smoother(self, frame):
        frame = frame - len(self.state_estimates)

        self.state_estimates[-frame:] = self.state_estimates_smoothed[-frame:]
        self.cov_estimates[-frame:] = self.cov_estimates_smoothed[-frame:]

        conf_ellipse(self.ax, self.state_estimates[frame, :2], self.cov_estimates[frame, :2, :2])

        self.update_trajectory(len(self.state_estimates) - 1)

        self.filter_points.set_data(self.measurements[:-frame, 0], self.measurements[:-frame, 1])
        self.smoother_points.set_data(self.measurements[-frame:, 0], self.measurements[-frame:, 1])

        frame = -frame - 1
        self.update_histogram(frame)

        return self.ax,

    def update(self, frame):
        if frame < len(self.state_estimates):
            ax = self.update_filter(frame)
        else:
            ax = self.update_smoother(frame - len(self.state_estimates))

        return ax,

    def animate(self):
        with tqdm(total=2 * len(self.state_estimates)) as pbar:
            ani = animation.FuncAnimation(
                self.fig, self.update, frames=range(0, 2 * len(self.state_estimates)), init_func=self.init
            )
            ani.save(f'{self.name}.gif', writer='Pillow', fps=20, progress_callback=lambda i, n: pbar.update())

        plt.close()


class PlotParzenAnimation:

    def __init__(self, states, measurements, state_estimates, particle_history, particle_cov_history, name="animation"):
        self.states = states
        self.measurements = measurements
        self.state_estimates = state_estimates
        self.particle_history = particle_history
        self.particle_cov_history = particle_cov_history
        self.name = name

        self.fig = plt.figure(figsize=(16, 6))
        self.gs = self.fig.add_gridspec(
            2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
            left=0.1, right=0.9, bottom=0.1, top=0.9,
            wspace=0.05, hspace=0.05
        )

        self.ax = self.fig.add_subplot(self.gs[1, 0])
        self.ax.plot(states[0, 0], states[0, 1], 'x', color='k', label="Start")
        self.ax.plot(states[:, 0], states[:, 1], '--', color='r', label="True trajectory")

        self.filter_points, = self.ax.plot([], [], '.', color='orange',label="Noisy observations (original)")

        self.ax.hlines(1, 1, 45, color='k', linestyle='solid', linewidth=1)
        self.ax.hlines(5, 1, 40, color='k', linestyle='solid', linewidth=1)
        self.ax.vlines(45, 1, 20, color='k', linestyle='solid', linewidth=1)
        self.ax.vlines(40, 5, 20, color='k', linestyle='solid', linewidth=1)

        self.ax_histx = self.fig.add_subplot(self.gs[0, 0], sharex=self.ax)
        self.ax_histy = self.fig.add_subplot(self.gs[1, 1], sharey=self.ax)
        self.line, = self.ax.plot([], [], '-', color='black', label="Estimated trajectory", alpha=0.5)
        self.mse_text = self.fig.text(0.75, 0.75, '', transform=self.fig.transFigure, fontsize=12)

        self.ax_histx.tick_params(axis="x", labelbottom=False)
        self.ax_histy.tick_params(axis="y", labelleft=False)

        self.contourf = None

    def update_trajectory(self, frame):
        self.line.set_xdata(self.state_estimates[:frame, 0])
        self.line.set_ydata(self.state_estimates[:frame, 1])

        if frame > 0:
            mse = np.mean((self.states[:frame, :2] - self.state_estimates[:frame, :2]) ** 2)
            self.mse_text.set_text(f'MSE: {mse:.2f}')

    def update_histogram(self, frame):
        self.ax_histx.clear()
        self.ax_histy.clear()
        self.ax_histx.set_ylim(0, .5)
        self.ax_histy.set_xlim(0, .5)

        X, Y = np.linspace(-5, 50, 100), np.linspace(-5, 50, 100)

        pdf_x = sum([norm.pdf(X, loc=mean[0], scale=np.sqrt(cov[0, 0])) for mean, cov in zip(self.particle_history[frame], self.particle_cov_history[frame])])
        pdf_y = sum([norm.pdf(Y, loc=mean[1], scale=np.sqrt(cov[1, 1])) for mean, cov in zip(self.particle_history[frame], self.particle_cov_history[frame])])
        #print(particle_history[frame][:,0].mean(), particle_history[frame][:,1].mean())

        pdf_x = pdf_x / np.sum(pdf_x)
        pdf_y = pdf_y / np.sum(pdf_y)

        self.ax_histx.plot(X, pdf_x)
        self.ax_histy.plot(pdf_y, Y)


    def init(self):
        self.ax.set_xlim(int(np.min(self.measurements[:, 0]) - 5), int(np.max(self.measurements[:, 0] + 5)))
        self.ax.set_ylim(int(np.min(self.measurements[:, 1]) - 5), int(np.max(self.measurements[:, 1] + 5)))
        self.ax_histx.set_ylim(0, .5)
        self.ax_histy.set_xlim(0, .5)
        return self.ax,

    def update_filter(self, frame):

        self.update_trajectory(frame)
        self.update_histogram(frame)
                
        X, Y = np.meshgrid(np.linspace(-5, 50, 100), np.linspace(-5, 50, 100))
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        pdf = sum([multivariate_normal.pdf(pos, mean=mean[:2], cov=cov[:2, :2]) for mean, cov in zip(self.particle_history[frame], self.particle_cov_history[frame])])

        self.contourf = self.ax.contourf(X, Y, pdf, levels=10, cmap='Blues')

        self.filter_points.set_data(self.measurements[:frame, 0], self.measurements[:frame, 1])
        
        return self.ax,

    def update(self, frame):
        
        ax = self.update_filter(frame)

        return ax,

    def animate(self):
        with tqdm(total=len(self.state_estimates)) as pbar:
            ani = animation.FuncAnimation(
                self.fig, self.update, frames=range(0, len(self.state_estimates)), init_func=self.init
            )
            ani.save(f'{self.name}.gif', writer='Pillow', fps=10, progress_callback=lambda i, n: pbar.update())

        plt.close()


def show_animation(trajectory, gif_path="animations/car_trajectory"):
    def display_animation(gif_path, style='style="max-width:100%;"'):
        display(HTML(f'<img src="{gif_path}" {style}>'))

    if not os.path.exists(gif_path + ".gif"):
        trajectory.animate(filename=gif_path)

    display_animation(gif_path + ".gif")

    plt.close()

def show_filter_animation(animation, gif_path):
    if not os.path.exists(gif_path + ".gif"):
        animation.animate()

    # Clear the redundant plot
    animation.ax.clear()

    # Display the animation without any additional plot
    display(HTML(f'<img src="{gif_path}.gif">'))

    plt.close()



### Unscented Transform ###

def illustrate_unscented_transform(ut_dict, xs, ys, xs_nl, ys_nl, title='title'):

    """
    Illustrates the unscented transform

    Parameters
    ----------
    ut_dict : dict
        Dictionary containing the unscented transform
    xs : numpy array
        x coordinates of the points
    ys : numpy array
        y coordinates of the points
    xs_nl : numpy array
        x coordinates of the transformed points
    ys_nl : numpy array
        y coordinates of the transformed points
    title : str
        Title of the plot
    """

    def get_ellipsis(cov):

        eigvals, eigvecs = np.linalg.eig(cov)
        eigvals = 2 * np.sqrt(eigvals)

        theta = np.linspace(0, 2*np.pi, 1000);
        ellipsis = (eigvals[None,:] * eigvecs) @ [np.sin(theta), np.cos(theta)]

        return ellipsis

    methods = ['merwe', 'julier', 'simplex']

    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    alpha = .4

    for i, method in enumerate(methods):
        mean = ut_dict[method]['mean']
        cov = ut_dict[method]['cov']
        sigmas = ut_dict[method]['sigmas']
        sigmas_f = ut_dict[method]['sigmas_f']

        ax[i].scatter(xs_nl, ys_nl, c='g', s=30, alpha=alpha, label='Transformed Points')
        ax[i].scatter(xs, ys, c='b', s=30, alpha=alpha, label='Original Points')

        # transformed points
        ax[i].scatter(xs_nl.mean(), ys_nl.mean(), c='purple', s=50, label='Mean of Transformed Points', marker='x')

        ellipsis = get_ellipsis(np.cov([xs_nl, ys_nl]))
        ax[i].plot(xs_nl.mean() + ellipsis[0,:], ys_nl.mean() + ellipsis[1,:], c='purple', label='Covariance of Transformed Points')

        # sigma points
        ax[i].scatter(sigmas[:,0], sigmas[:,1], c='r', s=50, label='Sigma Points')
        ax[i].scatter(sigmas_f[:,0], sigmas_f[:,1], c='k', s=50, label='Transformed Sigma Points')

        # UT estimate
        ax[i].scatter(mean[0], mean[1], c='y', s=100, label='UT Mean', marker='*')

        ellipsis = get_ellipsis(cov)
        ax[i].plot(mean[0] + ellipsis[0,:], mean[1] + ellipsis[1,:], c='y', label='UT Covariance')

        ax[i].set_title(method, fontsize=16)
        ax[i].set_xlabel('x')
        ax[i].set_ylabel('y')

    ax[0].legend(fontsize=12)

    fig.suptitle(title, fontsize=24)

    plt.tight_layout()
    plt.show()