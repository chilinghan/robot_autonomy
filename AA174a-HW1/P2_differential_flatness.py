import math
import typing as T

import numpy as np
from numpy import linalg
from scipy.integrate import cumtrapz  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from utils import save_dict, maybe_makedirs

class State:
    def __init__(self, x: float, y: float, V: float, th: float) -> None:
        self.x = x
        self.y = y
        self.V = V
        self.th = th

    @property
    def xd(self) -> float:
        return self.V*np.cos(self.th)

    @property
    def yd(self) -> float:
        return self.V*np.sin(self.th)

def basis(t):
    return np.array([1, t, t**2, t**3])

def basis_dot(t):
    return np.array([0, 1, 2*t, 3*t**2])

def basis_ddot(t):
    return np.array([0, 0, 2, 6*t])

def compute_traj_coeffs(initial_state: State, final_state: State, tf: float) -> np.ndarray:
    """
    Inputs:
        initial_state (State)
        final_state (State)
        tf (float) final time
    Output:
        coeffs (np.array shape [8]), coefficients on the basis functions

    Hint: Use the np.linalg.solve function.
    """
    ########## Code starts here ##########
    M = np.array([basis(0), basis_dot(0), basis(tf), basis_dot(tf)])
    M = np.kron(np.eye((2)), M)

    flat_outputs = np.array([[initial_state.x],
                              [initial_state.xd],
                              [final_state.x],
                              [final_state.xd],
                              [initial_state.y],
                              [initial_state.yd],
                              [final_state.y],
                              [final_state.yd]])
    coeffs = linalg.solve(M, flat_outputs)
    ########## Code ends here ##########
    return coeffs

def compute_traj(coeffs: np.ndarray, tf: float, N: int) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Inputs:
        coeffs (np.array shape [8]), coefficients on the basis functions
        tf (float) final_time
        N (int) number of points
    Output:
        t (np.array shape [N]) evenly spaced time points from 0 to tf
        traj (np.array shape [N,7]), N points along the trajectory, from t=0
            to t=tf, evenly spaced in time
    """
    t = np.linspace(0, tf, N) # generate evenly spaced points from 0 to tf
    traj = np.zeros((N, 7))
    ########## Code starts here ##########
    Mx, My = np.zeros(8), np.zeros(8)
    for i in range(N):
        time = t[i]

        Mx[:4], My[4:] = basis(time), basis(time)
        x, y = np.dot(Mx, coeffs), np.dot(My, coeffs)
        traj[i][0] = x
        traj[i][1] = y

        Mx[:4], My[4:] = basis_dot(time), basis_dot(time)
        x_dot, y_dot = np.dot(Mx, coeffs), np.dot(My, coeffs)
        traj[i][3] = x_dot
        traj[i][4] = y_dot

        theta = np.atan2(y_dot, x_dot)
        traj[i][2] = theta

        Mx[:4], My[4:] = basis_ddot(time), basis_ddot(time)
        x_ddot, y_ddot = np.dot(Mx, coeffs), np.dot(My, coeffs)
        traj[i][5] = x_ddot
        traj[i][6] = y_ddot
    ########## Code ends here ##########
    return t, traj

def compute_controls(traj: np.ndarray) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Input:
        traj (np.array shape [N,7])
    Outputs:
        V (np.array shape [N]) V at each point of traj
        om (np.array shape [N]) om at each point of traj
    """
    ########## Code starts here ##########
    V = np.zeros(len(traj))
    om = np.zeros(len(traj))
    for i in range(len(traj)):
        V[i] = linalg.norm([traj[i][3], traj[i][4]])
        x_dot = traj[i][3]
        y_dot = traj[i][4]
        x_ddot = traj[i][5]
        y_ddot = traj[i][6]
        om[i] = (y_ddot*x_dot - x_ddot*y_dot) * 1.0 / V[i]**2
    ########## Code ends here ##########

    return V, om

if __name__ == "__main__":
    # Constants
    tf = 25.

    # time
    dt = 0.005
    N = int(tf/dt)+1
    t = dt*np.array(range(N))

    # Initial conditions
    s_0 = State(x=0, y=0, V=0.5, th=-np.pi/2)

    # Final conditions
    s_f = State(x=5, y=5, V=0.5, th=-np.pi/2)

    coeffs = compute_traj_coeffs(initial_state=s_0, final_state=s_f, tf=tf)
    t, traj = compute_traj(coeffs=coeffs, tf=tf, N=N)
    V,om = compute_controls(traj=traj)

    maybe_makedirs('plots')

    # Plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(traj[:,0], traj[:,1], 'k-',linewidth=2)
    plt.grid(True)
    plt.plot(s_0.x, s_0.y, 'go', markerfacecolor='green', markersize=15)
    plt.plot(s_f.x, s_f.y, 'ro', markerfacecolor='red', markersize=15)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title("Path (position)")
    plt.axis([-1, 6, -1, 6])

    ax = plt.subplot(1, 2, 2)
    plt.plot(t, V, linewidth=2)
    plt.plot(t, om, linewidth=2)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc="best")
    plt.title('Original Control Input')
    plt.tight_layout()

    plt.savefig("plots/differential_flatness.png")
    plt.show()
