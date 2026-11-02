
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

C_A0 = 2.0  # mol/L
tau = 5.0   # min
k = 0.15    # 1/min

def reactor(t, y):
    C_A = y[0]
    dC_A_dt = (C_A0 - C_A)/tau - k*C_A
    return [dC_A_dt]

time = np.linspace(0, 30, 200)

solution = solve_ivp(
    reactor,
    (0, 30),
    [0.0],
    t_eval=time
)

plt.plot(solution.t, solution.y[0])
plt.xlabel("Time (min)")
plt.ylabel("Concentration (mol/L)")
plt.title("Dynamic CSTR Simulation")
plt.show()
