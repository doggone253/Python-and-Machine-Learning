import torch
import matplotlib.pyplot as plt

#Enable gradients on the vector to track operations for the plot
T = torch.linspace(250.0, 500.0, 500, requires_grad=True, dtype=torch.float64)

#Constants
R = 8.314
Ea = 45000.0
A = 1.0e6

#Compute the Arrhenius vector
k = A * torch.exp(-Ea / (R * T))

#Pass torch.ones_like(k) to compute individual derivatives (dk/dT) for each element
k.backward(torch.ones_like(k))

#Extract data for plotting
T_plot = T.detach().numpy()
k_plot = k.detach().numpy()
dk_dT_plot = T.grad.detach().numpy()
fig, ax1 = plt.subplots(figsize=(8, 5))

#Plotting results
color = '#1f77b4'
ax1.set_xlabel('Temperature (K)')
ax1.set_ylabel('Rate Constant k', color=color)
ax1.plot(T_plot, k_plot, color=color, label='Rate Constant (k)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--')
ax2 = ax1.twinx()
color = '#ff7f0e'
ax2.set_ylabel('Sensitivity dk/dT (PyTorch AutoGrad)', color=color)
ax2.plot(T_plot, dk_dT_plot, color=color, label='dk/dT')
ax2.tick_params(axis='y', labelcolor=color)
plt.title('Arrhenius Kinetics: Dynamic PyTorch AutoGrad Tracking')
fig.tight_layout()
plt.show()