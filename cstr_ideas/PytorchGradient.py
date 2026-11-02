
import torch

T = torch.tensor(
    350.0,
    requires_grad=True
)

R = 8.314
Ea = 45000.0
A = 1.0e6

# Arrhenius equation
k = A * torch.exp(-Ea / (R*T))

# Calculate dk/dT
k.backward()

print("Rate constant:", k.item())
print("dk/dT:", T.grad.item())
