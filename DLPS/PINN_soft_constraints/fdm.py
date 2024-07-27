#%%
import torch
import numpy as np
import matplotlib.pyplot as plt


def get_matrix_A(n, h, nu):
    A = torch.zeros((n, n))
    for i in range(1, n-1):
        A[i,i-1] = 1
        A[i,i] = -(h**2/nu + 2)
        A[i,i+1] = 1
    A[0,0] = 1
    A[-1,-1] = 1
    return A

def get_vector_b(n, h, initial_condition,nu):
    b = torch.zeros(n)
    for i in range(1, n):
        b[i] = (h**2) * torch.exp(torch.tensor((i-1)*h))/nu
    b[0] = initial_condition[0]
    b[-1] = initial_condition[1]
    return b

def solve_heat_equation(n, h, nu, initial_condition):
    A = get_matrix_A(n, h, nu)
    b = get_vector_b(n, h, initial_condition, nu)
    solution = torch.linalg.solve(A, b)
    return solution.view(-1)

n = 100
h = 2/(n-1)
nu = 0.001
initial_condition = [1, 0]
u = solve_heat_equation(n, h, nu, initial_condition)
print(u.shape)

plt.plot(np.linspace(-1, 1, n), u.detach().numpy().flatten())
# %%
