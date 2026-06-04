import numpy as np
import matplotlib.pyplot as plt

# Parameters
a = 10.0     # Lattice Spacing
E_cut = 20.0 # plane-wave energy cutoff
Num_k = 21   # num of k-points

# Basis Setup 
m_max = np.sqrt(2 * E_cut) * a / (2 * np.pi) # max G vector
m_max = int(m_max)

m = np.arange(-m_max, m_max +1) # Creating the reciprocal lattice vectors
G = (2* np.pi) * (m/a)

# real-space grid
N_grid = 4 * m_max + 1 # grid size
x = np.linspace(0, a, N_grid, endpoint=False) # Real space grid from 0 to a

# kpoints in the first brillouin zone
k_points = np.linspace(-np.pi/a, np.pi/a, Num_k, endpoint=False)



# Defining Potential & Kinetic Energy Functions
def get_V_ext(x, a, alpha = 1.0):
    dx = np.abs(x - a/2) #distance to center
    dx = np.minimum(dx, a - dx) #applying minimum image convention
    return -1.0 / np.sqrt(dx**2 + alpha)

def get_V_xc(n): #temporary exchange potential
    n_safe = np.maximum(n, 1e-12) #prevent division by zero
    lda_xc = -(3.0 / np.pi * n_safe)**(1/3)
    return lda_xc

def get_V_hartree(n, x, a, alpha=1.0):
    dx_periodic = np.minimum(x, a - x)
    v_interaction = 1.0 / np.sqrt(dx_periodic**2 + alpha)

    #convolution theorem using fast fourier transforms
    n_fft = np.fft.fft(n)
    v_fft = np.fft.fft(v_interaction)

    inverse_fft_step = np.fft.ifft(n_fft * v_fft)
    real_part = np.real(inverse_fft_step)
    dx = x[1] - x[0] 
    v_hartree_array = real_part * dx
    
    return v_hartree_array

def get_potential_matrix(V_x, x, G, a):
    N_G = len(G)
    dx = x[1] - x[0]

    V_matrix = np.zeros((N_G, N_G), dtype=complex)
    
    for i in range(N_G):
        for j in range(N_G):
            phase = np.exp(1j * (G[j] - G[i]) * x)
            #below is a numerical approximation for the potential energy matrix
            V_matrix[i, j] = (dx/a) * np.sum(V_x * phase) 

    return V_matrix

def get_kinetic_matrix(k_point, G):
    N_G = len(G) #Num of plane waves
    T_matrix = np.zeros((N_G, N_G)) #Creates N_G x N_G matrix with all zeros

    for i in range(N_G):
        T_matrix[i, i] = 0.5 * (k_point + G[i])**2
    ## This fills only the diagonal terms in the matrix
    ## (This is a result from projection of T operator onto
    ## our plane wave basis set (orthogonal))
    return T_matrix

def reconstruct_psi(coeffs, k_point, G, x, a):
    psi_x = np.zeros(len(x), dtype=complex) #Creates empty wavefunction in real space grid.
    for i in range(len(G)): 
        psi_x += coeffs[i] * np.exp(1j * (k_point + G[i]) * x) / np.sqrt(a) # adds one plane-wave component
    return psi_x



# Test / Temporary Hamiltonian

k_test = k_points[0]

T_test = get_kinetic_matrix(k_test, G)

V_ext_x = get_V_ext(x, a)
V_ext_matrix = get_potential_matrix(V_ext_x, x, G, a)
#Force hermitian
V_ext_matrix = 0.5 * (V_ext_matrix + V_ext_matrix.conj().T)

H_test = T_test + V_ext_matrix

eigenvalues, eigenvectors = np.linalg.eigh(H_test)

free_eigenvalues, free_eigenvectors = np.linalg.eigh(T_test)

print("Setup Complete")
print(f"m_max: {m_max}")
print(f"Number of plane waves: {len(G)}")
print(f"Real-space grid size: {len(x)}")

print("Hamiltonian shape:")
print(H_test.shape)

print("Free-electron eigenvalues:")
print(free_eigenvalues)

print("Eigenvalues with external potential:")
print(eigenvalues)

print("Energy shifts:")
print(eigenvalues - free_eigenvalues)


#2D array of our energy E(k)
all_energies = []

for k_point in k_points:
    T = get_kinetic_matrix(k_point, G)
    H = T + V_ext_matrix
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    all_energies.append(eigenvalues)

#Here we represent all_energies = [k_index, band_index]
#This means, for each k_point, we store the eigenvalues
all_energies = np.array(all_energies)

print("All energies shape:")
print(all_energies.shape)

print("First band energies:")
print(all_energies[:, 0]) #First band energy for every k-point

print("k-points and first band energies:")
for i in range(len(k_points)):
    print(k_points[i], all_energies[i, 0])



num_bands_to_plot = 5

for band_index in range(num_bands_to_plot):
    plt.plot(k_points, all_energies[:, band_index])
plt.xlabel("k")
plt.ylabel("Energy")
plt.title("1D Soft-Coulomb Hydrogen Chain: One-Electron Bands")
plt.grid(True)
plt.show()



