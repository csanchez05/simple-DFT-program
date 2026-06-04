import numpy as np

#physical params
a = 10.0     # Lattice Spacing
E_cut = 20.0 # plane-wave energy cutoff
Num_k = 10   # num of k-points

#reciprocal lattice vectors
m_max = np.sqrt(2 * E_cut) * a / (2 * np.pi) # max G vector
m_max = int(m_max)

m = np.arange(-m_max, m_max +1) # Creating the reciprocal lattice vectors
G = (2* np.pi) * (m/a)

# real-space grid
N_grid = 4 * m_max + 1 # grid size
x = np.linspace(0, a, N_grid, endpoint=False) # Real space grid from 0 to a

# kpoints in the first brillouin zone
k_points = np.linspace(-np.pi/a, np.pi/a, Num_k, endpoint=False)

def get_V_ext(x, a, alpha = 1.0):
    dx = np.abs(x - a/2) #distance to center
    dx = np.minimum(dx, a - dx) #applying minimum image convention
    return -1.0 / np.sqrt(dx**2 + alpha)

def get_V_xc(n): #temporary
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
    v_hartree_array = real_part * (x[1] - x[0])
    
    return v_hartree_array

def get_kinetic_matrix(k_point, G):
    N_G = len(G) #Num of plane waves
    T_matrix = np.zeros((N_G, N_G)) #Creates N_G x N_G matrix with all zeros
    for i in range(N_G):
        T_matrix[i, i] = 0.5 * (k_point + G[i])**2
    ## This fills only the diagonal terms in the matrix
    ## (This is a result from projection of T operator onto
    ## our plane wave basis set (orthogonal))
    return T_matrix

k_test = k_points[0]
T_test = get_kinetic_matrix(k_test, G)
print("Kinetic matrix shape:")
print(T_test.shape)
print("Kinetic matrix:")
print(T_test)



print("Setup Complete")
print(f"Number of plane waves: {len(G)}")
print(f"Real space grid size: {len(x)}")