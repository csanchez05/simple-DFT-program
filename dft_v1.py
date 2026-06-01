import numpy as np

#physical params
a = 10.0
E_cut = 20.0
Num_k = 10

#reciprocal lattice vectors
m_max = np.sqrt(2 * E_cut) * a / (2 * np.pi)
m_max = int(m_max)
m = np.arange(-m_max, m_max +1)
G = (2* np.pi /a) * m 

# real-space grid
N_grid = 4 * m_max + 1
x = np.linspace(0, a, N_grid, endpoint=False)

# kpoints in the first brillouin zone
k_points = np.linspace(-1/a, 1/a, Num_k, endpoint=False) * (np.pi)

def get_V_ext(x, a, alpha = 1.0):
    dx = np.abs(x - a/2) #distance to center
    dx = np.minimum(dx, a - dx) #applying minimum image convention
    return -1.0 / np.sqrt(dx**2 + alpha)
    pass

def get_V_xc(n):
    n_safe = np.maximum(n, 1e-12) #prevent division by zero
    lda_xc = -(3.0 / np.pi * n_safe)**(1/3)
    return lda_xc
    pass

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
    pass


print("Setup Complete")
print(f"Number of plane waves: {len(G)}")
print(f"Real space grid size: {len(x)}")