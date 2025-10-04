import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
from scipy.fft import fft, ifft, fftfreq
from scipy.special import hankel1 

# Constants
c0 = 1500         # Reference sound speed (m/s)
f0 = 200          # Center frequency (Hz)
omega0 = 2 * np.pi * f0
k0 = omega0 / c0
zs = 5       # Source depth (m)
W = 38     # Beamwidth
A = 1      # Source amplitude

print("k0",k0)

F = 512
dt = 1.0 / f0
t = np.arange(F) * dt
print("dt",dt)
print("t",t)

r_max = 20000.0             # 20 km
dr    = 100.0               # range step (m)
Nr    = int(np.round(r_max / dr)) + 1
r     = np.arange(Nr) * dr

# Retarded time = t - R/c0
tret = t[None, :] - (r[:, None] / c0)
print("Retard time", tret)

# Grid parameters
z_max = 10000      # Depth (m)
r_max = 20000     # Range (m)
dz = 0.5
dr = 10

nz = int(z_max / dz)
nr = int(r_max / dr)

z = np.linspace(0, z_max, nz)
r = np.linspace(0, r_max, nr)

# Index of refraction 
n_profile = np.ones((nr, nz)) 

# Gaussian source at r=0
phiz = A * np.exp(-(z - zs) ** 2 / W ** 2)
psi = np.zeros((nr, nz), dtype=complex)
psi[0, :] = phiz / np.max(np.abs(phiz))  # Normalization

# Vertical wavenumber s (kz)
kz = 2 * pi * fftfreq(nz, d=dz)
s2 = kz**2


for i in range(1, nr):
    
    psi_fft = fft(psi[i-1, :])
    
    psi_fft *= np.exp(-1j * dr * s2 / (2 * k0))
    
    psi_phys = ifft(psi_fft)
    
    psi[i, :] = psi_phys * np.exp(1j * k0 / 2 * (n_profile[i, :]**2 - 1) * dr)


#  p(r,z) = ψ(r,z) H(r)

R = np.tile(r[:, np.newaxis], (1, nz))  # Create 2D range array
H_r = hankel1(0, k0 * R)                # Hankel function for each range
p = psi * H_r                           

# Convert to dB
p = np.abs(psi)
p_dB = 20 * np.log10(p / np.max(p))




Tmin, Tmax = -0.1, 0.8

# Plot
plt.figure(figsize=(10, 5))
plt.imshow(p_dB, extent=[0, r_max/1000, z_max, 0], aspect='auto', cmap='jet', vmin=-60, vmax=0)
plt.xlabel('Retarded time (s)')
plt.ylabel('Range (km)')
plt.xlim(Tmin, Tmax)
plt.ylim(5, 20) 
plt.colorbar(label='dB')
plt.tight_layout()
plt.show()


