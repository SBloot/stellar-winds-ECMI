import numpy as np
import math
import matplotlib.pyplot as plt

#  Script to calculate mass-loss rate upper limits for a test star.
#  Authors: S. Bloot, R. D. Kavanagh, H. K. Vedantham
#  ASTRON, Nov 2024


from ffa_utils_parker import *

#Relevant constants in Gaussian units
C_c = 2.99792458e10
C_k = 1.380658e-16
C_h = 6.62606885e-27
C_G = 6.67428e-8
C_Msun = 1.99e33
C_Rsun = 6.96e10
C_au = 1.496e13
C_pc = 3.086e18
C_me = 9.1093897e-28
C_mp = 1.6726231e-24
year = 3.15567e7

A=0.09 # Helium fraction


# Define the range of base densities and wind temperatures to test
grid_rho=np.logspace(6, 18, 10000)
grid_temp=np.linspace(0.5,18,100)
frequency=120e6

# We evaluate the limits for three different magnetic field strengths (in Gauss)
B0s=[10,100,1000]

#Radius and mass of the star in solar units
Rstar=1
Mstar=1

all_maxes_iso_10=[]
all_maxes_iso_100=[]
all_maxes_iso_1000=[]


for k in range(len(grid_temp)):
     Mdots_iso=np.zeros(len(grid_rho))
     taus_iso_10=np.zeros(len(grid_rho))
     taus_iso_100=np.zeros(len(grid_rho))
     taus_iso_1000=np.zeros(len(grid_rho))

     # Calculate the velocity profile of the wind
     iso_rgrid, iso_vgrid=velocity_profile_isothermal(Mstar*C_Msun, Rstar*C_Rsun, 100*C_Rsun, 50000, grid_temp[k]*1e6, A=0.09)
     
     second_harmonic=True
     # For each base density, calculate the density profile of the wind
     # and the optical depth along the path of the ECMI emission
     for i in range(len(grid_rho)):
          iso_density, Mdots_iso[i]=density_profile(iso_rgrid, iso_vgrid, grid_rho[i], A=0.09)
          taus_iso_10[i]=optical_depth(iso_density, iso_rgrid, iso_vgrid, grid_temp[k]*1e6,Rstar*C_Rsun, frequency, A=0.09, second_harmonic=second_harmonic, B0= 10)
          taus_iso_100[i]=optical_depth(iso_density, iso_rgrid, iso_vgrid, grid_temp[k]*1e6,Rstar*C_Rsun, frequency, A=0.09, second_harmonic=second_harmonic, B0= 100)
          taus_iso_1000[i]=optical_depth(iso_density, iso_rgrid, iso_vgrid, grid_temp[k]*1e6,Rstar*C_Rsun, frequency, A=0.09, second_harmonic=second_harmonic, B0= 1000)
     # Find the highest mass loss rate for which the optical depth is less than 1
     Mdot_max_10=np.max(Mdots_iso[(taus_iso_10<1)])
     Mdot_max_100=np.max(Mdots_iso[(taus_iso_100<1)])
     Mdot_max_1000=np.max(Mdots_iso[(taus_iso_1000<1)])
     all_maxes_iso_10.append(Mdot_max_10)
     all_maxes_iso_100.append(Mdot_max_100)
     all_maxes_iso_1000.append(Mdot_max_1000)

np.save('test_star_iso_10', np.array([all_maxes_iso_10]))
np.save('test_star_iso_100', np.array([all_maxes_iso_100]))
np.save('test_star_iso_1000', np.array([all_maxes_iso_1000]))