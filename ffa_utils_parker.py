#  Utility functions to calculate the free-free absorption of ECMI emission in
#  a spherically symmetric stellar wind (Parker's model)
#  Authors: S. Bloot, R. D. Kavanagh, H. K. Vedantham
#  ASTRON, Nov 2024


import numpy as np
from scipy.interpolate import RegularGridInterpolator as interp2
from scipy.optimize import fsolve

# Relevant constants in Gaussian units
C_k = 1.380658e-16
C_h = 6.62606885e-27
C_G = 6.67428e-8
C_Msun = 1.99e33
C_Rsun = 6.96e10
C_me = 9.1093897e-28
C_mp = 1.6726231e-24
C_e = 4.8032068e-10
C_Ry = 2.17987e-11
year = 3.15567e7


def mean_molecular_weight(A): 
     # The mean weight of a particle in a fully ionised plasma.
     # A is the helium fraction of the plasma (N_He/N_H)
     mu = ((C_me/C_mp)*(1+2*A)+(1+4*A))/(2+3*A)
     return mu 

def Z2_from_A(A):
     # Calculate Z2 in the absorption coefficient given the helium fraction of the plasma. 
     # This factor assumes the density in the coefficient is the proton density.
     # A is the helium fraction of the plasma (N_He/N_H)
     return (1+2*A)*(1+4*A)

def n_p_from_rho(rho, A):
     # Get the number density of protons from the mass density rho in g/cm^-3
     # and the helium fraction A, assuming a fully ionized plasma.
     n_particles=rho/(mean_molecular_weight(A)*C_mp)
     proton_fraction=1/(1+2*A+A+1)
     n_proton=n_particles*proton_fraction
     return n_proton

def plasma_frequency(n_e):
     # Return plasma frequency in Hz given the electron density in cm^-3
     return (4*np.pi*n_e*C_e**2/C_me)**0.5 / (2*np.pi)


def velocity_profile_isothermal(M_star,R_star,R_max, number_grid, T, A):
     # Return the velocity structure of the isothermal Parker wind solution
     # M_star = Stellar mass in g
     # R_star = Stellar radius in cm
     # R_max = End of the grid in cm
     # number_grid = number of grid points
     # T = Coronal temperature (isothermal) in K
     # R_grid = Radius vector grid on which solution is needed (in cm)
     # A = Helium fraction
     # Returns the velocity in cm/s at the radii given by R_grid.

     R_grid = np.linspace(R_star,R_max,number_grid)

     # Calculate the location of the critical point
     cs2 = C_k*T/(mean_molecular_weight(A)*C_mp) # The isothermal sound speed of the wind, squared
     cs=np.sqrt(cs2)
     r_crit = C_G*M_star/(2*cs2) # The distance of the critical point

     # Solve the Parker wind model (see Kavanagh et al. (2020) for a description of the implementation of the model)
     def f(r, u):
          return u * ((2 * cs ** 2) / r - (C_G * M_star) / (r ** 2)) / (u ** 2 - cs ** 2)
     # Runge-Kutta solver 
     # - r: radii to solve velocity for (cm)
     def rk4(r):

          # Initialise values
          r_i = r_crit
          u_i = cs

          # List to store velocities
          u = []

          # Iterate
          for i in range(len(r)):

               # Runge-Kutta terms
               if (i == 0): 
                    k1 = u_i / r_i
                    dr = r[0] - r_i
               
               else: 
                    k1 = f(r_i, u_i)
                    dr = r[i] - r[i - 1]

               k2 = f(r_i + 0.5 * dr, u_i + 0.5 * k1 * dr)
               k3 = f(r_i + 0.5 * dr, u_i + 0.5 * k2 * dr)
               k4 = f(r_i + dr, u_i + k3 * dr)

               # Compute next velocity (cm s^-1)
               u_i += (dr / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
               u.append(u_i)

               # Increment distance
               r_i += dr

          # Wind velocity (cm s^-1)
          return np.array(u)
     r1 = np.flip(R_grid[np.where(R_grid < r_crit)])
     r2 = R_grid[np.where(R_grid > r_crit)]
     
     # Solve for velocity
     u1 = rk4(r1)
     u2 = rk4(r2)

     # Join solutions
     v_grid = np.concatenate((np.flip(u1), u2))

     return R_grid, v_grid



def density_profile(R_grid, v_grid, n_base, A):
     # Given the velocity profile of a wind, calculate the corresponding density profile.
     # R_grid =  the radial grid on which the profile is defined, in cm
     # v_grid = the velocity profile of the wind in cm/s
     # n_base = the total base particle density of the wind in cm^-3
     # A = the helium fraction in the stellar wind
     # Returns the proton density profile in cm^-3 and the mass-loss rate in solar units.
     rho_base=n_base*mean_molecular_weight(A)*C_mp
     Mdot=4.0 * np.pi * R_grid[0]**2.0 * rho_base * v_grid[0] / C_Msun * year
     rho_grid=Mdot/(4.0 * np.pi * R_grid**2.0 * v_grid / C_Msun * year)
     n_proton=n_p_from_rho(rho_grid, A)
     return n_proton, Mdot/(2e-14)


def Gaunt_factor(T,nu):
     # Returns the Gaunt factor for a plasma with a temperature T in K that can variable with distance,
     # at a frequency nu in Hz.
     # The calculation is based on van Hoof et al. (2014).
     # We assume the Gaunt factor is the same for hydrogen and helium
     log_g2 = np.log10(C_Ry/C_k/T)
     log_u = np.log10(C_h*nu/C_k/T)
     sols = np.loadtxt("gauntff.dat",comments="#")
     ugrid = np.arange(-16,13.2,0.2)
     g2grid = np.arange(-6,10.2,0.2)
     f = interp2(points=(ugrid,g2grid),values=sols)
     try:
          interp_values=np.zeros([len(T),2])
          interp_values[:,0]=log_u
          interp_values[:,1]=log_g2
     except:
          interp_values=[log_u, log_g2]
     return f(interp_values)

def alpha_ffa(n_proton,T,nu,A):
     # Compute the free free absorption coefficient for a thermal plasma
     # n_proton = proton density in cm^-3
     # T = temperature in Kelvin
     # nu = Frequency in Hz
     # A = helium fraction of the stellar wind
     # Returns the absorbtion coefficient on the grid, as well as the plasma frequency at each point along the grid.
     gff = Gaunt_factor(T,nu)    # Free free Guant factor 
     n_e = n_proton*(1+2*A)
     nu_p = plasma_frequency(n_e)    # Electron plasma frequency in Hz
     ref_ind = (1-nu_p**2/nu**2)**0.5   # Refractive index for an isotropic plasma
     # We use equation 5.19b from Rybicki and Lightman to calculate the absorbtion coefficient
     # then divide by the refractive index to correct for group velocity effects (Zheleznyakov 1996).
     return 0.018*T**-1.5*Z2_from_A(A)*n_proton**2*nu**-2*gff/ref_ind, nu_p


def optical_depth(n_proton, R_grid, v_grid, T,R_star, freq, A, B0, second_harmonic=True):
     # Compute the free-free absorption optical depth experienced by emission travelling radially outward
     # in a stellar wind from the location of emission, defined by the magnetic field.
     # n_proton = proton density profile of the wind in cm^-3
     # R_grid = radial distance grid in c,
     # v_grid = velocity profile of the wind in cm/s
     # T = wind temperature in K
     # R_star = radius of the star in cm
     # freq = frequency at which ECMI was detected
     # A = helium fraction in the stellar wind
     # B0 = stellar magnetic dipole field strength in Gauss
     # second_harmonic = True if the ECMI emission is the fundamental, False if it is the second harmonic
     
     n_total=n_proton*(1+2*A+A+1) # Total particle density profile

     # Calculate the magnetic field strength at the location of the emitter, 
     # for the fundamental or the secon harmonic.
     if second_harmonic:
          B_emit=freq/(2.8e6)/2
     else:
          B_emit=freq/(2.8e6)

     # Determine the radius where the magnetic field lines should open up 
     # by comparing the magnetic pressure and the dynamic pressure.
     test=(B0**2)*(R_grid/R_star)**-6 *0.00397887358 - n_total*(mean_molecular_weight(A)*C_mp)*1e-3*(1e-2)**-3*v_grid*1e-2
     try:
          radius_switch= R_grid[test<0.0][0]
     except:
          radius_switch=R_grid[-1]

     # Define the magnetic field profile as a closed field dipole that opens up beyond radius_switch.
     Br_1=B0*(R_grid/R_star)**-3.0
     B_switch=Br_1[R_grid>=radius_switch][0]
     Br_2=B_switch*(R_grid/(radius_switch))**-2.0
     BrGrid=Br_1*1.0
     BrGrid[R_grid>radius_switch]=Br_2[R_grid>radius_switch]

     R_emit=R_grid[BrGrid<B_emit][0] # The farthest radial distance from the surface where the emission could be produces

     alpha, nu_p = alpha_ffa(n_proton, T, freq, A)

     nu_p_at_location=nu_p[R_grid>Rlim_second][0]
     if nu_p_at_location>freq:
          tau=np.nan
     else:
          tau=np.trapz(y=alpha[R_grid>R_emit], x=R_grid[R_grid>R_emit])

     return tau


def magnetic_field_conservative(M_star):
     # A conservative estimate of the magnetic field (in Gauss) of an M dwarf with mass M_star in solar masses
     return 10**(-0.765*np.log10(M_star)+1.78+1.0)

def magnetic_field_average(M_star):
     # An average estimate of the magnetic field (in Gauss) of an M dwarf with mass M_star in solar masses
     return 10**(-0.765*np.log10(M_star)+1.78)


def coronal_temp(F_x):
     # Convert the surface X-ray flux in erg/s/cm^2 to a coronal temperature in MK, following Johnstone & Güdel (2015).
     return 0.11*np.array(F_x)**0.26

