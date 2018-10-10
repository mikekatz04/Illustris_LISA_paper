from astropy.cosmology import Planck15 as cosmo
import scipy.constants as ct
import numpy as np
import pdb

def mass_ratio_func(m1, m2):
		up = (m1 >= m2)
		down = (m1 < m2)
		return up* (m2/m1) + down* (m1/m2)

class MassiveBlackHoleBinaries:
	def evolve(self):
		self.formation_time = cosmo.age(self.z).value*1e9
		self.t_delay, self.e_f = self.calculate_timescale()
		self.coalescence_time = self.formation_time + self.t_delay
		return
		
	def readout(self, keys, fp_out):
		out = np.array([getattr(self, key) for key in keys]).T
		out_key = ''
		for key in keys:
			out_key+=key + '\t'

		np.savetxt(fp_out, out, header=out_key)

		return


class AnalyticApproximations:

	def half_radius(M_p,z):
		#from FD
		#factor_1=1.0
		factor_1=3.0/(3.**(0.73))
		M_p_0=M_p/((1.+z)**(-0.6))
		factor_2=(M_p_0/(1.E+8))**0.66
		
		half_radius=factor_1*factor_2*(1.+z)**(-0.71)
		
		return half_radius

	def velocity_dispersion(M_p,z):
	 
		factor_1=190.0
		
		M_p_0=M_p/((1.+z)**(-0.6))
		#velocity_dispersion=factor_1*factor_2*(1.+z)**(0.44)
		
		#velocity_dispersion=(M_p/1.E+8/1.66)**(1./5.)*200.  used before
		
		factor_2=(M_p_0/(1.E+8))**0.2
		
		velocity_dispersion=factor_1*factor_2*(1.+z)**(0.056)
		
		return velocity_dispersion

	def influence_radius(M_p,vel_disp):

		#inf_radius=G*M_p/(velocity_dispersion(M_p,z)**2)
		#inf_radius=35.*(M_p/1.E+8)**(0.56)
		#inf_radius=13.*(M_p/1.e8)/(velocity_dispersion(M_p,z)/200.)**2
		inf_radius = 10.8*(M_p/1e8)*(vel_disp/200.0)**-2 # Eq 3b in Merritt et al 2009
		return inf_radius

	"""
	def find_a_GW(self):
		f_e = f_e_func(e)

		#RHS refers to RHS of equation 64
		RHS = 55.0 * (self.r_infl/30)**(5./10.) * ((self.M+self.m)/1e8)**(-5/10) * f_e**(1/5.) * (4*self.q/(1 + self.q)**2)**(4/5)

		#return a_GW
		return self.a_h/RHS*(self.q>=1e-3) + self.a_h*(self.q<1e-3)




	def e_integral_func(self):
		return (self.e_f**(29/19) * (1 + (121./304)*self.e_f**2)**(1181./2299.))/(1 - self.e_f**2)**(3./2.)




	def GW_timescale_func(self):

		#from shane's gw guide
		
		self.a_crit = find_a_crit(self) #pc

		#convert a_GW to meters
		a_0 = a_crit *ct.parsec
		#convert masses to meters
		m1 = self.M *M_sun.value*ct.G/(ct.c**2)
		m2 = self.m *M_sun.value*ct.G/(ct.c**2)

		beta = 64./5. * m1*m2 * (m1+m2)
		if e_f == 0:
			tau_circ = a_0**4/(4*beta) # meters
			return tau_circ/(ct.c*ct.Julian_year) #c to meters and julian year to years
		
		c_0 = a_0 * (1 - e**2)/e**(12/19) * (1 + (121/304) * e**2)**(-870./2299.)

		e_integral = quad(e_integral_func, 0.0, e)

		tau_merge = (12./19.)*c_0**4/beta*e_integral

		
		return tau_merge/(ct.c*ct.Julian_year) #c to meters and julian year to years

	"""
