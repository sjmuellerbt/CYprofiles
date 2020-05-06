#!/usr/bin/python

#    Copyright (C) 2020  Sebastian Johannes Mueller
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

# ============================================================================
#            IMPORT PACKAGES
# ============================================================================
import matplotlib.pyplot as plt
import numpy as np
import math


# ============================================================================
#            CLASS DEFINITIONS
# ============================================================================
class Printing_Parameters : 
    """PrintingParameters class contains the printing parameters G (pressure gradient, negative) and R (channel radius)"""

    def __init__(self, pressureGradient=None, channelRadius=None, pressureDifference=None, channelLength=None, flowrate=None) :
        """Stores the given parameters. Not all need to be provided"""
        # Check for correct number of arguments
        if channelRadius == None or (pressureGradient == None and (pressureDifference == None or channelLength == None) and flowrate == None) :
            self.print_usage()
        else :
            self.R = channelRadius
            # Either pressure gradient or pressure difference and channellength or flowrate are needed
            if pressureGradient != None :
                # The pressure gradient has to be negative by definition
                self.G        = - math.fabs(pressureGradient)
                self.flowrate = None
            elif ( pressureDifference != None and channelLength != None) :
                if pressureGradient != None :
                    self.print_usage()
                else :
                    self.G        = - math.fabs(pressureDifference * (1.0 / channelLength))
                    self.flowrate = None
            elif flowrate != None :
                if pressureGradient != None :
                    self.print_usage()
                else :                
                    self.flowrate = flowrate
                    self.G        = None
    
    
    @staticmethod
    def print_usage() :
        """Method used to print input instructions."""
        print(" --- Printing_Parameters class usage ---")
        print("\nPrinting_Parameters() class requires at least two parameters:")
        print(" - channelRadius :\t radius of the cylindrical channel (in m)")
        print(" - pressureGradient : \t pressure gradient (negative) along the channel (in Pa/m)")
        print(" - Alternatively to pressureGradient, one of the following parameters must provided:")
        print("    - flowrate :\t\t flow rate in the channel (in m^3/s) (Note: the flow rate is iteratively converted into a pressure gradient, which takes some time (use option 'loud=1' for output))")
        print("    - pressureDifference :\t pressure drop along the channel of length (negative)")
        print("      and channelLength : \t the length of the channel\n\n")

      
class Profiles : 
    """Profiles class contains functions and parameters that define the solution for the shear rate and velocity profile in a channel"""
    
    def __init__(self, interpolation=None, printingParameters=None, samples=10000) :
        """"Initialize the parameters necessary for calculating the semi-analytical solution"""
        if interpolation == None or printingParameters == None :
            self.print_usage()
        else :
            # Interpolated viscosity model
            self.interpol = interpolation
            # Printing parameters    
            self.printparams = printingParameters
            # Save the number of interpolation intervals separately    
            self.Ninterpol = self.interpol.N
            # Initialize the number of relevant interpolation intervals
            self.Nchannel = self.Ninterpol
            # Initialize lists that store the parameters of the analytical solution
            self.r = np.zeros(self.Ninterpol+3)
            self.c = np.zeros(self.Ninterpol+2)
            # Number of intervals necessary to calculate velocity profile
            self.k = 0
            # The plotrange in terms of intervals: 0, ..., k
            self.krange = range(0,self.k+1,1)
            # Matrix to store the calculated values (for quicker printing and plotting)        
            self.samples     = samples
            self.samplerange = range(0,self.samples+1,1)
            self.data        = np.zeros((6, self.samples+1)) # radial position, velocity, shear rate, viscosity, interval (of interpolation)
            # Average values
            self.flowrate        = None
            self.avg_velocity    = None
            self.avg_shearrate   = None
            self.avg_viscosity   = None
            self.avg_shearstress = None
        
    
    @staticmethod
    def print_usage() :
        """Method used to print input instructions."""
        print(" --- Profiles class usage ---")
        print("\nProfiles() class requires at least two parameters:")
        print(" - interpolation :\t\t an instance of the Interpolation() class")
        print(" - printingParameters : \t an instance of the Printing_Parameters() class")
        print(" - (optional) samples :\t\t number of points calculated for plotting and data saving\n")
        print("The following methods must be called:")
        print(" - Profiles() :\t\t\t with all necessary parameters")
        print(" - calculate_profiles() :\t performs the interpolation \n")
        print("The following methods/variables are helpful:")
        print(" - save_profiles(filename) : \t save the profile data to a file (radial position, velocity, shear rate, viscosity)")
        print(" - plot_velocity(options) :\t plot the velocity profile")
        print(" - plot_shearrate(options) :\t plot the shear rate profile")
        print(" - plot_shearstress(options) :\t plot the shear stress profile")
        print(" - plot_viscosity(options) :\t plot the viscosity profile")
        print("   Options are:")
        print("   - draw_intervals=<int> : \t indicate the interval boundaries with lines")
        print("   - save_figure=<filename> : \t save the plot to a file\n\n")        
 
     
    def find_k(self) :
        """Determine the interval that includes the physical boundary"""
        k = 0
        while ( k <= self.Ninterpol and self.r[k] < self.printparams.R) :
            k += 1
        self.k = k
        self.krange = range(0,self.k+1,1) # 0, ..., k
 
       
    def get_last_interval(self) :
        """Return the last interval that includes the physical boundary (index k)"""
        self.find_k()
        return self.k
        
    def get_max_velocity(self) :
        """Return the maximum velocity"""
        return self.c[0]
        
    def get_max_shearrate(self) :
        """Return the maximum shear rate"""
        return self.calc_shearrate(self.printparams.R, self.k)


    def find_interval(self,r) :
        """Determine the index of the interval that includes the radial position r"""
        i = 0
        while( i < self.Ninterpol and self.r[i] < r) :
            i += 1
        return i
   
     
    def calc_ri(self) :
        """Calculate the interval positions"""
        for i in self.interpol.fullrange :
            self.r[i] = - 2.0 * 1.0/self.printparams.G * self.interpol.K[i] * (self.interpol.gammai[i]**(self.interpol.n[i]))        
    
        
    def calc_ci(self) :
        """Calculate the integrations constants of the velocity field"""
        self.find_k()
        self.c[self.k] = (- 0.5 * self.printparams.G * 1.0/self.interpol.K[self.k])**(1.0/self.interpol.n[self.k]) * self.interpol.n[self.k] * 1.0/(self.interpol.n[self.k] + 1.0) * self.printparams.R**(1.0 + 1.0/self.interpol.n[self.k])
        # The other integration constants are calculated using the already calculated oines
        #for i in range(self.k,-1,-1) : # count backwards from k-1
        for i in range(0,self.k,1) :
            #self.c[i] = self.c[i+1] - self.r[i] * self.interpol.gammai[i] * ( self.interpol.n[i+1]*1.0/(self.interpol.n[i+1]+1.0) - self.interpol.n[i]*1.0/(self.interpol.n[i]+1.0))
            self.c[i] = self.c[self.k]
            for j in range(i,self.k,1) :
                self.c[i] -= self.r[j] * self.interpol.gammai[j] * (self.interpol.n[j+1]/(1.0+self.interpol.n[j+1]) - self.interpol.n[j]/(1.0+self.interpol.n[j])) 
   
                 
    def calculate_profiles(self, loud=None, epsilon=1.0e-6) : 
        """Calculate all coefficients necessary for plotting"""
        if self.printparams.G == None and self.printparams.flowrate != None :
            self.find_pressureGradient(self.printparams.flowrate, loud=loud, epsilon=epsilon)
        self.calc_ri()
        self.calc_ci()
        self.calc_profile_data()
        self.calc_averages()
        
     
     
    def calc_velocity(self,r,i) :
        """Calculate the velocity in interval i at position r"""
        return ( - (- 0.5 * self.printparams.G * 1.0/self.interpol.K[i])**(1.0/self.interpol.n[i]) * self.interpol.n[i] * 1.0/(self.interpol.n[i] + 1.0) * r**(1.0 + 1.0/self.interpol.n[i]) ) + self.c[i]
 
 
    def calc_shearrate(self,r,i) : 
        """Calculate the shear rate in interval i at position r"""
        return (- 0.5 * r * self.printparams.G * 1.0/self.interpol.K[i])**(1.0/self.interpol.n[i])
     
     
    def calc_viscosity(self,r,i) :
        """Calculate the viscosity in interval i at position r"""
        return self.interpol.K[i] * (self.calc_shearrate(r,i))**(self.interpol.n[i]-1.0)
    
    
    def calc_flowrate(self) :
        """Calculates the flow rate for the approximated flow profile"""
        flowrate = 0
        # Add zeroth interval
        if self.k == 0 :
            flowrate += 0.5*self.c[0]*self.printparams.R*self.printparams.R - (-0.5*self.printparams.G*1.0/self.interpol.K[0])**(1.0/self.interpol.n[0])*self.interpol.n[0]*1.0/(1.0+self.interpol.n[0])*1.0/(3.0+1.0/self.interpol.n[0])*(self.printparams.R**(3.0+1.0/self.interpol.n[0]))
        else :
            flowrate += 0.5*self.c[0]*self.r[0]*self.r[0] - (-0.5*self.printparams.G*1.0/self.interpol.K[0])**(1.0/self.interpol.n[0])*self.interpol.n[0]*1.0/(1.0+self.interpol.n[0])*1.0/(3.0+1.0/self.interpol.n[0])*(self.r[0]**(3.0+1.0/self.interpol.n[0]))
        
        # Add k-th interval
        if self.k != 0 :
            flowrate += 0.5*self.c[self.k]*(self.printparams.R**2 - self.r[self.k-1]**2 ) - (-0.5*self.printparams.G*1.0/self.interpol.K[self.k])**(1.0/self.interpol.n[self.k])*self.interpol.n[self.k]*1.0/(1.0+self.interpol.n[self.k])*1.0/(3.0+1.0/self.interpol.n[self.k])*(self.printparams.R**(3.0+1.0/self.interpol.n[self.k]) - self.r[self.k-1]**(3.0+1.0/self.interpol.n[self.k]))
            # Add the intermediate intervals
            for j in range(0,self.k,1) :
                flowrate += 0.5*self.c[j]*(self.r[j]**2 - self.r[j-1]**2 ) - (-0.5*self.printparams.G*1.0/self.interpol.K[j])**(1.0/self.interpol.n[j])*self.interpol.n[j]*1.0/(1.0+self.interpol.n[j])*1.0/(3.0+1.0/self.interpol.n[j])*(self.r[j]**(3.0+1.0/self.interpol.n[j]) - self.r[j-1]**(3.0+1.0/self.interpol.n[j]))
        flowrate *= 2.0 * math.pi
        self.flowrate = flowrate
        return flowrate
    
    def calc_partial_flowrate(self, rpos) :
        """Calculates the flow rate for the approximated flow profile up to a specified radial position"""
        pflowrate = 0
        # Determine interpolation interval of the radial position
        ipos = self.find_interval(rpos)
        
        # Add zeroth interval
        if ipos == 0 :
            pflowrate += 0.5*self.c[0]*rpos*rpos - (-0.5*self.printparams.G*1.0/self.interpol.K[0])**(1.0/self.interpol.n[0])*self.interpol.n[0]*1.0/(1.0+self.interpol.n[0])*1.0/(3.0+1.0/self.interpol.n[0])*(rpos**(3.0+1.0/self.interpol.n[0]))
        else :
            pflowrate += 0.5*self.c[0]*self.r[0]*self.r[0] - (-0.5*self.printparams.G*1.0/self.interpol.K[0])**(1.0/self.interpol.n[0])*self.interpol.n[0]*1.0/(1.0+self.interpol.n[0])*1.0/(3.0+1.0/self.interpol.n[0])*(self.r[0]**(3.0+1.0/self.interpol.n[0]))
        
        # Add ipos-th interval
        if ipos != 0 :
            pflowrate += 0.5*self.c[self.k]*(rpos**2 - self.r[ipos-1]**2 ) - (-0.5*self.printparams.G*1.0/self.interpol.K[ipos])**(1.0/self.interpol.n[ipos])*self.interpol.n[ipos]*1.0/(1.0+self.interpol.n[ipos])*1.0/(3.0+1.0/self.interpol.n[ipos])*(rpos**(3.0+1.0/self.interpol.n[ipos]) - self.r[ipos-1]**(3.0+1.0/self.interpol.n[ipos]))
            # Add the intermediate intervals
            for j in range(0,ipos,1) :
                pflowrate += 0.5*self.c[j]*(self.r[j]**2 - self.r[j-1]**2 ) - (-0.5*self.printparams.G*1.0/self.interpol.K[j])**(1.0/self.interpol.n[j])*self.interpol.n[j]*1.0/(1.0+self.interpol.n[j])*1.0/(3.0+1.0/self.interpol.n[j])*(self.r[j]**(3.0+1.0/self.interpol.n[j]) - self.r[j-1]**(3.0+1.0/self.interpol.n[j]))
        pflowrate *= 2.0 * math.pi
        return pflowrate    
        

    def find_r_from_gamma(self, gamma) : 
        """Determine the radial position corresponding to a shear rate gamma"""
        # Find the interplation itnerval corresponding to the given shear rate
        i = 0
        while( i < self.Ninterpol and self.interpol.gammai[i] < gamma) :
            i += 1       
        # Calculate the corresponding radial position            
        rpos = gamma**(self.interpol.n[i])*1.0/(- 0.5 * self.printparams.G * 1.0/self.interpol.K[i])
        return rpos       

        
    def find_pressureGradient(self, flowrate, epsilon=None, maxiter=10000, loud=None) :
        """Calculate the pressure gradient from the flow rate iteratively until the relative error is less than epsilon"""
        # Inital values for pressure gradient intervals
        Glow = -1.0e0
        Ghigh = -1.0e1
        # Calculate initial flow rates
        self.printparams.G = Glow
        self.calculate_profiles()
        Qlow = self.calc_flowrate()       
        self.printparams.G = Ghigh
        self.calculate_profiles()
        Qhigh = self.calc_flowrate()
        # Find interval, such that Qlow < flowrate < Qhigh
        while Qlow < flowrate and Qhigh < flowrate :
            if loud != None :
                print("Intervals too low", Glow, Ghigh)
            Glow  = Ghigh
            self.printparams.G = Glow
            self.calculate_profiles()
            Qlow = self.calc_flowrate()           
            Ghigh = 10*Ghigh
            self.printparams.G = Ghigh
            self.calculate_profiles()
            Qhigh = self.calc_flowrate()
        while Qlow > flowrate and Qhigh > flowrate :
            if loud != None :
                print("Intervals too high", Glow, Ghigh)
            Ghigh = Glow
            self.printparams.G = Ghigh
            self.calculate_profiles()
            Qhigh = self.calc_flowrate()        
            Glow  = Glow/10.0
            self.printparams.G = Glow
            self.calculate_profiles()
            Qlow = self.calc_flowrate()           
        # Do bisection until desired accuracy is reached
        Gmid = 0.5 * (Ghigh + Glow)
        self.printparams.G = Gmid
        self.calculate_profiles()
        Qmid = self.calc_flowrate()        
        eps = (Qmid - flowrate)/flowrate
        index = 1
        if loud != None :
            print("iter\tGlow\t\tGhigh\t\tQlow\t\tQmid\t\tQhigh\t\tepsilon")
        while math.fabs(eps) > epsilon :
            Gmid = 0.5 * (Ghigh + Glow)
            self.printparams.G = Gmid
            self.calculate_profiles()
            Qmid = self.calc_flowrate()
            if Qmid > flowrate :
                Ghigh = Gmid
            elif Qmid <= flowrate :
                Glow = Gmid
            self.printparams.G = Glow
            self.calculate_profiles()
            Qlow = self.calc_flowrate()
            self.printparams.G = Ghigh
            self.calculate_profiles()
            Qhigh = self.calc_flowrate()
            eps = (Qmid - flowrate)/flowrate
            if loud != None :
                print("%s\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E"%(index,Glow,Ghigh,Qlow,Qmid,Qhigh,eps))
            index += 1
            if index >= maxiter :
                print("Error: Conversion of flowrate to pressure gradient didnt converge after %s iterations."%(maxiter))
                break
        # Set final pressure gradient between Glow and Ghigh
        Gmid = 0.5 * (Ghigh + Glow)
        self.printparams.G = Gmid
        self.calculate_profiles()
        Qmid = self.calc_flowrate()
        if loud != None :
            print("Conversion of flowrate to pressure gradient converged after %s iterations."%(index))
            print("flowrate (set): %.6E"%(flowrate))
            print("flowrate (get): %.6E"%(Qmid))
        
                
    def calc_averages(self) :
        """Calculate the average velocity, shear rate and viscosity in the channel"""
        area       = math.pi * self.printparams.R * self.printparams.R
        flowrate   = 0.0
        avg_shear  = 0.0
        avg_visc   = 0.0
        # Add zeroth interval
        if self.k == 0 :
            flowrate  += 0.5*self.c[0]*self.printparams.R*self.printparams.R - (-0.5*self.printparams.G*1.0/self.interpol.K[0])**(1.0/self.interpol.n[0])*self.interpol.n[0]*1.0/(1.0+self.interpol.n[0])*1.0/(3.0+1.0/self.interpol.n[0])*(self.printparams.R**(3.0+1.0/self.interpol.n[0]))
            avg_shear += (-0.5*self.printparams.G*1.0/self.interpol.K[0])**(1.0/self.interpol.n[0])*1.0/(2.0+1.0/self.interpol.n[0])*(self.printparams.R**(2.0+1.0/self.interpol.n[0]) - self.r[0]**(2.0+1.0/self.interpol.n[0]))            
            avg_visc  += self.interpol.K[0]*(0.5*self.printparams.G*1.0/self.interpol.K[0])**(1.0-1.0/self.interpol.n[0])*1.0/(3.0-1.0/self.interpol.n[0])*(self.printparams.R**(3.0-1.0/self.interpol.n[0]))
        else :
            flowrate  += 0.5*self.c[0]*self.r[0]*self.r[0] - (-0.5*self.printparams.G*1.0/self.interpol.K[0])**(1.0/self.interpol.n[0])*self.interpol.n[0]*1.0/(1.0+self.interpol.n[0])*1.0/(3.0+1.0/self.interpol.n[0])*(self.r[0]**(3.0+1.0/self.interpol.n[0]))
            avg_shear += (-0.5*self.printparams.G*1.0/self.interpol.K[0])**(1.0/self.interpol.n[0])*1.0/(2.0+1.0/self.interpol.n[0])*self.r[0]**(2.0+1.0/self.interpol.n[0])
            avg_visc  += self.interpol.K[0]*(-0.5*self.printparams.G*1.0/self.interpol.K[0])**(1.0-1.0/self.interpol.n[0])*1.0/(3.0-1.0/self.interpol.n[0])*(self.r[0]**(3.0-1.0/self.interpol.n[0]))
        # Add k-th interval
        if self.k != 0 :
            flowrate  += 0.5*self.c[self.k]*(self.printparams.R**2 - self.r[self.k-1]**2 ) - (-0.5*self.printparams.G*1.0/self.interpol.K[self.k])**(1.0/self.interpol.n[self.k])*self.interpol.n[self.k]*1.0/(1.0+self.interpol.n[self.k])*1.0/(3.0+1.0/self.interpol.n[self.k])*(self.printparams.R**(3.0+1.0/self.interpol.n[self.k]) - self.r[self.k-1]**(3.0+1.0/self.interpol.n[self.k]))
            avg_shear += (-0.5*self.printparams.G*1.0/self.interpol.K[self.k])**(1.0/self.interpol.n[self.k])*1.0/(2.0+1.0/self.interpol.n[self.k])*(self.printparams.R**(2.0+1.0/self.interpol.n[self.k]) - self.r[self.k-1]**(2.0+1.0/self.interpol.n[self.k]))
            if self.interpol.n[self.k] == 1.0/3.0 :            
                avg_visc  += self.interpol.K[self.k]*(-0.5*self.printparams.G*1.0/self.interpol.K[self.k])**(1.0-1.0/self.interpol.n[self.k])*math.log(self.printparams.R *1.0 / self.r[self.k-1])
            else :
                avg_visc  += self.interpol.K[self.k]*(-0.5*self.printparams.G*1.0/self.interpol.K[self.k])**(1.0-1.0/self.interpol.n[self.k])*1.0/(3.0-1.0/self.interpol.n[self.k])*(self.printparams.R**(3.0-1.0/self.interpol.n[self.k]) - self.r[self.k-1]**(3.0-1.0/self.interpol.n[self.k]))    
            # Add the intermediate intervals
            for j in range(0,self.k,1) :
                flowrate  += 0.5*self.c[j]*(self.r[j]**2 - self.r[j-1]**2 ) - (-0.5*self.printparams.G*1.0/self.interpol.K[j])**(1.0/self.interpol.n[j])*self.interpol.n[j]*1.0/(1.0+self.interpol.n[j])*1.0/(3.0+1.0/self.interpol.n[j])*(self.r[j]**(3.0+1.0/self.interpol.n[j]) - self.r[j-1]**(3.0+1.0/self.interpol.n[j]))
                avg_shear += (-0.5*self.printparams.G*1.0/self.interpol.K[j])**(1.0/self.interpol.n[j])*1.0/(2.0+1.0/self.interpol.n[j])*(self.r[j]**(2.0+1.0/self.interpol.n[j]) - self.r[j-1]**(2.0+1.0/self.interpol.n[j]))
                if self.interpol.n[self.k] == 1.0/3.0 :            
                    avg_visc  += self.interpol.K[j]*(-0.5*self.printparams.G*1.0/self.interpol.K[j])**(1.0-1.0/self.interpol.n[j])*math.log(self.r[j]*1.0/self.r[j-1])
                else :
                    avg_visc  += self.interpol.K[j]*(-0.5*self.printparams.G*1.0/self.interpol.K[j])**(1.0-1.0/self.interpol.n[j])*1.0/(3.0-1.0/self.interpol.n[j])*(self.r[j]**(3.0-1.0/self.interpol.n[j]) - self.r[j-1]**(3.0-1.0/self.interpol.n[j]))  
        flowrate  *= 2.0 * math.pi
        avg_shear *= 2.0 * math.pi
        avg_visc  *= 2.0 * math.pi
        # Store in class members
        self.flowrate = flowrate
        self.avg_shearrate = avg_shear* 1.0/area
        self.avg_viscosity = avg_visc* 1.0/area
        self.avg_velocity = self.flowrate * 1.0/area
        self.avg_shearstress = - self.printparams.G * self.printparams.R * 1.0/3.0           
    

    def calc_discrete_averages(self) :
        """Calculate the average velocity, shear rate, shear stress, viscosity and flow rate from the sampled data"""
        area       = math.pi * self.printparams.R * self.printparams.R
        flowrate   = 0
        avg_shear  = 0
        avg_visc   = 0
        avg_vel    = 0
        avg_stress = 0
        for i in range(1,self.samples+1,1) :
            ringarea    = math.pi * (self.data[0][i]*self.data[0][i] - self.data[0][i-1]*self.data[0][i-1])
            flowrate   += ringarea * self.data[1][i-1]
            avg_shear  += ringarea * self.data[2][i-1]
            avg_visc   += ringarea * self.data[3][i-1]
            avg_stress += ringarea * self.data[2][i-1] * self.data[3][i-1]
        avg_vel     = flowrate * 1.0/area
        avg_shear  /= area
        avg_visc   /= area
        avg_stress /= area
        return avg_vel,avg_shear,avg_stress,avg_visc,flowrate        
        
    def calc_profile_data(self) : 
        """Calculate and store the velocity, shear rate, viscosity and shear stress profiles"""
        # Determine the range in r (radial position)
        for i in self.samplerange :
            # Find corresponding interval to radial position
            r = i*1.0/self.samples * self.printparams.R
            interval = self.find_interval(r)
            # Radial position
            self.data[0][i] = r
            # Velocity
            self.data[1][i] = self.calc_velocity(r, interval)
            # Shear rate
            self.data[2][i] = self.calc_shearrate(r, interval)
            # Viscosity
            self.data[3][i] = self.interpol.calc_powerlaw_visc(self.calc_shearrate(r, interval),interval)
            # Shear stress
            self.data[4][i] = self.data[2][i] * self.data[3][i]
            # interpolation interval
            self.data[5][i] = interval
            

    def save_profiles(self, filename) : 
        """Save the calculated velocity, shear rate and viscosity profiles to a file"""
        datafile = open(filename, 'w')
        datafile.write("#r\tvelocity(m/s)\tshearrate(1/s)\tviscosity(Pa*s)\tshearstress\tinterval\n")
        for r in self.samplerange :
            datafile.write("%.6E\t%.6E\t%.6E\t%.6E\t%.6E\t%i\n" % (self.data[0][r],self.data[1][r],self.data[2][r],self.data[3][r],self.data[4][r],self.data[5][r]))
            
            
    def save_averages(self, filename) :
        """Save the averaged quantities"""
        datafile = open(filename, 'w')
        datafile.write("Averaged quantities:\n")
        datafile.write("flowrate \t = %.6E m^3/s\n" % (self.flowrate))
        datafile.write("average velocity \t = %.6E m/s\n" % (self.avg_velocity))
        datafile.write("average shear rate \t = %.6E 1/s\n" % (self.avg_shearrate))
        datafile.write("average shear stress \t = %.6E 1/s\n" % (self.avg_shearstress))
        datafile.write("average viscosity \t = %.6E Pa*s\n" % (self.avg_viscosity))       


    def plot_velocity(self, draw_intervals=None, save_figure=None, ymin=None, ymax=None, xlabel=None, ylabel=None, title=None) : 
        """Plot the calculated velocity profiles and/or save the data to a file"""
        # Create plot
        fig = plt.figure(1, figsize=(12,9))
        ax = fig.add_subplot(111)
        # Add plot title
        if title == None :
            fig.suptitle('Velocity profile', fontsize=20)
        else :
            fig.suptitle(title, fontsize=20)
        # Add axis labels
        if xlabel == None :
            ax.set_xlabel(r'radial position $r$ in $\mathrm{m}$', fontsize=18)
        else : 
            ax.set_xlabel(xlabel, fontsize=18)
        if ylabel == None :
            ax.set_ylabel(r'velocity $u$ in $\frac{\mathrm{m}}{\mathrm{s}}$', fontsize=18) 
        else :
            ax.set_ylabel(ylabel, fontsize=18)
        # Set plot ranges
        ax.set_xlim(-self.printparams.R, self.printparams.R);
        if ymin == None and ymax == None :
            plotymin = 0.0
            plotymax = 1.1 * self.c[0]
        else :
            plotymin = ymin
            plotymax = ymax
        ax.set_ylim(plotymin, plotymax);
        # Draw the interpolated intervals
        if draw_intervals != None :
            # Dummy plot for interpolation intervals legend
            plt.plot([], linewidth=0.5, color='0.25', label=r'intervals: $R_i$')            
            for i in self.krange :
                ax.axvline(self.r[i], linewidth=0.5, color='0.25')
                ax.axvline(-self.r[i], linewidth=0.5, color='0.25')
        # Print the fluid profile
        ax.plot(self.data[0][self.samplerange], self.data[1][self.samplerange], 'b-', label='Piecewise solution')
        ax.plot(-self.data[0][self.samplerange], self.data[1][self.samplerange], 'b-') # Mirror image
        # Further plot options
        # Key options
        ax.legend()        
        # Invert x axis labels (only positive radial position, but mirrored for better visibility)
        ax.set_xticklabels([str("%.2e"%(abs(x))) for x in ax.get_xticks()])
        # Format y axis labels
        ax.set_yticklabels([str("%.2e"%(y)) for y in ax.get_yticks()])
        # Show plot in Console    
        plt.show()        
        # Save the plot as PNG file
        if save_figure != None :
            fig.savefig(save_figure)
            
                
    def plot_shearrate(self, draw_intervals=None, save_figure=None, ymin=None, ymax=None, xlabel=None, ylabel=None, title=None) : 
        """Plot the calculated shear rate profiles"""
        # Create plot
        fig = plt.figure(1, figsize=(12,9))
        ax = fig.add_subplot(111)
        # Add plot title
        if title == None :
            fig.suptitle('Shear rate profile', fontsize=20)
        else :
            fig.suptitle(title, fontsize=20)
        # Add axis labels
        if xlabel == None :
            ax.set_xlabel(r'radial position $r$ in $\mathrm{m}$', fontsize=18)
        else : 
            ax.set_xlabel(xlabel, fontsize=18)
        if ylabel == None :
            ax.set_ylabel(r'shear rate $\dot{\gamma}$ in $\frac{1}{\mathrm{s}}$', fontsize=18)
        else :
            ax.set_ylabel(ylabel, fontsize=18)
        # Set plot ranges
        ax.set_xlim(-self.printparams.R, self.printparams.R);
        if ymin == None and ymax == None :
            plotymin = 0.0
            plotymax = 1.1 * self.calc_shearrate(self.printparams.R,self.k)
        else :
            plotymin = ymin
            plotymax = ymax
        ax.set_ylim(plotymin, plotymax);
        # Draw the interpolated intervals
        if draw_intervals != None :
            # Dummy plot for interpolation intervals legend
            plt.plot([], linewidth=0.5, color='0.25', label=r'intervals: $R_i$')            
            for i in self.krange :
                ax.axvline(self.r[i], linewidth=0.5, color='0.25')
                ax.axvline(-self.r[i], linewidth=0.5, color='0.25')
        # Print the fluid profile
        ax.plot(self.data[0][self.samplerange], self.data[2][self.samplerange], 'b-', label='Piecewise solution')
        ax.plot(-self.data[0][self.samplerange], self.data[2][self.samplerange], 'b-') # Mirror image
        # Further plot options
        # Key options
        ax.legend()        
        # Invert x axis labels (only positive radial position, but mirrored for better visibility)
        ax.set_xticklabels([str("%.2e"%(abs(x))) for x in ax.get_xticks()])
        # Format y axis labels
        ax.set_yticklabels([str("%.2e"%(y)) for y in ax.get_yticks()])
        # Show plot in Console    
        plt.show()        
        # Save the plot as PNG file
        if save_figure != None :
            fig.savefig(save_figure)
            
            
    def plot_viscosity(self, draw_intervals=None, save_figure=None, ymin=None, ymax=None, xlabel=None, ylabel=None, title=None, logarithmic=None) : 
        """Plot the calculated viscosity profiles"""       
        # Create plot
        fig = plt.figure(1, figsize=(12,9))
        ax = fig.add_subplot(111)
        # Add plot title
        if title == None :
            fig.suptitle('Viscosity profile', fontsize=20)
        else :
            fig.suptitle(title, fontsize=20)
        # Add axis labels
        if xlabel == None :
            ax.set_xlabel(r'radial position $r$ in $\mathrm{m}$', fontsize=18)
        else : 
            ax.set_xlabel(xlabel, fontsize=18)
        if ylabel == None :
            ax.set_ylabel(r'dynamic viscosity $\eta$ in $\mathrm{Pa\,s}$', fontsize=18)
        else :
            ax.set_ylabel(ylabel, fontsize=18)
        # Set plot ranges
        ax.set_xlim(-self.printparams.R, self.printparams.R);
        if ymin == None and ymax == None :
            plotymin = 0.9 * self.calc_viscosity(self.printparams.R,self.k)
            plotymax = 1.1 * self.interpol.K[0]
        else :
            plotymin = ymin
            plotymax = ymax
        ax.set_ylim(plotymin, plotymax);
        # Draw the interpolated intervals
        if draw_intervals != None :
            # Dummy plot for interpolation intervals legend
            plt.plot([], linewidth=0.5, color='0.25', label=r'intervals: $R_i$')            
            for i in self.krange :
                ax.axvline(self.r[i], linewidth=0.5, color='0.25')
                ax.axvline(-self.r[i], linewidth=0.5, color='0.25')
        # Print the fluid profile
        if logarithmic != None :
            ax.semilogy(self.data[0][self.samplerange], self.data[3][self.samplerange], 'b-', label='Piecewise solution')
            ax.semilogy(-self.data[0][self.samplerange], self.data[3][self.samplerange], 'b-') # Mirror image
        else :
            ax.plot(self.data[0][self.samplerange], self.data[3][self.samplerange], 'b-', label='Piecewise solution')
            ax.plot(-self.data[0][self.samplerange], self.data[3][self.samplerange], 'b-') # Mirror image
        # Further plot options
        # Key options
        ax.legend()        
        # Invert x axis labels (only positive radial position, but mirrored for better visibility)
        ax.set_xticklabels([str("%.2e"%(abs(x))) for x in ax.get_xticks()])
        # Format y axis labels
        ax.set_yticklabels([str("%.2e"%(y)) for y in ax.get_yticks()])
        # Show plot in Console    
        plt.show()        
        # Save the plot as PNG file
        if save_figure != None :
            fig.savefig(save_figure)
            
            
    def plot_shearstress(self, draw_intervals=None, save_figure=None, ymin=None, ymax=None, xlabel=None, ylabel=None, title=None) : 
        """Plot the shear stress profiles calculated from viscosity and shear rate profiles"""
        # Create plot
        fig = plt.figure(1, figsize=(12,9))
        ax = fig.add_subplot(111)
        # Add plot title
        if title == None :
            fig.suptitle('Shear stress profile', fontsize=20)
        else :
            fig.suptitle(title, fontsize=20)
        # Add axis labels
        if xlabel == None :
            ax.set_xlabel(r'radial position $r$ in $\mathrm{m}$', fontsize=18)
        else : 
            ax.set_xlabel(xlabel, fontsize=18)
        if ylabel == None :
            ax.set_ylabel(r'shear stress $\tau$ in $\mathrm{Pa}$', fontsize=18)
        else :
            ax.set_ylabel(ylabel, fontsize=18)
        # Set plot ranges
        ax.set_xlim(-self.printparams.R, self.printparams.R);
        if ymin == None and ymax == None :
            plotymin = 0.0
            plotymax = 1.1 * self.calc_shearrate(self.printparams.R,self.k)*self.calc_viscosity(self.printparams.R,self.k)
        else :
            plotymin = ymin
            plotymax = ymax
        ax.set_ylim(plotymin, plotymax);
        # Draw the interpolated intervals
        if draw_intervals != None :
            # Dummy plot for interpolation intervals legend
            plt.plot([], linewidth=0.5, color='0.25', label=r'intervals: $R_i$')            
            for i in self.krange :
                ax.axvline(self.r[i], linewidth=0.5, color='0.25')
                ax.axvline(-self.r[i], linewidth=0.5, color='0.25')
        # Print the fluid profile
        ax.plot(self.data[0][self.samplerange], self.data[4][self.samplerange], 'b-', label='Piecewise solution')
        ax.plot(-self.data[0][self.samplerange], self.data[4][self.samplerange], 'b-') # Mirror image
        # Further plot options
        # Key options
        ax.legend()        
        # Invert x axis labels (only positive radial position, but mirrored for better visibility)
        ax.set_xticklabels([str("%.2e"%(abs(x))) for x in ax.get_xticks()])
        # Format y axis labels
        ax.set_yticklabels([str("%.2e"%(y)) for y in ax.get_yticks()])
        # Show plot in Console    
        plt.show()        
        # Save the plot as PNG file
        if save_figure != None :
            fig.savefig(save_figure)            
            


class Interpolation : 
    """Interpolation class interpolates given analytical function with continuous, piecewise defined, power-law functions"""

    def __init__(self, gamma0=None, gammaN=None, Ninterpol=None, gammaStart=None, gammaEnd=None, analytical=None, samples=10000) :
        """Create a new Interpolation with given or default parameters"""
        # Check if correct number of arguments is given, otherwise print instructions
        if gamma0 == None or gammaN == None  or Ninterpol == None or analytical == None : 
            self.print_usage()
        else :
            # Bounds of range of the interpolation functions (only power-law intervals)
            self.gamma0     = gamma0
            self.gammaN     = gammaN
            # Number of interpolation intervals
            self.N          = Ninterpol
            # Bounds of full range of viscosity interpolation (including Newtonian limits)
            if gammaStart == None or gammaEnd == None :
                self.gammaStart = gamma0 * 1.0e-1
                self.gammaEnd   = gammaN * 1.0e1
            else :  
                self.gammaStart = gammaStart
                self.gammaEnd   = gammaEnd
            # Lists storing the parameters of the interpolation functions
            self.n          = np.ones(self.N+2)
            self.K          = np.zeros(self.N+2)
            self.gammai     = np.zeros(self.N+3)
            # Index range for full interpolation (including Newtonian limits)
            self.fullrange  = range(0,self.N+2,1)
            # The analytical form to be interpolated
            self.analytical = analytical
            # The number of samples used for data saving and plotting
            self.samples     = samples
            self.samplerange = range(0,self.samples+1,1)
            self.data        = np.zeros((4, self.samples+1), dtype=np.float64) # shear rate, viscosity(interpolated), viscosity(analytical), interval
    
        
    @staticmethod
    def print_usage() :
        """Method used to print input instructions."""
        print("\n --- Interpolation class usage --- \n")
        print("Interpolation() class requires at least 4 parameters:")
        print(" - gamma0 :\t\t\t lower shear rate limit for interpolation")
        print(" - gammaN :\t\t\t upper shear rate limit for interpolation")
        print(" - Ninterpol :\t\t\t number of interpolation intervals")
        print(" - analytical :\t\t\t analytical viscosity model as instance of Analytical_viscosity() class")
        print(" - (optional) gammaStart :\t lower shear rate limit for plotting")
        print(" - (optional) gammaEnd :\t upper shear rate limit for plotting")
        print(" - (optional) samples :\t\t number of points calculated for plotting and data saving\n")
        print("The following methods must be called:")
        print(" - Interpolation(<params>) :\t\t with all necessary parameters")
        print(" - calculate_interpolation() :\t\t performs the interpolation \n")
        print("The following methods are helpful:")
        print(" - save_interpolation(filename) : \t\t save the viscosity-shearrate data to a file")
        print(" - save_interpolation_parameters(filename) :\t save the interpolation parameters to a file")
        print(" - plot_interpolation(options) :\t\t plot the viscosity-shearrate relationship")
        print("   Options are:")
        print("   - plot_interpolation=<int> : \t\t plot the interpolated viscosity")
        print("   - plot_analytical=<int> : \t\t\t plot the analytical viscosity (numbers give the order in the plot)")
        print("   - draw_intervals=<int> : \t\t\t indicate the interval boundaries with lines")
        print("   - draw_last_interval=<int> : \t\t indicate the last interval used for profile calculation with a thick line (provide last interval with Profiles().get_last_interval())")
        print("   - save_figure=<filename> : \t\t\t save the plot to a file\n\n")

    
    
    def set_interpolation_range(self, gamma0, gammaN) :
        """Set the interpolation range bounds"""
        self.gamma0 = gamma0
        self.gammaN = gammaN
        
        
    def set_full_range(self, gammaStart, gammaEnd) :
        """Set the bounds of the full gamma range, which includes the interpolation range"""
        self.gammaStart = gammaStart
        self.gammaEnd   = gammaEnd
    
    
    def set_interpolation_intervals(self, Nintervals) :
        """Set the number of interpolation intervals in the interpolation range"""
        self.N = Nintervals
        self.reinit_params()
        
        
    def set_analytical_viscostiy(self, analytical) :
        """Set the analytical viscosity model"""
        self.analytical = analytical
        
        
    def reinit_params(self) :
        """Re-initializes the parameter arrays, if number of interpolation intervals, N, changes"""
        self.n          = np.ones(self.N+2)
        self.K          = np.zeros(self.N+2)
        self.gammai     = np.zeros(self.N+3)
        self.fullrange  = range(0,self.N+2,1)        


    def calc_gammai(self, i) :
        """Calculates the upper bound of interval, assuming log-equidistant partitioning of the range"""
        # Equidistant in log scale
        gammai = self.gamma0 * (self.gammaN *1.0/self.gamma0)**(1.0*i * 1.0/self.N)
        return gammai


    def calc_ni(self, i) :
        """Calculates the power-law exponent of interpolation interval i"""
        # The 0th and (n+1)th interval are Newtonian, the exponent equals 1
        if i==0 : 
            return 1.0
        elif i==self.N+1 :
            return 1.0
        else:
            # The exponents of the other intervals are calculated using the range and the analytical form
            ni = 1.0 - self.N * 1.0/(math.log(self.gammaN * 1.0/self.gamma0)) * math.log( self.analytical.calc_visc(self.gammai[i-1]) / self.analytical.calc_visc(self.gammai[i]) )
            return ni
 
   
    def calc_Ki(self,i) :
        """Calculates the power-law consistency parameter of interpolation interval <i>"""
        # The Newtonian regions have a constant viscosity, corresponding to the viscosity
        # at the beginning of the first interval (i=1) and
        # at the end of the last interval (i=N)
        if i==0 :
            return self.analytical.calc_visc(self.gamma0)
        elif i==self.N+1 :
            return self.analytical.calc_visc(self.gammaN)
        else :
            Ki = math.sqrt(self.analytical.calc_visc(self.gammai[i-1]) * self.analytical.calc_visc(self.gammai[i])) * (self.gammai[i-1] * self.gammai[i])**(0.5*(1.0-self.n[i]))
            return Ki
          
          
    def calc_powerlaw_visc(self, gamma, i) :
        """Calculate the interpolated viscosity at shear rate <gamma> in interval <i>"""
        visc = self.K[i] * gamma**(self.n[i] - 1.0)
        return visc

    def find_interval(self,gamma) :
        """Determine the index of the interval that includes the shear rate <gamma>"""
        i = 0
        while( self.gammai[i] < gamma) :
            i += 1
        return i

   
    def calculate_interpolation(self) :
        """Fills the parameter lists for the interpolation function parameters"""
        # Each list is filled in a separate for-loop, because the calculation requires the previously calculated values       
        # First, the complete shear rate (gamma) range is defined, which includes the interpolated range and the Newtonian regions
        self.gammai[-1]       = self.gammaStart # gammai[-1] equals gammai[N+2]
        self.gammai[self.N+1] = self.gammaEnd
        for i in range(0,self.N+1,1) :
            self.gammai[i]    = self.calc_gammai(i)         
        # Calculate the power-law exponents and consistency parameters of the interpolation functions
        # This range includes the Newtonian regions
        for i in self.fullrange :
            self.n[i] = self.calc_ni(i)
            
        for i in self.fullrange :
            self.K[i] = self.calc_Ki(i)
        # Store data in array
        self.store_data()

        
    def store_data(self) : 
        """Store the plotting data for viscosity(shear rate) in data array"""
        # Determine the range in r (radial position)
        for i in self.samplerange :
            # Find corresponding interval to shear rate gamma (equidistant on logarithmic scale)
            gamma = self.gammaStart * (self.gammaEnd*1.0/self.gammaStart)**(i*1.0/self.samples)
            interval = int(self.find_interval(gamma))
            # Shear rate
            self.data[0][i] = gamma
            # Viscosity (interpolated)
            self.data[1][i] = self.calc_powerlaw_visc(gamma, interval)
            # Viscosity (analytical)
            self.data[2][i] = self.analytical.calc_visc(gamma)
            # interpolation interval
            self.data[3][i] = interval
        
        
    def save_interpolation(self, filename) :
        """Save the interpolated and analytical viscosity as function of the shear rate"""
        if filename == None :
            print('Provide filename and path: save_interpolation("path/to/data.dat")')
        else :
            filehandle = open(filename, 'w')
            filehandle.write("# shearrate\tvisc(interpol)\tvisc(analytical)\tinterval")
            for i in self.samplerange :
                filehandle.write("%.6E\t%.6E\t%.6E\t%i\n" % (self.data[0][i],self.data[1][i],self.data[2][i],self.data[3][i]) )


    def save_interpolation_parameters(self, filename) :
        """Save the interpolation function parameters of each interval to a file"""
        if filename == None :
            print('Provide filename and path: save_interpolation_params("path/to/data.dat")')
        else :        
            filehandle = open(filename, 'w')
            filehandle.write("# i\tgammai\t\tKi\t\tni\n")
            for i in self.fullrange :
                filehandle.write("%i\t%.6E\t%.6E\t%.6E\n" % (i, self.gammai[i], self.K[i], self.n[i]))

    
    def plot_interpolation(self, plot_interpolation=1, plot_analytical=None, draw_intervals=None, draw_last_interval=None, draw_special_shearrate=None, save_figure=None, ymin=None, ymax=None, xlabel=None, ylabel=None, title=None) :
        """Plot the interpolation function (and optional the analytical form) in a log-log plot"""
        # Create plot
        fig = plt.figure(1, figsize=(12,9))
        fig.add_subplot(111)
        # Add plot title
        if title == None :
            fig.suptitle('Viscosity interpolation', fontsize=20)
        else :
            fig.suptitle(title, fontsize=20)
        # Add axis labels
        if xlabel == None :
            plt.xlabel(r'shear rate $\dot{\gamma}$ in $\frac{1}{\mathrm{s}}$', fontsize=18)
        else : 
            plt.xlabel(xlabel, fontsize=18)
        if ylabel == None :
            plt.ylabel(r'Dynamic viscosity $\eta(\dot{\gamma})$ in $\mathrm{Pa\,s}$', fontsize=18)
        else :
            plt.ylabel(ylabel, fontsize=18)

        # Set plot ranges
        plt.xlim(self.gammaStart, self.gammaEnd);
        if ymin != None and ymax != None : 
            plot_ymin = ymin
            plot_ymax = ymax
        else :
            plot_ymin = 1.0e-1*self.K[self.N+1]
            plot_ymax = 1.0e1*self.K[0]           
        plt.ylim(plot_ymin, plot_ymax);  

        # Draw the interpolated intervals
        if draw_intervals != None :
            # Dummy plot for interpolation intervals legend
            plt.loglog([], linewidth=0.5, color='0.25', label=r'Interpolation intervals: $\dot{\Gamma}_i$')
            # Draw intervals as vertical lines
            for i in self.fullrange :
                plt.axvline(self.gammai[i], linewidth=0.5, color='0.25')
        
        # Draw the k-th interval from flow profile calculation
        if draw_last_interval != None : 
            plt.axvline(self.gammai[draw_last_interval], linewidth=2.0, color='green', label=r'Last interval of profile calculation: $\dot{\Gamma}_{k=%i} = %.3E}$'%(draw_last_interval, self.gammai[draw_last_interval]))

        # Draw an additional vertical line at a specific shear rate       
        if draw_special_shearrate != None : 
            plt.axvline(draw_special_shearrate, linewidth=2.0, color='cyan', label=r'Survival shear rate: $\dot{\gamma}_{crit} = %.3E}$'%(draw_special_shearrate))    
    
        # Determine the plot order: interpolation in front of or behind analytical
        if plot_analytical != None :
            if plot_interpolation < plot_analytical :
                # Print the analytical viscosity model
                plt.loglog(self.data[0][self.samplerange], self.data[2][self.samplerange], 'r-', label=r'Carreau-Yasuda model: $\tilde{\eta}(\dot{\gamma})$') # : $\tilde{\eta}(\dot{\gamma}) = \eta_\infty + \frac{\eta_o - \eta_\infty}{(1 + (K\dot{\gamma})^{a_1})^{a_2(a_1)}}$            
                # Print the interpolation function
                plt.loglog(self.data[0][self.samplerange], self.data[1][self.samplerange], 'b-', label=r'Interpolation: $\eta(\dot{\gamma})$') # : $\eta(\dot{\gamma})$       
            else :
                # Print the interpolation function
                plt.loglog(self.data[0][self.samplerange], self.data[1][self.samplerange], 'b-', label=r'Interpolation: $\eta(\dot{\gamma})$') # : $\eta(\dot{\gamma})$       
                # Print the analytical viscosity model
                plt.loglog(self.data[0][self.samplerange], self.data[2][self.samplerange], 'r-', label=r'Carreau-Yasuda model: $\tilde{\eta}(\dot{\gamma})$') # : $\tilde{\eta}(\dot{\gamma}) = \eta_\infty + \frac{\eta_o - \eta_\infty}{(1 + (K\dot{\gamma})^{a_1})^{a_2(a_1)}}$
        else : 
            # Print only the interpolation function
            plt.loglog(self.data[0][self.samplerange], self.data[1][self.samplerange], 'b-', label=r'Interpolation: $\eta(\dot{\gamma})$') # : $\eta(\dot{\gamma})$                   

        # Key options
        plt.legend(loc='lower left')
        
        # Show plot in Console    
        plt.show()
        
        # Print plot to file
        if save_figure != None :
            fig.savefig(save_figure)            
        
  

class Analytical_Viscosity :    
    """Analytical_viscosity class holds the parameters for the analytical viscosity model"""
    def __init__(self, eta0=None, etainf=None, K=None, a1=None, a2=None) :
        """Initialize the parameters of the Carreau-Yasuda model"""
        # Check for correct number of arguments
        if eta0 == None or etainf == None or K == None or a1 == None or a2 == None :
            self.print_usage()
        else :
            self.eta0   = eta0
            self.etainf = etainf
            self.K      = K
            self.a1     = a1
            self.a2     = a2
   
    @staticmethod     
    def print_usage() :
        """Method used to print input instructions."""
        print(" --- Analytical_Viscosity class usage ---")
        print("\nAnalytical_Viscosity() class requires 5 parameters (Carreau-Yasuda viscosity model):")
        print(" - eta0 :\t viscosity in the limit of zero shear rate")
        print(" - etainf :\t viscosity in the limit of infinite shear rate")
        print(" - K :\t\t time constant or inverse of corner shear rate")
        print(" - a1 :\t\t first exponent")
        print(" - a2 :\t\t second exponent\n\n")
        
    
    def set_parameters(self, eta0, etainf, K, a1, a2) :
        """Set the parameters of the Carreau-Yasuda model"""
        self.eta0   = eta0
        self.etainf = etainf
        self.K      = K
        self.a1     = a1
        self.a2     = a2
        
        
    def calc_visc(self, gamma) :
        """Calculate the viscosity at given shear rate for the analytical viscosity model"""
        visc = self.etainf + (self.eta0 - self.etainf) / ( (1.0 + (self.K * gamma)**(self.a1) )**(self.a2*1.0/self.a1) )
        return visc
        
        
    def save_parameters(self, filename, data_format='%.6E') :
        """Save the data of the analytical viscosity model to a file"""
        datafile = open(filename, 'w')
        datafile.write(("eta0\t = \t"+data_format+"\n") % self.eta0)
        datafile.write(("etainf\t = \t"+data_format+"\n") % self.etainf)
        datafile.write(("K\t = \t"+data_format+"\n") % self.K)
        datafile.write(("a1\t = \t"+data_format+"\n") % self.a1)
        datafile.write(("a2\t = \t"+data_format+"\n") % self.a2)
        datafile.write("\n# The viscosity is calculated from the shear rate (shear) using the formula: etainf + (eta0-etainf)/((1.0+(K*shear)^a1)^(a2/a1))\n")            
