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
#            IMPORT GENERAL PACKAGES AND CUSTOM CLASSES
# ============================================================================
# os package used for creating output directory (Note: this may differ for other operating systems)
import os
from CYprofiles import Interpolation, Analytical_Viscosity, Printing_Parameters, Profiles


# ============================================================================
#            GENERAL DESCRIPTION AND TIPPS
# ============================================================================
# The programm is structured in 4 main classes:
#    - Interpolation() class does the interpolation of the analytical form of the viscosity in dependence of the shear rate
#    - Analytical_Viscosity() class holds the parameters of the analytical model (Carreau-Yasuda model)
#    - Printing_Parameters() class holds the parameters of the printer (pressure and radius)
#    - Profiles() class does the actual calculation of the flow profiles for a given Interpolation() and Printing_Parameters()

# To obtain flow profiles, the following steps need to be performed:
#    1. define the parameters of the analytical viscosity model
#    2. define the parameters for the interpolation
#    3. calculate the viscosity interpolation
#    4. define the printing parameters
#    5. define the profiles parameters
#    6. calculate the flow profiles
#    7. (optional) save or plot the calculated data

# Print the implemented usage instructions for each class
#Analytical_Viscosity().print_usage()
#Interpolation().print_usage()
#Printing_Parameters().print_usage()
#Profiles().print_usage()

# ============================================================================
#            OUTPUT DIRECTORY
# ============================================================================
# --- Specify output directory ---
# This part sets the name of the output directory and checks whether it exists and, if not, creates a new directory
outDir = "tmp/"
if not os.path.exists(outDir) :
    os.mkdir(outDir)


# ============================================================================
#            STEP 1: PERFORMING THE INTERPOLATION
# ============================================================================
# --- Carreau-Yasuda model parameters ---
eta0   = 1.0e2          # viscosity in the limit of zero shear rates (in Pa*s)
etainf = 1.0e-3         # viscosity in the limit of infinite shear rates (in Pa*s)
K      = 1.0e-3         # consistency parameter (sometimes its inverse "corner shear rate") (in s)
a1     = 0.3            # first exponent
a2     = 0.9            # second exponent

# --- Initialize Carreau-Yasuda viscosity model ---
analytical = Analytical_Viscosity(eta0=eta0, etainf=etainf, K=K, a1=a1, a2=a2)

# --- Interpolation parameters ---
gamma0 = 1.0e-6         # start shear rate for the interpolation
gammaN = 1.0e6          # end shear rate for the interpolation
Ninterpol = 100         # number of interpolation intervals

# --- Initialize interpolation ---
interpol = Interpolation(gamma0=gamma0, gammaN=gammaN, Ninterpol=Ninterpol, analytical=analytical)

# --- Perform the interpolation ---
interpol.calculate_interpolation()

# --- Plot the viscosity interpolation ---
interpol.plot_interpolation(plot_analytical=1, plot_interpolation=2, draw_intervals=1, title="Viscosity interpolation")

# --- Save the viscosity interpolation data ---
interpol.save_interpolation(outDir+"viscosity-shearrate.dat")
interpol.analytical.save_parameters(outDir+"analytical-parameters.dat")
interpol.save_interpolation_parameters(outDir+"interpolation-parameters.dat")


# ============================================================================
#            STEP 2: CALCULATING THE FLOW PROFILE
# ============================================================================
# --- Printing parameters ---
rchannel = 1.0e-4       # radius of the cylindrical channel (in m)
# (a) pressure gradient
pgrad    = -1.0e7       # pressure gradient (in Pa/m) along the channel (by definition negative!)
# (b) pressure difference and channel length
#pressure = -1.0e5       # pressure difference (in Pa) along the channel (by definition negative!)
#lchannel = 1.0e-2       # length of the nozzle (in m) 
# (c) flow rate
#flowrate = 1.0e-9       # flow rate (in m^3/s)

# --- Initialize printing parameters ---
# Create the printing parameters with all necessary parameters
# (a) with a pressure gradient
printparams = Printing_Parameters(pressureGradient=pgrad, channelRadius=rchannel)                                    # with a pressure gradient
# (b) with a pressure difference and a channel length
#printparams = Printing_Parameters(pressureDifference=pressure, channelLength=lchannel, channelRadius=rchannel)      # with pressure difference and channel length
# (c) with a flow rate
#printparams = Printing_Parameters(flowrate=flowrate, channelRadius=rchannel)                                        # with a flow rate

# --- Initiliaze flow profiles ---
fluidprofiles = Profiles(interpolation=interpol, printingParameters=printparams)

# --- Perform the profile calculation ---
fluidprofiles.calculate_profiles()

# --- Plot the profiles ---
fluidprofiles.plot_velocity(draw_intervals=1, save_figure=outDir+"velocity.png")
fluidprofiles.plot_shearrate(draw_intervals=1, save_figure=outDir+"shearrate.png")
fluidprofiles.plot_viscosity(draw_intervals=1, save_figure=outDir+"viscosity.png")
fluidprofiles.plot_shearstress(draw_intervals=1, save_figure=outDir+"shearstress.png")

# --- Save the profiles data ---
fluidprofiles.save_profiles(outDir+"profiles.dat")
fluidprofiles.save_averages(outDir+"averages.dat")

# -- Plot the viscosity interpolation again, including the last interval used for profile calculation ---
# --- Plot the viscosity interpolation ---
interpol.plot_interpolation(plot_analytical=1, plot_interpolation=2, draw_intervals=1, title="Viscosity interpolation", save_figure=outDir+"interpolation.png", draw_last_interval=fluidprofiles.get_last_interval())
