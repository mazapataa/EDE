import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg to avoid Qt backend issues
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from fgivenx import plot_contours, samples_from_getdist_chains
from simplemc.models.SplineGHDECosmology import SplineGHDECosmology

# Set up LaTeX for text rendering
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\boldmath'

# Initialize the cosmology model
T = SplineGHDECosmology(mean=-2.0, nodes=1, interp='lineal')

def EoS_1node(z, theta):
    Om0 = theta[0]
    h = theta[1]
    c_value = theta[2]
    node_value = theta[3]
    Ode0 = 1 - Om0  # Assuming flat universe (Ok = 0)

    # Initialize a new cosmology model instance
    T = SplineGHDECosmology(mean=-2.0, nodes=1, interp='lineal')
    T.Om = Om0
    T.h = h
    T.c_par.setValue(c_value)
    T.updateParams([T.c_par])
    
    # Set the node value
    T.params[0].setValue(node_value)
    T.updateParams([T.params[0]])

    # Compute the Equation of State (EoS) values
    eos_values = [T.EoS(zi) for zi in z]
    return eos_values

# Get redshift values
z = T.zvals

# Load samples from GetDist chains
file_root1 = '/home/alfonsozapata/Downloads/spline_1/'
file_root1 += 'SplGen1_phy_Union3+DESI+RiessH0_21_nested_multi'
samples1, weights1 = samples_from_getdist_chains(['Om', 'h', 'c', 'amp_0'], file_root1, settings={'ignore_rows': 0.0})

# Create the contour plot
cbar1 = plot_contours(EoS_1node, z, samples1, weights=weights1, contour_line_levels=[1, 2], colors=plt.cm.Blues_r)
cbar1 = plt.colorbar(cbar1, ticks=[0, 1, 2])
cbar1.set_ticklabels(['', r'$1\sigma$', r'$2\sigma$'])

# Add labels and title
plt.ylabel(r'$\rm w_{de}(z)$', fontsize=18)
plt.xlabel(r'$z$', fontsize=18)
plt.title('1-Node: DESI+Union3+Riess', fontsize=16)
plt.ylim(-2, 0)

# Add a horizontal line for LCDM (w = -1)
lcdm = np.full(100, -1.0)
plt.plot(z, lcdm, linestyle='-', color='darkred', label=r'$\Lambda$CDM ($w = -1$)')

# Add grid and legend
plt.grid()
plt.legend(fontsize=12)

plt.show()
