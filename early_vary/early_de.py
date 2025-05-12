import matplotlib
matplotlib.use('TkAgg')  # Try TkAgg to avoid Qt issues; revert to Qt5Agg if Qt is fixed
import numpy as np
import matplotlib.pyplot as plt

# Constants and parameters
Omega_ede0 = 0.7       
Omega_eEDE_values = np.linspace(0.001, 0.2, 8)
Omega_m0 = 0.3         
w0 = -1.0            
a_eq = 1/3400          

z = np.logspace(-3, 3, 100)  
a = 1/(1 + z)           

def Omega_ede(a, Omega_eEDE):
    term1 = (Omega_ede0 - Omega_eEDE * (1 - a**(-3*w0)))
    term2 = (Omega_ede0 + Omega_m0 * a**(3*w0))
    term3 = Omega_eEDE * (1 - a**(-3*w0))
    return term1/term2 + term3

def dlnOmega_dlna(a, Omega_eEDE):
    eps = 1e-6
    a_plus = a * (1 + eps)
    a_minus = a * (1 - eps)
    Omega_plus = Omega_ede(a_plus, Omega_eEDE)
    Omega_minus = Omega_ede(a_minus, Omega_eEDE)
    return (np.log(Omega_plus) - np.log(Omega_minus)) / (2 * eps)

def w_ede(a, Omega_eEDE):
    term1 = -1/(3*(1 - Omega_ede(a, Omega_eEDE))) * dlnOmega_dlna(a, Omega_eEDE)
    term2 = a_eq/(3*(a + a_eq))
    return term1 + term2

# Create plot
fig = plt.figure(figsize=(12, 6))

# Color map for different Omega_eEDE values
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=min(Omega_eEDE_values), vmax=max(Omega_eEDE_values))

# Plot Omega_ede(z) for all values
ax1 = plt.subplot(1, 2, 1)
for i, Omega_eEDE in enumerate(Omega_eEDE_values):
    Omega_ede_vals = np.array([Omega_ede(ai, Omega_eEDE) for ai in a])
    plt.loglog(z, Omega_ede_vals, '-', lw=2, color=cmap(norm(Omega_eEDE)))

plt.axhline(Omega_ede0, color='k', linestyle='--', alpha=0.5,label=f'$\\Omega_{0}$ = {Omega_ede0:.1f}')
plt.xlabel('Redshift (z)', fontsize=12)
plt.ylabel(r'$\Omega_{\rm ede}(z)$', fontsize=12)
plt.title('Early Dark Energy Density', fontsize=14)
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.3)



# Plot w_ede(z) for all values
ax2 = plt.subplot(1, 2, 2)
for i, Omega_eEDE in enumerate(Omega_eEDE_values):
    w_ede_vals = np.array([w_ede(ai, Omega_eEDE) for ai in a])
    plt.semilogx(z, w_ede_vals, '-', lw=2, color=cmap(norm(Omega_eEDE)))

plt.axhline(w0, color='k', linestyle='--', alpha=0.5, label='$w_0$')
plt.axhline(-1/3, color='g', linestyle='--', alpha=0.5, label='$-1/3$')
plt.xlim(1e-2, 1e4)  
plt.ylim(-1.1, 0)    
plt.xlabel('Redshift (z)', fontsize=12)
plt.ylabel(r'$w_{\rm ede}(z)$', fontsize=12)
plt.title('EDE Equation of State', fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend(fontsize=10)

# Add colorbar for second subplot
cbar2 = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax2)
cbar2.set_label(r'$\Omega_e^{\rm EDE}$', fontsize=12)

plt.tight_layout()
plt.show()