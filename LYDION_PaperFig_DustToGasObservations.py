import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

Dsun = 1 / 162
xsun = 8.69

def D_Dsun_Remy2014(Z_Zsun):
    if Z_Zsun < 0.2:
        D_Dsun = 0.2 * ((Z_Zsun / 0.2)**3.1)
    else:
        D_Dsun = Z_Zsun

    return D_Dsun

D_Dsun_Remy2014 = np.vectorize(D_Dsun_Remy2014)

# Remy-Ruyer+ (2014) provide data down to 12 + log10(O/H) ~ 7.0, 
# which corresponds to Z/Zsun ~ 0.02:
Z_Zsun_min_Remy = 0.02

def D_Dsun_Galliano2021(Z_Zsun):

    # Compute x = 12 + log_10(O/H) from Z/Zsun:
    x = xsun + np.log10(Z_Zsun)

    # Scaling relations of Galliano+ (2021):
    if x > 7.3:
        log10_D = ( 11471.808  -  5669.5959 * x  +  1045.9713 * (x**2)
                  - 85.434332 * (x**3) +  2.6078774 * (x**4) )
    else:
        log10_D = - 13.230 + x 

    # Compute D/Dsun from log10_D:
    D_Dsun = (10**log10_D) / Dsun 

    return D_Dsun

D_Dsun_Galliano2021 = np.vectorize(D_Dsun_Galliano2021)


# Minimum Z/Zsun for Galliano+ (2021) scaling relation is set 
# by the minimum x = 12 + log_10(O/H) = 7.1 in their paper (see their 
# fig. 13):
xmin_Galliano2021   = 7.1
Z_Zsun_min_Galliano = 10**(xmin_Galliano2021 - xsun)


######## Algera et al. (2026) data, for z ~ 7. #################

REBELS_names = np.array([
    "REBELS-05", "REBELS-08", "REBELS-12", "REBELS-14",
    "REBELS-15", "REBELS-18", "REBELS-25", "REBELS-29",
    "REBELS-32", "REBELS-34", "REBELS-38", "REBELS-39"
])

# Gas-phase metallicity: 12 + log10(O/H)
OH_REBELS = np.array([
    8.51, 8.22, 8.23, 7.90,
    7.78, 8.50, 8.62, 8.73,
    8.48, 8.33, 8.28, 8.02
])

# log10(Mdust / MH2)
# For REBELS-15 and REBELS-34 these are upper limits.
logDtG_REBELS = np.array([
    -3.16, -3.01, -3.10, -2.89,
    -2.58, -3.26, -2.45, -3.11,
    -3.26, -3.10, -2.81, -3.12
])

is_upper_limit_REBELS = np.array([
    False, False, False, False,
    True,  False, False, False,
    False, True,  False, False
])

Z_Zsun_REBELS = 10**(OH_REBELS - xsun)
D_Dsun_REBELS = 10**(logDtG_REBELS) / Dsun

# Uncertainties from Algera et al. (2025), Table 1.
# Metallicities are quoted as symmetric errors in 12 + log10(O/H).
OH_REBELS_err = np.array([
    0.16, 0.20, 0.13, 0.12,
    0.30, 0.13, 0.17, 0.15,
    0.13, 0.29, 0.18, 0.29
])

# Asymmetric errors on log10(Mdust / MH2).
# For upper limits, set to NaN because we do not plot y-error bars for limits.
logDtG_REBELS_err_upper = np.array([
    0.56, 0.59, 0.67, 0.62,
    np.nan, 0.63, 0.63, 0.57,
    0.57, np.nan, 0.48, 0.61
])

logDtG_REBELS_err_lower = np.array([
    0.47, 0.47, 0.54, 0.50,
    np.nan, 0.49, 0.54, 0.47,
    0.46, np.nan, 0.48, 0.50
])

# Convert metallicity errors from dex to linear Z/Zsun error bars.
Z_Zsun_REBELS_lower = 10**((OH_REBELS - OH_REBELS_err) - xsun)
Z_Zsun_REBELS_upper = 10**((OH_REBELS + OH_REBELS_err) - xsun)

Z_Zsun_REBELS_err_lower = Z_Zsun_REBELS - Z_Zsun_REBELS_lower
Z_Zsun_REBELS_err_upper = Z_Zsun_REBELS_upper - Z_Zsun_REBELS

# Convert log(DtG) errors from dex to linear D/Dsun error bars.
D_Dsun_REBELS_lower = 10**(logDtG_REBELS - logDtG_REBELS_err_lower) / Dsun
D_Dsun_REBELS_upper = 10**(logDtG_REBELS + logDtG_REBELS_err_upper) / Dsun

D_Dsun_REBELS_err_lower = D_Dsun_REBELS - D_Dsun_REBELS_lower
D_Dsun_REBELS_err_upper = D_Dsun_REBELS_upper - D_Dsun_REBELS

########

# Algera+ also provide an estimate of dust-to-gas ratio for other high-z galaxies,
# like B14-65666 at z = 7.15. They give the following parameters:

alpha_CII     = 31.0

OH_B146566    = 8.18
Mdust_B146566 = 1e7 
L_CII_B146566 = 11e8 
M_gas_B146566 = alpha_CII * L_CII_B146566

Z_Zsun_B146566 = 10**(OH_B146566 - xsun)
D_Dsun_B146566 = (Mdust_B146566 / M_gas_B146566) / Dsun


######## Mitsuhashi et al. (2026) data, z > 12 #################

# GHZ2: z = 12.3
# Metallicity: Z/Zsun = 0.05 (+0.12/-0.03) from Zavala+2025 line diagnostics
Z_Zsun_GHZ2           = 0.05
Z_Zsun_GHZ2_err_lower = 0.03
Z_Zsun_GHZ2_err_upper = 0.12

logMdust_GHZ2  = 5.0                             # 3-sigma upper limit
logMgas_GHZ2   = 8.8
logMgas_GHZ2_err = 0.3
D_Dsun_GHZ2    = (10**logMdust_GHZ2 / 10**logMgas_GHZ2) / Dsun  # upper limit

# GS-z14-0: z = 14.2
# Metallicity: Z/Zsun = 0.17 (+0.01/-0.01), but +0.2 dex floor on uncertainty
Z_Zsun_GSz14           = 0.17
Z_Zsun_GSz14_err_lower = Z_Zsun_GSz14 - Z_Zsun_GSz14 / 10**0.2
Z_Zsun_GSz14_err_upper = Z_Zsun_GSz14 * 10**0.2 - Z_Zsun_GSz14

logMdust_GSz14 = 5.3                             # 3-sigma upper limit
logMgas_GSz14  = 8.7
logMgas_GSz14_err = 0.3
D_Dsun_GSz14   = (10**logMdust_GSz14 / 10**logMgas_GSz14) / Dsun  # upper limit

# Gas mass uncertainty propagates directly to D/G uncertainty (dust is upper limit):
# log(D/G) = logMdust - logMgas - log(Dsun), so delta_log(D/G) = delta_logMgas
D_Dsun_GHZ2_upper  = (10**logMdust_GHZ2  / 10**(logMgas_GHZ2  - logMgas_GHZ2_err))  / Dsun
D_Dsun_GSz14_upper = (10**logMdust_GSz14 / 10**(logMgas_GSz14 - logMgas_GSz14_err)) / Dsun

# Error bars point upward only (gas mass uncertainty; dust is a limit so no lower bar)
D_Dsun_GHZ2_err_upper  = D_Dsun_GHZ2_upper  - D_Dsun_GHZ2
D_Dsun_GSz14_err_upper = D_Dsun_GSz14_upper - D_Dsun_GSz14

######## Takechi et al. (2026) MACS0416-Y1, z = 8.312 ########

# MACS0416-Y1 / Y1 at z = 8.312.
# Takechi+2026:
# 12 + log10(O/H) = 7.86 (+0.09/-0.08)
# log10(Mdust/Mgas) = -3.60 (+0.29/-0.22)
OH_Y1           = 7.86
OH_Y1_err_lower = 0.08
OH_Y1_err_upper = 0.09

logDtG_Y1           = -3.60
logDtG_Y1_err_lower = 0.22
logDtG_Y1_err_upper = 0.29

Z_Zsun_Y1 = 10**(OH_Y1 - xsun)
D_Dsun_Y1 = 10**logDtG_Y1 / Dsun

# Convert metallicity errors from dex to linear Z/Zsun error bars.
Z_Zsun_Y1_lower = 10**((OH_Y1 - OH_Y1_err_lower) - xsun)
Z_Zsun_Y1_upper = 10**((OH_Y1 + OH_Y1_err_upper) - xsun)

Z_Zsun_Y1_err_lower = Z_Zsun_Y1 - Z_Zsun_Y1_lower
Z_Zsun_Y1_err_upper = Z_Zsun_Y1_upper - Z_Zsun_Y1

# Convert log(DtG) errors from dex to linear D/Dsun error bars.
D_Dsun_Y1_lower = 10**(logDtG_Y1 - logDtG_Y1_err_lower) / Dsun
D_Dsun_Y1_upper = 10**(logDtG_Y1 + logDtG_Y1_err_upper) / Dsun

D_Dsun_Y1_err_lower = D_Dsun_Y1 - D_Dsun_Y1_lower
D_Dsun_Y1_err_upper = D_Dsun_Y1_upper - D_Dsun_Y1

######## Schneider et al. (2016) local metal-poor galaxies ########

Schneider2016_names = np.array([
    "I Zw 18",
    "SBS 0335-052"
])

# Schneider+2016 Table 1:
# Z is given in units of 10^-2 Zsun.
Z_Zsun_Schneider2016 = np.array([
    3.09e-2,   # I Zw 18
    3.80e-2    # SBS 0335-052
])

# Total gas masses from Table 1, in units of Msun.
# These include ionized + atomic + inferred molecular gas.
Mgas_Schneider2016 = np.array([
    9.15e7,    # I Zw 18
    27.8e7     # SBS 0335-052
])

# Dust masses from Table 1, in units of Msun.
Mdust_Schneider2016 = np.array([
    3.4e2,     # I Zw 18:  (3.4 ± 1.0) x 10^2 Msun
    3.8e4      # SBS 0335-052: (3.8 ± 0.6) x 10^4 Msun
])

Mdust_Schneider2016_err = np.array([
    1.0e2,
    0.6e4
])

D_Dsun_Schneider2016 = (Mdust_Schneider2016 / Mgas_Schneider2016) / Dsun

# Only propagate the quoted dust-mass uncertainty.
# No gas-mass or metallicity uncertainty is plotted here.
D_Dsun_Schneider2016_err = (Mdust_Schneider2016_err / Mgas_Schneider2016) / Dsun

######## Peroux & Howk (2020) DLA data, z ~ 1-5 #################

plot_Peroux2020 = True
color_Peroux_by_redshift = False

Peroux2020_file = "Peroux_SupplementalTable4_Fig7_10.asc"

Peroux_data = np.genfromtxt(Peroux2020_file,
                            skip_header=14,
                            dtype=None,
                            encoding='utf-8',
                            names=['QSO_ID', 'z_abs', 'NHI', 'sig_NHI',
                                   'MH', 'sig_MH', 'log_Z',
                                   'log_DTG', 'log_DTM'])

z_abs_Peroux2020    = Peroux_data['z_abs']
MH_Peroux2020       = Peroux_data['MH']
logDtG_Peroux2020   = Peroux_data['log_DTG']

Z_Zsun_Peroux2020 = 10**MH_Peroux2020
D_Dsun_Peroux2020 = 10**logDtG_Peroux2020 / Dsun


###############################################################
#######                                                 #######
#######           P L O T   C O S M E T I C S           #######
#######                                                 #######
###############################################################

# COMMENT THIS OUT IF YOU DON'T HAVE THE REQUIRED FONTS:

#plt.rc('font',**{'sans-serif':['RomanD'],
                 #'size': 27} )
#plt.rc('font',**{'sans-serif':['AVHershey Simplex'],
                #'size': 27} )
plt.rc('font',**{'sans-serif':['AVHershey Complex'],
                'size': 37} )
#plt.rc('font',**{'sans-serif':['Complex'],
                # 'size': 27} )

params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

# The stuff below need not be removed:

params = {'legend.fontsize': 19 }
plt.rcParams.update(params)

plt.rcParams["legend.markerscale"] = 0.75

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.major.width'] = 1.7
plt.rcParams['xtick.minor.width'] = 1.7
plt.rcParams['xtick.major.size'] = 20
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['ytick.major.width'] = 1.7
plt.rcParams['ytick.minor.width'] = 1.7
plt.rcParams['ytick.major.size'] = 20
plt.rcParams['ytick.minor.size'] = 8
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'

plt.rcParams['legend.frameon'] = 'False'

plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['grid.linewidth'] = 2.5
plt.rcParams['grid.linestyle'] = '--'

# Make ticks appear on all sides and point inwards
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True

###############################################################
#######                                                 #######
#######               P L O T  I T !                    #######
#######                                                 #######
###############################################################

fig, ax = plt.subplots(figsize=(16, 9))

ax.fill_between([1e-4, 3e-2], 0.5, 1.1, 
                color='limegreen', zorder=0, alpha = 0.3, edgecolor='none', linewidth = 0.0)
ax.text(8e-4, 0.6, 'Metallicity of stars in UFDs', color='green', fontsize=28)

ax.fill_between([4e-4, 5e-2], 0.5*(0.5/1.1), 0.5, 
                color='lightcoral', zorder=0, alpha = 0.3, edgecolor='none', linewidth = 0.0)
ax.text(8e-4, 0.27, 'Metallicity of stars in old GCs', color='red', fontsize=28)

####### LYDION assumed dust-to-gas ###############

Z_Zsun_LYDION = np.array([0.01, 0.01])
D_Dsun_LYDION = np.array([1e-3, 0.01])

ax.scatter(Z_Zsun_LYDION[0], D_Dsun_LYDION[0], marker='s', s=400, 
           facecolors='black', edgecolors='black', linewidths=0.0,
           zorder=20)

ax.scatter(Z_Zsun_LYDION[1], D_Dsun_LYDION[1], marker='s', s=400, 
           facecolors='black', edgecolors='black', linewidths=0.0,
           zorder=20)

# One compact label, kept inside the figure.
# Adjust these if needed.
lydion_text_x = 0.00135*0.4
lydion_text_y = 3.0e-2

ax.text(
    lydion_text_x,
    lydion_text_y,
    r'LYDION simulations' + '\n' + 'in this paper',
    color='black',
    fontsize=30,
    ha='left',
    va='center',
    zorder=25
)

# Connector geometry:
# one horizontal trunk from the label, one vertical splitter,
# then two horizontal arrows to the two LYDION points.
x_trunk_start = 0.003
x_split       = 0.0063
x_arrow_end   = 0.0092   # stop slightly before the square marker

y_low   = D_Dsun_LYDION[0]
y_high  = D_Dsun_LYDION[1]
y_trunk = 0.67*lydion_text_y  # this keeps the connector attached when text moves

connector_lw = 2.0
arrow_size   = 22

# Main horizontal trunk from label to splitter
ax.plot(
    [x_trunk_start, x_split],
    [y_trunk, y_trunk],
    color='black',
    lw=connector_lw,
    solid_capstyle='butt',
    zorder=24
)

# Vertical splitter: extend all the way from lower point to trunk
ax.plot(
    [x_split, x_split],
    [y_low, y_trunk],
    color='black',
    lw=connector_lw,
    solid_capstyle='butt',
    zorder=24
)

# Filled arrowhead to lower LYDION point
arrow_low = FancyArrowPatch(
    (x_split, y_low),
    (x_arrow_end, y_low),
    arrowstyle='-|>',
    mutation_scale=arrow_size,
    linewidth=connector_lw,
    edgecolor='black',
    facecolor='black',
    shrinkA=0,
    shrinkB=0,
    zorder=24
)
ax.add_patch(arrow_low)

# Filled arrowhead to upper LYDION point
arrow_high = FancyArrowPatch(
    (x_split, y_high),
    (x_arrow_end, y_high),
    arrowstyle='-|>',
    mutation_scale=arrow_size,
    linewidth=connector_lw,
    edgecolor='black',
    facecolor='black',
    shrinkA=0,
    shrinkB=0,
    zorder=24
)
ax.add_patch(arrow_high)

####### Plot Peroux & Howk (2020) DLA data #######

if plot_Peroux2020:

    if color_Peroux_by_redshift:

        sc_Peroux = ax.scatter(Z_Zsun_Peroux2020, D_Dsun_Peroux2020,
                               c=z_abs_Peroux2020,
                               cmap='viridis',
                               marker='*',
                               s=55,
                               edgecolors='black',
                               linewidths=0.0,
                               alpha=0.85,
                               zorder=1,
                               label='Peroux & Howk (2020)')

        cbar_Peroux = fig.colorbar(sc_Peroux, ax=ax, pad=0.02)
        cbar_Peroux.set_label(r'Absorber redshift $z_{\rm abs}$',
                              labelpad=12)
        cbar_Peroux.ax.tick_params(which='both',
                                   direction='in',
                                   length=8,
                                   width=1.5)

    else:

        ax.scatter(Z_Zsun_Peroux2020, D_Dsun_Peroux2020,
                   marker='s',
                   s=80,
                   facecolors='lightgray',
                   edgecolors='black',
                   linewidths=0.0,
                   alpha=1.0,
                   zorder=1,
                   label='Peroux & Howk (2020)')
        
###################################################

Z_Zsun_Remy2014     = np.logspace(np.log10(Z_Zsun_min_Remy), 0.0, 300)
Z_Zsun_Galliano2021 = np.logspace(np.log10(Z_Zsun_min_Galliano), 0.0, 300)

# Remy+ (2014) scaling:
D_Dsun_remy  = D_Dsun_Remy2014(Z_Zsun_Remy2014)

ax.plot(Z_Zsun_Remy2014, D_Dsun_remy, label='Remy et al. (2014)', 
        color='black', linewidth = 3.0, linestyle = '--', dash_capstyle = 'round')

# Galliano+ (2021) scaling, including 95% confidence interval:
D_Dsun_galliano           = D_Dsun_Galliano2021(Z_Zsun_Galliano2021)
D_Dsun_Galliano2021_upper = 10**(0.70)  * D_Dsun_galliano
D_Dsun_Galliano2021_lower = 10**(-0.68) * D_Dsun_galliano

ax.fill_between(Z_Zsun_Galliano2021, D_Dsun_Galliano2021_lower, D_Dsun_Galliano2021_upper,
                color='orangered', alpha=0.3, edgecolor='none', linewidth = 0.0)

ax.plot(Z_Zsun_Galliano2021, D_Dsun_galliano, label='Galliano et al. (2021)', 
        color='orangered', linewidth = 3.0)

####### Plot REBELS data from Algera et al. (2025) #######

det = ~is_upper_limit_REBELS
lim =  is_upper_limit_REBELS

# Error bars for detections.
ax.errorbar(Z_Zsun_REBELS[det], D_Dsun_REBELS[det],
            xerr=[Z_Zsun_REBELS_err_lower[det],
                  Z_Zsun_REBELS_err_upper[det]],
            yerr=[D_Dsun_REBELS_err_lower[det],
                  D_Dsun_REBELS_err_upper[det]],
            fmt='none',
            ecolor='black',
            elinewidth=1.6,
            capsize=4,
            capthick=1.6,
            zorder=9)

# For upper limits, only show metallicity error bars.
ax.errorbar(Z_Zsun_REBELS[lim], D_Dsun_REBELS[lim],
            xerr=[Z_Zsun_REBELS_err_lower[lim],
                  Z_Zsun_REBELS_err_upper[lim]],
            yerr=None,
            fmt='none',
            ecolor='black',
            elinewidth=1.6,
            capsize=4,
            capthick=1.6,
            zorder=9)

ax.scatter(Z_Zsun_REBELS[det], D_Dsun_REBELS[det],
           marker='o',
           s=200,
           facecolors='red',
           edgecolors='black',
           linewidths=2.0,
           zorder=10,
           label='REBELS-IFU')

ax.scatter(Z_Zsun_REBELS[lim], D_Dsun_REBELS[lim],
           marker='v',
           s=200,
           facecolors='red',
           edgecolors='black',
           linewidths=2.0,
           zorder=10,
           label='REBELS-IFU upper limits')

############ Mitsuhashi+ (2026) z > 12 data ######################

# Plot Mitsuhashi+ (2026) z > 12 data:
ax.scatter(
    [Z_Zsun_GHZ2, Z_Zsun_GSz14],
    [D_Dsun_GHZ2, D_Dsun_GSz14],
    marker='v',               # downward triangle = upper limit
    s=200,
    facecolors='dodgerblue',
    edgecolors='black',
    linewidths=2.0,
    zorder=10,
    label='Mitsuhashi+ (2026), z > 12'
)


# Error bars for Misuhashi+ (2026) z > 12 data:
ax.errorbar(
    [Z_Zsun_GHZ2, Z_Zsun_GSz14],
    [D_Dsun_GHZ2, D_Dsun_GSz14],
    xerr=[[Z_Zsun_GHZ2_err_lower, Z_Zsun_GSz14_err_lower],
          [Z_Zsun_GHZ2_err_upper, Z_Zsun_GSz14_err_upper]],
    yerr=[[0, 0],
          [D_Dsun_GHZ2_err_upper, D_Dsun_GSz14_err_upper]],
    fmt='none',
    ecolor='dodgerblue',
    elinewidth=1.6,
    capsize=4,
    capthick=1.6,
    zorder=9
)

# Add downward arrows to emphasize that these are upper limits.
# On a log y-axis, it is often best to choose the arrow length as a fraction
# of the plotted value.

arrow_frac = 0.35  # adjust to taste

yerr_arrow_GHZ2  = D_Dsun_GHZ2  * arrow_frac
yerr_arrow_GSz14 = D_Dsun_GSz14 * arrow_frac

uplim_container = ax.errorbar(
    [Z_Zsun_GHZ2, Z_Zsun_GSz14],
    [D_Dsun_GHZ2, D_Dsun_GSz14],
    yerr=[[yerr_arrow_GHZ2, yerr_arrow_GSz14], [0, 0]],
    fmt='none',
    ecolor='dodgerblue',
    elinewidth=1.8,
    capsize=0,
    uplims=True,
    zorder=11
)

upper_limit_arrow_size = 10
upper_limit_arrow_width = 1.3

for artist in uplim_container[1]:
    artist.set_color('dodgerblue')
    artist.set_markeredgecolor('dodgerblue')
    artist.set_markerfacecolor('dodgerblue')
    artist.set_markersize(upper_limit_arrow_size)
    artist.set_markeredgewidth(upper_limit_arrow_width)

for artist in uplim_container[2]:
    artist.set_color('dodgerblue')

ax.text(1.9e-2, 6e-2, 'Mitsuhashi+ (2026)\nz ~ 12 - 14', color='dodgerblue', fontsize=22)

######## Plot Takechi et al. (2026) MACS0416-Y1 ########

ax.errorbar(
    Z_Zsun_Y1,
    D_Dsun_Y1,
    xerr=[[Z_Zsun_Y1_err_lower], [Z_Zsun_Y1_err_upper]],
    yerr=[[D_Dsun_Y1_err_lower], [D_Dsun_Y1_err_upper]],
    fmt='none',
    ecolor='purple',
    elinewidth=1.8,
    capsize=4,
    capthick=1.8,
    zorder=12
)

ax.scatter(
    Z_Zsun_Y1,
    D_Dsun_Y1,
    marker='D',
    s=260,
    facecolors='purple',
    edgecolors='black',
    linewidths=2.0,
    zorder=13,
    label='Takechi+ (2026), z = 8.3'
)

ax.text(
    0.2,
    0.016,
    'Takechi+ (2026)\nz = 8.3',
    color='purple',
    fontsize=22,
    ha='left',
    va='center',
    zorder=14
)

######## Plot Schneider et al. (2016) local galaxies ########

ax.errorbar(
    Z_Zsun_Schneider2016,
    D_Dsun_Schneider2016,
    xerr=None,
    yerr=D_Dsun_Schneider2016_err,
    fmt='none',
    ecolor='deeppink',
    elinewidth=1.8,
    capsize=4,
    capthick=1.8,
    zorder=9
)

ax.scatter(
    Z_Zsun_Schneider2016,
    D_Dsun_Schneider2016,
    marker='D',
    s=230,
    facecolors='deeppink',
    edgecolors='black',
    linewidths=2.0,
    zorder=10,
    label='Schneider+ (2016), local dwarfs'
)

# Optional labels:
ax.text(0.036, D_Dsun_Schneider2016[0] * 1.0,
        'IZw18', color='deeppink', fontsize=24)

#ax.text(0.042, D_Dsun_Schneider2016[1] * 1.15,
#        'SBS 0335-052', color='dodgerblue', fontsize=24)

############

#ax.scatter(Z_Zsun_B146566, D_Dsun_B146566,
#           marker='D',
#           s=200,
#           facecolors='deeppink',
#           edgecolors='black',
#           linewidths=2.0,
#           zorder=10,
#           label='B14-65666 (z=7.15)')

##################################################
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Gas metallicity ($Z / Z_{\odot}$)', labelpad=10)
ax.set_ylabel(r'Dust-to-gas ratio ($\mathfrak{D} / \mathfrak{D}_{\odot}$)', labelpad=10)
#ax.legend()

yl = [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0]
ylabels = ['10$^{-5}$', '10$^{-4}$', '10$^{-3}$', '0.01', '0.1', '1']
xlabels = ylabels 
ax.set_yticks(yl)
ax.set_yticklabels(ylabels)
ax.set_xticks(yl)
ax.set_xticklabels(xlabels)

ax.set_xlim(3e-4, 2.0)
ax.set_ylim(1e-5, 4.0)

# Text for Remy+ (2014) scaling relation:
ax.text(1.3e-2, 1.1e-4, 'Remy-Ruyer+ (2014), z ~ 0', color='black', fontsize=22, rotation=48)

# Text for Galliano+ (2021) scaling relation and confidence interval:
dx = 1e-2
ax.text(4e-2 + dx, 0.5*3e-4, 'Galliano+ (2021), z ~ 0', 
        color='orangered', fontsize=22, rotation=54)
ax.text(6e-2 + dx, 0.5*2.5e-4, '95% interval', 
        color='orangered', fontsize=22, rotation=54)

# Text for REBELS data:
ax.text(0.17, 0.7, 'Algera+ (2026),\nz ~ 7', color='red', fontsize=22, rotation=0)

# Set different lengths for major and minor ticks
ax.minorticks_on()
ax.tick_params(axis='both', which='major', direction='in', length=20, width=2.5)
ax.tick_params(axis='both', which='minor', direction='in', length=10, width=2.5)

plt.tight_layout()
plt.savefig("LYDION_Figure_ObservedDustToGasScaling.pdf", dpi=300, bbox_inches='tight')
plt.show()