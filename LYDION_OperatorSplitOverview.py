import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def add_box(ax, x, y, w, h, text, fontsize=18,
            fc='white', ec='black', lw=2.0, ls='solid'):

    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls
    )
    ax.add_patch(patch)

    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center', fontsize=fontsize)

    return patch


def add_arrow(ax, x1, y1, x2, y2, lw=2.0, shrinkA=6, shrinkB=6):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>',
        mutation_scale=16,
        linewidth=lw,
        color='black',
        shrinkA=shrinkA,
        shrinkB=shrinkB
    )
    ax.add_patch(arrow)


###############################################################
#######                                                 #######
#######           P L O T   C O S M E T I C S           #######
#######                                                 #######
###############################################################

# COMMENT THIS OUT IF YOU DON'T HAVE THE REQUIRED FONTS:

# plt.rc('font', **{'sans-serif':['RomanD'],
#                   'size': 27})
# plt.rc('font', **{'sans-serif':['AVHershey Simplex'],
#                   'size': 27})
plt.rc('font', **{'sans-serif':['AVHershey Complex'],
                  'size': 37})
# plt.rc('font', **{'sans-serif':['Complex'],
#                   'size': 27})

params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

params = {'legend.fontsize': 19}
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
plt.rcParams['xtick.major.pad'] = '15'
plt.rcParams['ytick.major.pad'] = '15'

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

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 1.5)
ax.set_ylim(0, 1.0)
ax.axis('off')

#fig.suptitle(
#"LYDION Radiation Hydrodynamics:\n"
#"Operator-split method for a global step $\\Delta t_{Hydro}$",
#fontsize=32
#)

ax.text(0.75, 0.98, "LYDION Radiation Dust-Hydrodynamics:\n"
                   "Operator-split method for a global step $\\Delta t_{Hydro}$",
        ha='center', va='top', fontsize=28)

# ============================================================
# Layout parameters: left/main column
# ============================================================

x_left     = 0.10
box_width  = 0.55
box_height = 0.10
box_gap    = 0.06
y_top      = 0.78

# ============================================================
# Text inside each main box & colors of boxes
# ============================================================

labels = [
    "(1) Gravity solve & kick over\nhalf-step $\\Delta t_{Hydro}/2$",
    "(2) Advect gas & dust\nover full step $\\Delta t_{Hydro}$",
    "(3) Gravity solve & kick over\nhalf-step $\\Delta t_{Hydro}/2$",
    "(4) Stellar wind feedback over\nfull step $\\Delta t_{Hydro}$ (optional)",
    "(5) RT & photo-thermochemistry\nsubcycling over $\\Delta t_{Hydro}$",
]

colors = ['lightblue', 'lightgreen', 'lightblue', 'coral', 'pink']

# ============================================================
# Compute main-box positions
# ============================================================

box_positions = []

for i in range(len(labels)):
    y = y_top - i * (box_height + box_gap)
    box_positions.append((x_left, y))

# ============================================================
# Draw the main boxes
# ============================================================

box_patches = []

for (x, y), label, color in zip(box_positions, labels, colors):
    patch = add_box(ax, x, y, box_width, box_height, label, fontsize=19, fc = color)
    box_patches.append(patch)

# ============================================================
# Draw vertical arrows between main boxes
# ============================================================

for i in range(len(box_positions) - 1):

    x, y_upper = box_positions[i]
    _, y_lower = box_positions[i + 1]

    x_start = x + box_width / 2
    y_start = y_upper

    x_end = x + box_width / 2
    y_end = y_lower + box_height

    add_arrow(ax, x_start, y_start, x_end, y_end, lw=2.0, shrinkA=6, shrinkB=6)

# ============================================================
# Right-side sub-cycling boxes
# ============================================================

x_right = 0.75
right_box_width = 0.65

# (5.1) Top RT advection box
y_right1 = 0.78
right_box_h1 = 0.09

rt_patch = add_box(
    ax, x_right, y_right1, right_box_width, right_box_h1,
    "(5.1) Stellar emission & M1 advection of\n"
    "stellar, IR, and Ly$\\alpha$ photons over $\\Delta t_{RT}$,\n"
    "where $\\sum \\Delta t_{RT} = \\Delta t_{Hydro}$",
    fontsize=18
)

# (5.2) One large photo-thermochemistry box
y_big = 0.42
big_box_height = 0.3

big_patch = FancyBboxPatch(
    (x_right, y_big), right_box_width, big_box_height,
    boxstyle="round,pad=0.01,rounding_size=0.02",
    facecolor='dimgray', edgecolor='black', linewidth=2.0
)
ax.add_patch(big_patch)

# Title for the big box
ax.text(
    x_right + right_box_width/2,
    y_big + big_box_height - 0.045,
    "(5.2) Photo-thermochemistry\nsubcycling over $\\sum \, \\Delta t_{sub} = \\Delta t_{RT}$",
    ha='center', va='center', fontsize=19, color='white'
)

# Ordered text inside the big box
subcycle_text = (
    "(a) Photon absorption & diffuse emission\n"
    "(b) Radiation pressure & gas-dust drag\n"
    "(c) Dust growth & destruction\n"
    "(d) Gas heating & cooling\n"
    "(e) Chemistry"
)

ax.text(
    x_right + 0.015,
    y_big + big_box_height - 0.095,
    subcycle_text,
    ha='left', va='top', fontsize=18, color = 'white'
)

# ============================================================
# Connector from main step (5) to right-side top box (5.1)
# ============================================================

x5, y5 = box_positions[4]

x_start = x5 + box_width + 0.01
y_start = y5 + box_height / 2

x_end = x_right - 0.004
y_end = y_right1 + right_box_h1 / 2

dx1 = 0.035
dx2 = 0.040

x_vertical    = x_start + dx1
x_arrow_start = x_end - dx2

ax.plot([x_start, x_vertical], [y_start, y_start],
        color='black', lw=2.0, solid_capstyle='butt')

ax.plot([x_vertical, x_vertical], [y_start, y_end],
        color='black', lw=2.0, solid_capstyle='butt')

ax.plot([x_vertical, x_arrow_start], [y_end, y_end],
        color='black', lw=2.0, solid_capstyle='butt')

add_arrow(ax, x_arrow_start, y_end, x_end, y_end,
          lw=2.0, shrinkA=0, shrinkB=4)

# ============================================================
# Arrow from (5.1) down to (5.2)
# ============================================================

x_mid = x_right + right_box_width / 2
y_rt_bottom = y_right1
y_big_top   = y_big + big_box_height

add_arrow(ax, x_mid, y_rt_bottom, x_mid, y_big_top,
          lw=2.0, shrinkA=6, shrinkB=6)


# ============================================================
# Final box: Lyα frequency solve over Δt_RT
# ============================================================

y_bottom_box = 0.24
bottom_box_height = 0.12

lya_patch = add_box(
    ax, x_right, y_bottom_box, right_box_width, bottom_box_height,
    "(5.3) Ly$\\alpha$ frequency diffusion\n& Doppler shift solve over $\\Delta t_{RT}$",
    fontsize=18
)

# Arrow from (5.2) down to final Lyα frequency-solve box
y_big_bottom = y_big
y_bottom_top = y_bottom_box + bottom_box_height

add_arrow(ax, x_mid, y_big_bottom, x_mid, y_bottom_top,
          lw=2.0, shrinkA=6, shrinkB=6)


ax.text( 0.74, 0.15, "Steps (5.1)-(5.3) are subcycled within\nstep (5) until the total RT time equals $\\Delta t_{Hydro}$",
         fontsize = 18)


###################################

plt.tight_layout()
plt.savefig("LYDION_OperatorSplitting.pdf", bbox_inches='tight')
plt.show()