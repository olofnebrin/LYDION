import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import shutil   # For checking if LaTeX is installed (shutil.which)
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np


# Default colormap choices for different physical quantities, 
# with fallbacks if cmastro colormaps are not available:
CMAP_DEFAULTS = {
    "density":      ("cma:laguna",    "viridis"),
    "temperature":  ("coolwarm",   "coolwarm"),
    "dust":         ("cma:emph",   "viridis"),
    "chemistry":    ("magma",      "magma"),
}


def HasFont(fontname: str) -> bool:
    """ This function checks if the font is available on the system. """
    try:
        path = fm.findfont(fm.FontProperties(family = fontname), fallback_to_default = False)
        return True
    except Exception:
        return False
    
def HasLaTeX() -> bool:
    """ This function checks if LaTeX is installed on the system. """
    return (shutil.which("pdflatex") is not None) or (shutil.which("latex") is not None)

def cmap_or(name: str,  fallback: str = "viridis") -> str:
    """ Return color map (cmap) name if available, else fallback 
        to the specified cmap."""
    try:
        mpl.colormaps[name]  
        return name
    except Exception:
        return fallback


def cmap_for(kind: str) -> str:
    preferred, fallback = CMAP_DEFAULTS.get(kind, ("viridis", "viridis"))
    return cmap_or(preferred, fallback)

def CinematicAxes( figsize = (8, 8) ):
    """ Create a matplotlib figure and axes with a cinematic style (dark background, no axes/ticks). """

    fig = plt.figure( figsize = figsize )
    ax  = fig.add_axes([0, 0, 1, 1])      # Full canvas
    ax.set_axis_off()                     # Turn off axes, ticks, and labels for a cinematic look

    return fig, ax

def AddTimeLabel( ax, text, pos = (0.03, 0.97), fontsize = 24 ):
    """ Add a time label to the given axes with a cinematic style (white text with black outline). """
    style = dict( ha = "left", va = "top", transform=ax.transAxes, color = "white",
                  path_effects = [pe.withStroke(linewidth=3, foreground="black")] )
    
    return ax.text(pos[0], pos[1], text, fontsize=fontsize, **style)

def InsetColorBar(fig, ax, mappable,
                  rect=(0.88, 0.20, 0.03, 0.40),   # (left, bottom, width, height) in FIGURE coords
                  label=None,
                  labelsize=18,
                  tick_labelsize=18,
                  outline_width=1.2,
                  labelcolor = "white",
                  tickcolors = "white"):
    """
    Add a colorbar in fixed FIGURE coordinates (won't clip in cinematic full-canvas plots).
    """
    cax = fig.add_axes(rect)
    cb  = fig.colorbar(mappable, cax=cax)

    # Cinematic styling
    cax.set_facecolor("black")
    cb.outline.set_edgecolor("white")
    cb.outline.set_linewidth(outline_width)

    cb.ax.tick_params(axis="y",
                      colors=tickcolors,
                      labelcolor= labelcolor,
                      labelsize=tick_labelsize,
                      length=0,          # disable default tick marks (we draw our own)
                      width=0,
                      pad=10)

    if label is not None:
        cb.set_label(label, color=labelcolor, fontsize=labelsize, labelpad=10)

    return cb


def ColorBarFullWidthTicks(cb, values, lw=1.5, color="white"):
    """
    Draw full-width horizontal tick lines across a vertical colorbar.
    values are in the colorbar's data units (works for log norms too).
    """
    ax = cb.ax
    x0, x1 = ax.get_xlim()
    for v in values:
        ax.hlines(v, x0, x1, colors=color, linewidth=lw, zorder=5)


def AddStar( ax, x, y, size = 250 ):
    """ Add a star marker to the given axes at the specified position 
        with a cinematic style (white star with black outline). """
    
    ax.scatter([x], [y], marker = "*", s = size, c = "white",
               edgecolors = "black", linewidths = 1.5, zorder = 10 )


def AddScaleBar( ax,
                 length,                   # e.g. 1.0
                 label    = "1 pc",
                 pos      = (0.08, 0.08),  # Axes fraction
                 lw       = 2.0,
                 color    = "white",
                 fontsize = 22,
                 cap_frac = 0.02):         # Cap height in axes fraction
    
    """ Draw a bracket-style scale bar (└──┘) in axes coordinates.

    - Length is in data units (e.g. pc, if your axes are in pc).
    - pos is the left end of the bar in axes-fraction coordinates.
    - cap_frac controls the cap height relative to the axes. """

    x0, y0 = pos
    xlim = ax.get_xlim()
    dx_data = xlim[1] - xlim[0]
    if dx_data <= 0:
        return None

    bar_frac = length / dx_data
    x1 = x0 + bar_frac

    # Clamp inside axes
    if x1 > 0.98:
        x1 = 0.98
        x0 = x1 - bar_frac

    trans = ax.transAxes
    y1 = y0 + cap_frac

    # Single polyline: up-left cap -> down -> across -> up-right cap
    verts = [(x0, y1), (x0, y0), (x1, y0), (x1, y1)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
    patch = PathPatch(
        Path(verts, codes),
        transform=trans,
        fill = False,
        lw = lw,
        edgecolor = color,
        capstyle = "round",
        joinstyle = "round",
        antialiased = True,
        zorder = 20,
    )
    ax.add_patch(patch)

    # Label (same outline style as time label)
    ax.text(
        0.5 * (x0 + x1), y1 + 0.01, label,
        transform = trans, ha = "center", va = "bottom",
        color = color, fontsize = fontsize,
        path_effects=[pe.withStroke(linewidth = 2.5, foreground = "black")],
        zorder = 21,
    )

    return (x0, x1, y0)



def ApplyFontStyle( preferred_font = "AVHershey Complex",
                         math_font = "GreekC",
                         font_size = 37,
                             theme = "default"    # "default" or "cinematic"
                     ):
    """ This function applies the specified font and styling to matplotlib plots. 
        It checks for the availability of the preferred font and LaTeX, 
        and falls back to appropriate alternatives if necessary. """
    
    plt.rcParams.update({
        "legend.fontsize": 0.6 * font_size,  # Scale legend font size with the main font size
        "legend.markerscale": 0.75,
        "legend.frameon": False,

        "axes.unicode_minus": False,
        "axes.linewidth": 2.5,

        "grid.linewidth": 2.5,
        "grid.linestyle": "--",

        "xtick.major.width": 1.7,
        "xtick.minor.width": 1.7,
        "xtick.major.size": 20,
        "xtick.minor.size": 8,
        "ytick.major.width": 1.7,
        "ytick.minor.width": 1.7,
        "ytick.major.size": 20,
        "ytick.minor.size": 8,
        "xtick.major.pad": 15,
        "ytick.major.pad": 15,

        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

    # CINEMATIC THEME: A dark theme which allows for more cinematic visuals
    if theme == "cinematic":
        plt.rcParams.update({
            "figure.facecolor":  "black",
            "axes.facecolor":    "black",
            "savefig.facecolor": "black",

            # Hide ticks/labels by default 
            "xtick.bottom": False,
            "xtick.top":    False,
            "ytick.left":   False,
            "ytick.right":  False,

            # Make remaining text visible on black
            "text.color":      "white",
            "axes.labelcolor": "white",
            "axes.edgecolor":  "white",
            "xtick.color":     "white",
            "ytick.color":     "white",
        })

        # Remove grid lines::
        plt.rcParams["axes.linewidth"] = 0.0
        plt.rcParams["axes.grid"]      = False


    # FONT FALL-BACK:
    if HasFont(preferred_font):
        plt.rc("font", **{"family": "sans-serif", "sans-serif": [preferred_font], "size": font_size})
        mpl.rcParams["text.usetex"] = False
    elif HasLaTeX():
        mpl.rcParams.update({"text.usetex": True, "font.size": font_size})
    else:
        mpl.rcParams.update({"text.usetex": False})  # keep defaults

    # MATH FALL-BACK (only used if not using LaTeX):
    if not mpl.rcParams.get("text.usetex", False):
        mpl.rcParams["mathtext.default"] = "regular"
        if HasFont(math_font):
            mpl.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": math_font})
        else:
            mpl.rcParams.update({"mathtext.fontset": "stix"})


def MirrorQuadrant( A, mirror_r = True, mirror_z = True ):
    """ For axisymmetric simulations, it can be nice to mirror 
        quadrants to show the full volume, for better visual appeal. 
        This function takes a quadrant field A defined on (Nr, Nz) 
        and mirrors it across the r=0 and/or z=0 axes to create a larger array.

        Returns:

        B with shape:
        (2Nr, Nz)    if mirror_r only
        (Nr, 2Nz)    if mirror_z only
        (2Nr, 2Nz)   if both
    """
    B = A
    if mirror_r:
        B = np.concatenate([B[::-1, :], B], axis=0)   # r -> -r
    if mirror_z:
        B = np.concatenate([B[:, ::-1], B], axis=1)   # z -> -z
    return B

def MirrorEdges(edges):
    """ Given monotonic edges [0 ... L], return [-L ... L] edges for mirrored plot. """
    # Edges: (N+1,) with edges[0] = 0
    neg = - edges[:0:-1]          # exclude 0 to avoid duplicate
    return np.concatenate([neg, edges])