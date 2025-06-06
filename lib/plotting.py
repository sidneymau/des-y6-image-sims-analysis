import functools
import logging
import shutil
import warnings

from cycler import cycler
import matplotlib as mpl
import matplotlib.cbook as cbook
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.image as mimage
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.style as mstyle
from mpl_toolkits.axes_grid1 import Divider, Size
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.texmanager import TexManager
from scipy.ndimage import gaussian_filter


logger = logging.getLogger(__name__)

#---------#
# Styling #
#---------#

def set_style(restore_defaults=True):
    if restore_defaults:
        logger.info(f"restoring matplotlib defaults")
        mpl.rcdefaults()

    logger.info("using plotting style")
    mstyle.use("lib/style.mplstyle")

    # if "drafter" in mpl.colormaps:
    #     logger.info(f"using drafter colormap")
    #     mpl.rcParams["image.cmap"] = "drafter"
    # else:
    #     warnings.warn(f"drafter colormap not found")

    return None


def set_type(font="monospace", latex=False):
    logger.info(f"setting font in {font}")
    mpl.rcParams["font.family"] = font

    if latex:
        logger.info(f"setting type in latex")
        latex_path = shutil.which("latex")

        if latex_path is None:
            warnings.warn(f"No executable found for 'latex'")
        else:
            logger.debug(f"latex is {latex_path}")
            mpl.rcParams["text.usetex"] = True
            # mpl.rcParams["text.latex.preamble"] = "\everymath{\\tt}"
            mpl.rcParams["text.latex.preamble"] = (
                r"\renewcommand*\familydefault{\ttdefault}"  # use typewriter text as default
                r"\usepackage[noendash,LGRgreek]{mathastext}"  # typeset math as text
                r"\MTfamily{\ttdefault}\MTgreekfont{\ttdefault}\Mathastext"  # use typewriter text for math
            )


        if font not in TexManager._font_families:
            warnings.warn(f"{font} is invalid latex font")
    # else:
    #     # mpl.rcParams["font.monospace"] = "Courier New"
    #     mpl.rcParams["font.monospace"] = "Nimbus Mono PS"
    #     mpl.rcParams["text.usetex"] = False
    #     mpl.rcParams["mathtext.fontset"] = "custom"
    #     mpl.rcParams["mathtext.default"] = "regular"
    #     mpl.rcParams["mathtext.fallback"] = None

    return None


def setup(restore_defaults=True, latex=False):
    set_style(restore_defaults=restore_defaults)
    set_type(latex=latex)


# US Letter
MAX_WIDTH = 8.5
MAX_HEIGHT = 11


def cm_to_in(l, /):
    return l / 2.54


def in_to_pt(l, /):
    return l * 6 * 12


#------#
# Axes #
#------#

def make_axes(
    nrows,
    ncols,
    *,
    fig_width=None,
    fig_height=None,
    width=2,
    height=2,
    margin=1,
    gutter=1,
    horizontal_margin=None,
    vertical_margin=None,
    left_margin=None,
    right_margin=None,
    top_margin=None,
    bottom_margin=None,
    horizontal_gutter=None,
    vertical_gutter=None,
    cbar_width=1/8,
    cbar_pad=1/8,
    sharex=None,
    sharey=None,
    squeeze=True,
):

    logger.info(f"making figure of size ({fig_width}, {fig_height})")
    fig = plt.figure(figsize=(fig_width, fig_height))
    # fig = make_figure(figsize=(fig_width, fig_height))

    fig_width, fig_height = fig.get_size_inches()
    if fig_width > MAX_WIDTH:
        warnings.warn(f"Figure width ({fig_width}) greater than maximum width ({MAX_WIDTH})")
    if fig_height > MAX_HEIGHT:
        warnings.warn(f"Figure height ({fig_height}) greater than maximum height ({MAX_HEIGHT})")

    if horizontal_gutter is None:
        horizontal_gutter = gutter

    if vertical_gutter is None:
        vertical_gutter = gutter

    if left_margin is None:
        if horizontal_margin is None:
            left_margin = margin
        else:
            left_margin = horizontal_margin

    if right_margin is None:
        if horizontal_margin is None:
            right_margin = margin
        else:
            right_margin = horizontal_margin

    if top_margin is None:
        if vertical_margin is None:
            top_margin = margin
        else:
            top_margin = vertical_margin

    if bottom_margin is None:
        if vertical_margin is None:
            bottom_margin = margin
        else:
            bottom_margin = vertical_margin

    if left_margin is None:
        raise ValueError(f"unspecified left margin!")
    if right_margin is None:
        raise ValueError(f"unspecified right margin!")
    if top_margin is None:
        raise ValueError(f"unspecified top margin!")
    if bottom_margin is None:
        raise ValueError(f"unspecified bottom margin!")

    if cbar_pad is not None:
        horizontal_gutter -= cbar_pad
        right_margin -= cbar_pad

    if cbar_width is not None:
        horizontal_gutter -= cbar_width
        right_margin -= cbar_width

    h = [Size.Fixed(left_margin)]
    h_idx = []
    for i in range(ncols):
        h_idx.append(len(h))
        h.append(Size.Fixed(width))
        h.append(Size.Fixed(cbar_pad))
        h.append(Size.Fixed(cbar_width))
        if i == ncols - 1:
            h.append(Size.Fixed(right_margin))
        else:
            h.append(Size.Fixed(horizontal_gutter))
    logger.info(f"made {len(h)} horizontal locations")

    v = [Size.Fixed(bottom_margin)]
    v_idx = []
    for i in range(nrows):
        v_idx.append(len(v))
        v.append(Size.Fixed(height))
        if i == nrows - 1:
            v.append(Size.Fixed(top_margin))
        else:
            v.append(Size.Fixed(vertical_gutter))
    logger.info(f"made {len(v)} vertical locations")

    total_width = sum(_h.fixed_size for _h in h)
    total_height = sum(_v.fixed_size for _v in v)
    logger.info(f"made axes of total size ({total_width}, {total_height})")

    if total_width > fig_width:
        warnings.warn(f"Total axes width ({total_width}) greater than figure width ({fig_width})")
    elif total_width < fig_width:
        warnings.warn(f"Total axes width ({total_width}) less than figure width ({fig_width})")

    if total_height > fig_height:
        warnings.warn(f"Total axes height ({total_height}) greater than figure height ({fig_height})")
    elif total_height < fig_height:
        warnings.warn(f"Total axes height ({total_height}) less than figure height ({fig_height})")

    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)

    # return array of primary axes
    vaxes = []
    for v_id in v_idx:
        haxes = []
        for h_id in h_idx:
            ax = fig.add_axes(
                divider.get_position(),
                axes_locator=divider.new_locator(nx=h_id, ny=v_id),
            )
            cax = fig.add_axes(
                divider.get_position(),
                axes_locator=divider.new_locator(nx=h_id + 2, ny=v_id),
            )
            cax.set_visible(False)
            ax.cax = cax
            haxes.append(ax)
        vaxes.append(haxes)
    vaxes = vaxes[::-1]

    axes = np.array(vaxes)

    if sharex == "all":
        head_ax = axes[0, 0]
        for ax in axes.flat:
            ax.sharex(head_ax)
    elif sharex == "row":
        for row in axes:
            head_ax = row[0]
            for ax in row:
                ax.sharex(head_ax)
    elif sharex == "col":
        for col in axes.T:
            head_ax = col[0]
            for ax in col:
                ax.sharex(head_ax)
    elif sharex:
        raise ValueError(f"sharex for {sharex} not supported")

    if sharey == "all":
        head_ax = axes[0, 0]
        for ax in axes.flat:
            ax.sharey(head_ax)
    elif sharey == "row":
        for row in axes:
            head_ax = row[0]
            for ax in row:
                ax.sharey(head_ax)
    elif sharey == "col":
        for col in axes.T:
            head_ax = col[0]
            for ax in col:
                ax.sharey(head_ax)
    elif sharey:
        raise ValueError(f"sharex for {sharex} not supported")

    # https://github.com/matplotlib/matplotlib/blob/v3.8.4/lib/matplotlib/gridspec.py
    # turn off redundant tick labeling
    if sharex in ["col", "all"]:
        for ax in axes.flat:
            ax._label_outer_xaxis(skip_non_rectangular_axes=True)
    if sharey in ["row", "all"]:
        for ax in axes.flat:
            ax._label_outer_yaxis(skip_non_rectangular_axes=True)

    logger.info(f"made {axes.size} axes")

    # NOTE this currently does not work because we bypass an internal gridspec...
    # fig.align_labels()
    # fig.align_titles()


    if squeeze:
        _axes = axes.item() if axes.size == 1 else axes.squeeze()
    else:
        _axes = axes

    return fig, _axes


def add_colorbar(
    axes,
    *args,
    **kwargs
):
    logger.info(f"adding colorbar to {axes}")
    axes.cax.set_visible(True)

    ax_fig = axes.get_figure()
    cb = ax_fig.colorbar(*args, cax=axes.cax, **kwargs)

    return cb


def imshow(
    axes,
    *args,
    **kwargs,
):
    kwargs = cbook.normalize_kwargs(kwargs, mimage.AxesImage)

    im = axes.imshow(*args, aspect="auto", **kwargs)

    axes.xaxis.set_major_locator(mticker.MaxNLocator(nbins="auto", integer=True))
    axes.xaxis.set_minor_locator(mticker.NullLocator())
    axes.yaxis.set_major_locator(mticker.MaxNLocator(nbins="auto", integer=True))
    axes.yaxis.set_minor_locator(mticker.NullLocator())

    return im


def mesh(
    axes,
    *args,
    edges=False,
    **kwargs,
):
    kwargs = cbook.normalize_kwargs(kwargs, mcoll.QuadMesh)

    if edges:
        # Draw mesh edges, styled as gridlines by default
        if "edgecolor" not in kwargs:
            kwargs["edgecolor"] = mpl.rcParams["grid.color"]
        if "linewidth" not in kwargs:
            kwargs["linewidth"] = mpl.rcParams["grid.linewidth"] / 2

    mesh = axes.pcolormesh(*args, **kwargs)

    return mesh


def _get_bin_centers_linear(bins):
    return 0.5 * (bins[:-1] + bins[1:])

def _get_bin_centers_log(bins):
    return (bins[:-1] * bins[1:])**(0.5)

def get_bin_centers(bins, log=False):
    match log:
        case False:
            centers = _get_bin_centers_linear(bins)
        case True:
            centers = _get_bin_centers_log(bins)
        case _:
            raise ValueError()
    return centers


def contour1d(axs, data, bins, *, smoothing=1, log=False, **kwargs):
    bin_centers = get_bin_centers(bins, log=log)
    lines = axs.plot(
        bin_centers,
        gaussian_filter(data, sigma=smoothing),
        **kwargs
    )

    return lines


def contour(axs, data, bins, *, smoothing=1, log=[False, False], **kwargs):
    bin_centers = (
        get_bin_centers(bins[0], log=log[0]),
        get_bin_centers(bins[1], log=log[1])
    )
    contours = axs.contour(
        bin_centers[0],
        bin_centers[1],
        gaussian_filter(data.T, sigma=smoothing),
        **kwargs
    )

    return contours


def watermark(
    fig,
    text="preliminary",
):
    fig_width, fig_height = fig.get_size_inches()
    angle = np.degrees(np.arctan2(fig_height, fig_width))
    size = in_to_pt(np.hypot(fig_width, fig_height) / 12)
    fig.text(
        0.5,
        0.5,
        text,
        color="k",
        alpha=0.1,
        fontsize=size,
        rotation=angle,
        horizontalalignment="center",
        verticalalignment="center",
    )

    return None


def set_palette(
    *,
    axes=None,
    palette=None,
):
    cycle = cycler(color=palette)

    axes.set_prop_cycle(cycle)

    return None


def sharexy(ax1, ax2):
    """
    See https://github.com/matplotlib/matplotlib/blob/v3.10.3/lib/matplotlib/axes/_base.py#L1275-L1311
    """
    # ax1.xaxis.major = ax2.yaxis.major  # Ticker instances holding
    # ax1.xaxis.minor = ax2.yaxis.minor  # locator and formatter.
    y0, y1 = ax2.get_ylim()
    ax1.set_xlim(y0, y1, emit=False, auto=ax2.get_autoscaley_on())
    # ax1.xaxis._scale = ax2.yaxis._scale
    ax1.set_xscale(ax2.get_yscale())


def shareyx(ax1, ax2):
    """
    See https://github.com/matplotlib/matplotlib/blob/v3.10.3/lib/matplotlib/axes/_base.py#L1275-L1311
    """
    # ax1.yaxis.major = ax2.xaxis.major  # Ticker instances holding
    # ax1.yaxis.minor = ax2.xaxis.minor  # locator and formatter.
    x0, x1 = ax2.get_xlim()
    ax1.set_ylim(x0, x1, emit=False, auto=ax2.get_autoscalex_on())
    # ax1.yaxis._scale = ax2.xaxis._scale
    ax1.set_yscale(ax2.get_xscale())


#----------------------#
# Colormaps & Palettes #
#----------------------#


def _reverser(func, x):
    # adapted from matplotlib.colors.LinearSegmentedColormap._reverser
    return func(1 - x)

def reversed(cmap):
    # adapted from matplotlib.colors.LinearSegmentedColormap.reversed
    # Using a partial object keeps the cmap picklable.
    data_r = {
        key: (functools.partial(_reverser, data))
        for key, data in cmap._segmentdata.items()}

    new_cmap = mcolors.LinearSegmentedColormap(
        cmap.name,
        data_r,
        cmap.N,
        cmap._gamma,
    )

    # Reverse the over/under values too
    new_cmap._rgba_over = cmap._rgba_under
    new_cmap._rgba_under = cmap._rgba_over
    new_cmap._rgba_bad = cmap._rgba_bad

    return new_cmap


def _truncator(func, light, dark, x):
    # adapted from matplotlib.colors.LinearSegmentedColormap._reverser
    if (light > 1) or (light < 0):
        raise ValueError(f"light {light} not in [0, 1]")
    if (dark > 1) or (dark < 0):
        raise ValueError(f"dark {dark} not in [0, 1]")

    return func(x * (light - dark) + dark)


def truncated(cmap, light, dark):
    # adapted from matplotlib.colors.LinearSegmentedColormap.reversed
    # Using a partial object keeps the cmap picklable.
    data_t = {
        key: (functools.partial(_truncator, data, light, dark))
        for key, data in cmap._segmentdata.items()
    }

    new_cmap = mcolors.LinearSegmentedColormap(
        cmap.name,
        data_t,
        cmap.N,
        cmap._gamma,
    )

    # Truncate the over/under values too
    new_cmap._rgba_over = cmap._rgba_under
    new_cmap._rgba_under = cmap._rgba_over
    new_cmap._rgba_bad = cmap._rgba_bad

    return new_cmap


def cubehelix_colormap(
    *,
    start=None,
    rot=None,
    gamma=None,
    hue=None,
    light=1,
    dark=0,
    name=None,
    reverse=False,
):
    """
    cubehelix color scheme by Dave Green (https://people.phy.cam.ac.uk/dag9/CUBEHELIX/)
    """
    # Note: this relies on an internal matplotlib function, so may need to be
    # updated in the future
    cdict = mpl._cm.cubehelix(gamma=gamma, s=start, r=rot, h=hue)

    cmap = mcolors.LinearSegmentedColormap(name, cdict)

    cmap = truncated(cmap, light, dark)

    if reverse:
        cmap = reversed(cmap)

    return cmap


def cubehelix_palette(
    n_colors=6,
    start=0,
    rot=0.4,
    gamma=1.0,
    hue=0.8,
    light=0.85,
    dark=0.15,
    reverse=False,
):
    cmap = cubehelix_colormap(
        start=start,
        rot=rot,
        gamma=gamma,
        hue=hue,
        light=light,
        dark=dark,
        name=None,
        reverse=reverse,
    )

    x = np.linspace(0, 1, n_colors)
    palette = cmap(x)[:, :3].tolist()

    return palette


_kwargs = {
    "rot": -0.1,
    "gamma": 1,
    "hue": 1,
    "light": 0.85,
    "dark": 0.15,
    "reverse": True,
}
_mdet_kwargs = {
    "start": 0,
}
_sims_kwargs = {
    "start": 1,
}

_cmap_mdet = cubehelix_colormap(
    **_kwargs,
    **_mdet_kwargs,
     name="des-y6-mdet",
)
mpl.colormaps.register(cmap=_cmap_mdet)

_cmap_sims = cubehelix_colormap(
    **_kwargs,
    **_sims_kwargs,
   name="des-y6-sims",
)
mpl.colormaps.register(cmap=_cmap_sims)


# takes arg of n_colors
mdet_palette = functools.partial(
    cubehelix_palette,
    **_kwargs,
    **_mdet_kwargs,
)

sims_palette = functools.partial(
    cubehelix_palette,
    **_kwargs,
    **_sims_kwargs,
)

mdet_color = mdet_palette(3)[1]
sims_color = sims_palette(3)[1]


