"""Centralized matplotlib style configuration used across plots."""
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

# Import plotting config if needed, or just set defaults here
# from general_configuration import config

def set_plot_style(context="paper", style="ticks", rc_overrides=None, use_constrained_layout=True):
    """
    Configures matplotlib to produce high-quality scientific plots.
    This function sets default parameters for font size, line width, colors, etc.

    Args:
        context: Seaborn context (e.g., "paper", "talk").
        style: Seaborn style (e.g., "ticks", "whitegrid").
        rc_overrides: Optional dict of matplotlib rcParams to override defaults.
        use_constrained_layout: Toggle matplotlib constrained layout globally.
    """
    # rcParams changes are global for the current Python process.
    sns.set_context(context)
    sns.set_style(style)

    # Enable LaTeX rendering for text in the figure
    plt.rcParams['text.usetex'] = False
    # Set SVG font type to 'none' to export text as text
    plt.rcParams['svg.fonttype'] = 'none'

    # General font settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.titleweight'] = 'normal'
    plt.rcParams['axes.labelweight'] = 'bold'

    # Line settings
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['lines.linestyle'] = '-'
    plt.rcParams['lines.color'] = 'C0'
    plt.rcParams['lines.marker'] = 'None'
    plt.rcParams['lines.markerfacecolor'] = 'black'
    plt.rcParams['lines.markeredgecolor'] = 'black'
    plt.rcParams['lines.markeredgewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 5
    plt.rcParams['lines.dash_joinstyle'] = 'round'
    plt.rcParams['lines.dash_capstyle'] = 'butt'
    plt.rcParams['lines.solid_joinstyle'] = 'round'
    plt.rcParams['lines.solid_capstyle'] = 'projecting'
    plt.rcParams['lines.antialiased'] = True

    # Patch settings
    plt.rcParams['patch.linewidth'] = 1.0
    plt.rcParams['patch.facecolor'] = 'none'
    plt.rcParams['patch.edgecolor'] = 'black'
    plt.rcParams['patch.force_edgecolor'] = False
    plt.rcParams['patch.antialiased'] = True

    # Hatch settings
    plt.rcParams['hatch.color'] = 'black'
    plt.rcParams['hatch.linewidth'] = 1.0

    # Boxplot settings
    plt.rcParams['boxplot.notch'] = False
    plt.rcParams['boxplot.vertical'] = True
    plt.rcParams['boxplot.whiskers'] = 1.5
    plt.rcParams['boxplot.bootstrap'] = None
    plt.rcParams['boxplot.patchartist'] = False
    plt.rcParams['boxplot.showmeans'] = False
    plt.rcParams['boxplot.showcaps'] = True
    plt.rcParams['boxplot.showbox'] = True
    plt.rcParams['boxplot.showfliers'] = True
    plt.rcParams['boxplot.meanline'] = False

    plt.rcParams['boxplot.flierprops.color'] = 'black'
    plt.rcParams['boxplot.flierprops.marker'] = 'o'
    plt.rcParams['boxplot.flierprops.markerfacecolor'] = 'none'
    plt.rcParams['boxplot.flierprops.markeredgecolor'] = 'none'
    plt.rcParams['boxplot.flierprops.markeredgewidth'] = 1.0
    plt.rcParams['boxplot.flierprops.markersize'] = 6
    plt.rcParams['boxplot.flierprops.linestyle'] = 'none'
    plt.rcParams['boxplot.flierprops.linewidth'] = 1.0

    plt.rcParams['boxplot.boxprops.color'] = 'none'
    plt.rcParams['boxplot.boxprops.linewidth'] = 1.0
    plt.rcParams['boxplot.boxprops.linestyle'] = '-'

    plt.rcParams['boxplot.whiskerprops.color'] = 'black'
    plt.rcParams['boxplot.whiskerprops.linewidth'] = 1.0
    plt.rcParams['boxplot.whiskerprops.linestyle'] = '-'

    plt.rcParams['boxplot.capprops.color'] = 'black'
    plt.rcParams['boxplot.capprops.linewidth'] = 1.0
    plt.rcParams['boxplot.capprops.linestyle'] = '-'

    plt.rcParams['boxplot.medianprops.color'] = 'black'
    plt.rcParams['boxplot.medianprops.linewidth'] = 1.0
    plt.rcParams['boxplot.medianprops.linestyle'] = '-'

    plt.rcParams['boxplot.meanprops.color'] = 'C2'
    plt.rcParams['boxplot.meanprops.marker'] = '^'
    plt.rcParams['boxplot.meanprops.markerfacecolor'] = 'C2'
    plt.rcParams['boxplot.meanprops.markeredgecolor'] = 'C2'
    plt.rcParams['boxplot.meanprops.markersize'] = 6
    plt.rcParams['boxplot.meanprops.linestyle'] = '--'
    plt.rcParams['boxplot.meanprops.linewidth'] = 1.0

    # Axes settings
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.grid.axis'] = 'both'
    plt.rcParams['axes.grid.which'] = 'major'
    plt.rcParams['axes.titlelocation'] = 'left'
    plt.rcParams['axes.titlecolor'] = 'black'
    plt.rcParams['axes.titlepad'] = 6.0
    plt.rcParams['axes.labelpad'] = 4.0
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['axes.axisbelow'] = 'line'
    plt.rcParams['axes.formatter.limits'] = (-4, 4)
    plt.rcParams['axes.formatter.use_locale'] = False
    plt.rcParams['axes.formatter.use_mathtext'] = False
    plt.rcParams['axes.formatter.min_exponent'] = 0
    plt.rcParams['axes.formatter.useoffset'] = True
    plt.rcParams['axes.formatter.offset_threshold'] = 4
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.unicode_minus'] = True
    plt.rcParams['axes.prop_cycle'] = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    plt.rcParams['axes.xmargin'] = 0.05
    plt.rcParams['axes.ymargin'] = 0.05
    plt.rcParams['axes.zmargin'] = 0.05
    plt.rcParams['axes.autolimit_mode'] = 'data'

    # Ticks settings
    plt.rcParams['xtick.top'] = False
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.labeltop'] = False
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.major.size'] = 2
    plt.rcParams['xtick.minor.size'] = 2
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['xtick.minor.width'] = 0.5
    plt.rcParams['xtick.major.pad'] = 2
    plt.rcParams['xtick.minor.pad'] = 2
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['xtick.labelcolor'] = 'inherit'
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['xtick.minor.visible'] = False
    plt.rcParams['xtick.major.top'] = True
    plt.rcParams['xtick.major.bottom'] = True
    plt.rcParams['xtick.minor.top'] = True
    plt.rcParams['xtick.minor.bottom'] = True
    plt.rcParams['xtick.alignment'] = 'center'

    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.right'] = False
    plt.rcParams['ytick.labelleft'] = True
    plt.rcParams['ytick.labelright'] = False
    plt.rcParams['ytick.major.size'] = 2
    plt.rcParams['ytick.minor.size'] = 2
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['ytick.minor.width'] = 0.25
    plt.rcParams['ytick.major.pad'] = 2
    plt.rcParams['ytick.minor.pad'] = 2
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['ytick.labelcolor'] = 'inherit'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['ytick.minor.visible'] = False
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.minor.left'] = True
    plt.rcParams['ytick.minor.right'] = True
    plt.rcParams['ytick.alignment'] = 'center_baseline'

    # Legend settings
    plt.rcParams['legend.loc'] = 'upper right'
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.facecolor'] = 'inherit'
    plt.rcParams['legend.edgecolor'] = 'black'
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['legend.shadow'] = False
    plt.rcParams['legend.numpoints'] = 1
    plt.rcParams['legend.scatterpoints'] = 1
    plt.rcParams['legend.markerscale'] = 1.0
    plt.rcParams['legend.labelcolor'] = 'black'
    plt.rcParams['legend.title_fontsize'] = 10
    plt.rcParams['legend.borderpad'] = 0.4
    plt.rcParams['legend.labelspacing'] = 0.5
    plt.rcParams['legend.handlelength'] = 2.0
    plt.rcParams['legend.handleheight'] = 0.7
    plt.rcParams['legend.handletextpad'] = 0.8
    plt.rcParams['legend.borderaxespad'] = 0.1
    plt.rcParams['legend.columnspacing'] = 2.0

    # Figure settings
    plt.rcParams['figure.titlesize'] = 11
    plt.rcParams['figure.titleweight'] = 'normal'
    plt.rcParams['figure.labelsize'] = 10
    plt.rcParams['figure.labelweight'] = 'normal'
    plt.rcParams['figure.figsize'] = (5/2.54, 8/2.54)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['figure.edgecolor'] = 'white'
    plt.rcParams['figure.frameon'] = False
    plt.rcParams['figure.subplot.left'] = 0.125
    plt.rcParams['figure.subplot.right'] = 0.9
    plt.rcParams['figure.subplot.bottom'] = 0.11
    plt.rcParams['figure.subplot.top'] = 0.88
    plt.rcParams['figure.subplot.wspace'] = 0.2
    plt.rcParams['figure.subplot.hspace'] = 0.2
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['figure.constrained_layout.use'] = use_constrained_layout
    plt.rcParams['figure.constrained_layout.h_pad'] = 0.04167
    plt.rcParams['figure.constrained_layout.w_pad'] = 0.04167
    plt.rcParams['figure.constrained_layout.hspace'] = 0.02
    plt.rcParams['figure.constrained_layout.wspace'] = 0.02

    # Savefig settings
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.edgecolor'] = 'white'
    plt.rcParams['savefig.format'] = 'svg'
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.05
    plt.rcParams['savefig.transparent'] = False
    plt.rcParams['savefig.orientation'] = 'portrait'

    # PS backend settings
    plt.rcParams['ps.papersize'] = 'letter'
    plt.rcParams['ps.useafm'] = False
    plt.rcParams['ps.usedistiller'] = False
    plt.rcParams['ps.distiller.res'] = 6000
    plt.rcParams['ps.fonttype'] = 3

    # PDF backend settings
    plt.rcParams['pdf.compression'] = 6
    plt.rcParams['pdf.fonttype'] = 3
    plt.rcParams['pdf.use14corefonts'] = False
    plt.rcParams['pdf.inheritcolor'] = False

    # SVG backend settings
    # plt.rcParams['svg.image_inline'] = True
    # plt.rcParams['svg.fonttype'] = 'path'
    # plt.rcParams['svg.hashsalt'] = None

    if rc_overrides:
        plt.rcParams.update(rc_overrides)
