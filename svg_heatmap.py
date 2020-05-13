from typing import List, Tuple, Union

from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex, Normalize
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib as mpl
import binascii
import numpy as np
from io import BytesIO


def heatmap(data: Union[np.ndarray, pd.DataFrame, list], vmin=None, vmax=None, cmap: str = 'magma',
            cbar: bool = True, cbar_kws=None, size: Tuple[int, int] = (400, 300), precision: int = 2,
            delim: str = '\n', ) -> str:
    """Plot rectangular data as a color-encoded matrix.

    This  will draw the heatmap into a new SVG of the specified size. Part of
    this svg space will be taken and used to plot a colormap, unless ``cbar``
    is False.

    Parameters
    ----------
    data : rectangular dataset
        2D dataset that can be coerced into an ndarray. If a Pandas DataFrame
        is provided, the index/column information will be used to label the
        columns and rows.
    vmin, vmax : floats, optional
        Values to anchor the colormap, otherwise they are inferred from the
        data and other keyword arguments.
    cmap : matplotlib colormap name or object, or list of colors, optional
        The mapping from data values to color space. If not provided, the
        default will depend on whether ``center`` is set.
    cbar : boolean, optional
        Whether to draw a colorbar.
    cbar_kws : dict of key, value mappings, optional
        Keyword arguments for `fig.colorbar`.
    size: Tuple[int, int], optional
        SVG size in pixels
    precision: int, optional
        number of decimals to use for coordinates
    delim: str, optional
        delimiter for SVG elements

    Returns
    -------
    svg : str
        SVG as str

    Examples
    --------

    Plot a heatmap for a numpy array:

    .. plot::
        :context: close-figs

        >>> import numpy as np; np.random.seed(0)
        >>> from svg_heatmap import heatmap
        >>> uniform_data = np.random.rand(10, 12)
        >>> svg = heatmap(uniform_data)
    """

    def get_text_size(text: str, rotated: bool = False) -> Tuple[float, float]:
        width, height = len(str(text)) * letter_w, letter_h
        return (height, width) if rotated else (width, height)

    def get_ticks(orient='x', margin=1.0) -> List[str]:
        def get_x_tick(loc: float, label: str) -> str:
            transforms = ''
            x = x_margin + x_size * (loc + .5)
            # distance to x_label
            y = size[1] - (2 * font_size if x_label not in ('', None) else 0)

            label_size = get_text_size(label)[0 if rotate_x_ticks else 1]
            y_space = y_margin - (size[1] - y)
            y -= (y_space - label_size - margin * font_size)

            if rotate_x_ticks:
                rotation_coords = round(x, precision), round(y - .5 * letter_h, precision)
                transforms = 'transform="rotate(270 {}, {})"'.format(*rotation_coords)
            return text_base.format(round(x, precision), round(y, precision), transforms, label)

        def get_y_tick(loc: float, label: str) -> str:
            y = y_size * loc + letter_h
            # distance to y_label
            x = (2 * font_size if y_label not in ('', None) else 0)
            x_space = x_margin - x
            x += (x_space - get_text_size(label)[0] - margin * font_size)
            return text_base.format(round(x, precision), round(y, precision), '', label)

        locations = range(np.shape(data)[0 if orient == 'x' else 1])
        if (orient == 'y' and y_tick_labels is None) or (orient == 'x' and x_tick_labels is None):
            labels = list(map(str, locations))
        else:
            labels = y_tick_labels if orient == 'y' else x_tick_labels

        text_base = '<text x="{}"y="{}"{}>{}</text>'
        tick_fun = get_x_tick if orient == 'x' else get_y_tick
        ticks = [tick_fun(loc, label) for loc, label in zip(locations, labels)]
        return ['<g font-family="monospace"font-size="{}">'.format(font_size), *ticks, '</g>']

    def get_label(orient='x') -> str:
        if orient == 'x':
            label, transforms = y_label, ''
            y = size[1] - font_size
            x = .5 * size[0] + .5 * x_margin - .5 * get_text_size(label)[0] - .5 * x_margin_left
        else:
            label = x_label
            x = font_size
            y = .5 * size[1] - .5 * y_margin + .5 * get_text_size(label)[0]
            transforms = 'transform="rotate(270 {}, {})"'.format(x, y - .5 * letter_h)

        if label is None:
            return ''

        text_base = '<text x="{}"y="{}"{}font-family="monospace"font-size="{}">{}</text>'
        return text_base.format(x, y, transforms, font_size, label)

    def get_grid() -> List[str]:
        def _get_rect(x, y, w, h, color):
            x, y, w, h = np.round([x, y, w, h], precision)
            return '<rect x="{}"y="{}"width="{}"height="{}"style="fill:{};"/>'.format(x, y, w, h, color)

        for x in range(np.shape(data)[0]):
            for y in range(np.shape(data)[1]):
                color = to_hex(cmap_fun(norm(data[x, y])))
                yield _get_rect(x_margin + x * x_size, y * y_size, x_size, y_size, color)

    def get_margin(orient='x') -> float:
        labels = y_tick_labels if orient == 'x' else x_tick_labels
        rotated = orient == 'y' and not rotate_x_ticks
        tick_label_space = np.max([get_text_size(str(l), rotated=rotated)[0] for l in labels])
        label = y_label if orient == 'x' else x_label
        label_space = 2 * font_size if label not in ('', None) else 0
        return label_space + tick_label_space + font_size

    def get_colobar_img(scaling: float = 1.25) -> str:
        def encode_plot(fig: plt.Figure):
            with BytesIO() as buf:
                fig.canvas.print_png(buf)
                img_data = binascii.b2a_base64(buf.getvalue()).decode()
            img_html = '<image x="{}"y="{}"height="{}"width="{}"xlink:href="data:image/png;base64,{}">'
            return img_html.format(size[0] - cbar_w, 0, cbar_h, cbar_w, img_data)

        cbar_width_in = cbar_w / cbar_dpi * scaling
        cbar_height_in = cbar_h / cbar_dpi * scaling
        fig, ax = plt.subplots(figsize=(cbar_width_in, cbar_height_in), dpi=cbar_dpi)
        plt.axis('off')
        cax = fig.add_axes([0, 0, 0.25, 1])
        mpl.colorbar.ColorbarBase(cax, cmap=cmap_fun, norm=norm, orientation='vertical', **cbar_kws)
        cb = encode_plot(fig)
        plt.close()
        return cb

    if cbar_kws is None:
        cbar_kws = {}
    if isinstance(data, pd.DataFrame):
        y_label, x_label = data.columns.name, data.index.name
        x_tick_labels, y_tick_labels = data.columns.values, data.index.values
        rotate_x_ticks = True
        data = data.T.values
    else:
        x_label, y_label = '', ''
        x_tick_labels, y_tick_labels = range(np.shape(data)[0]), range(np.shape(data)[1])
        rotate_x_ticks = False

    vmin = np.min(data) if vmin is None else vmin
    vmax = np.max(data) if vmax is None else vmax

    font_size, cbar_w = 4 * round(np.log10(np.max(size))), 15 * round(np.log10(np.max(size))) if cbar else 0
    letter_h, letter_w = np.floor(font_size * 1.1875), .61 * font_size
    cbar_dpi = 30 * round(np.log10(np.max(size)))

    cmap_fun = get_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax)

    x_margin, y_margin = get_margin('x'), get_margin('y')
    x_margin_left = cbar_w + .5 * font_size
    cbar_h = size[1] - y_margin

    x_size = (size[0] - x_margin - x_margin_left) / np.shape(data)[0]
    y_size = (size[1] - y_margin) / np.shape(data)[1]

    x_ticks, y_ticks = get_ticks('x'), get_ticks('y')
    x_label, y_label = get_label('x'), get_label('y')

    grid = get_grid()
    colorbar_img = get_colobar_img() if cbar else ''

    svg_base = '<svg width="{}" height="{}">{{}}</svg>'.format(*size)
    content_strs = [*[delim.join(c) for c in (grid, x_ticks, y_ticks)], x_label, y_label, colorbar_img]

    return svg_base.format(delim.join(content_strs))
