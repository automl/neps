import logging

import ConfigSpace as CS
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


def set_seed_get_config(configspace, i):
    configspace.seed(i)
    return configspace.sample_configuration()


def add_color(arch, color_map):
    """Add a node attribute called color, which color the nodes based on the op type"""
    for i, (_, data) in enumerate(arch.nodes(data=True)):
        try:
            arch.nodes[i]["color"] = color_map[data["op_name"]]
        except KeyError:
            logging.warning(
                "node operation "
                + color_map[data["op_name"]]
                + " is not found in the color_map! Skipping"
            )
    return arch


def encoding_to_nx(encoding: str, color_map: dict = None):
    """Convert a feature encoding (example 'input~maxpool3x3~conv3x3') to a networkx graph
    for WL features up to h=1.
    color_map: dict. When defined, supplement the encoding motifs with a color information that can be
    useful for later plotting.
    WARNING: this def is not tested for higher-order WL features, and thus the code might break."""
    g_nx = nx.DiGraph()
    nodes = encoding.split("~")
    for i, n in enumerate(nodes):
        if color_map is None:
            g_nx.add_node(i, op_name=n)
        else:
            try:
                g_nx.add_node(i, op_name=n, color=color_map[n])
            except KeyError:
                logging.warning(
                    "node operation " + n + " is not found in the color_map! Skipping"
                )
                g_nx.add_node(i, op_name=n)
        if i > 0:
            g_nx.add_edge(0, i)
    return g_nx


def _preprocess(X, y=None):
    from nasbowl.initial_design.generate_test_graphs import prune

    tmp = []
    valid_indices = []
    for idx, c in enumerate(X):
        node_labeling = list(nx.get_node_attributes(c, "op_name").values())
        try:
            res = prune(nx.to_numpy_array(c), node_labeling)
            if res is None:
                continue
            c_new, label_new = res
            c_nx = nx.from_numpy_array(c_new, create_using=nx.DiGraph)
            for i, n in enumerate(label_new):
                c_nx.nodes[i]["op_name"] = n
        except KeyError:
            print("Pruning error!")
            c_nx = c
        tmp.append(c_nx)
        valid_indices.append(idx)
    if y is not None:
        y = y[valid_indices]
    if y is None:
        return tmp
    return tmp, y


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x2:.2f}",
    textcolors=["black", "white"],
    threshold=None,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x2:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def gradient(outputs, inputs, grad_outputs=None, retain_graph=True, create_graph=True):
    """
    Compute the gradient of `outputs` with respect to `inputs`
    gradient(x.sum(), x)
    gradient((x * y).sum(), [x, y])
    """
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    grads = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs,
        allow_unused=True,
        retain_graph=retain_graph,
        create_graph=create_graph,
    )
    grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
    return torch.cat([x.contiguous().view(-1) for x in grads])


def jacobian(outputs, inputs, create_graph=False):
    """
    Compute the Jacobian of `outputs` with respect to `inputs`
    jacobian(x, x)
    jacobian(x * y, [x, y])
    jacobian([x * y, x.sqrt()], [x, y])
    """
    if torch.is_tensor(outputs):
        outputs = [outputs]
    else:
        outputs = list(outputs)

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    jac = []
    for output in outputs:
        output_flat = output.view(-1)
        output_grad = torch.zeros_like(output_flat)
        for i in range(len(output_flat)):
            output_grad[i] = 1
            jac += [gradient(output_flat, inputs, output_grad, True, create_graph)]
            output_grad[i] = 0
    return torch.stack(jac)


def hessian(output, inputs, out=None, allow_unused=False, create_graph=False):
    """
    Compute the Hessian of `output` with respect to `inputs`
    hessian((x * y).sum(), [x, y])
    """
    assert output.ndimension() == 0

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    n = sum(p.numel() for p in inputs)
    if out is None:
        out = output.new_zeros(n, n)

    ai = 0
    for i, inp in enumerate(inputs):
        [grad] = torch.autograd.grad(
            output, inp, create_graph=True, allow_unused=allow_unused
        )
        grad = torch.zeros_like(inp) if grad is None else grad
        grad = grad.contiguous().view(-1)

        for j in range(inp.numel()):
            if grad[j].requires_grad:
                row = gradient(
                    grad[j], inputs[i:], retain_graph=True, create_graph=create_graph
                )[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            out[ai, ai:].add_(row.type_as(out))  # ai's row
            if ai + 1 < n:
                out[ai + 1 :, ai].add_(row[1:].type_as(out))  # ai's column
            del row
            ai += 1
        del grad

    return out


def config_to_array(configuration):
    """

    :param configuration:
    :return:
    """
    if isinstance(configuration, CS.Configuration):
        configuration = np.array(
            [configuration[k] for k in configuration], dtype=np.float
        )
    if isinstance(configuration, dict):
        configuration = np.array(list(configuration.values()))
    return configuration
