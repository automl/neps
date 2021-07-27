import copy
from copy import deepcopy

from networkx.drawing.nx_pydot import graphviz_layout
from tqdm import tqdm

from nasbowl.models.gp import ComprehensiveGP
from nasbowl.utils.nasbowl_utils import *
from nasbowl.utils.nasbowl_utils import _preprocess
from nasbowl.kernel_operators import WeisfilerLehman
from nasbowl.misc.find_stuctures import find_wl_feature


class Interpreter:
    def __init__(self, X=None, y=None, h=1, gp=None, thres=25, occur_thres=10):
        assert 0 < thres < 100
        if gp is None:
            assert X is not None
            assert y is not None
            X, y = _preprocess(X, y)
            self.X_ = copy.deepcopy(X)
            self.y_ = copy.deepcopy(y)
            k = WeisfilerLehman(oa=False, h=h, requires_grad=True)
            self.gp = ComprehensiveGP(X, y, [k])
        else:
            self.gp = deepcopy(gp)
            self.X_ = copy.deepcopy(gp.x)
            self.y_ = copy.deepcopy(gp.y_)
        self.gp.fit(wl_subtree_candidates=())
        self.grads, self.grad_vars, self.incidences = self.gp.dmu_dphi(self.X_)
        feat_list = self.gp.domain_kernels[0].feature_value(self.X_)[1]

        grad_ = self.grads.cpu().detach().numpy()[:, 1].flatten()
        # Compute the predictive gradients
        # Set up the upper and lower percentiles:
        self.feat_list = []
        self.grad_ = []
        self.pruning = []
        for i, feat in enumerate(feat_list):
            # if feat.count('~') > 3 or self.incidences[i, 1:].sum() < 10:
            if self.incidences[i, 1:].sum() < occur_thres:
                continue
            if feat.count("~") == 0:
                if 'input' in feat or 'output' in feat or 'add' in feat:
                    continue
            self.feat_list.append(feat)
            self.grad_.append(grad_[i])
            self.pruning.append(i)

        self.color_map = {
            'input': 'black',
            'output': 'gray',
            'conv1x1-bn-relu': 'c',
            'nor_conv_1x1': 'c',
            'maxpool3x3': 'blue',
            'avg_pool_3x3': 'purple',  # avg pooling is only in nasbench201
            'conv3x3-bn-relu': 'orange',
            'nor_conv_3x3': 'orange',  # this appears in nasbench201
            ### DARTS ops
            'dil_conv_3x3': 'r',
            'dil_conv_5x5': 'g',
            'input1': 'k',
            'input2': 'k',
            'max_pool_3x3': 'blue',
            'sep_conv_3x3': 'orange',
            'sep_conv_5x5': 'yellow',
            'add': 'brown',
        }

        self.darts = ['input', 'dil_conv_3x3', 'dil_conv_5x5',
                      'max_pool_3x3', 'avg_pool_3x3', 'sep_conv_3x3',
                      'sep_conv_5x5', 'add', 'output']
        self.n101 = ['input', 'output', 'conv1x1-bn-relu', 'maxpool3x3',
                     'conv3x3-bn-relu']
        self.n201 = ['nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3',
                     'input', 'output']

        self.feat_graphlets = [encoding_to_nx(fe, color_map=self.color_map) for fe in self.feat_list]
        up = np.percentile(self.grad_, 100. - thres)
        lo = np.percentile(self.grad_, thres)

        # Identify the good and bad features
        self.bad_idx = np.argwhere(self.grad_ >= up).flatten()
        self.good_idx = np.argwhere(self.grad_ <= lo).flatten()

    def set_threshold(self, threshold):
        assert 0 < threshold < 100
        up = np.percentile(self.grad_, 100. - threshold)
        lo = np.percentile(self.grad_, threshold)

        # Identify the good and bad features
        self.bad_idx = np.argwhere(self.grad_ >= up).flatten()
        self.good_idx = np.argwhere(self.grad_ <= lo).flatten()

    def explain(self, X_s, y_s, percentile):
        """
        Predict the performance, along side the interpretation of that prediction
        """
        from networkx.algorithms import isomorphism
        # Do unsupervised WL decomposigion
        X_s = add_color(X_s, self.color_map)
        embedding = self.gp.domain_kernels[0].feature_value([X_s])[0].flatten()
        embedding = [embedding[i] for i in self.pruning]
        good_match = []
        bad_match = []
        for i in self.good_idx:
            if embedding[int(i)] != 0: good_match.append(int(i))
        for i in self.bad_idx:
            if embedding[int(i)] != 0: bad_match.append(int(i))

        # for n in range(len(X_s)):
        #     X_s.nodes[n]['color'] = 'gray'

        def _process(X_s, match_set, is_good):
            special_edges = []
            special_nodes = []
            for i in match_set:
                f = self.feat_graphlets[int(i)]
                if len(f) == 1:  ### h = 0 features
                    for i in range(len(X_s)):
                        if X_s.nodes[i]['op_name'] == f.nodes[0]['op_name']:
                            special_nodes.append(i)
                else:
                    gm = isomorphism.DiGraphMatcher(X_s, f, node_match=lambda x, y: x['op_name'] == y['op_name'])
                    print(gm.subgraph_is_isomorphic())
                    mapping = gm.mapping
                    if not len(mapping):
                        from misc.draw_nx import draw_graph
                        draw_graph(X_s)
                        draw_graph(f)
                    rev_mapping = {v: k for k, v in mapping.items()}
                    if len(mapping):
                        root_node = rev_mapping[0]
                        special_edges += [(root_node, m) for m in list(mapping.keys())[1:]]
                        for i in list(mapping.keys()):
                            # color the nodes
                            special_nodes.append(i)
                        # special edges
            return X_s, special_nodes, special_edges

        plt.figure(figsize=[5, 5])
        if percentile > 80:
            plt.title('Valid Acc.: %.2f %%. Percentile: %.2f' % ((1. - np.exp(-y_s)) * 100., percentile),
                      color='green')
        elif percentile < 20:
            plt.title('Valid Acc.: %.2f %%. Percentile: %.2f' % ((1. - np.exp(-y_s)) * 100., percentile),
                      color='red')
        else:
            plt.title('Valid Acc.: %.2f %%. Percentile: %.2f' % ((1. - np.exp(-y_s)) * 100., percentile))
        pos = graphviz_layout(X_s)
        special_edges = []
        special_nodes = []
        if len(bad_match):
            X_s, sn, se = _process(X_s, bad_match, False)
            if len(se):
                nx.draw_networkx_edges(
                    X_s, pos, edgelist=se, edge_color="red",
                    width=3, alpha=0.7, style="dashed",
                )
                special_edges += se
            if len(sn):
                nodes = nx.draw_networkx_nodes(
                    sn, pos, node_color=[X_s.nodes[i]['color'] for i in sn],
                    linewidths=3,

                )
                nodes.set_edgecolor('red')
                special_nodes += sn

        if len(good_match):
            X_s, sn, se = _process(X_s, good_match, True)
            if len(se):
                nx.draw_networkx_edges(
                    X_s, pos, edgelist=se, edge_color="green",
                    width=3, alpha=0.7, style="dashed",
                )
                special_edges += se
            if len(sn):
                nodes = nx.draw_networkx_nodes(
                    sn, pos, node_color=[X_s.nodes[i]['color'] for i in sn],
                    linewidths=3,
                )
                nodes.set_edgecolor('green')
                special_nodes += sn

        non_se = [x for x in list(X_s.edges) if x not in special_edges]
        nx.draw_networkx_edges(
            X_s, pos, edgelist=non_se, edge_color="gray",
            width=3, alpha=0.7, style="dashed",
        )
        label = {n: i['op_name'] for n, i in X_s.nodes(data=True)}
        nodes = [i for i in X_s.nodes if i not in special_nodes]
        color = [data["color"] for v, data in X_s.nodes(data=True) if v not in special_nodes]
        nx.draw_networkx_nodes(nodes, pos, node_color=color, labels=label, edge_color='gray', )

        # plt.show()

    def plot_motifs(self):
        """Plot the motifs (the WL features) extracted by the explainer in the train set, """

        # define a color map for the nodes
        G = nx.DiGraph()
        B = nx.DiGraph()
        good_features = [self.feat_graphlets[i] for i in self.good_idx]
        bad_features = [self.feat_graphlets[i] for i in self.bad_idx]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis('off')
        plt.subplot2grid((5, 2), (0, 0), rowspan=4)
        plt.title('Best motifs')
        for g in good_features:
            G = nx.disjoint_union(G, g)
        color = [data["color"] for v, data in G.nodes(data=True)]
        pos = graphviz_layout(G, )
        nx.draw(G, pos, node_size=70, node_color=color, with_labels=False)
        # plt.savefig('pos_motif_searched_on' + e.task + '.png')

        plt.subplot2grid((5, 2), (0, 1), rowspan=4)
        plt.title('Worst motifs')
        for b in bad_features:
            B = nx.disjoint_union(B, b)
        color = [data["color"] for v, data in B.nodes(data=True)]
        pos = graphviz_layout(B, )
        nx.draw(B, pos, node_size=70, node_color=color, with_labels=False)
        # plt.savefig('neg_motif_searched_on' + e.task + '.png')
        plt.subplot2grid((5, 2), (4, 0), colspan=2)
        plt.axis('off')
        plt.legend(*self.get_legend('darts'), frameon=False, loc='lower center', ncol=3, prop={'size': 9})
        plt.show()

    def validate(self, X_s, y_s):
        """
        Can be slow!
        """
        good_features = [self.feat_list[i] for i in self.good_idx]
        bad_features = [self.feat_list[i] for i in self.bad_idx]
        good_perf = []
        bad_perf = []
        for i, x_s in tqdm(enumerate(X_s)):
            for g in good_features:
                if g.count('~') == 0:
                    continue
                if find_wl_feature(x_s, (g,), self.gp.domain_kernels[0]):
                    good_perf.append(y_s[i])
                    break
            for b in bad_features:
                if b.count('~') == 0:
                    continue
                if find_wl_feature(x_s, (b,), self.gp.domain_kernels[0]):
                    bad_perf.append(y_s[i])
                    break
        return good_perf, bad_perf

    def get_legend(self, select=None):
        from matplotlib.lines import Line2D
        if select == 'n101':
            specs = {k: v for k, v in self.color_map.items() if k in self.n101}
        elif select == 'n201':
            specs = {k: v for k, v in self.color_map.items() if k in self.n201}
        elif select == 'darts':
            specs = {k: v for k, v in self.color_map.items() if k in self.darts}
        else:
            specs = self.color_map

        colors = list(specs.values())
        lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='None', marker='o') for c in colors]
        labels = list(specs.keys())
        return lines, labels
        # for k, v in specs.items():
        #     plt.plot(1, 1, ".", markersize=15, label=k, color=v)
        # ax1 = plt.gca()
        # fig2 = plt.figure()
        # ax2 = fig2.add_subplot()
        # ax2.axis('off')
        # res = ax1.get_legend_handles_labels()
        # # plt.figure(figsize=[10, 10])
        # legend = ax2.legend(*res, frameon=False, loc='lower center', ncol=5, prop={'size': 6})
        # fig = legend.figure
        # fig.canvas.draw()
        # bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # plt.show()


if __name__ == '__main__':
    from benchmarks.nas101 import NAS101Cifar10
    from bayesopt.generate_test_graphs import random_sampling
    import seaborn as sns

    # Change this line to your own data directory containing!
    e = NAS101Cifar10('../data')

    ori_task = 'ImageNet16-120'
    e.task = ori_task
    X = random_sampling(300, 'nasbench101')[0]
    X_s = random_sampling(1000, 'nasbench101')[0]

    y = torch.tensor([e.eval(x)[0] for x in X]).float()
    # This is not visible to the interpreter
    y_s = torch.tensor([e.eval(x)[0] for x in X_s]).float()
    y_s_np = y_s.numpy()

    interpreter = Interpreter(X, y, h=1)
    # for i, x_s in enumerate(X_s):
    #     perf_percentile = percentileofscore(y_s_np, y_s_np[i])
    #     interpreter.explain(x_s, y_s[i].item(), perf_percentile)
    # preds = interpreter.gp.predict(X_s)

    interpreter.plot_motifs()
    # Cross validate over different datasets
    if isinstance(e, NAS101Cifar10):
        y_s = torch.tensor([e.eval(x)[0] for x in X_s]).float()
        y_s_np = y_s.numpy()
        g, b = interpreter.validate(X_s, y_s_np)
        g = 1. - np.exp(-np.array(g).flatten())
        b = 1. - np.exp(-np.array(b).flatten())
        y = 1. - np.exp(-y_s_np)
        data = [g, y, b]
        plt.figure(figsize=[2.5, 2.5])
        plt.title('N101')
        sns.kdeplot(g, shade=True, label='good', color='green', clip=(g.min(), g.max()), )
        sns.kdeplot(y, shade=True, label='all', color='blue', clip=(y.min(), y.max()), alpha=0.2)
        sns.kdeplot(b, shade=True, label='bad', color='red', clip=(b.min(), b.max()), alpha=0.2)
        plt.axvline(np.median(g), linestyle='--', color='green')
        plt.axvline(np.median(y), linestyle='--', color='blue')
        plt.axvline(np.median(b), linestyle='--', color='red')
        plt.xlabel('Val Acc')
        plt.ylabel('PDF')
        plt.xlim([0.8, None])
        plt.legend()
        plt.savefig('valid_searched_on_N101.png', bbox_inches='tight')
    else:
        for task in ['cifar10-valid', 'cifar100', 'ImageNet16-120']:
            e.task = task
            y_s = torch.tensor([e.eval(x)[0] for x in X_s]).float()
            y_s_np = y_s.numpy()
            g, b = interpreter.validate(X_s, y_s_np)
            g = 1. - np.exp(-np.array(g).flatten())
            b = 1. - np.exp(-np.array(b).flatten())
            y = 1. - np.exp(-y_s_np)
            data = [g, y, b]
            plt.figure(figsize=[2.5, 2.5])
            plt.title(task)
            sns.kdeplot(g, shade=True, label='good', color='green', clip=(g.min(), g.max()), )
            sns.kdeplot(y, shade=True, label='all', color='blue', clip=(y.min(), y.max()), alpha=0.2)
            sns.kdeplot(b, shade=True, label='bad', color='red', clip=(b.min(), b.max()), alpha=0.2)
            plt.axvline(np.median(g), linestyle='--', color='green')
            plt.axvline(np.median(y), linestyle='--', color='blue')
            plt.axvline(np.median(b), linestyle='--', color='red')
            if task == 'cifar10-valid':
                plt.xlim([0.65, 0.95])
            elif task == 'cifar100':
                plt.xlim([0.35, 0.8])
            plt.xlabel('Val Acc')
            plt.ylabel('PDF')
            plt.legend()
            plt.savefig('valid_searched_on' + ori_task + '_on' + task + '.png', bbox_inches='tight')
            # plt.show()
