"""Record encoders, importance-aware means, gated aggregation and prediction network."""
import copy
import math
import torch
from torch import nn
from .stage2_contract import DEFAULT_IMPORTANCE_LAMBDA, validate_network


class RowEncoder(nn.Module):
    def __init__(self, numeric_dim, vocab_sizes, hidden=32, embedding_dim=8):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim, padding_idx=0) for size in vocab_sizes
        ])
        size = numeric_dim + len(vocab_sizes) * embedding_dim
        if not size:
            raise ValueError("An encoder needs at least one field")
        self.net = nn.Sequential(nn.Linear(size, hidden), nn.GELU(), nn.LayerNorm(hidden))

    def forward(self, numeric, categorical):
        pieces = [numeric]
        pieces.extend(emb(categorical[..., i]) for i, emb in enumerate(self.embeddings))
        return self.net(torch.cat(pieces, dim=-1))


class TabularEncoder(nn.Module):
    def __init__(self, numeric_dim, vocab_sizes, numeric_knots, hidden=64, embedding_dim=8):
        super().__init__()
        if numeric_dim != 2 * len(numeric_knots):
            raise ValueError("Expected numeric values followed by their missing indicators")
        self.fields = len(numeric_knots)
        left, spans, columns = [], [], []
        for i, knots in enumerate(numeric_knots):
            for a, b in zip(knots[:-1], knots[1:]):
                if b <= a:
                    raise ValueError("Numeric knots must be strictly increasing")
                left.append(a)
                spans.append(b-a)
                columns.append(i)
        self.register_buffer("left", torch.tensor(left, dtype=torch.float32))
        self.register_buffer("spans", torch.tensor(spans, dtype=torch.float32))
        self.register_buffer("columns", torch.tensor(columns, dtype=torch.long))
        self.embeddings = nn.ModuleList(nn.Embedding(size, embedding_dim, padding_idx=0) for size in vocab_sizes)
        size = numeric_dim + len(left)
        self.net = nn.Sequential(nn.Linear(size + embedding_dim * len(vocab_sizes), hidden),
                                 nn.GELU(), nn.LayerNorm(hidden))

    def numeric_features(self, numeric):
        # Raw signed-log values retain tail differences beyond the fitted bins.
        piecewise = ((numeric[..., self.columns] - self.left) / self.spans).clamp(0, 1)
        present = 1 - numeric[..., self.fields + self.columns]
        return torch.cat([numeric, piecewise * present], -1)

    def forward(self, numeric, categorical):
        pieces = [self.numeric_features(numeric)]
        pieces.extend(emb(categorical[..., i]) for i, emb in enumerate(self.embeddings))
        return self.net(torch.cat(pieces, -1))


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2*width),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(2*width, width))

    def forward(self, x):
        return x + self.net(x)


DEFAULT = dict(hidden=64, row_width=128, depth=4, dropout=.1)


def validate_config(config):
    if set(config) - set(DEFAULT):
        raise ValueError('Unknown module configuration fields')
    result = dict(DEFAULT, **config)
    for field in ('hidden','row_width'):
        if type(result[field]) is not int or not 8 <= result[field] <= 256:
            raise ValueError(f'Invalid {field}')
    if type(result['depth']) is not int or not 0 <= result['depth'] <= 4:
        raise ValueError('Invalid depth')
    if not isinstance(result['dropout'], (int,float)) or not 0 <= result['dropout'] <= .5:
        raise ValueError('Invalid dropout')
    return result


class FieldEncoder(nn.Module):
    def __init__(self, spec, config, tabular=False):
        super().__init__()
        width = config['row_width']
        self.input = (TabularEncoder(**spec, hidden=width, embedding_dim=8) if tabular else
                      RowEncoder(spec['numeric_dim'], spec['vocab_sizes'], width, 8))
        self.blocks = nn.Sequential(*(ResidualBlock(width,config['dropout']) for _ in range(config['depth'])))
        self.output = nn.Sequential(nn.Linear(width,config['hidden']),nn.GELU(),nn.LayerNorm(config['hidden']))

    def forward(self, numeric, categorical):
        return self.output(self.blocks(self.input(numeric,categorical)))


HIDDEN = 64


class ShrinkageMean(nn.Module):
    """Normalized importance/uniform mixture, without a learned projection.

    Positive-mass groups use alpha=(1-lambda)/n + lambda*s/sum(s).
    All-zero groups use a uniform mean; empty groups return an absent message.
    """
    def __init__(self, shrinkage=DEFAULT_IMPORTANCE_LAMBDA):
        super().__init__()
        if (isinstance(shrinkage, bool) or not isinstance(shrinkage, (int, float))
                or not math.isfinite(shrinkage) or not 0 <= shrinkage <= 1):
            raise ValueError('shrinkage must be finite and lie in [0,1]')
        self.shrinkage = float(shrinkage)

    def forward(self, values, parent, scores, receivers):
        if parent.ndim != 1 or parent.shape != scores.shape or len(values) != len(parent):
            raise ValueError('Misaligned packed aggregation inputs')
        if not torch.isfinite(scores).all() or (scores < 0).any():
            raise ValueError('Importance must be finite and nonnegative')
        count = scores.new_zeros(receivers).index_add_(0, parent, torch.ones_like(scores))
        maximum = scores.new_zeros(receivers)
        maximum.scatter_reduce_(0, parent, scores, reduce='amax', include_self=True)
        # Scale before summing to avoid overflow for finite SQL scores.
        group_max = maximum[parent]
        scaled = scores/torch.where(group_max > 0, group_max, torch.ones_like(group_max))
        mass = scores.new_zeros(receivers).index_add_(0, parent, scaled)
        uniform = count[parent].reciprocal()
        group_mass = mass[parent]
        relative = scaled/torch.where(group_mass > 0, group_mass, torch.ones_like(group_mass))
        relative = torch.where(group_mass > 0, relative, uniform)
        alpha = (1-self.shrinkage)*uniform+self.shrinkage*relative
        message = values.new_zeros((receivers, values.shape[-1]))
        message.index_add_(0, parent, values*alpha[:, None])
        return message, (count > 0)[:, None]


class GatedSum(nn.Module):
    """Feature-wise gates conditioned on each receiver and each table message."""
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.gate = nn.Linear(2*hidden, hidden)
        self.projection = nn.Linear(hidden, hidden)
        nn.init.constant_(self.gate.bias, -2.)

    def forward(self, context, sources):
        delta = torch.zeros_like(context)
        for message, present in sources:
            gate = self.gate(torch.cat([context, message], -1)).sigmoid()
            delta = delta+gate*self.projection(message)*present
        return delta


class Stage2Network(nn.Module):
    def __init__(self, root_spec, node_specs, nodes, horizon_days, operators):
        super().__init__()
        validate_network(operators)
        self.model_spec = copy.deepcopy(dict(root_spec=root_spec, node_specs=node_specs,
            nodes=nodes, horizon_days=horizon_days, operators=operators))
        self.nodes = copy.deepcopy(nodes)
        self.root = FieldEncoder(root_spec, validate_config({'hidden':64, 'row_width':128, 'depth':4}), tabular=True)
        cfg = validate_config({'hidden':64, 'row_width':64, 'depth':1})
        self.records = nn.ModuleDict({'node_'+name: FieldEncoder(spec, cfg)
                                     for name, spec in node_specs.items()})
        self.intra = ShrinkageMean(operators.get('importance_lambda', DEFAULT_IMPORTANCE_LAMBDA))
        self.inter = GatedSum()
        self.head = nn.Sequential(nn.Linear(64,64), nn.GELU(), nn.Dropout(.1), nn.Linear(64,1))

    def auxiliary(self, batch, root=None):
        """Aggregate updated children first, then gate messages using the root."""
        root = self.root(batch['sql_num'], batch['sql_cat']) if root is None else root
        vectors = {node['name']: self.records['node_'+node['name']](
            batch[node['name']+'_num'], batch[node['name']+'_cat']) for node in self.nodes}
        incoming = {node['name']: [] for node in self.nodes}
        incoming['root'] = []
        for node in reversed(self.nodes):
            name, parent = node['name'], node['parent']
            context = vectors[name]
            updated = context+self.inter(context, incoming[name])
            receivers = len(root) if parent == 'root' else len(vectors[parent])
            incoming[parent].append(self.intra(updated, batch[name+'_parent'],
                                               batch[name+'_importance'], receivers))
        return self.inter(root, incoming['root'])

    def forward(self, batch):
        root = self.root(batch['sql_num'], batch['sql_cat'])
        return self.head(root+self.auxiliary(batch, root)).squeeze(-1)
