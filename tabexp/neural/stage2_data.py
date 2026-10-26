"""Train-only codecs, packed batches, SQL scheduling and cutoff-safe relational inputs."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re

import duckdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .snapshot import quote
from .stage2_contract import ROOT, DEFAULT_IMPORTANCE_LAMBDA, DEFAULT_MAX_HOPS, occurrences, record_budget, validate_plan


def official_test_metrics(loader, probabilities, rows, entity, time_col="timestamp"):
    """Unmask labels ONLY at scoring, and verify canonical prediction order."""
    table = loader.task.get_table("test", mask_input_cols=False)
    keys = [entity, time_col]
    expected = table.df[keys].reset_index(drop=True)
    if not expected.equals(rows[keys].reset_index(drop=True)):
        raise ValueError("Test prediction keys/order differ from official label table")
    return loader.evaluate(probabilities, split="test", table=table)


def category_strings(series):
    # Nullable integer columns can become float after a LEFT JOIN in another
    # split; 1 and 1.0 must retain the same category identity.
    def canonical(value):
        if pd.isna(value):
            return None
        if isinstance(value, (float, np.floating)) and np.isfinite(value) and value.is_integer():
            return str(int(value))
        return str(value)
    return series.astype(object).map(canonical)


@dataclass
class Codec:
    categorical: list
    max_categories: int = 1024
    numeric: list = field(default_factory=list)
    vocabulary: dict = field(default_factory=dict)
    medians: list = field(default_factory=list)
    scales: list = field(default_factory=list)
    numeric_knots: list = field(default_factory=list)

    def fit(self, frame):
        return self.fit_streaming(frame)

    def _numeric(self, frame):
        columns = []
        for col in self.numeric:
            series = frame[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                value = series.astype("datetime64[ns]").astype("int64").astype(float) / 1e9
                value[series.isna()] = np.nan
            else:
                value = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            columns.append(value)
        return np.stack(columns, -1) if columns else np.zeros((len(frame), 0))

    def fit_streaming(self,frame):
        """Exact per-column statistics with no full float64 numeric matrix."""
        if frame.columns.duplicated().any():
            raise ValueError('Duplicate input columns')
        self.numeric = [c for c in frame if c not in self.categorical]
        for col in self.categorical:
            values = category_strings(frame[col]).dropna().value_counts()
            words = sorted(values.index,key=lambda x:(-values[x],x))[:self.max_categories]
            self.vocabulary[col] = {word:i+2 for i,word in enumerate(words)}
        self.medians,self.scales,self.numeric_knots = [],[],[]
        for col in self.numeric:
            saved = self.numeric
            self.numeric = [col]
            try:
                x = self._numeric(frame)[:,0]
            finally:
                self.numeric = saved
            valid = x[np.isfinite(x)]
            median = float(np.median(valid)) if len(valid) else 0.
            scale = float(np.subtract(*np.percentile(valid,[75,25]))) if len(valid) else 1.
            scale = scale if scale>1e-6 else 1.
            self.medians.append(median)
            self.scales.append(scale)
            t = (valid-median)/scale
            t = np.sign(t)*np.log1p(np.abs(t))
            knots = np.unique(np.quantile(t,np.linspace(0,1,17))) if len(t) else np.array([0.])
            if len(knots)<2:
                knots = np.array([float(knots[0])-1.,float(knots[0])+1.])
            self.numeric_knots.append(knots.tolist())
        return self

    def transform_streaming(self,frame,numeric_out=None,categorical_out=None):
        """Fill arrays/memmaps columnwise using the same transform contract."""
        n = np.zeros((len(frame),2*len(self.numeric)),np.float32) if numeric_out is None else numeric_out
        c = np.zeros((len(frame),len(self.categorical)),np.int64) if categorical_out is None else categorical_out
        columns = list(self.numeric)
        for i,col in enumerate(columns):
            self.numeric = [col]
            try:
                x = self._numeric(frame)[:,0]
            finally:
                self.numeric = columns
            missing = ~np.isfinite(x)
            x = (np.where(missing,self.medians[i],x)-self.medians[i])/self.scales[i]
            x = np.sign(x)*np.log1p(np.abs(x))
            n[:,i],n[:,len(columns)+i] = x,missing
        for i,col in enumerate(self.categorical):
            values = category_strings(frame[col]).map(self.vocabulary[col]).fillna(1).to_numpy(dtype=np.int64,copy=True)
            values[frame[col].isna().to_numpy()] = 0
            c[:,i] = values
        return n,c

    def transform(self, frame):
        x = self._numeric(frame)
        missing = ~np.isfinite(x)
        x = np.where(missing, np.asarray(self.medians), x)
        x = (x - np.asarray(self.medians)) / np.asarray(self.scales)
        x = np.sign(x) * np.log1p(np.abs(x))
        numeric = np.concatenate([x, missing.astype(float)], -1).astype(np.float32)
        cats = []
        for col in self.categorical:
            s = frame[col]
            v = category_strings(s).map(self.vocabulary[col]).fillna(1).to_numpy(dtype=np.int64, copy=True)
            v[s.isna().to_numpy()] = 0
            cats.append(v)
        categorical = np.stack(cats, -1) if cats else np.zeros((len(frame), 0), dtype=np.int64)
        return numeric, categorical

    @property
    def spec(self):
        result = {"numeric_dim": 2 * len(self.numeric),
                  "vocab_sizes": [len(self.vocabulary[c]) + 2 for c in self.categorical]}
        result["numeric_knots"] = self.numeric_knots
        return result


class IndexDataset(Dataset):
    def __init__(self,size):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self,index):
        return index


def make_loader(self,labels,batch_size,shuffle=False,generator=None):
    labels = np.asarray(labels,np.float32) if labels is not None else None
    def collate(indices):
        batch = self.batch(indices)
        return (batch,torch.from_numpy(labels[indices].copy())) if labels is not None else batch
    return DataLoader(IndexDataset(self.size),batch_size=batch_size,shuffle=shuffle,generator=generator,
                      num_workers=0,collate_fn=collate)


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _expression_ast(conn, expression):
    serialized = conn.execute('SELECT json_serialize_sql(?)',
                              ['SELECT ('+expression+') FROM records r']).fetchone()[0]
    document = json.loads(serialized)
    if document.get('error'):
        raise ValueError('SQL expression cannot be analyzed')
    statements = document['statements']
    if len(statements) != 1:
        raise ValueError('Expected one expression')
    ast = statements[0]['node']['select_list'][0]
    # Unknown AST classes, whole-row/star references and lambda scoping need
    # special handling; do not guess which enhanced columns they depend on.
    supported = {'FUNCTION', 'COLUMN_REF', 'CONSTANT', 'WINDOW', 'SUBQUERY',
                 'COMPARISON', 'CONJUNCTION', 'CASE', 'CAST', 'BETWEEN',
                 'OPERATOR', 'COLLATE'}
    if any(n.get('class') not in supported for n in _walk(ast) if 'class' in n):
        raise ValueError('Unsupported expression class')
    return ast


def schedule_enhancement(conn, fields, importance):
    """Return early/late field names, or a reason for eager fallback.

Early fields are either dependencies of f.column in the scoring expression,
or expressions whose value may change if records is filtered (windows,
subqueries reading records/features, unqualified/whole-row references).
"""
    all_names = [field['name'] for field in fields]
    fallback = {'mode':'eager', 'early':all_names, 'late':[], 'reason':None}
    try:
        score_ast = _expression_ast(conn, importance)
        field_asts = {field['name']:_expression_ast(conn, field['sql']) for field in fields}
        # SQL macros can hide references/windows not present in the call AST.
        # Keep their old execution semantics rather than assuming row locality.
        macros = {row[0].casefold() for row in conn.execute(
            "SELECT function_name FROM duckdb_functions() WHERE function_type IN ('macro','table_macro')").fetchall()}
        for ast in [score_ast, *field_asts.values()]:
            if any(node.get('class') == 'FUNCTION' and node.get('function_name', '').casefold() in macros
                   for node in _walk(ast)):
                raise ValueError('SQL macro dependencies require eager execution')
        known = {name.casefold():name for name in all_names}
        dependencies = set()
        for node in _walk(score_ast):
            if node.get('type') == 'BASE_TABLE' and node.get('table_name', '').casefold() == 'features':
                raise ValueError('Scoring subquery reads the full features relation')
            if node.get('class') != 'COLUMN_REF':
                continue
            columns = node['column_names']
            if len(columns) < 2:
                raise ValueError('Scoring uses an unqualified or whole-row reference')
            if columns[0].casefold() == 'f':
                if len(columns) != 2:
                    raise ValueError('Scoring uses a structured feature reference')
                column = columns[1].casefold()
                if column == '_node' or (not fields and column == '_constant'):
                    continue
                if column not in known:
                    # This can also be a shadowed alias inside a subquery.
                    raise ValueError('Cannot resolve scoring feature scope')
                dependencies.add(known[column])
        early = set(dependencies)
        for field in fields:
            ast = field_asts[field['name']]
            for node in _walk(ast):
                if node.get('type') == 'BASE_TABLE' and node.get('table_name', '').casefold() == 'features':
                    raise ValueError('Enhancement reads the runtime features relation')
                if (node.get('class') == 'WINDOW' or
                    node.get('type') == 'BASE_TABLE' and node.get('table_name', '').casefold() in ('records','features') or
                    node.get('class') == 'COLUMN_REF' and len(node['column_names']) < 2):
                    early.add(field['name'])
        return {'mode':'deferred', 'early':[n for n in all_names if n in early],
                'late':[n for n in all_names if n not in early], 'reason':None}
    except (duckdb.Error, ValueError, KeyError, TypeError, IndexError) as error:
        fallback['reason'] = str(error)
        return fallback


INTERNAL = {'_seed', '_cutoff', '_record', '_parent', '_node', '_time', '_importance'}


def expression_sql(conn, expression, allowed_tables):
    """Accept scalar SELECT expressions, not author-supplied execution scripts.

This is a design contract, not a sandbox for adversarial SQL. Execute reviewed
plans only. Source columns are censored independently of expressions.
"""
    if re.search(r';|--|/\*|\b(random|uuid|now|current_timestamp|current_date|read_\w*|\w*scan|query|query_table|glob|nextval|setval)\b', expression, re.I):
        raise ValueError('SQL must be a deterministic scalar expression over safe tables')
    sql = 'SELECT ('+expression+') FROM records r'
    statements = conn.extract_statements(sql)
    if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
        raise ValueError('Only scalar SELECT expressions are supported')
    if conn.get_table_names(sql)-set(allowed_tables)-{'records', 'features'}:
        raise ValueError('SQL references a table outside the safe catalog')
    return expression


def context_from_loader(loader, *, default_budget=64, hop_decay=.5, per_table=None,
                        intra='shrinkage_mean', inter='gated_sum', final='root_residual',
                        importance_lambda=DEFAULT_IMPORTANCE_LAMBDA, max_hops=DEFAULT_MAX_HOPS):
    """Export system-owned schema/task context. No model API is called."""
    blocked = set(tuple(x) for x in (getattr(loader.task, 'remove_columns', None) or []))
    blocked.add((loader.task.entity_table, loader.task.target_col))
    tables = {}
    with duckdb.connect() as conn:
        for name, spec in loader.dataset.manifest.tables.items():
            source = conn.read_parquet(str(Path(loader.dataset.db_dir)/(name+'.parquet')))
            columns = [col for col in source.columns if (name, col) not in blocked]
            dtypes = {col:str(dtype) for col,dtype in zip(source.columns,source.types) if col in columns}
            # Parquet physical row number is a stable identity, even for event
            # tables without a declared PK. It is never a neural feature.
            key = spec.pkey or '_physical_row'
            if spec.pkey is None:
                if key in columns or 'file_row_number' in columns:
                    raise ValueError('Reserved physical identity column collision')
                columns.append(key)
                dtypes[key] = 'BIGINT'
            tables[name] = {'key': key, 'time': spec.time_col, 'columns': columns,
                            'fkeys': {k:v for k,v in spec.fkeys.items() if k in columns}, 'dtypes':dtypes}
    kind = 'regression' if 'regression' in str(loader.task.task_type).lower() else 'classification'
    network = {'intra': intra, 'inter': inter, 'final': final}
    network['importance_lambda'] = importance_lambda
    return {'task': {'dataset': loader.dataset_name, 'task': loader.task_name, 'kind': kind,
            'entity': loader.task.entity_col, 'clock': loader.task.time_col,
            'target': loader.task.target_col, 'root_table': loader.task.entity_table,
            'horizon_days': float(loader.task.timedelta.total_seconds()/86400.)},
            'tables': tables, 'sampling': {'default_budget': default_budget,
            'per_table': per_table or {}, 'hop_decay': hop_decay, 'max_hops': max_hops},
            'network': network, 'input_columns': 'original_plus_enhancements'}


@contextmanager
def database(loader, context, *, memory_limit='3GB'):
    """Open label-free sources. prepare() applies each anchor cutoff first."""
    conn = duckdb.connect(config={'memory_limit': memory_limit, 'threads': 2})
    try:
        for name, desc in context['tables'].items():
            path = str(Path(loader.dataset.db_dir)/(name+'.parquet')).replace("'", "''")
            physical = desc['key'] == '_physical_row'
            projections = [quote(c) for c in desc['columns'] if c != '_physical_row']
            if physical:
                projections.append('file_row_number AS _physical_row')
            sql = 'SELECT '+','.join(projections)+f" FROM read_parquet('{path}',file_row_number={'true' if physical else 'false'})"
            if desc['time']:
                stamp = str(pd.Timestamp(loader.dataset.test_timestamp)).replace("'", "''")
                sql += ' WHERE '+quote(desc['time'])+f"<=TIMESTAMP '{stamp}'"
            conn.execute('CREATE TEMP VIEW '+quote('_s2_source_'+name)+' AS '+sql)
        yield conn
    finally:
        conn.close()


class PackedStore:
    """Flat records and parent indices; no global tree-shaped padding."""
    make_loader = make_loader

    def __init__(self, root, nodes, arrays):
        self.root, self.nodes, self.arrays = root, nodes, arrays
        self.size = len(root['sql_num'])
        self.ids = {}
        for name, data in arrays.items():
            order = np.argsort(data['seed'], kind='stable')
            offsets = np.r_[0, np.cumsum(np.bincount(data['seed'], minlength=self.size))]
            self.ids[name] = (order, offsets)

    def batch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if ((indices < 0) | (indices >= self.size)).any():
            raise IndexError('Anchor index outside store')
        result = {k: torch.from_numpy(v[indices].copy()) for k,v in self.root.items()}
        # Per-example maps allow reordered and repeated anchors in a batch.
        maps = {'root': [{int(seed): i} for i,seed in enumerate(indices)]}
        for node in self.nodes:
            name = node['name']; data = self.arrays[name]
            order, offsets = self.ids[name]
            selections = [order[offsets[s]:offsets[s+1]] for s in indices]
            positions = np.concatenate(selections) if len(indices) else np.empty(0, np.int64)
            parent, local_maps, start = [], [], 0
            for i, chosen in enumerate(selections):
                parent.extend(maps[node['parent']][i][int(p)] for p in data['parent'][chosen])
                local_maps.append({int(original): start+j for j,original in enumerate(chosen)})
                start += len(chosen)
            maps[name] = local_maps
            for field in ('num', 'cat', 'importance'):
                result[name+'_'+field] = torch.from_numpy(data[field][positions].copy())
            result[name+'_parent'] = torch.tensor(parent, dtype=torch.long)
        return result

    def close(self):
        self.root = {}; self.arrays = {}; self.ids = {}


class Stage2Executor:
    def __init__(self, plan, preprocessing=None):
        self.plan = validate_plan(plan)
        self.feature_fields = self._encoding_fields()
        self.codecs = {name: Codec(**state) for name,state in (preprocessing or {}).items()}
        self.execution_stats = []

    def _encoding_fields(self):
        """Original visible attributes are inputs; plan fields are additions.

        Benchmark targets/remove_columns were already removed from the catalog.
        """
        result = {}
        context = self.plan['context']
        for node in occurrences(self.plan):
            name = node['name']
            additions = self.plan['enhancements'][name]
            desc = context['tables'][node['table']]
            # Real PK/FK values are original attributes; synthetic row offsets
            # are execution identities only and must never become features.
            identities = {'_physical_row'}
            originals = []
            reserved = {f['name'] for f in additions}
            expressions = {re.sub(r'[\s"`]', '', f['sql']).lower() for f in additions}
            for index, column in enumerate(desc['columns']):
                if column in identities:
                    continue
                # Existing explicit projections already encode this attribute.
                if ('r.'+column).lower() in expressions:
                    continue
                alias = 'raw__'+column if re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',column) else 'raw__column'+str(index)
                if alias in reserved:
                    raise ValueError('Enhancement collides with original column: '+alias)
                dtype = desc['dtypes'][column].upper()
                temporal = dtype.startswith(('TIMESTAMP', 'DATE', 'TIME'))
                numeric = temporal or any(t in dtype for t in ('INT','FLOAT','DOUBLE','DECIMAL','BOOLEAN','REAL','HUGEINT')) and not any(t in dtype for t in ('[]','STRUCT','MAP','LIST'))
                expression = 'r.'+quote(column)
                if temporal:
                    expression = 'epoch('+expression+')'
                originals.append({'name':alias, 'kind':'numeric' if numeric else 'categorical', 'sql':expression})
            result[name] = originals + additions
        return result

    def state_dict(self):
        if ROOT not in self.codecs:
            raise ValueError('Prepare train first')
        return {name: asdict(codec) for name,codec in self.codecs.items()}

    def _feature_expressions(self, conn, fields):
        expressions = []
        for field in fields:
            expression_sql(conn, field['sql'], self.plan['context']['tables'])
            cast = 'DOUBLE' if field['kind'] == 'numeric' else 'VARCHAR'
            expressions.append('CAST(('+field['sql']+') AS '+cast+') AS '+quote(field['name']))
        return expressions

    def _enhance(self, conn, name, frame, *, fields=None):
        conn.register('records', frame)
        fields = self.feature_fields[name] if fields is None else fields
        expressions = self._feature_expressions(conn, fields)
        if not expressions:
            expressions = ['1.0 AS _constant']
        return conn.execute('SELECT '+','.join(expressions)+' FROM records r ORDER BY r._node').fetchdf()

    def _select(self, name, raw, weights):
        weights = np.asarray(np.ma.filled(weights, np.nan), dtype=np.float64)
        if weights.shape != (len(raw),) or not np.isfinite(weights).all() or (weights < 0).any() or (weights > np.finfo(np.float32).max).any():
            raise ValueError('Importance must be finite, nonnegative, float32-representable')
        raw = raw.copy()
        raw['_importance'] = weights.astype(np.float32)
        chosen = raw.sort_values(['_parent','_importance','_record'], ascending=[True,False,True], kind='stable').groupby('_parent', sort=False).head(record_budget(self.plan, name)).index
        return raw.loc[chosen].sort_values(['_parent','_time','_record'], kind='stable', na_position='last')

    def _eager_sample_and_enhance(self, conn, name, raw):
        """Original behavior, retained for ambiguous SQL and equivalence tests."""
        enhanced = self._enhance(conn, name, raw)
        if len(enhanced) != len(raw):
            raise ValueError('Enhancement must return one row per candidate record')
        conn.register('features', enhanced.assign(_node=raw['_node'].to_numpy()))
        score = self.plan['importance'][name]['sql']
        expression_sql(conn, score, self.plan['context']['tables'])
        weights = conn.execute('SELECT CAST(('+score+') AS DOUBLE) AS weight FROM records r JOIN features f ON f._node=r._node ORDER BY r._node').fetchnumpy()['weight']
        selected = self._select(name, raw, weights)
        return selected.reset_index(drop=True), enhanced.loc[selected.index].reset_index(drop=True)

    def _sample_and_enhance(self, conn, name, raw):
        fields = self.feature_fields[name]
        score = self.plan['importance'][name]['sql']
        schedule = schedule_enhancement(conn, fields, score)
        if schedule['mode'] == 'eager':
            selected, enhanced = self._eager_sample_and_enhance(conn, name, raw)
        else:
            # Keep records bound to the complete candidate set while evaluating
            # the score and candidate-sensitive intermediate expressions.
            conn.register('records', raw)
            expression_sql(conn, score, self.plan['context']['tables'])
            early_fields = [f for f in fields if f['name'] in schedule['early']]
            early = self._feature_expressions(conn, early_fields)
            if not fields:
                early = ['1.0 AS _constant']
            feature_sql = 'SELECT r._node'+(','+','.join(early) if early else '')+' FROM records r'
            # MATERIALIZED evaluates dependencies once, including those reused
            # as encoded features. Scores need not reference f at all.
            query = ('WITH features AS MATERIALIZED ('+feature_sql+') SELECT r._node,'
                     'CAST(('+score+') AS DOUBLE) AS _importance,f.* EXCLUDE (_node) '
                     'FROM records r JOIN features f ON f._node=r._node ORDER BY r._node')
            scored = conn.execute(query).fetchdf()
            if len(scored) != len(raw) or not np.array_equal(scored['_node'], raw['_node']):
                raise ValueError('Scoring must return one aligned value per candidate record')
            selected = self._select(name, raw, scored['_importance'].to_numpy())
            kept = scored.loc[selected.index].reset_index(drop=True)
            enhanced = kept.drop(columns=['_node','_importance'])
            late_fields = [f for f in fields if f['name'] in schedule['late']]
            if late_fields:
                # Preserve original candidate IDs/order and the full records
                # relation for correlated historical queries. Only the outer
                # rows are filtered; a reference to records inside a subquery
                # or window is scheduled early above.
                conn.register('_s2_selected', selected[['_node']])
                expressions = self._feature_expressions(conn, late_fields)
                late = conn.execute('SELECT r._node,'+','.join(expressions)+
                    ' FROM records r WHERE r._node IN (SELECT _node FROM _s2_selected) ORDER BY r._node').fetchdf().set_index('_node')
                if len(late) != len(selected):
                    raise ValueError('Enhancement must return one row per selected record')
                late = late.loc[selected['_node']].reset_index(drop=True)
                for field in late_fields:
                    enhanced[field['name']] = late[field['name']]
            if fields:
                enhanced = enhanced[[field['name'] for field in fields]]
            selected = selected.reset_index(drop=True)
        self.execution_stats.append({'name':name, 'candidates':len(raw), 'selected':len(selected), **schedule})
        return selected, enhanced

    def _groups(self, rows):
        return rows.groupby('cutoff', sort=True)

    def _store_frame(self, name, raw, enhanced, frames, features):
        frames[name].append(raw)
        features[name].append(enhanced)

    def prepare(self, conn, rows, *, fit=False):
        if fit and self.codecs:
            raise ValueError('Refitting is forbidden')
        if not fit and ROOT not in self.codecs:
            raise ValueError('Fit preprocessing on train first')
        context = self.plan['context']; task = context['task']; catalog = context['tables']
        rows = rows[[task['entity'], task['clock']]].reset_index(drop=True).copy()
        if not len(rows) or rows.isna().any().any() or rows.duplicated().any():
            raise ValueError('Anchors must be nonempty, nonnull, unique entity/time pairs')
        rows.columns = ['entity', 'cutoff']; rows.insert(0, '_seed', np.arange(len(rows)))
        if rows.cutoff.dt.tz is not None:
            raise ValueError('Use timezone-naive UTC timestamps, as in RelBench')
        frames = {n['name']: [] for n in occurrences(self.plan)}
        features = {name: [] for name in frames}
        counts = {name: 0 for name in frames}
        by_name = {n['name']:n for n in occurrences(self.plan)}
        self.execution_stats = []
        for cutoff, anchors in self._groups(rows):
            stamp = str(cutoff).replace("'", "''")
            for table, desc in catalog.items():
                sql = 'SELECT '+','.join(quote(c) for c in desc['columns'])+' FROM '+quote('_s2_source_'+table)
                if desc['time']:
                    op = '<=' if table == task['root_table'] else '<'
                    sql += ' WHERE '+quote(desc['time'])+op+f"TIMESTAMP '{stamp}'"
                conn.execute('CREATE OR REPLACE TEMP VIEW '+quote(table)+' AS '+sql)
            conn.register('_s2_anchors', anchors)
            current = {}
            for node in occurrences(self.plan):
                name, table = node['name'], node['table']; desc = catalog[table]
                key = quote(desc['key'])
                time = 'r.'+quote(desc['time']) if desc['time'] else 'CAST(NULL AS TIMESTAMP)'
                if name == ROOT:
                    sql = f'SELECT a._seed,a.cutoff AS _cutoff,a._seed AS _parent,a.entity AS _record,{time} AS _time,r.* FROM _s2_anchors a LEFT JOIN {quote(table)} r ON r.{key}=a.entity ORDER BY a._seed'
                else:
                    parent = current[node['parent']]
                    conn.register('_s2_parents', parent)
                    parent_desc = catalog[by_name[node['parent']]['table']]
                    fk = quote(node['fk_column'])
                    incoming = node.get('direction', 'incoming' if node['fk_table'] == table else 'outgoing') == 'incoming'
                    if incoming:
                        join = f'r.{fk}=p.{quote(parent_desc["key"])}'
                    else:
                        join = f'p.{fk}=r.{key}'
                    if desc['time']:
                        join += ' AND r.'+quote(desc['time'])+'<p._cutoff'
                    sql = f'SELECT p._seed,p._cutoff,p._node AS _parent,r.{key} AS _record,{time} AS _time,r.* FROM _s2_parents p JOIN {quote(table)} r ON {join} ORDER BY p._node,r.{key}'
                raw = conn.execute(sql).fetchdf()
                if set(desc['columns']) & INTERNAL:
                    raise ValueError('Source schema collides with internal columns')
                if raw[['_parent','_record']].duplicated().any() or (name == ROOT and len(raw) != len(anchors)):
                    raise ValueError('Source identities are not unique; invalid catalog key')
                raw['_node'] = np.arange(len(raw))  # temporary row alignment
                if name == ROOT:
                    enhanced = self._enhance(conn, name, raw)
                    if len(enhanced) != len(raw):
                        raise ValueError('Enhancement must return one row per candidate record')
                else:
                    raw, enhanced = self._sample_and_enhance(conn, name, raw)
                raw['_node'] = np.arange(counts[name], counts[name]+len(raw)) if name != ROOT else raw['_seed']
                counts[name] += len(raw)
                self._store_frame(name, raw, enhanced, frames, features)
                current[name] = raw
        return self._finish(counts, frames, features, fit)

    def _finish(self, counts, frames, features, fit):
        context = self.plan['context']; task = context['task']
        arrays = {}; root = None
        for node in occurrences(self.plan):
            name = node['name']
            raw = pd.concat(frames[name], ignore_index=True)
            data = pd.concat(features[name], ignore_index=True)
            if name == ROOT:
                order = np.argsort(raw['_seed'].to_numpy(), kind='stable')
                data = data.iloc[order].reset_index(drop=True)
            if fit:
                categorical = [f['name'] for f in self.feature_fields[name] if f['kind'] == 'categorical']
                self.codecs[name] = Codec(categorical, 1024).fit_streaming(data)
            num, cat = self.codecs[name].transform_streaming(data)
            if name == ROOT:
                root = {'sql_num': num, 'sql_cat': cat}
            else:
                arrays[name] = {'num': num, 'cat': cat, 'seed': raw['_seed'].to_numpy(np.int64),
                    'parent': raw['_parent'].to_numpy(np.int64), 'importance': raw['_importance'].to_numpy(np.float32)}
        self.model_spec = {'root_spec': self.codecs[ROOT].spec,
            'node_specs': {n['name']:self.codecs[n['name']].spec for n in self.plan['structure']['nodes']},
            'nodes': self.plan['structure']['nodes'], 'horizon_days': task['horizon_days'],
            'operators': context['network']}
        self.coverage = {name: len(data['seed']) for name,data in arrays.items()}
        return PackedStore(root, self.plan['structure']['nodes'], arrays)
