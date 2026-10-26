"""Plan identity and strict contracts for task features, PK-FK structure and importance."""
import copy
import hashlib
import json
import math
import re


def plan_hash(plan):
    return hashlib.sha256(json.dumps(plan,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()


FORMAT = 'stage2_records_v1'
ROOT = 'root'
DEFAULT_IMPORTANCE_LAMBDA = 0.25
DEFAULT_MAX_HOPS = 2


def validate_max_hops(value):
    if type(value) is not int or value < 0:
        raise ValueError('max_hops must be a nonnegative integer')
    return value


def keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or set(value)-set(required)-set(optional):
        raise ValueError('Missing or unsupported keys; expected '+str(tuple(required)))


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', value):
        raise ValueError('Expected a plain identifier: '+str(value))
    return value


def validate_structure(context, structure):
    """Select table occurrences and real FK edges, in parent-before-child order.

Occurrences preserve relation roles and unfold cycles/repeated tables into a
finite computation tree. No designer-supplied arbitrary join predicates.
"""
    keys(structure, ('nodes',))
    if not isinstance(structure['nodes'], list):
        raise ValueError('nodes must be a list')
    tables = context['tables']
    seen = {ROOT: context['task']['root_table']}
    depths = {ROOT: 0}
    max_hops = validate_max_hops(context['sampling']['max_hops'])
    for node in structure['nodes']:
        keys(node, ('name', 'table', 'parent', 'fk_table', 'fk_column'), ('direction',))
        name = identifier(node['name'])
        if name in seen or name == 'sql' or name.startswith('_'):
            raise ValueError('Duplicate or reserved occurrence name')
        if node['parent'] not in seen or node['table'] not in tables:
            raise ValueError('Unknown table or parent; parents must precede children')
        depths[name] = depths[node['parent']]+1
        if depths[name] > max_hops:
            raise ValueError(f'Occurrence {name} exceeds max_hops={max_hops}')
        parent, child = seen[node['parent']], node['table']
        fk_table = node['fk_table']
        if fk_table not in (parent, child):
            raise ValueError('FK must connect the declared parent and child')
        other = child if fk_table == parent else parent
        if tables[fk_table]['fkeys'].get(node['fk_column']) != other:
            raise ValueError('Selection is not a catalog PK-FK edge')
        direction = node.get('direction')
        if parent == child and direction is None:
            raise ValueError('Self-table edges need explicit incoming/outgoing direction')
        if direction is not None and (direction not in ('incoming', 'outgoing') or
                fk_table != (parent if direction == 'outgoing' else child)):
            raise ValueError('Edge direction does not match FK owner')
        seen[name] = child
    return copy.deepcopy(structure)


def validate_enhancements(context, structure, enhancements):
    """Explicit typed SQL expressions, evaluated for each current record r.

Subqueries over cutoff-safe tables may generate historical statistics. SQL is
not allowed to choose records, joins in the computation tree, or tensor layout.
"""
    expected = {ROOT} | {n['name'] for n in structure['nodes']}
    if not isinstance(enhancements, dict) or set(enhancements) != expected:
        raise ValueError('Enhancement must cover exactly root and selected occurrences')
    for occurrence, fields in enhancements.items():
        if not isinstance(fields, list):
            raise ValueError('Enhancement fields must be a list')
        if occurrence != ROOT and fields:
            raise ValueError('SQL augmentation applies to the task table only')
        seen = set()
        for field in fields:
            keys(field, ('name', 'kind', 'sql'))
            name = identifier(field['name'])
            if name.startswith('_') or name in seen:
                raise ValueError('Duplicate or reserved feature name')
            if field['kind'] not in ('numeric', 'categorical'):
                raise ValueError('Feature kind must be numeric or categorical')
            if not isinstance(field['sql'], str) or not field['sql'].strip():
                raise ValueError('Feature SQL must be a nonempty expression')
            seen.add(name)
    return copy.deepcopy(enhancements)


def validate_importance(structure, importance):
    """One nonnegative scalar SQL expression per auxiliary occurrence.

Use r for current raw record and f for its enhanced fields. Runtime checks
finite/nonnegative values. An explicit constant 1.0 is the uniform policy.
"""
    if not isinstance(importance, dict) or set(importance) != {n['name'] for n in structure['nodes']}:
        raise ValueError('Importance must cover exactly auxiliary occurrences')
    for value in importance.values():
        keys(value, ('sql',))
        if not isinstance(value['sql'], str) or not value['sql'].strip():
            raise ValueError('Importance SQL must be a nonempty expression')
    return copy.deepcopy(importance)


def validate_context(context):
    keys(context, ('task', 'tables', 'sampling', 'network'), ('input_columns',))
    if context.get('input_columns') != 'original_plus_enhancements':
        raise ValueError('Encode original visible fields together with root enhancements')
    task = context['task']
    keys(task, ('dataset', 'task', 'kind', 'entity', 'clock', 'target', 'root_table', 'horizon_days'),
         ('target_bounds', 'training_budget'))
    if task['kind'] not in ('classification', 'regression'):
        raise ValueError('Only binary classification and scalar regression are supported')
    if not isinstance(task['horizon_days'], (int, float)) or isinstance(task['horizon_days'], bool) or not math.isfinite(task['horizon_days']) or task['horizon_days'] <= 0:
        raise ValueError('Invalid prediction horizon')
    for table, desc in context['tables'].items():
        identifier(table)
        if table in ('records', 'features') or table.startswith('_s2_'):
            raise ValueError('Reserved runtime table name')
        keys(desc, ('key', 'time', 'columns', 'fkeys'), ('dtypes',))
        if 'dtypes' in desc and set(desc['dtypes']) != set(desc['columns']):
            raise ValueError('Catalog dtypes must match visible columns')
        if desc['key'] not in desc['columns'] or (desc['time'] is not None and desc['time'] not in desc['columns']):
            raise ValueError('Catalog key/time missing from columns')
        for column in desc['columns']:
            identifier(column)
        for col, dest in desc['fkeys'].items():
            if col not in desc['columns'] or dest not in context['tables']:
                raise ValueError('Invalid catalog FK')
    if task['root_table'] not in context['tables']:
        raise ValueError('Unknown root table')
    # Task labels are anchor metadata only, never exposed as source attributes.
    if task['target'] in context['tables'][task['root_table']]['columns']:
        raise ValueError('Task target must be excluded from root source columns')
    sampling = context['sampling']
    keys(sampling, ('default_budget', 'per_table', 'hop_decay', 'max_hops'))
    validate_max_hops(sampling['max_hops'])
    for budget in [sampling['default_budget'], *sampling['per_table'].values()]:
        if type(budget) is not int or budget <= 0:
            raise ValueError('Sampling budgets must be positive integers')
    if set(sampling['per_table'])-set(context['tables']):
        raise ValueError('Budget names must be catalog tables')
    if not isinstance(sampling['hop_decay'], (int, float)) or isinstance(sampling['hop_decay'], bool) or not 0 < sampling['hop_decay'] <= 1:
        raise ValueError('hop_decay must lie in (0,1]')
    validate_network(context['network'])
    return context


def validate_network(network):
    keys(network, ('intra', 'inter', 'final'), ('importance_lambda',))
    expected = {'intra':'shrinkage_mean', 'inter':'gated_sum', 'final':'root_residual'}
    if any(network[k] != value for k, value in expected.items()):
        raise ValueError('Expected shrinkage_mean, gated_sum and root_residual')
    if 'importance_lambda' in network:
        value = network['importance_lambda']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('importance_lambda must be finite and lie in [0,1]')
    return network


def make_plan(context, structure, enhancements, importance):
    validate_context(context)
    structure = validate_structure(context, structure)
    enhancements = validate_enhancements(context, structure, enhancements)
    importance = validate_importance(structure, importance)
    context = copy.deepcopy(context)
    context['network'].setdefault('importance_lambda', DEFAULT_IMPORTANCE_LAMBDA)
    return {'input_format': FORMAT, 'context': context,
            'structure': structure, 'enhancements': enhancements, 'importance': importance}


def validate_plan(plan):
    keys(plan, ('input_format', 'context', 'structure', 'enhancements', 'importance'))
    if plan['input_format'] != FORMAT:
        raise ValueError('Unsupported stage-two plan')
    return make_plan(plan['context'], plan['structure'], plan['enhancements'], plan['importance'])


def occurrences(plan):
    task = plan['context']['task']
    return [{'name': ROOT, 'table': task['root_table'], 'parent': None}] + plan['structure']['nodes']


def record_budget(plan, name):
    nodes = {n['name']: n for n in plan['structure']['nodes']}
    node, depth = nodes[name], 1
    parent = node['parent']
    while parent != ROOT:
        depth += 1
        parent = nodes[parent]['parent']
    policy = plan['context']['sampling']
    base = policy['per_table'].get(node['table'], policy['default_budget'])
    return max(1, math.floor(base*policy['hop_decay']**(depth-1)))
