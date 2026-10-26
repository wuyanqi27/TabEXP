"""Small, validation-only exploration ledger and stage-two design boundary.

This module has no model/network side effects. Experiences are claims with
references, not automatic estimates of relational/neural importance.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


FORMAT = 'stage1_experience_v1'
PROGRAM_KEYS = ('feature_queries', 'model_choice', 'model_config')
EXPERIENCE_KEYS = {'insight', 'status', 'trial_ids', 'query_refs', 'stage2_use'}


def write_json(path, value):
    """Atomic snapshot; the JSONL ledger remains the append-only source."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def parse_object(content):
    text = content.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    return value


def validation_ledger(trials):
    result = []
    for trial in trials:
        if trial.get('split') != 'val':
            continue
        score = trial.get('primary_score')
        result.append({
            'trial_id': trial['trial_id'],
            'primary_score': score if isinstance(score, (float, int)) and math.isfinite(score) else None,
            'error': trial.get('error'),
            'query_names': [q['name'] for q in trial.get('feature_queries', [])],
            'timings_seconds': trial.get('timings_seconds', {}),
        })
    return result


def validate_experiences(value, ledger):
    if set(value) != {'experiences'} or not isinstance(value['experiences'], list):
        raise ValueError('Expected exactly {"experiences": [...]}')
    if len(value['experiences']) > 3:
        raise ValueError('At most 3 new experiences per round')
    known = {t['trial_id']: set(t['query_names']) for t in ledger}
    for item in value['experiences']:
        if not isinstance(item, dict) or set(item) != EXPERIENCE_KEYS:
            raise ValueError('Experience keys must be '+str(sorted(EXPERIENCE_KEYS)))
        if not isinstance(item['insight'], str) or not item['insight'].strip():
            raise ValueError('A nonempty insight is required')
        if item['status'] not in ('observation', 'hypothesis'):
            raise ValueError('status must be observation or hypothesis')
        refs = item['trial_ids']
        if not isinstance(refs, list) or not refs or any(type(i) is not int or i not in known for i in refs):
            raise ValueError('trial_ids must cite existing validation trials')
        queries = item['query_refs']
        allowed = set().union(*(known[i] for i in refs))
        if not isinstance(queries, list) or any(not isinstance(q, str) or q not in allowed for q in queries):
            raise ValueError('query_refs must belong to the cited validation trials')
        uses = item['stage2_use']
        if not isinstance(uses, list) or not uses or any(u not in ('structure', 'enhancements', 'importance') for u in uses):
            raise ValueError('stage2_use must reference the three stage-two interfaces')
    return value['experiences']


def make_package(context, trials, best_program, experiences, fixed_config, task_description='', exploration_memory=None):
    program = {key: best_program[key] for key in PROGRAM_KEYS if key in best_program}
    return {'format': FORMAT, 'context': context, 'task_description': task_description,
            'fixed_predictor': {'model_choice': best_program.get('model_choice', 'catboost'), 'model_config': fixed_config},
            'best_program': program, 'validation_trials': validation_ledger(trials),
            'experiences': experiences, 'exploration_memory': exploration_memory or [],
            'test_information_provided': False,
            'neural_feedback_provided': False}


def load_package(path):
    """Accept our own exported interface, never arbitrary result/trace files."""
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    required = {'format', 'context', 'task_description', 'fixed_predictor', 'best_program', 'validation_trials',
                'experiences', 'exploration_memory', 'test_information_provided', 'neural_feedback_provided'}
    if set(value) != required or value['format'] != FORMAT:
        raise ValueError('Expected a stage1_experience_v1 experience_package.json')
    if value['test_information_provided'] is not False or value['neural_feedback_provided'] is not False:
        raise ValueError('Design must not receive test/neural feedback')
    if set(value['best_program']) - set(PROGRAM_KEYS):
        raise ValueError('Unexpected fields in saved best program')
    ledger = value['validation_trials']
    for trial in ledger:
        if set(trial) != {'trial_id', 'primary_score', 'error', 'query_names', 'timings_seconds'}:
            raise ValueError('Unexpected fields in validation summary')
    for item in value['experiences']:
        if set(item) != EXPERIENCE_KEYS | {'id', 'turn', 'fingerprint', 'timestamp'}:
            raise ValueError('Unexpected fields in experience')
        validate_experiences({'experiences': [{k: item[k] for k in EXPERIENCE_KEYS}]}, ledger)
    return value


def assemble_stage2(content, context, directory):
    from tabexp.neural.stage2_contract import make_plan
    modules = parse_object(content)
    required = {'structure', 'enhancements', 'importance'}
    if not required <= set(modules) or set(modules)-required-{'design_notes'}:
        raise ValueError('Return only structure/enhancements/importance, optionally design_notes')
    plan = make_plan(context, modules['structure'], modules['enhancements'], modules['importance'])
    for name in required:
        write_json(Path(directory)/(name+'.json'), modules[name])
    write_json(Path(directory)/'design_notes.json', modules.get('design_notes'))
    write_json(Path(directory)/'plan.json', plan)
    return plan
