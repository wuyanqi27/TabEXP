"""Explicit generate -> execute/evaluate -> memory loop, with append-only blocks."""
from dataclasses import dataclass, field, asdict
import copy
import json
import math
from pathlib import Path
import re
import time

from . import prompts

FIXED_TREE = dict(n_estimators=200, learning_rate=.05, max_depth=6, l2_leaf_reg=3.)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def parse_object(value):
    if isinstance(value, str):
        text = value.strip()
        if text.startswith('```'):text = text.split('\n', 1)[1].rsplit('```', 1)[0]
        value = json.loads(text)
    if not isinstance(value, dict):raise ValueError('Return one JSON object')
    return value


def error_feedback(error):
    message = str(error)
    # Keep schema/SQL details; conversion errors may quote an actual data cell.
    if 'Conversion Error' in message or 'could not convert' in message.lower():
        message = re.sub(r"(['\"]).*?\1", '<redacted value>', message, flags=re.S)
    return message[:6000]


@dataclass(frozen=True)
class SearchConfig:
    rounds: int = 5
    patience: int = 3
    max_repairs: int = 5
    min_improvement: float = 0.

    def __post_init__(self):
        if self.rounds < 1 or self.patience < 0 or self.max_repairs < 0 or self.min_improvement < 0:
            raise ValueError('Positive round budget and nonnegative stopping/repair parameters required')


@dataclass
class Evaluation:
    score: float
    metrics: dict
    feature_statistics: dict
    categorical_features: list = field(default_factory=list)
    feature_importance: list = field(default_factory=list)
    confusion_matrix: object = None

    def feedback(self):
        if not math.isfinite(self.score):raise ValueError('Evaluation did not produce a finite validation score')
        return dict(validation_score=float(self.score), metrics=self.metrics,
                    train_feature_statistics=self.feature_statistics,
                    feature_importance=self.feature_importance, confusion_matrix=self.confusion_matrix)


class AugmentationLoop:
    """llm(phase, system_prompt, payload) and evaluator.evaluate(queries) are injected.

    No agent, tool registration, arbitrary execution by the LLM, or test path.
    Every valid round appends its block, even if its score does not improve.
    Best selection and early stopping are independent of cumulative input state.
    """
    def __init__(self, bundle, evaluator, llm, output, config=None):
        self.bundle = copy.deepcopy(bundle)
        self.evaluator, self.llm = evaluator, llm
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=False)
        self.config = config or SearchConfig()
        self.queries = copy.deepcopy(bundle['base_queries'])
        self.memory, self.trials, self.best_program = [], [], None
        self.best_score, self.valid_rounds, self.attempts = -math.inf, 0, 0
        self.no_improvement = 0
        write_json(self.output/'input_bundle.json', self.bundle)
        write_json(self.output/'protocol.json', dict(format='paper_stage1_v1', **asdict(self.config),
            predictor=dict(model_choice='catboost', model_config=FIXED_TREE),
            cumulative_features=True, failed_attempts_count=False,
            test_information_provided=False, neural_feedback_provided=False))

    def _status(self, state, **extra):
        write_json(self.output/'status.json', dict(status=state, scored_rounds=self.valid_rounds,
            planned_rounds=self.config.rounds, attempts=self.attempts,
            best_validation_score=None if not math.isfinite(self.best_score) else self.best_score, **extra))

    def _proposal(self, response):
        value = parse_object(response)
        if set(value)-{'hypothesis','feature_queries','reason'}:raise ValueError('Return only hypothesis, feature_queries and optional reason; evaluator settings are fixed')
        if not isinstance(value.get('hypothesis'), str) or not value['hypothesis'].strip():raise ValueError('State one information hypothesis')
        queries = value.get('feature_queries')
        if not isinstance(queries, list) or not queries:raise ValueError('Submit at least one NEW feature query')
        names = {q['name'] for q in self.queries}
        for query in queries:
            if not isinstance(query,dict) or set(query)-{'name','sql','reason'} or not {'name','sql'}<=set(query):raise ValueError('Each query needs name/sql, optionally reason')
            name = query['name']
            if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z][A-Za-z_0-9]*',name):raise ValueError('Use plain query names starting with a letter')
            if name in names:raise ValueError('Cannot replace or duplicate an existing block: '+name)
            if not isinstance(query['sql'],str) or not query['sql'].strip():raise ValueError('SQL must be nonempty')
            names.add(name)
        return value

    def _summarize(self, proposal, feedback, repairs):
        payload=dict(task=self.bundle['task_definition'], hypothesis=proposal['hypothesis'],
            added_queries=proposal['feature_queries'], evaluation=feedback,
            repaired_errors=repairs,previous_summaries=[m['summary'] for m in self.memory])
        try:
            value=parse_object(self.llm('memory',prompts.MEMORY,payload))
            if set(value)!={'findings','issues','next_direction'}:raise ValueError('Memory needs findings/issues/next_direction')
            if not isinstance(value['next_direction'],str) or any(not isinstance(value[k],list) or any(not isinstance(x,str) for x in value[k]) for k in ['findings','issues']):raise ValueError('Invalid memory value types')
            return value
        except Exception as exc:
            write_json(self.output/'rounds'/f'{self.valid_rounds:03d}'/'memory_error.json',dict(error=error_feedback(exc)))
            return dict(findings=['A valid validation result was obtained; see its recorded feedback.'],
                issues=['LLM summary unavailable; evaluation was retained.'],next_direction='Use the recorded scores and training diagnostics.')

    def run(self):
        started=time.time();self._status('running')
        try:
            while self.valid_rounds < self.config.rounds:
                base_payload=dict(schema=dict(task=self.bundle['context']['task'],tables=self.bundle['context']['tables']),task_definition=self.bundle['task_definition'],
                    train_field_profile=self.bundle['field_profile'],current_feature_queries=self.queries,
                    exploration_memory=[dict(round=m['round'],validation_score=m['feedback']['validation_score'],
                        metrics=m['feedback']['metrics'],summary=m['summary']) for m in self.memory],
                    latest_train_feature_diagnostics=self.memory[-1]['feedback']['train_feature_statistics'] if self.memory else None,
                    remaining_scored_rounds=self.config.rounds-self.valid_rounds)
                previous=None;error=None;repairs=[]
                for repair in range(self.config.max_repairs+1):
                    self.attempts+=1;directory=self.output/'attempts'/f'{self.attempts:04d}'
                    payload=copy.deepcopy(base_payload)
                    if repair:payload.update(failed_candidate=previous,error=error)
                    phase='repair' if repair else 'generate'
                    self._status('running',phase=phase)
                    write_json(directory/'request.json',dict(phase=phase,payload=payload))
                    try:
                        response=self.llm(phase,prompts.REPAIR if repair else prompts.GENERATE,payload)
                        write_json(directory/'response.json',dict(content=response))
                        previous=response
                        proposal=self._proposal(response)
                        additions=[dict(name=q['name'],sql=q['sql']) for q in proposal['feature_queries']]
                        candidate=self.queries+additions
                        evaluation=self.evaluator.evaluate(candidate)
                        feedback=evaluation.feedback()
                        write_json(directory/'evaluation.json',feedback)
                        break
                    except (ValueError,TypeError,KeyError,ArithmeticError) as exc:
                        error=error_feedback(exc)
                        repairs.append(error)
                        write_json(directory/'error.json',dict(error=error,counts_as_round=False))
                else:raise RuntimeError('Repair budget exhausted before a scored round; run is incomplete')
                self.valid_rounds+=1;self.queries=candidate
                program=dict(feature_queries=copy.deepcopy(candidate),model_choice='catboost',
                    model_config=dict(FIXED_TREE,categorical_features=evaluation.categorical_features))
                improved=evaluation.score>self.best_score
                significant=evaluation.score>self.best_score+self.config.min_improvement
                if improved:
                    self.best_score=evaluation.score;self.best_program=program
                    write_json(self.output/'best_program.json',program)
                self.no_improvement=0 if significant else self.no_improvement+1
                trial=dict(trial_id=self.valid_rounds,split='val',primary_score=float(evaluation.score),
                    metrics=evaluation.metrics,error=None,is_best=improved,attempt_id=self.attempts,
                    hypothesis=proposal['hypothesis'],**program)
                self.trials.append(trial)
                with (self.output/'trials.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(trial,allow_nan=False)+'\n')
                write_json(self.output/'last_program.json',program)
                self._status('running',phase='memory')
                summary=self._summarize(proposal,feedback,repairs)
                self.memory.append(dict(round=self.valid_rounds,feedback=feedback,repaired_errors=repairs,summary=summary))
                write_json(self.output/'exploration_memory.json',self.memory)
                self._export()
                if self.config.patience and self.no_improvement>=self.config.patience:
                    self._status('complete',stop_reason='validation_patience',seconds=time.time()-started);return self.best_program
            self._status('complete',stop_reason='scored_round_budget',seconds=time.time()-started)
            return self.best_program
        except Exception as exc:
            self._status('incomplete',error=error_feedback(exc),seconds=time.time()-started)
            raise

    def _export(self):
        from tabexp.experience import make_package
        experiences=[]
        for memory in self.memory:
            trial=self.trials[memory['round']-1]
            for number,text in enumerate(memory['summary']['findings']):
                experiences.append(dict(insight=text,status='hypothesis',trial_ids=[trial['trial_id']],
                    query_refs=[],stage2_use=['structure','enhancements','importance'],
                    id=f"r{memory['round']}_{number}",turn=memory['round']-1,fingerprint=f"r{memory['round']}_{number}",timestamp=''))
        package=make_package(self.bundle['context'],self.trials,self.best_program,experiences,FIXED_TREE,
            self.bundle['task_definition'].get('description',''), exploration_memory=copy.deepcopy(self.memory))
        write_json(self.output/'experience_package.json',package)
        write_json(self.output/'stage2_handoff.json',dict(format='paper_stage1_handoff_v1',
            root_program='best_program.json',root_policy='execute all queries verbatim; do not summarize or regenerate root SQL',
            experience_package='experience_package.json',
            llm_remaining_decisions=['PK-FK paths','neighbor importance rules'],
            tree_predictions_used=False,test_information_provided=False))
