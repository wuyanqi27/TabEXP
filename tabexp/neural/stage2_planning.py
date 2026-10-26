"""Two explicit LLM decisions: freeze PK--FK inputs, then assign importance."""
import copy
import json
from pathlib import Path
from tabexp.stage1.core import parse_object,write_json,error_feedback

FILTER_PROMPT='''Select relational inputs for a fixed predictor. Return ONLY JSON
with structure, enhancements, and optionally design_notes. Do NOT generate
importance scores in this step. The task, catalog and exploration experience
are supplied as data; no test or neural feedback is available.
structure={"nodes":[{"name":"occurrence","table":"real_table","parent":"root",
"fk_table":"real_fk_owner","fk_column":"real_fk_column"}]}.
Use actual PK--FK edges; fk_table owns the FK. Parents precede children.
Self-table edges require direction incoming/outgoing. Return a finite hierarchy.
Obey context.sampling.max_hops in the supplied experience package: root has
depth zero and each PK--FK edge adds one hop. This limit applies to relational
aggregation, not joins inside task-table SQL features. Zero allows root only.
Select the smallest sufficient set of paths that adds information beyond the
augmented task table. Semantic relevance or reachability alone is insufficient.
Exclude redundant paths and paths known to lack pre-cutoff records. Do not
invent coverage or isolated gains. Mark unverified usefulness as a hypothesis.
A connector node may be retained to reach an informative descendant. Explain
retained incremental signals and the main excluded branches in design_notes.
enhancements maps root and each occurrence to lists of {name,kind,sql}, with
kind numeric/categorical. root may be []; the system preserves all frozen
stage-one root features. All original visible attributes, including real PK/FK
IDs, are automatically encoded for each selected table. Set auxiliary lists to
[]: do not select fields or invent auxiliary derived features. Internal physical
row numbers, labels and benchmark-prohibited columns are never input features.
Never change the network, sampling budgets, training settings or root program.
On error make only the minimal interface/execution repair, not a new design.
'''

IMPORTANCE_PROMPT='''Assign importance scores to records in a FROZEN relational
input plan. Return ONLY JSON {"importance":{"occurrence":{"sql":"..."}}},
optionally with design_notes. Cover exactly the selected auxiliary occurrences,
not root. Do not change paths, fields, the root program, budgets or the network.
Each score is a deterministic scalar DuckDB expression using raw r and selected
enhanced f fields. r._cutoff is prediction time; r._time is record time, possibly
NULL. Visible catalog tables are already censored at prediction time. Scores
must be finite nonnegative float32 values. Use 1.0 if uniform scoring is best
justified. Ground scoring in task semantics and supplied exploration evidence;
tree feature importance does not establish individual record usefulness.
No sample rows, test results or neural feedback are supplied. On execution error
repair only the scoring expression, keeping all frozen inputs unchanged.
'''

def generate_stage2_split(package,respond,output,*,attempts=6,assemble=None,probe=None,abort_errors=()):
    """respond(phase, prompt, payload, attempt_dir); probe is execution-only.

    Uniform scores are solely an internal filtering probe, never an LLM score
    decision. Importance repair cannot reopen the committed filtering decision.
    """
    if attempts<1:raise ValueError('Positive attempt budget required')
    if assemble is None:
        from tabexp.experience import assemble_stage2
        assemble=assemble_stage2
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'design_input.json',package)
    frozen=None;coverage=None
    for phase,prompt in [('pk_fk_filter',FILTER_PROMPT),('importance',IMPORTANCE_PROMPT)]:
        previous=None;error=None
        for number in range(1,attempts+1):
            directory=output/phase/f'attempt_{number}';directory.mkdir(parents=True)
            payload=dict(experience_package=copy.deepcopy(package))
            if frozen is not None:
                payload.update(frozen_structure=copy.deepcopy(frozen['structure']),frozen_enhancements=copy.deepcopy(frozen['enhancements']),training_probe_coverage=coverage)
            if error:payload.update(failed_candidate=previous,error=error)
            write_json(output/'status.json',dict(status='running',phase=phase,attempt=number))
            write_json(directory/'request.json',payload)
            try:
                previous=respond(phase,prompt,payload,directory)
                value=parse_object(previous);write_json(directory/'response.json',value)
                if phase=='pk_fk_filter':
                    if not {'structure','enhancements'}<=set(value) or set(value)-{'structure','enhancements','design_notes'}:
                        raise ValueError('Filtering returns only structure/enhancements and optional design_notes, never importance')
                    modules=copy.deepcopy(value)
                    modules['importance']={n['name']:dict(sql='1.0') for n in modules['structure']['nodes']}
                else:
                    if 'importance' not in value or set(value)-{'importance','design_notes'}:
                        raise ValueError('Importance step may return only importance and optional design_notes; filtering is frozen')
                    modules=dict(structure=copy.deepcopy(frozen['structure']),enhancements=copy.deepcopy(frozen['enhancements']),importance=value['importance'],design_notes=value.get('design_notes'))
                plan=assemble(json.dumps(modules),copy.deepcopy(package['context']),directory)
                if frozen is not None and (plan['structure']!=frozen['structure'] or plan['enhancements']!=frozen['enhancements']):
                    raise ValueError('Importance must preserve frozen structure and enhancements exactly')
                if probe is not None:probe(directory)
                write_json(output/(phase+'.json'),plan)
                break
            except Exception as exc:
                error=error_feedback(exc);write_json(directory/'error.json',dict(error=error))
                if isinstance(exc,abort_errors):
                    write_json(output/'status.json',dict(status='failed',phase=phase,error=error));raise
        else:
            write_json(output/'status.json',dict(status='failed',phase=phase,error=error))
            raise RuntimeError('Split planning exhausted '+phase+' repairs: '+str(error))
        if phase=='pk_fk_filter':
            frozen=copy.deepcopy(plan)
            path=directory/'probe_coverage.json'
            coverage=json.loads(path.read_text()) if path.exists() else None
    write_json(output/'plan.json',plan)
    write_json(output/'status.json',dict(status='complete',phases=['pk_fk_filter','importance'],filtering_frozen_before_importance=True,probe_performed=probe is not None))
    return output/'plan.json'
