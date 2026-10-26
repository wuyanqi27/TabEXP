"""Minimal TabEXP exploration, planning and fresh neural training."""
import copy
import json
import logging
from pathlib import Path
import time
import statistics
from tempfile import TemporaryDirectory

import numpy as np

from .stage1.core import write_json,parse_object,AugmentationLoop,SearchConfig
from .stage1.data import prepare_bundle
from .stage1.backend import RelBenchEvaluator,SQLFeatures,prepare_matrices
from .stage1.transport import JSONModel
from .neural.snapshot import MetadataLoader,quote
from .neural.stage2_data import Stage2Executor,database
from .neural.disk import DiskExecutor
from .neural.stage2_model import Stage2Model
from .neural.stage2_planning import generate_stage2_split
from .neural.stage2_contract import validate_max_hops
from .neural.stage2_data import official_test_metrics
from .experience import load_package,assemble_stage2

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def model_client(args,folder):
    return JSONModel(folder,model=args.model,base_url=args.base_url,key_env=args.key_env,
                     timeout=args.api_timeout,max_tokens=args.max_tokens,retries=args.api_retries)

def explore(args):
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    bundle=prepare_bundle(args.dataset,args.task,output/'input',profile_rows=args.profile_rows)
    llm=model_client(args,output/'llm_calls')
    with RelBenchEvaluator(bundle,output/'evaluation',threads=args.threads,train_rows=args.train_rows) as evaluator:
        loop=AugmentationLoop(bundle,evaluator,llm,output/'stage1',SearchConfig(rounds=args.rounds,patience=args.patience,max_repairs=args.repairs))
        loop.run()
    return output/'stage1'

def root_fields(queries,columns,categories,context):
    mapping={q['name']:q['sql'].strip().rstrip(';') for q in queries};task=context['task'];result=[]
    for name in columns:
        block,column=name.split('__',1)
        expr='(WITH eval_table AS (SELECT r.'+quote(task['entity'])+' AS '+quote(task['entity'])+',r._cutoff AS '+quote(task['clock'])+') SELECT max(q.'+quote(column)+') FROM ('+mapping[block]+') q)'
        result.append(dict(name=name,kind='categorical' if name in categories else 'numeric',sql=expr))
    return result

class RootProgramExecutor(DiskExecutor):
    """Execute root feature programs on complete cutoff cohorts before chunking."""
    def _enhance(self,conn,name,raw,*,fields=None):
        if name!='root' or fields is not None:return super()._enhance(conn,name,raw,fields=fields)
        frame=self.root_frame.iloc[raw['_seed'].to_numpy(np.int64)].reset_index(drop=True)
        original=[f for f in self.feature_fields[name] if f['name'] not in frame]
        if original:
            extra=Stage2Executor._enhance(self,conn,name,raw,fields=original)
            for column in extra:frame[column]=extra[column]
        return frame.reindex(columns=[f['name'] for f in self.feature_fields[name]])

def plan(args):
    package=load_package(args.experience);context=package['context']
    context['sampling']['max_hops']=validate_max_hops(args.max_hops)
    context['input_columns']='original_plus_enhancements'
    context['network']=dict(intra='shrinkage_mean',inter='gated_sum',final='root_residual',importance_lambda=args.importance_lambda)
    loader=MetadataLoader(context['task']['dataset'],context['task']['task'])
    rows=loader.get_table('train').df.head(8).reset_index(drop=True)
    output=Path(args.output)
    queries=package['best_program']['feature_queries']
    with database(loader,context) as conn:
        roots=SQLFeatures(conn,context).collect(queries,rows,'train_probe')
    # Only determine native dtype here; fit all preprocessing later on train.
    # Preserve categories learned by the full stage-one training evaluator;
    # eight probe rows can all be missing for an otherwise categorical field.
    categories=package['best_program'].get('model_config',{}).get('categorical_features')
    if categories is None:
        _,_,categories=prepare_matrices(roots,roots)
    fields=root_fields(queries,list(roots),categories,context)
    llm=model_client(args,output.parent/(output.name+'_llm_calls'))
    def assemble(content,ctx,directory):
        value=parse_object(content);value['enhancements']['root']=copy.deepcopy(fields)
        for node in value['structure']['nodes']:
            if value['enhancements'][node['name']]:
                raise ValueError('Auxiliary enhancements must be []; original fields are automatic')
        return assemble_stage2(json.dumps(value),ctx,directory)
    def probe(directory):
        p=read(directory/'plan.json');ex=RootProgramExecutor(p);ex.root_frame=roots
        # In-memory extraction for eight anchors with full-cohort root features.
        with database(loader,context) as conn:
            store=Stage2Executor.prepare(ex,conn,rows,fit=True)
        write_json(directory/'probe_coverage.json',ex.coverage);store.close()
    result=generate_stage2_split(package,lambda phase,prompt,payload,directory:llm(phase,prompt,payload),output,attempts=args.repairs+1,assemble=assemble,probe=probe)
    write_json(output/'root_program.json',package['best_program'])
    return result

def train(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    # Temporary packed arrays bound memory within one run, not a reusable cache.
    with TemporaryDirectory(prefix='tabexp_inputs_') as folder:
        return _train(args,output,Path(folder))

def _train(args,output,workspace):
    import torch
    started=time.perf_counter()
    p=read(args.plan);task=p['context']['task'];loader=MetadataLoader(task['dataset'],task['task'])
    full=loader.get_table('train').df.reset_index(drop=True)
    if args.train_indices:ids=np.load(args.train_indices)
    elif len(full)>args.train_rows:
        from sklearn.model_selection import train_test_split
        _,ids=train_test_split(np.arange(len(full)),test_size=args.train_rows,random_state=42,stratify=full[task['target']] if task['kind']=='classification' else None);ids=np.sort(ids)
    else:ids=np.arange(len(full))
    if len(ids)==0 or len(np.unique(ids))!=len(ids) or ids.min()<0 or ids.max()>=len(full):raise ValueError('Invalid training indices')
    rows={'train':full.iloc[ids].reset_index(drop=True),'val':loader.get_table('val').df.reset_index(drop=True)}
    np.save(output/'train_indices.npy',ids);del full
    # Test labels remain masked until final scoring.
    if not args.no_test:rows['test']=loader.get_table('test').df.reset_index(drop=True)
    program=read(args.program) if args.program else None
    training_protocol=dict(epochs=args.epochs,patience=args.patience,batch_size=args.batch_size,learning_rate=args.learning_rate,weight_decay=args.weight_decay,train_rows=len(ids),test_evaluation=not args.no_test,api_calls=0)
    stores={};model=Stage2Model(p);model.inputs=RootProgramExecutor(p) if program else DiskExecutor(p)
    for split,frame in rows.items():
        folder=workspace/split
        with database(loader,p['context'],memory_limit=args.memory_limit) as conn:
            conn.execute('SET threads='+str(args.threads))
            if program:model.inputs.root_frame=SQLFeatures(conn,p['context'],timeout=args.sql_timeout).collect(program['feature_queries'],frame,split)
            model.inputs.configure(folder,chunk_rows=args.chunk_rows)
            stores[split]=model.prepare(conn,frame,fit=split=='train')
        if split=='train':
            write_json(output/'preprocessing.json',model.inputs.state_dict());write_json(output/'model_spec.json',model.inputs.model_spec)
    prep=model.inputs.state_dict();spec=model.inputs.model_spec;runs=[];torch.set_num_threads(args.threads)
    write_json(output/'plan.json',p)
    write_json(output/'protocol.json',dict(seeds=args.seeds,**training_protocol))
    for seed in args.seeds:
        folder=output/('seed'+str(seed));folder.mkdir(exist_ok=False)
        candidate=Stage2Model(p,prep);candidate.inputs.model_spec=spec
        fit=candidate.fit(stores['train'],rows['train'][task['target']].to_numpy(np.float32),stores['val'],rows['val'][task['target']].to_numpy(np.float32),seed=seed,device=args.device,epochs=args.epochs,patience=args.patience,batch_size=args.batch_size,learning_rate=args.learning_rate,weight_decay=args.weight_decay,checkpoint=folder/'checkpoint.pt',logger=logging.getLogger('seed'+str(seed)))
        candidate.save(folder/'model.pt');metrics=None
        if not args.no_test:
            pred=candidate.predict(stores['test'],device=args.device);np.save(folder/'test_predictions.npy',pred)
            metrics=official_test_metrics(loader,pred,rows['test'],task['entity'],task['clock'])
        result=dict(seed=seed,training=fit,test_metrics=metrics);write_json(folder/'results.json',result);runs.append(result)
        del candidate;torch.cuda.empty_cache()
    metric='roc_auc' if task['kind']=='classification' else 'mae'
    values=[r['test_metrics'][metric] for r in runs if r['test_metrics'] is not None]
    result=dict(status='complete',metric=metric,runs=runs,mean=statistics.mean(values) if values else None,sample_std=statistics.stdev(values) if len(values)>1 else None,elapsed_seconds=time.perf_counter()-started)
    write_json(output/'results.json',result)
    for store in stores.values():store.close()
    return output/'results.json'
