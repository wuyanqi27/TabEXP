"""Train-only metadata, fixed original features, and aggregate diagnostics."""
from pathlib import Path
import inspect
import json


def quote(name):return '"'+name.replace('"','""')+'"'


def visible(conn,context,cutoff):
    for table,desc in context['tables'].items():
        sql='SELECT * FROM '+quote('_s2_source_'+table)
        if desc['time']:
            sql+=' WHERE '+quote(desc['time'])+(' <= ' if table==context['task']['root_table'] else ' < ')+"TIMESTAMP '"+str(cutoff).replace("'","''")+"'"
        conn.execute('CREATE OR REPLACE TEMP VIEW '+quote(table)+' AS '+sql)


def field_profile(conn,context,cutoff,max_rows=100000):
    if max_rows<1:raise ValueError('Positive profiling row limit required')
    visible(conn,context,cutoff);tables={}
    for name,desc in context['tables'].items():
        total=conn.execute('SELECT count(*) FROM '+quote(name)).fetchone()[0]
        conn.execute('CREATE OR REPLACE TEMP VIEW _profile AS SELECT * FROM '+quote(name)+' LIMIT '+str(int(max_rows)))
        profiled=min(total,max_rows);fields={}
        for col in desc['columns']:
            n,distinct=conn.execute('SELECT count('+quote(col)+'),count(DISTINCT '+quote(col)+') FROM _profile').fetchone()
            dtype=desc['dtypes'][col]
            item=dict(dtype=dtype,nonmissing_count=n,n_unique=distinct,missing_fraction=1-n/profiled if profiled else None)
            if col==desc['key'] or col in desc['fkeys']:
                item['role']='identifier; values omitted'
            elif dtype in ['VARCHAR','BOOLEAN'] and distinct<=32:
                free_text=any(word in col.lower() for word in ['comment','title','summar','description','criteria','population','reason','explanation','detail'])
                longest=conn.execute('SELECT max(length(CAST('+quote(col)+' AS VARCHAR))) FROM _profile').fetchone()[0] or 0
                if not free_text and longest<=80:
                    values=conn.execute('SELECT '+quote(col)+',count(*) FROM _profile GROUP BY 1 ORDER BY 2 DESC,CAST('+quote(col)+' AS VARCHAR)').fetchall()
                    item['categories']=[dict(value=v,count=c) for v,c in values]
                else:item['values_omitted']='free text or long strings'
            elif any(x in dtype for x in ['INT','FLOAT','DOUBLE','DECIMAL']):
                values=conn.execute('SELECT min('+quote(col)+'),quantile_cont('+quote(col)+',0.5),quantile_cont('+quote(col)+',0.95),max('+quote(col)+') FROM _profile').fetchone()
                import math
                item['numeric_summary']={k:(float(v) if v is not None and math.isfinite(float(v)) else None) for k,v in zip(['min','median','p95','max'],values)}
            fields[col]=item
        tables[name]=dict(total_visible_rows=total,profiled_rows=profiled,fields=fields)
    return dict(format='paper_train_profile_v1',cutoff=str(cutoff),max_rows_per_table=max_rows,
        policy='deterministic prefix of training-cutoff-visible rows; static availability follows benchmark',
        raw_sample_rows_included=False,validation_or_test_rows_used=False,table_profiles=tables)


def target_definition(loader):
    task=loader.task
    description=str(getattr(task,'description',''))
    manifest=Path(loader.dataset.db_dir).parent/'tasks'/loader.task_name/'manifest.yaml'
    if manifest.exists():
        import yaml
        description=yaml.safe_load(manifest.read_text(encoding='utf-8')).get('description',description)
    definition=dict(dataset=loader.dataset_name,task=loader.task_name,
        description=description,entity_table=task.entity_table,
        entity_col=task.entity_col,target_col=task.target_col,time_col=task.time_col,
        horizon=str(task.timedelta))
    sql=getattr(task,'_sql',None)
    if sql:definition['label_generation_sql']=sql
    elif hasattr(task,'label_definition'):definition['label_definition']=task.label_definition
    else:definition['label_generation_code']=inspect.getsource(type(task).make_table)
    definition['usage']='Defines the label only. Future label-window observations must never become input features.'
    return definition


def base_queries(context,profile,fields=None):
    task=context['task'];table=task['root_table'];desc=context['tables'][table]
    chosen=[]
    for col in desc['columns']:
        if col in {desc['key'],*desc['fkeys']} or col.startswith('_'):continue
        dtype=desc['dtypes'][col].upper()
        if fields is not None:
            if col not in fields:continue
        elif not (any(x in dtype for x in ['INT','FLOAT','DOUBLE','DECIMAL','BOOL','DATE','TIMESTAMP']) or 'categories' in profile['table_profiles'][table]['fields'][col]):continue
        if any(x in dtype for x in ['STRUCT','[]','MAP','BLOB']):continue
        expression='r.'+quote(col)
        if 'DATE' in dtype or 'TIMESTAMP' in dtype:expression='epoch('+expression+')'
        chosen.append(expression+' AS '+quote(col))
    if fields is not None and set(fields)-set(desc['columns']):raise ValueError('Unknown original field')
    entity,clock=task['entity'],task['clock']
    keys='t.'+quote(entity)+',t.'+quote(clock)
    if not chosen:return [dict(name='task_original',sql='SELECT '+keys+',1.0 AS intercept FROM eval_table t')]
    sql='SELECT '+keys+','+','.join(chosen)+' FROM eval_table t LEFT JOIN '+quote(table)+' r ON r.'+quote(desc['key'])+'=t.'+quote(entity)
    return [dict(name='task_original',sql=sql)]


def feature_statistics(frame):
    import numpy as np
    import pandas as pd
    columns={}
    for name in frame:
        values=frame[name];n=int(values.notna().sum());distinct=int(values.nunique(dropna=True))
        item=dict(nonmissing_count=n,missing_fraction=1-n/len(values) if len(values) else None,
            n_unique_nonmissing=distinct,all_missing=n==0,constant_nonmissing=distinct<=1)
        if pd.api.types.is_numeric_dtype(values):
            finite=values.to_numpy(dtype=float,na_value=np.nan);finite=finite[np.isfinite(finite)]
            if len(finite):item['numeric_summary']=dict(zip(['min','median','p95','max'],map(float,np.quantile(finite,[0,.5,.95,1]))))
        columns[name]=item
    return dict(fitted_split='train',rows=len(frame),fields=len(frame.columns),columns=columns,
        raw_sample_rows_included=False)


def prepare_bundle(dataset,task,output,profile_rows=100000,original_fields=None):
    from tabexp.neural.snapshot import MetadataLoader
    from tabexp.neural.stage2_data import context_from_loader,database
    from .core import write_json
    output=Path(output)
    if output.exists():raise FileExistsError('Prepare output already exists: '+str(output))
    loader=MetadataLoader(dataset,task);context=context_from_loader(loader)
    train=loader.get_table('train').df
    cutoff=train[loader.task.time_col].max()
    with database(loader,context,memory_limit='3GB') as conn:profile=field_profile(conn,context,cutoff,profile_rows)
    bundle=dict(format='paper_stage1_input_v1',context=context,task_definition=target_definition(loader),
        field_profile=profile,base_queries=base_queries(context,profile,original_fields))
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'input_bundle.json',bundle)
    return bundle
