"""Fixed CatBoost and cutoff-safe feature SQL. No agents or model-selected tools."""
from contextlib import ExitStack
import json
from pathlib import Path
import re

from .core import Evaluation, FIXED_TREE
from .data import visible,feature_statistics


def merge_blocks(blocks,keys):
    """Join feature blocks without labels, duplicate keys or silent row expansion."""
    result=keys.copy().reset_index(drop=True)
    if result.duplicated().any():raise ValueError('Task entity--timestamp keys must be unique')
    joins=list(keys.columns)
    for name,frame in blocks.items():
        if not set(joins)<=set(frame):raise ValueError('Query '+name+' must return both entity and timestamp keys')
        if frame.duplicated(joins).any():raise ValueError('Query '+name+' returned duplicate entity--timestamp keys')
        fields=[col for col in frame if col not in joins]
        if not fields:raise ValueError('Query '+name+' returned no feature columns')
        if any(not re.fullmatch(r'[A-Za-z][A-Za-z_0-9]*',str(col)) for col in fields):raise ValueError('Use plain feature aliases')
        renamed=frame.rename(columns={col:name+'__'+col for col in fields})
        result=result.merge(renamed,on=joins,how='left',validate='one_to_one',sort=False)
    return result.drop(columns=joins)


class SQLFeatures:
    """Execute exact feature blocks with cutoff-safe, key-aligned results."""
    def __init__(self,conn,context,timeout=300):
        self.conn,self.context=conn,context
        self.timeout=timeout

    def collect(self,queries,rows,split):
        import pandas as pd
        from tabexp.sql import run_feature_queries as _run_feature_queries
        entity,clock=self.context['task']['entity'],self.context['task']['clock']
        keys=rows[[entity,clock]].reset_index(drop=True);blocks={}
        for query in queries:
            parts=[]
            for cutoff,group in keys.groupby(clock,sort=True):
                visible(self.conn,self.context,cutoff)
                self.conn.register('eval_table',group.reset_index(drop=True))
                try:
                    validate_query(self.conn,query['sql'],self.context)
                    result=_run_feature_queries(self.conn,[query],entity_col=entity,allowed_entities=group[entity],
                        n_entities=len(group),sql_timeout_seconds=self.timeout)
                except Exception as exc:
                    from .core import error_feedback
                    raise ValueError(error_feedback(exc)) from None
                parts.append(result[query['name']])
            frame=pd.concat(parts,ignore_index=True)
            # Validate cardinality before concatenating an executed block.
            merge_blocks({query['name']:frame},keys)
            blocks[query['name']]=frame
        return merge_blocks(blocks,keys)


def validate_query(conn,sql,context):
    import duckdb
    statements=conn.extract_statements(sql)
    if len(statements)!=1 or statements[0].type!=duckdb.StatementType.SELECT:
        raise ValueError('Only one SELECT statement per feature block is allowed')
    if re.search(r'\b(read_\w+|\w*scan|query|query_table|glob|random|uuid|now|nextval|setval)\s*\(|\bcurrent_timestamp\b|\bcurrent_date\b',sql,re.I):
        raise ValueError('External reads and volatile SQL functions are not allowed')
    unknown=conn.get_table_names(sql)-set(context['tables'])-{'eval_table'}
    if unknown:raise ValueError('SQL may reference only the supplied catalog and eval_table: '+', '.join(sorted(unknown)))


def prepare_matrices(train,val):
    """Native categories; numeric medians fitted on TRAIN only and reused for val."""
    import numpy as np
    import pandas as pd
    if set(val.columns)!=set(train.columns):raise ValueError('Feature column schema changed between train and validation; return fixed aliases')
    val=val.reindex(columns=train.columns)
    train=train.copy();val=val.copy();categories=[]
    for name in train:
        if pd.api.types.is_timedelta64_dtype(train[name]):raise ValueError('Return numeric days/hours for temporal features, not INTERVAL objects')
        if pd.api.types.is_numeric_dtype(train[name]) or pd.api.types.is_datetime64_any_dtype(train[name]) or not train[name].notna().any():
            if pd.api.types.is_datetime64_any_dtype(train[name]):
                for frame in [train,val]:
                    missing=frame[name].isna();frame[name]=pd.to_datetime(frame[name]).astype('int64').astype(float)/1e9;frame.loc[missing,name]=np.nan
            for frame in [train,val]:frame[name]=pd.to_numeric(frame[name],errors='coerce').replace([np.inf,-np.inf],np.nan)
            median=train[name].median();median=0. if pd.isna(median) else float(median)
            for frame in [train,val]:frame[name]=frame[name].fillna(median).astype(float)
        else:
            categories.append(name)
            for frame in [train,val]:frame[name]=frame[name].astype(object).map(lambda v:'__NA__' if pd.isna(v) else str(v))
    return train,val,categories


class RelBenchEvaluator:
    """One validation-only evaluator per process; fixed learner, no search hooks."""
    def __init__(self,bundle,output,threads=4,memory_limit='3GB',sql_timeout=300,train_rows=200000):
        self.bundle,self.output=bundle,Path(output)
        self.threads,self.memory_limit,self.sql_timeout=threads,memory_limit,sql_timeout
        self.train_rows=train_rows
        if train_rows<1:raise ValueError('Positive training row budget required')

    def __enter__(self):
        from tabexp.neural.snapshot import MetadataLoader
        from tabexp.neural.stage2_data import database
        import numpy as np
        self.loader=MetadataLoader(self.bundle['context']['task']['dataset'],self.bundle['context']['task']['task'])
        self.train=self.loader.get_table('train').df.reset_index(drop=True)
        self.val_table=self.loader.get_table('val');self.val=self.val_table.df.reset_index(drop=True)
        self.stack=ExitStack()
        try:
            conn=self.stack.enter_context(database(self.loader,self.bundle['context'],memory_limit=self.memory_limit))
            conn.execute('SET threads='+str(int(self.threads)))
            self.sql=SQLFeatures(conn,self.bundle['context'],self.sql_timeout)
            ids=np.arange(len(self.train))
            if len(ids)>self.train_rows:
                from sklearn.model_selection import train_test_split
                target=self.loader.task.target_col
                _,ids=train_test_split(ids,test_size=self.train_rows,random_state=42,
                    stratify=self.train[target] if self.bundle['context']['task']['kind']=='classification' else None)
                ids=np.sort(ids);self.train=self.train.iloc[ids].reset_index(drop=True)
            self.output.mkdir(parents=True,exist_ok=True);np.save(self.output/'train_indices.npy',ids)
            return self
        except Exception:self.stack.close();raise

    def __exit__(self,*args):self.stack.close()

    def evaluate(self,queries):
        import numpy as np
        from catboost import CatBoostClassifier,CatBoostRegressor,Pool
        context=self.bundle['context'];target=self.loader.task.target_col
        raw=self.sql.collect(queries,self.train,'train');val=self.sql.collect(queries,self.val,'val')
        diagnostics=feature_statistics(raw);X,V,categories=prepare_matrices(raw,val)
        y=self.train[target].to_numpy();classification=context['task']['kind']=='classification'
        if classification and len(np.unique(y))!=2:raise ValueError('This backend supports binary classification or regression')
        config=dict(iterations=FIXED_TREE['n_estimators'],depth=FIXED_TREE['max_depth'],
            learning_rate=FIXED_TREE['learning_rate'],l2_leaf_reg=FIXED_TREE['l2_leaf_reg'],
            random_seed=42,thread_count=self.threads,allow_writing_files=False,verbose=False)
        model=(CatBoostClassifier if classification else CatBoostRegressor)(**config)
        if not categories and not np.isfinite(X.to_numpy(dtype=float)).all():raise ValueError('Nonfinite predictor input')
        try:model.fit(X,y,cat_features=categories)
        except Exception as exc:
            from .core import error_feedback
            raise ValueError(error_feedback(exc)) from None
        prediction=model.predict_proba(V)[:,1] if classification else model.predict(V)
        if not np.isfinite(prediction).all():raise ValueError('Nonfinite validation predictions')
        metrics={k:(float(v) if np.isfinite(float(v)) else None) for k,v in self.loader.evaluate(prediction,split='val',table=self.val_table).items()}
        score=metrics['roc_auc'] if classification else -metrics['mae']
        confusion=None
        if classification:
            from sklearn.metrics import confusion_matrix
            confusion=confusion_matrix(self.val[target],prediction>=.5,labels=[0,1]).tolist()
        indices=np.linspace(0,len(V)-1,min(256,len(V)),dtype=int)
        values=model.get_feature_importance(Pool(V.iloc[indices],cat_features=categories),type='ShapValues')
        importance=np.abs(values[:,:-1]).mean(0);importance_type='mean absolute SHAP'
        order=np.argsort(-np.asarray(importance))[:20]
        ranked=[dict(feature=str(X.columns[i]),importance=float(importance[i]),type=importance_type) for i in order]
        return Evaluation(float(score),metrics,diagnostics,categories,ranked,confusion)
