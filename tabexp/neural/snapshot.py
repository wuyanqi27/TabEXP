"""Local dataset metadata adapters; no implicit downloads or project imports."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

def quote(name):return '"'+str(name).replace('"','""')+'"'

class LocalTask:
    def __init__(self,dataset,name,spec):
        self.dataset,self.name,self.spec=dataset,name,spec
        self.entity_table=spec['entity_table'];self.entity_col=spec['entity_col']
        self.target_col=spec['target_col'];self.time_col=spec.get('time_col','timestamp')
        self.task_type=spec['kind'];self.timedelta=pd.Timedelta(days=spec['horizon_days'])
        self.remove_columns=[tuple(x) for x in spec.get('remove_columns',[])]
        self.description=spec.get('description','');self._sql=spec.get('label_sql')
        self.label_definition=spec.get('label_definition','Labels are supplied by the dataset task specification.')

    def get_table(self,split,mask_input_cols=True):
        frame=pd.read_parquet(self.dataset.root/self.spec['splits'][split])
        if split=='test' and mask_input_cols:frame=frame.drop(columns=[self.target_col],errors='ignore')
        return SimpleNamespace(df=frame)

    def evaluate(self,prediction,table):
        from sklearn.metrics import roc_auc_score,mean_absolute_error
        y=table.df[self.target_col].to_numpy();prediction=np.asarray(prediction)
        if len(y)!=len(prediction) or not np.isfinite(prediction).all():raise ValueError('Invalid predictions')
        if 'regression' in self.task_type:return {'mae':float(mean_absolute_error(y,prediction))}
        return {'roc_auc':float(roc_auc_score(y,prediction))}

class MetadataLoader:
    """Accept a dataset.json directory, or a locally installed RelBench 3 dataset.

    External paths may contain data only. All custom execution code lives here;
    relbench, when used, is a declared third-party library, never project code.
    """
    def __init__(self,dataset_name,task_name,logger=None):
        self.dataset_name,self.task_name,self.logger=str(dataset_name),task_name,logger
        path=Path(dataset_name).expanduser()
        if not path.is_dir() and os.environ.get('RELBENCH_DATA_ROOT'):
            path=Path(os.environ['RELBENCH_DATA_ROOT'])/str(dataset_name)
        if (path/'dataset.json').is_file():
            config=json.loads((path/'dataset.json').read_text(encoding='utf-8'))
            tables={n:SimpleNamespace(pkey=v.get('key'),time_col=v.get('time'),fkeys=v.get('fkeys',{})) for n,v in config['tables'].items()}
            self.dataset=SimpleNamespace(root=path.resolve(),db_dir=path.resolve()/config.get('db_dir','db'),manifest=SimpleNamespace(tables=tables),test_timestamp=pd.Timestamp(config['test_timestamp']))
            self.task=LocalTask(self.dataset,task_name,config['tasks'][task_name])
        elif (path/'manifest.yaml').is_file():
            from relbench import load_dataset
            self.dataset=load_dataset(str(path.resolve()));self.task=self.dataset.load_task(task_name)
        else:
            raise FileNotFoundError('Provide a LOCAL dataset directory containing dataset.json or manifest.yaml; no download is attempted: '+str(path))
        if not hasattr(self.dataset,'db_dir') or not hasattr(self.dataset,'manifest'):
            raise ValueError('Parquet dataset metadata is required')

    def get_table(self,split='train'):
        if split not in ('train','val','test'):raise ValueError('Unknown split')
        return self.task.get_table(split)

    def evaluate(self,predictions,split='val',table=None):
        table=self.get_table(split) if table is None else table
        if len(predictions)!=len(table.df):raise ValueError('Prediction/label length mismatch')
        metrics=self.task.evaluate(predictions,table)
        if 'nmae' in metrics and 'mae' not in metrics:
            scale=float(getattr(self.task,'nmae_std',float('nan')))
            if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid NMAE scale')
            metrics=dict(metrics,mae=float(metrics['nmae'])*scale)
        return metrics
