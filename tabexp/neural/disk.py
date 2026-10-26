"""Temporary chunked storage for cutoff-safe relational input execution."""
import logging
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from tabexp.neural import stage2_data as original
from tabexp.neural.stage2_data import Codec


def chunked_groups(rows,chunk_rows):
    for cutoff,anchors in rows.groupby('cutoff',sort=True):
        for start in range(0,len(anchors),chunk_rows): yield cutoff,anchors.iloc[start:start+chunk_rows]


class DiskExecutor(original.Stage2Executor):
    def _groups(self, rows):
        if not hasattr(self, 'directory'):
            return super()._groups(rows)
        return chunked_groups(rows, self.chunk_rows)

    def _store_frame(self, name, raw, enhanced, frames, features):
        if not hasattr(self, 'directory'):
            return super()._store_frame(name, raw, enhanced, frames, features)
        self._persist(name, raw, enhanced)

    def _finish(self, counts, frames, features, fit):
        if not hasattr(self, 'directory'):
            return super()._finish(counts, frames, features, fit)
        return self._finish_disk(counts)

    def configure(self,directory,chunk_rows=2048):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        self.chunk_rows=chunk_rows;self.parts={n['name']:[] for n in original.occurrences(self.plan)}
        self.anchor_count=0

    def _persist(self,name,raw,enhanced):
        folder=self.directory/name;folder.mkdir(exist_ok=True)
        index=len(self.parts[name]);meta=folder/f'{index:06d}_meta.parquet';features=folder/f'{index:06d}_fields.parquet'
        columns=['_seed'] if name=='root' else ['_seed','_node','_parent','_importance']
        raw[columns].to_parquet(meta,index=False)
        enhanced.to_parquet(features,index=False)
        self.parts[name].append((meta,features))
        if name=='root':
            self.anchor_count+=len(raw)
            if index==0 or (index+1)%10==0:
                logging.getLogger('stage2').info('disk %s anchors=%d shard=%d',self.directory.name,self.anchor_count,index+1)

    def _finish_disk(self,counts):
        root=None;arrays={};self.coverage={}
        with duckdb.connect() as conn:
            conn.execute("SET memory_limit='1GB'");conn.execute('SET threads=2')
            for node in original.occurrences(self.plan):
                name=node['name'];parts=self.parts[name];folder=self.directory/name;count=counts[name]
                if name=='root':size=count
                if name not in self.codecs:
                    cats=[f['name'] for f in self.feature_fields[name] if f['kind']=='categorical']
                    relation=conn.read_parquet([str(p[1]) for p in parts],union_by_name=True)
                    codec=Codec(cats,1024);codec.numeric=[c for c in relation.columns if c not in cats]
                    for col in relation.columns:
                        frame=relation.project('"'+col+'"').fetchdf()
                        one=Codec([col] if col in cats else [],1024).fit_streaming(frame)
                        if col in cats:codec.vocabulary[col]=one.vocabulary[col]
                        else:
                            codec.medians.extend(one.medians);codec.scales.extend(one.scales);codec.numeric_knots.extend(one.numeric_knots)
                    self.codecs[name]=codec
                codec=self.codecs[name]
                def allocate(field,shape,dtype):
                    path=folder/(field+'.npy')
                    if np.prod(shape)==0:
                        a=np.empty(shape,dtype=dtype);np.save(path,a);return a
                    return np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)
                num=allocate('numeric',(count,2*len(codec.numeric)),np.float32)
                cat=allocate('categorical',(count,len(codec.categorical)),np.int64)
                if name!='root':
                    data=dict(num=num,cat=cat,seed=allocate('seed',(count,),np.int64),
                        parent=allocate('parent',(count,),np.int64),importance=allocate('importance',(count,),np.float32))
                start=0
                for meta_path,fields_path in parts:
                    meta=pd.read_parquet(meta_path);frame=pd.read_parquet(fields_path);end=start+len(meta)
                    if name=='root':
                        n,c=codec.transform_streaming(frame);ids=meta['_seed'].to_numpy(np.int64);num[ids]=n;cat[ids]=c
                    else:
                        assert np.array_equal(meta['_node'].to_numpy(),np.arange(start,end))
                        codec.transform_streaming(frame,numeric_out=num[start:end],categorical_out=cat[start:end])
                        data['seed'][start:end]=meta['_seed'];data['parent'][start:end]=meta['_parent']
                        data['importance'][start:end]=meta['_importance']
                    start=end
                assert start==count
                if name=='root':root={'sql_num':num,'sql_cat':cat}
                else:arrays[name]=data;self.coverage[name]=count
        self.model_spec=dict(root_spec=self.codecs['root'].spec,
            node_specs={n['name']:self.codecs[n['name']].spec for n in self.plan['structure']['nodes']},
            nodes=self.plan['structure']['nodes'],horizon_days=self.plan['context']['task']['horizon_days'],operators=self.plan['context']['network'])
        return original.PackedStore(root,self.plan['structure']['nodes'],arrays)


