"""Local bounded SELECT execution; no agent or repository helpers."""
import threading
import duckdb

def run_feature_queries(conn,feature_queries,*,entity_col=None,allowed_entities=None,
                        sql_timeout_seconds=300,n_entities=None,**unused):
    results={}
    allowed=set(allowed_entities.dropna().tolist()) if allowed_entities is not None else None
    for query in feature_queries:
        statements=conn.extract_statements(query['sql'])
        if len(statements)!=1 or statements[0].type!=duckdb.StatementType.SELECT:
            raise ValueError('Feature blocks must contain exactly one SELECT')
        expired=threading.Event()
        def interrupt():
            expired.set();conn.interrupt()
        timer=threading.Timer(sql_timeout_seconds,interrupt);timer.daemon=True;timer.start()
        try:
            # Task-table augmentation has one row per anchor, checked again by
            # the key-aligned merger. Fetch at most one excess row on failure.
            cursor=conn.execute(query['sql'])
            reader=cursor.fetch_record_batch(rows_per_batch=65536)
            batches=[];count=0;limit=n_entities if n_entities is not None else 1000000
            for batch in reader:
                count+=len(batch)
                if count>limit:raise ValueError('Feature query exceeds its anchor row budget')
                batches.append(batch)
            import pyarrow as pa
            frame=pa.Table.from_batches(batches,schema=reader.schema).to_pandas()
        finally:
            timer.cancel();timer.join()
        if expired.is_set():raise TimeoutError('Feature query exceeded its time budget')
        if frame.columns.duplicated().any():raise ValueError('Duplicate output column names')
        if entity_col and allowed is not None:
            if entity_col not in frame:raise ValueError('Feature query omitted entity key')
            if not frame[entity_col].isin(allowed).all():raise ValueError('Feature query returned an unrequested entity')
        results[query['name']]=frame
    return results
