"""Small OpenAI-compatible JSON transport; no tool calling or agent dependencies."""
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from .core import write_json


class JSONModel:
    def __init__(self,output,model='gpt-6.1-sol',base_url='https://api.openai.com/v1',
                 key_env='OPENAI_API_KEY',timeout=180,max_tokens=32768,retries=2):
        self.output=Path(output);self.model=model;self.base_url=base_url.rstrip('/')
        self.key_env,self.timeout,self.max_tokens,self.retries=key_env,timeout,max_tokens,retries
        self.calls=0

    def __call__(self,phase,system,payload):
        key=os.environ.get(self.key_env)
        if not key:raise ValueError(self.key_env+' is required; never pass a literal key as a CLI argument')
        self.calls+=1;directory=self.output/f'{self.calls:04d}_{phase}'
        body=dict(model=self.model,messages=[dict(role='system',content=system),
            dict(role='user',content=json.dumps(payload,ensure_ascii=False,allow_nan=False))],
            response_format={'type':'json_object'},max_completion_tokens=self.max_tokens,
            reasoning_effort='medium')
        write_json(directory/'request.json',body)
        for retry in range(self.retries+1):
            started=time.perf_counter()
            request=urllib.request.Request(self.base_url+'/chat/completions',data=json.dumps(body).encode(),
                headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
            try:
                with urllib.request.urlopen(request,timeout=self.timeout) as response:value=json.load(response)
                write_json(directory/'response.json',value)
                write_json(directory/'usage.json',dict(seconds=time.perf_counter()-started,usage=value.get('usage'),transport_retries=retry))
                if value['choices'][0].get('finish_reason')=='length':raise ValueError('JSON output truncated; reduce repetition while retaining the information block')
                return value['choices'][0]['message']['content']
            except urllib.error.HTTPError as exc:
                write_json(directory/f'http_error_{retry}.json',dict(status=exc.code))
                if exc.code not in [429,500,502,503,504] or retry==self.retries:raise RuntimeError(f'LLM HTTP {exc.code}; run preserved, no unbounded retries') from None
            except (TimeoutError,urllib.error.URLError):
                if retry==self.retries:raise RuntimeError('LLM transport failed after bounded retries') from None
            time.sleep(min(2**retry,5))
