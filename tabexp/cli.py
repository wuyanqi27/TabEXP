"""Run each stage explicitly, with keys read from environment variables only."""
import argparse
import logging
from . import pipeline
from .neural.stage2_contract import DEFAULT_MAX_HOPS

def main():
    parser=argparse.ArgumentParser(prog='tabexp');sub=parser.add_subparsers(dest='command',required=True)
    def api(p):
        p.add_argument('--model',default='gpt-6.1-sol');p.add_argument('--base-url',default='https://api.openai.com/v1')
        p.add_argument('--key-env',default='OPENAI_API_KEY');p.add_argument('--api-timeout',type=float,default=180)
        p.add_argument('--api-retries',type=int,default=2);p.add_argument('--max-tokens',type=int,default=32768)
        p.add_argument('--repairs',type=int,default=5)
    p=sub.add_parser('explore');p.add_argument('--dataset',required=True);p.add_argument('--task',required=True);p.add_argument('--output',required=True)
    p.add_argument('--rounds',type=int,default=10);p.add_argument('--patience',type=int,default=3);p.add_argument('--train-rows',type=int,default=200000)
    p.add_argument('--profile-rows',type=int,default=100000);p.add_argument('--threads',type=int,default=4);api(p)
    p=sub.add_parser('plan');p.add_argument('--experience',required=True);p.add_argument('--output',required=True);p.add_argument('--importance-lambda',type=float,default=.25);api(p)
    p.add_argument('--max-hops',type=int,default=DEFAULT_MAX_HOPS,help='Maximum PK-FK distance from root (default: 2; 0: root only)')
    p=sub.add_parser('train');p.add_argument('--plan',required=True);p.add_argument('--program',help='Frozen stage-one best_program.json; required for full-cohort root evaluation')
    p.add_argument('--output',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--seeds',type=int,nargs='+',default=[1])
    p.add_argument('--train-rows',type=int,default=200000);p.add_argument('--train-indices');p.add_argument('--epochs',type=int,default=10);p.add_argument('--patience',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=512);p.add_argument('--learning-rate',type=float,default=.001);p.add_argument('--weight-decay',type=float,default=1e-5)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--chunk-rows',type=int,default=2048);p.add_argument('--memory-limit',default='3GB');p.add_argument('--sql-timeout',type=float,default=300)
    p.add_argument('--no-test',action='store_true')
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    print(getattr(pipeline,args.command)(args))

if __name__=='__main__':main()
