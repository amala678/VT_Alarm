import argparse
from pathlib import Path
from .data import download,build_cache,synthetic

def main():
    p=argparse.ArgumentParser(description='Offline VTaC research experiments')
    sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('download'); d.add_argument('--data',default='data/vtac'); d.add_argument('--limit-per-split',type=int); d.add_argument('--split-column')
    c=sub.add_parser('prepare'); c.add_argument('--data',default='data/vtac'); c.add_argument('--cache',default='cache/vtac'); c.add_argument('--seconds',type=int,choices=[10,30,60],default=30); c.add_argument('--split-column')
    t=sub.add_parser('train'); t.add_argument('--cache',default='cache/vtac'); t.add_argument('--out',default='runs/main'); t.add_argument('--epochs',type=int,default=40); t.add_argument('--seeds',type=int,nargs='+',default=[11,22,33]); t.add_argument('--target',type=float,default=.95); t.add_argument('--quick',action='store_true'); t.add_argument('--device',default='cpu')
    e=sub.add_parser('evaluate'); e.add_argument('--cache',default='cache/vtac'); e.add_argument('--out',default='runs/main'); e.add_argument('--bootstrap',type=int,default=1000); e.add_argument('--device',default='cpu')
    s=sub.add_parser('demo'); s.add_argument('--out',default='demo_run'); s.add_argument('--epochs',type=int,default=3); s.add_argument('--bootstrap',type=int,default=100)
    args=p.parse_args()
    if args.command=='download': download(args.data,args.limit_per_split,args.split_column)
    elif args.command=='prepare': build_cache(args.data,args.cache,args.seconds,args.split_column)
    elif args.command=='train':
        from .experiment import train
        train(args.cache,args.out,args.seeds,args.epochs,args.target,args.quick,args.device)
    elif args.command=='evaluate':
        from .experiment import evaluate
        if args.bootstrap<1: p.error('--bootstrap must be positive')
        evaluate(args.cache,args.out,args.bootstrap,args.device)
    else:
        from .experiment import train,evaluate
        out=Path(args.out); synthetic(out/'cache')
        train(out/'cache',out/'models',epochs=args.epochs,quick=True)
        evaluate(out/'cache',out/'models',repetitions=args.bootstrap)

if __name__=='__main__': main()
