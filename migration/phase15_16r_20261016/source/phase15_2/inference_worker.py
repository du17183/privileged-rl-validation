"""Exact frozen Phase15.1 inference engine with Phase15.2 socket isolation."""
import argparse
from phase15_2.common import isolated
import phase15_1.inference_opt.worker as original

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--socket',required=True)
    p.add_argument('--backend',choices=['graph','unroll','cached'],default='graph');a=p.parse_args()
    original.isolated=isolated
    original.main(a)
