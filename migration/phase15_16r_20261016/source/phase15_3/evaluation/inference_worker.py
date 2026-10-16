"""Unmodified frozen inference engine; socket is confined to Phase15.3."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
import phase15_1.inference_opt.worker as original
def isolated(path):
    p=Path(path).resolve();root=(Path.cwd()/'phase15_3/results').resolve()
    if not p.is_relative_to(root):raise RuntimeError('Phase15.3 socket escaped namespace')
    return p
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--socket',required=True);p.add_argument('--backend',choices=['graph','unroll','cached'],default='graph')
    a=p.parse_args();original.isolated=isolated;original.main(a)
