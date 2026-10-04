from . import main
import argparse
p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True)
p.add_argument('--updates',type=int,default=5000);p.add_argument('--batch',type=int,default=128)
main(p.parse_args())
