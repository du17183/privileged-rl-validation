"""The baseline delegates to byte-preserved Phase15 inference code."""
import argparse
from phase15_1.common import RESULTS


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--socket', required=True)
    p.add_argument('--backend', default='baseline')
    a = p.parse_args()
    if a.backend == 'baseline':
        from phase15 import inference_server
        inference_server.RESULTS = RESULTS
        inference_server.main(a)
    else:
        from phase15_1.inference_opt.worker import main
        main(a)
