"""Execute a bounded, resumable range of the ten frozen negative-control runs."""
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--start',type=int,required=True); parser.add_argument('--end',type=int,required=True); args=parser.parse_args()
    assert 1<=args.start<=args.end<=10
    for i in range(args.start,args.end+1):
        subprocess.run([sys.executable,str(Path(__file__).with_name('experiment.py')),'--permutation',str(i)],check=True)

if __name__=='__main__': main()
