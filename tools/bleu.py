# This script is pretty much a hack, can't seem to run it using perl.

import argparse
import subprocess
import codecs

def report_bleu(gen_file, tgt_path):
    res = subprocess.check_output(
        "perl tools/multi-bleu.perl %s" % (tgt_path),
        stdin=codecs.open(gen_file, 'r', 'utf-8'), shell=True
    ).decode("utf-8")

    msg = ">> " + res.strip()
    print(msg)
    with open(args.gen+".bleu", 'w') as f:
        f.write(msg+'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gen', '-gen', type=str)
    parser.add_argument('--tgt', '-tgt', type=str)
    args = parser.parse_args()

    report_bleu(args.gen, args.tgt)
