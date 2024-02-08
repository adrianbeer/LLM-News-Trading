import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--foo', action='store_true', default=False)
args = parser.parse_args()

print(args.foo)