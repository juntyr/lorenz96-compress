import argparse
import numpy as np

from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--dt", default=0.01, type=float)
parser.add_argument("-k", default=36, type=int)
parser.add_argument("-e", "--ensemble-size", default=11, type=int)
parser.add_argument("-o", "--output", default="state", type=Path)

args = parser.parse_args()

for i in range(args.ensemble_size):
    file_path = args.output.with_name(f"{args.output.name}_{i}")

    data = np.fromfile(file_path, dtype=np.double).reshape(-1, args.k)

    print(file_path)

    for j in range(data.shape[0]):
        print(f"n={j}: {data[j]}")

    print()
