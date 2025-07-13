#!/usr/bin/env python
import argparse, os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--he_dir',  required=True)
    p.add_argument('--ihc_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--size', type=int, default=256)
    args = p.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    he_paths = sorted(Path(args.he_dir).glob('*'))
    for he_path in tqdm(he_paths, desc='Pairing'):
        fname = he_path.name
        ihc_path = Path(args.ihc_dir) / fname
        if not ihc_path.exists():
            print(f'Skip {fname}: IHC missing'); continue

        A = Image.open(he_path).convert('RGB').resize((args.size, args.size))
        B = Image.open(ihc_path).convert('RGB').resize((args.size, args.size))
        combo = Image.new('RGB', (args.size*2, args.size))  # [A | B]
        combo.paste(A, (0,0)); combo.paste(B, (args.size,0))
        combo.save(out_root / fname.replace('.png', '.jpg'), quality=95)

if __name__ == '__main__':
    main()
