#!/usr/bin/env python3
"""Package the generated KeyFree-Bench data into compact archives for release.

The raw PNG dumps are large (~1.3 GB).  For convenient reproduction we ship
JPEG-compressed copies (quality 95, no chroma subsampling) that keep the *same
file names*, so every script reads them without modification (PIL/OpenCV detect
the codec from the content, not from the extension).  Binary masks are stored
losslessly as PNG.  For bit-exact pixels, regenerate from the scripts in code/.

Usage:
    python tools/package_dataset.py --out /tmp/pkg
"""
import argparse
import os
import zipfile
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f'{BASE}/data'


def dump(paths, dst_dir, quality=95, lossless_suffixes=('mask',)):
    """Copy images into dst_dir with JPEG compression (name preserved)."""
    for p in paths:
        name = os.path.basename(p)
        keep_lossless = any(k in name for k in lossless_suffixes)
        out = os.path.join(dst_dir, name)
        if keep_lossless:
            Image.open(p).save(out, optimize=True)
        else:
            Image.open(p).convert('RGB').save(out, quality=quality, subsampling=0)
    print(f'  {len(paths)} images -> {dst_dir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/pkg')
    ap.add_argument('--quality', type=int, default=95)
    args = ap.parse_args()

    work = '/tmp/pkg_work'
    os.makedirs(args.out, exist_ok=True)

    groups = {
        'KeyFree-Bench-T1-samedomain': [
            f'{D}/bigset/images', f'{D}/bigset/meta.json', 'bigset/images',
        ],
        'KeyFree-Bench-T2-crossdomain': [
            f'{D}/xdomain', None, 'xdomain',
        ],
        'KeyFree-Bench-T3-crosseditor': [
            f'{D}/hires_expanded', None, 'hires_expanded',
        ],
    }

    for name, (src, meta, rel) in groups.items():
        wd = os.path.join(work, rel)
        if os.path.exists(wd):
            os.system(f'rm -rf "{wd}"')
        os.makedirs(wd, exist_ok=True)

        if src.endswith('images'):                       # T1: one image folder
            dump([os.path.join(src, f) for f in sorted(os.listdir(src))],
                 wd, args.quality)
            if meta and os.path.exists(meta):
                os.system(f'cp "{meta}" "{wd}/../meta.json"')
        else:                                            # T2/T3: tree
            for root, _, files in os.walk(src):
                imgs = [os.path.join(root, f) for f in sorted(files)
                        if f.lower().endswith(('.png', '.jpg'))]
                if imgs:
                    sub = os.path.join(wd, os.path.relpath(root, src))
                    os.makedirs(sub, exist_ok=True)
                    dump(imgs, sub, args.quality)
                for f in files:
                    if f.endswith('.json'):
                        sub = os.path.join(wd, os.path.relpath(root, src))
                        os.makedirs(sub, exist_ok=True)
                        os.system(f'cp "{os.path.join(root, f)}" "{sub}/{f}"')

        zip_path = os.path.join(args.out, f'{name}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for root, _, files in os.walk(wd):
                for f in files:
                    p = os.path.join(root, f)
                    z.write(p, os.path.relpath(p, wd))
        print(f'{name}.zip  {os.path.getsize(zip_path) / 1e6:.1f} MB')

    print('done ->', args.out)


if __name__ == '__main__':
    main()
