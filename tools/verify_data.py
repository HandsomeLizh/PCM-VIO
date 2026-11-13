"""Verify prepared data identity against the handoff manifests (no image decoding).

Pose/IMU bytes and the complete ordered image inventory are checked, plus first/last
image bytes. Interior image bytes are protected by the ground archive SHA256 in the
shared package, not by this fast check. KITTI images remain external shared data.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(root, seq):
    root = Path(root)
    images = sorted((root / 'sequences' / seq / 'image_2').glob('*.png'))
    if not images:
        raise FileNotFoundError(f'No images for {seq} in {root}')
    names = '\n'.join(p.name for p in images) + '\n'
    return {
        'frames': len(images), 'image_names_sha256': hashlib.sha256(names.encode()).hexdigest(),
        'first_image_sha256': sha256(images[0]), 'last_image_sha256': sha256(images[-1]),
        'poses_sha256': sha256(root / 'poses' / f'{seq}.txt'),
        'imu_sha256': sha256(root / 'imus' / f'{seq}.mat'),
    }


def verify(root, manifest, sequences=None):
    expected = json.loads(Path(manifest).read_text())
    for seq in sequences or expected['sequences']:
        actual = fingerprint(root, seq)
        if actual != expected['sequences'][seq]:
            changed = [k for k, v in actual.items() if expected['sequences'][seq].get(k) != v]
            raise ValueError(f'Dataset fingerprint mismatch for {seq}: {changed}')
    return expected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--manifest', required=True)
    args = p.parse_args()
    data = verify(args.data_dir, args.manifest)
    print('Verified', data['dataset'], len(data['sequences']), 'sequences')


if __name__ == '__main__':
    main()
