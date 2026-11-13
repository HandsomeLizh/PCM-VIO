"""Convert M2DGR-Plus and M3DGR (ground robot, RealSense colour camera + its own IMU) to the KITTI layout
that dataset/KITTI_dataset.py, train.py, train_crvio.py and the testers read:

  <out>/sequences/<seq>/image_2/%06d.png   frames at 15 Hz, resized to 512 x 256 like the KITTI frames
  <out>/imus/<seq>.mat                     imu_data_interp: [ax ay az wx wy wz], 10 samples per frame
                                           interval + 1 (the KITTI convention), camera-IMU frame
  <out>/poses/<seq>.txt                    3 x 4 camera poses (camera frame, first frame = identity)
  <out>/meta/<seq>.json                    time offset, camera-from-GT-body rotation, checks

The two ground truths use different body frames (GNSS-INS in local ENU for M2DGR-Plus, a motion-capture
body for M3DGR). Both are brought to the camera frame so the two datasets can be mixed: per sequence, the
clock offset and the rotation camera <- GT body are estimated from the camera-IMU gyro against the GT
angular velocity. The lever arm between the GT body and the camera is unknown and left at zero.

Needs the rosbags package (separate venv): /root/VIO/.venv-bag/bin/python tools/convert_ground_robot.py
"""
import argparse
import io
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image
from scipy.spatial.transform import Rotation as R, Slerp

M2 = '/remote-home/share/ums_lzj/M2DGR-Plus'
M3 = '/remote-home/share/ums_lzj/M3DGR'
# name -> (bag, ground truth, image topic, frame stride to 15 Hz, GT kind; ECEF vs local is detected from the data)
M2_BAG = {'switch': 'switch2', 'tree': 'tree3'}      # bag names that differ from the ground-truth names
SEQS = {f'm2_{n}': (f'{M2}/raw/{n[0].upper() + n[1:]}/{M2_BAG.get(n, n)}.bag', f'{M2}/groundtruth/{n}.txt',
                    '/camera/color/image_raw', 1, 'ecef')
        for n in ['anomaly', 'bridge2', 'building1', 'building2', 'parking1', 'parking2', 'street1', 'street2',
                  'switch', 'tree']}
SEQS.update({f'm3_{n.lower().replace("-", "")}': (f'{M3}/extracted/{n}/{n}.bag', f'{M3}/ground_truth/{n}.txt',
                                                  '/camera/color/image_raw/compressed', 2, 'local')
             for n in ['Dynamic01', 'Occlusion01', 'Sha-turn01', 'Varying-illu01', 'Wheel-float01']})
IMU_TOPIC = '/camera/imu'
OUT_W, OUT_H = 512, 256
GRID = 0.02          # s, grid for the gyro / GT alignment
MAX_GAP = 0.5        # s, longest ground-truth gap allowed inside a sequence


def stamp(m):
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def ecef_to_enu(p):
    x, y, z = p[0]
    lon, lat = np.arctan2(y, x), np.arctan2(z, np.hypot(x, y))
    Renu = np.array([[-np.sin(lon), np.cos(lon), 0],
                     [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                     [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]])
    return (p - p[0]) @ Renu.T


def load_gt(path, kind):
    d = np.loadtxt(path, comments='#')
    d = d[np.argsort(d[:, 0], kind='stable')]
    keep = np.concatenate([[True], np.diff(d[:, 0]) > 1e-4])      # mocap rows a few microseconds apart
    d = d[keep]
    # Keep the longest stretch without a ground-truth gap over MAX_GAP (switch: gaps up to 92 s,
    # Wheel-float01: 3 s near the start); interpolating across a gap would invent the motion.
    cut = np.flatnonzero(np.diff(d[:, 0]) > MAX_GAP) + 1
    d = max(np.split(d, cut), key=lambda x: x[-1, 0] - x[0, 0])
    t, p, q = d[:, 0], d[:, 1:4], d[:, 4:8]
    if np.linalg.norm(p[0]) > 1e6:   # ECEF (most M2DGR-Plus files); anomaly / switch / tree are already local
        p = ecef_to_enu(p)            # the M2DGR-Plus orientation is already body -> local ENU (checked on street2)
    else:
        p = p - p[0]
    return t, p, R.from_quat(q)


class GTInterp:
    def __init__(self, t, p, rot):
        self.t, self.p, self.slerp = t, p, Slerp(t, rot)

    def __call__(self, tq):
        p = np.stack([np.interp(tq, self.t, self.p[:, k]) for k in range(3)], -1)
        return p, self.slerp(tq)


def read_imu(reader):
    conns = [c for c in reader.connections if c.topic == IMU_TOPIC]
    t, a, g = [], [], []
    for c, _, raw in reader.messages(connections=conns):
        m = reader.deserialize(raw, c.msgtype)
        t.append(stamp(m))
        a.append([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
        g.append([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
    t, a, g = map(np.asarray, (t, a, g))
    o = np.argsort(t, kind='stable')
    t, a, g = t[o], a[o], g[o]
    keep = np.concatenate([[True], np.diff(t) > 1e-6])
    return t[keep], a[keep], g[keep]


def align(gt, t_imu, gyro):
    """GT time = sensor time + offset; omega_cam = R_cb omega_body."""
    tg = np.arange(gt.t[0] + 1.0, gt.t[-1] - 1.0, GRID)
    _, rot = gt(np.concatenate([tg, tg[-1:] + GRID]))
    w_body = (rot[:-1].inv() * rot[1:]).as_rotvec() / GRID
    def gyro_at(ts):
        return np.stack([np.interp(ts, t_imu, gyro[:, k]) for k in range(3)], -1)
    mag_b = np.linalg.norm(w_body, axis=1)
    best = (-2, 0.0)
    for off in np.arange(-2.0, 2.0, 0.005):
        mag_c = np.linalg.norm(gyro_at(tg + GRID / 2 - off), axis=1)
        c = np.corrcoef(mag_b, mag_c)[0, 1]
        if c > best[0]:
            best = (c, off)
    corr, off = best
    w_cam = gyro_at(tg + GRID / 2 - off)
    sel = mag_b > 0.05
    U, _, Vt = np.linalg.svd(w_cam[sel].T @ w_body[sel])
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R_cb = U @ D @ Vt
    res = np.sqrt(((w_cam[sel] - w_body[sel] @ R_cb.T) ** 2).sum(1).mean())
    return off, R_cb, float(corr), float(res), float(np.sqrt((w_cam[sel] ** 2).sum(1).mean()))


def heading_fix(gt, off, R_cb, t_imu, acc, h=1.0):
    """Rotation about the camera's up axis that makes straight driving point along +z (camera forward).

    The gyro fixes the up axis well, but a ground robot turns almost only about that axis, so the rotation
    about it is weakly determined (parking1 / parking2 came out 33-45 deg off). A differential-drive robot
    only moves forward or backward, so the GT motion on straight stretches gives that angle. This assumes
    the camera looks along the driving direction (within ~3 deg on the well-conditioned sequences).
    Also returns how consistent the straight-driving heading is (resultant length of the doubled angles,
    1 = all along one axis): anomaly / building1 / parking1 / parking2 are 0.1-0.6 because their GT heading
    jumps by up to 80 deg within the sequence; they are not used."""
    a = acc[(t_imu > gt.t[0] - off) & (t_imu < gt.t[-1] - off)].mean(0)
    up = a / np.linalg.norm(a)                 # gravity, camera frame (the gyro's main axis is noise on parking1)
    tg = np.arange(gt.t[0] + 1.0, gt.t[-1] - 1.0 - h, 0.05)
    p0, r0 = gt(tg)
    p1, r1 = gt(tg + h)
    d = (r0.inv().apply(p1 - p0)) @ R_cb.T                     # displacement over h, camera frame
    rate = (r0.inv() * r1).magnitude() / h
    sel = (np.linalg.norm(d, axis=1) > 0.2 * h) & (rate < 0.02)      # >= 0.2 m/s, < 1.1 deg/s
    theta, consistency = axial_mean_heading(d[sel], up)
    return R.from_rotvec(-theta * up).as_matrix(), float(np.degrees(-theta)), int(sel.sum()), consistency


def decode(m, compressed):
    if compressed:
        img = Image.open(io.BytesIO(bytes(m.data))).convert('RGB')
    else:
        arr = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.step)[:, :m.width * 3]
        arr = arr.reshape(m.height, m.width, 3)
        img = Image.fromarray(arr[..., ::-1] if m.encoding == 'bgr8' else arr)
    return img.resize((OUT_W, OUT_H), Image.BILINEAR)    # squash, as the KITTI frames are squashed to 2:1


def convert(name, out):
    from rosbags.highlevel import AnyReader
    bag, gt_path, topic, stride, kind = SEQS[name]
    gt = GTInterp(*load_gt(gt_path, kind))
    with AnyReader([Path(bag)]) as reader:
        t_imu, acc, gyro = read_imu(reader)
        off, R_cb, corr, res, w_rms = align(gt, t_imu, gyro)
        R_fix, heading_deg, n_straight, heading_consistency = heading_fix(gt, off, R_cb, t_imu, acc)
        R_cb = R_fix @ R_cb
        conns = [c for c in reader.connections if c.topic == topic]
        img_dir = Path(out) / 'sequences' / name / 'image_2'
        img_dir.mkdir(parents=True, exist_ok=True)
        times, k = [], 0
        lo, hi = max(gt.t[0] - off, t_imu[0]) + 0.05, min(gt.t[-1] - off, t_imu[-1]) - 0.05
        for i, (c, _, raw) in enumerate(reader.messages(connections=conns)):
            if i % stride:
                continue
            m = reader.deserialize(raw, c.msgtype)
            ts = stamp(m)
            if not lo <= ts <= hi:
                continue
            if times and ts - times[-1] < 0.5 / 15:     # dropped or duplicated frames: keep a 15 Hz-ish stream
                continue
            decode(m, compressed=topic.endswith('compressed')).save(img_dir / f'{k:06d}.png')
            times.append(ts)
            k += 1
    times = np.asarray(times)

    # Camera poses: T_w_c = T_w_b(t + off) * T_b_c, T_b_c = (R_cb^T, 0).
    p, rot = gt(times + off)
    R_wc = (rot * R.from_matrix(R_cb.T)).as_matrix()
    T = np.tile(np.eye(4), (len(times), 1, 1))
    T[:, :3, :3], T[:, :3, 3] = R_wc, p
    T = np.linalg.inv(T[0]) @ T
    pose_dir = Path(out) / 'poses'
    pose_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(pose_dir / f'{name}.txt', T[:, :3, :].reshape(len(T), 12), fmt='%.9e')

    # 10 IMU samples per frame interval, linearly interpolated, + the last frame time.
    tq = np.concatenate([np.linspace(a, b, 10, endpoint=False) for a, b in zip(times[:-1], times[1:])] + [times[-1:]])
    imu = np.concatenate([np.stack([np.interp(tq, t_imu, x[:, j]) for j in range(3)], -1) for x in (acc, gyro)], -1)
    (Path(out) / 'imus').mkdir(parents=True, exist_ok=True)
    sio.savemat(Path(out) / 'imus' / f'{name}.mat', {'imu_data_interp': imu.astype(np.float64)})

    # Check: gyro integrated over each frame interval vs the GT relative rotation in the camera frame.
    rel_gt = (R.from_matrix(T[:-1, :3, :3]).inv() * R.from_matrix(T[1:, :3, :3]))
    w = imu[:, 3:][:-1].reshape(len(times) - 1, 10, 3)
    dt = np.diff(times)[:, None, None] / 10
    rel_imu = R.from_rotvec(np.zeros((len(times) - 1, 3)))
    for j in range(10):
        rel_imu = rel_imu * R.from_rotvec(w[:, j] * dt[:, 0])
    rot_err = np.degrees((rel_gt.inv() * rel_imu).magnitude())
    gt_gaps = np.diff(gt.t)
    in_range = (gt.t[1:] >= times[0] + off) & (gt.t[:-1] <= times[-1] + off)
    meta = dict(name=name, bag=bag, gt=gt_path, frames=len(times), duration_s=float(times[-1] - times[0]),
                frame_dt_median=float(np.median(np.diff(times))), frame_dt_max=float(np.diff(times).max()),
                path_m=float(np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1).sum()),
                time_offset_s=float(off), gyro_corr=corr, gyro_fit_residual_rad_s=res, gyro_rms_rad_s=w_rms,
                R_cam_from_gtbody=R_cb.tolist(), R_cam_from_gtbody_rotvec_deg=np.degrees(R.from_matrix(R_cb).as_rotvec()).tolist(),
                gt_max_gap_s=float(gt_gaps[in_range].max()) if in_range.any() else None,
                check_gyro_vs_gt_rel_rot_deg_median=float(np.median(rot_err)),
                check_gyro_vs_gt_rel_rot_deg_p95=float(np.percentile(rot_err, 95)),
                gt_rel_rot_deg_median=float(np.median(np.degrees(rel_gt.magnitude()))),
                heading_fix_deg=heading_deg, heading_fix_samples=n_straight, check_heading_consistency=heading_consistency,
                check_vertical_motion_share=vertical_share(T), check_straight_heading_deg=straight_heading(T),
                lever_arm='not modelled (zero)')
    (Path(out) / 'meta').mkdir(parents=True, exist_ok=True)
    with open(Path(out) / 'meta' / f'{name}.json', 'w') as f:
        json.dump(meta, f, indent=1)
    return meta


def vertical_share(T):
    """Mean |y| of the unit frame-to-frame motion in the camera frame (y is down): small for a ground robot."""
    rel = np.linalg.inv(T[:-1]) @ T[1:]
    t = rel[:, :3, 3]
    n = np.linalg.norm(t, axis=1)
    return float(np.abs(t[n > 0.005, 1] / n[n > 0.005]).mean())


def axial_mean_heading(d, up):
    """Mean heading of displacements d (camera frame) about the axis `up`, relative to +z, in rad.
    Forward and backward driving are the same axis, so the mean is taken over doubled angles."""
    b = np.array([0.0, 0.0, 1.0]) - up * up[2]
    b /= np.linalg.norm(b)
    c = np.cross(up, b)
    th = np.arctan2(d @ c, d @ b)
    S, C = np.sin(2 * th).mean(), np.cos(2 * th).mean()
    return 0.5 * np.arctan2(S, C), float(np.hypot(S, C))


def straight_heading(T):
    """Heading of the frame-to-frame motion on straight stretches, camera frame, deg (0 = along +z)."""
    rel = np.linalg.inv(T[:-1]) @ T[1:]
    t = rel[:, :3, 3]
    sel = (np.linalg.norm(t, axis=1) > 0.01) & (np.degrees(R.from_matrix(rel[:, :3, :3]).magnitude()) < 0.15)
    return float(np.degrees(axial_mean_heading(t[sel], np.array([0.0, -1.0, 0.0]))[0]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='/root/VIO/data_ground')
    p.add_argument('--seqs', nargs='+', default=sorted(SEQS))
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    with Pool(a.workers) as pool:
        for meta in pool.starmap(convert, [(s, a.out) for s in a.seqs]):
            print(json.dumps({k: v for k, v in meta.items() if k not in ('R_cam_from_gtbody', 'bag', 'gt')}), flush=True)


if __name__ == '__main__':
    main()
