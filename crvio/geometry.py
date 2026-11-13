"""Differentiable rotation helpers.

Conventions follow the baseline exactly (utils/utils.py):
  * pose vector = [rx, ry, rz, tx, ty, tz], camera frame of the interval start.
  * R = Rz(rz) @ Ry(ry) @ Rx(rx).
"""
import torch


def euler_to_matrix(euler):
    """(..., 3) Euler angles -> (..., 3, 3), same as eulerAnglesToRotationMatrix."""
    rx, ry, rz = euler.unbind(-1)
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    row0 = torch.stack([cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx], -1)
    row1 = torch.stack([sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx], -1)
    row2 = torch.stack([-sy, cy * sx, cy * cx], -1)
    return torch.stack([row0, row1, row2], -2)


def so3_exp(phi):
    """(..., 3) rotation vector -> (..., 3, 3), Rodrigues with a safe small-angle series."""
    theta2 = (phi * phi).sum(-1, keepdim=True)
    small = theta2 < 1e-8
    theta2_safe = torch.where(small, torch.ones_like(theta2), theta2)
    theta = torch.sqrt(theta2_safe)
    a = torch.where(small, 1.0 - theta2 / 6.0, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24.0, (1.0 - torch.cos(theta)) / theta2_safe)
    x, y, z = phi.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).view(*phi.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(K.shape)
    return eye + a.unsqueeze(-1) * K + b.unsqueeze(-1) * (K @ K)


def so3_log(R):
    """(..., 3, 3) -> (..., 3) rotation vector, i.e. Log(R)^vee.

    Gradients stay finite at the identity because the small-angle branch only
    uses the squared sine (acos / sqrt are never evaluated there). Frame-to-frame
    KITTI rotations are far from pi, so no dedicated near-pi branch is needed;
    theta / sin(theta) is simply capped there.
    """
    w = 0.5 * torch.stack([R[..., 2, 1] - R[..., 1, 2],
                           R[..., 0, 2] - R[..., 2, 0],
                           R[..., 1, 0] - R[..., 0, 1]], -1)
    s2 = (w * w).sum(-1, keepdim=True)                      # sin^2(theta)
    cos = 0.5 * (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0).unsqueeze(-1)
    small = s2 < 1e-8
    s2_safe = torch.where(small, torch.ones_like(s2), s2)
    s = torch.sqrt(s2_safe)
    factor = torch.atan2(s, cos) / s
    series = 1.0 + s2 / 6.0 + 3.0 * s2 * s2 / 40.0          # arcsin(s) / s
    return torch.where(small, series, factor.clamp(max=1e3)) * w


def rotation_error(R_hat, R_gt):
    """Log(R_gt^T R_hat)^vee, the local rotation residual used by L_mot and L_prior."""
    return so3_log(R_gt.transpose(-1, -2) @ R_hat)
