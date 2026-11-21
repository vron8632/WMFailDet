#!/usr/bin/env python3
"""G0 密钥化方案: key-based forensic watermark (per-image key + coord-PRF 校验位).

威胁模型 (论文 §Threat Model 直接采用本文件定义):
  * 嵌入方 = 内容所有方/平台, 持 master_key (32B) 与内容登记 ID (image_uid);
  * 检测方持同一 master_key + 待检图像的内容 ID -> 可**盲**检测/盲定位:
    不需要原图、不需要原始 payload、不需要原始水印图;
  * 无密钥第三方无法派生校验序列 -> 无法检测; 也无法用"重新嵌入"伪造干净状态;
  * 与领域标准一致 (WAM/EditGuard 检测同样需密钥);
  * 不再使用全局固定 seed=42: 该 seed 一旦随算法公开即等价于公开密钥,
    攻击者可用 re-embedding 把被编辑区域的校验位刷回"通过"(见 r7 实验 C 的量化对照).

密钥层级 (RFC 5869 HKDF-SHA256, 纯标准库):
  master_key (32B, owner 持有)
    └─ image_key = HKDF-SHA256(ikm=master_key, salt=SHA256(image_uid),
                               info=b'wm/image-key/v1', L=32)
         ├─ 校验序列 (check pattern)
         └─ payload (256 bit, 模拟"密钥加密后的内容 ID/消息")
  per-image 派生的意义: 一张图的密钥/pattern 泄露不牵连其他图, 攻击者也无法把
  从某图学到的 pattern 迁移到另一图.

校验序列的两种派生方式:
  pattern='matrix'  seed = HKDF(image_key, info=b'wm/check-matrix/v1')[0:4]
                    -> np.random.RandomState(seed).randint(0, 2, (n_h, n_w))
                    与 r3–r6 (固定 seed=42) **完全同构**, 只把固定 seed 换成密钥派生值.
                    用途: "密钥化不改变结论"的等价性复现 (r7 实验 A).
  pattern='coord'   第 i 行 = SHAKE256(image_key || 'row/v1' || i), 第 j 块取该流的第 j 位
                    -> 校验位只依赖块坐标 (i, j), 与图像尺寸 n_h,n_w 及裁剪原点解耦,
                       支撑 G3 几何同步: 检测方可对偏移 (off_i, off_j) 搜索对齐.

与旧实现 (run_signal_ra.embed_signal / check_bits) 的关系:
  QIM 规则 (δ=15, bit=1 -> floor(c/δ)δ+δ/2, bit=0 -> round(c/δ)δ) 逐字一致,
  差别只在"期望序列从哪来"; 因此 r7 的数值可直接与 r3/r6 对照.
"""
import hashlib
import hmac
import os

import numpy as np
import cv2

MASTER_KEY_LEN = 32
DELTA_DEFAULT = 15
MSGLEN_DEFAULT = 256
PAY_COEF_LOW = (1, 0)     # 低频 payload 系数 (C1 结论: 低频更抗重绘)
CHECK_COEF_LOW = (1, 1)   # 低频校验系数

__all__ = [
    'hkdf_sha256', 'new_master_key', 'image_key', 'check_pattern', 'payload_bits',
    'block_grid', 'embed_keyed', 'fail_map', 'fail_map_exp', 'check_fail_rate',
    'qim_bit', 'auc', 'smooth_fail_map', 'iou_at', 'random_keys',
]


# ---------------------------------------------------------------- KDF
def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
    """RFC 5869 HKDF-SHA256 (extract + expand), 无第三方依赖."""
    prk = hmac.new(salt if salt else b'\x00' * 32, ikm, hashlib.sha256).digest()
    okm, t, i = b'', b'', 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t
        i += 1
    return okm[:length]


def new_master_key(entropy: bytes = None) -> bytes:
    return entropy if entropy is not None else os.urandom(MASTER_KEY_LEN)


def image_key(master_key: bytes, image_uid) -> bytes:
    """per-image 派生密钥; image_uid = 内容登记 ID (字符串或 bytes)."""
    uid = image_uid.encode() if isinstance(image_uid, str) else bytes(image_uid)
    return hkdf_sha256(master_key, hashlib.sha256(uid).digest(), b'wm/image-key/v1', 32)


def random_keys(master_key: bytes, n: int) -> list:
    """攻击者视角的 n 个随机密钥 (与 owner 密钥独立)."""
    return [hkdf_sha256(master_key + b'|attacker-salt', b'atk', b'wm/atk/%d/v1' % i, 32)
            for i in range(n)]


# ---------------------------------------------------------------- 期望序列
def _bits_from_stream(stream: bytes, n: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(stream, np.uint8))[:n].astype(np.uint8)


def check_pattern(key: bytes, n_h: int, n_w: int, pattern: str = 'coord',
                  off=(0, 0), h: int = None, w: int = None) -> np.ndarray:
    """返回 (n_h, n_w) 期望校验位.

    off=(off_i, off_j): 该图像块网格相对嵌入时网格的偏移 (G3 同步用);
    coord 模式下偏移只平移取值位置, 不重排整张序列, 因此裁剪后仍可对齐.
    h, w 仅为兼容旧接口保留, 未使用.
    """
    if pattern == 'matrix':
        seed = int.from_bytes(
            hkdf_sha256(key, b'\x00' * 32, b'wm/check-matrix/v1', 4), 'big')
        return np.random.RandomState(seed).randint(0, 2, (n_h, n_w)).astype(np.uint8)
    if pattern == 'coord':
        oi, oj = int(off[0]), int(off[1])
        need = n_w + oj
        row_bytes = (need + 7) // 8
        out = np.empty((n_h, n_w), np.uint8)
        for i in range(n_h):
            stream = hashlib.shake_256(
                key + b'|row/v1|' + (oi + i).to_bytes(4, 'big', signed=True)).digest(row_bytes)
            out[i] = _bits_from_stream(stream, need)[oj:oj + n_w]
        return out
    raise ValueError(f'unknown pattern {pattern!r}')


def payload_bits(key: bytes, msglen: int = MSGLEN_DEFAULT) -> np.ndarray:
    stream = hashlib.shake_256(key + b'|payload/v1').digest((msglen + 7) // 8)
    return _bits_from_stream(stream, msglen)


def _msg_index(key: bytes, n_h: int, n_w: int, msglen: int, keyed_perm: bool) -> np.ndarray:
    """块 -> payload 位索引. 默认公开映射 (与 r3–r6 一致); keyed_perm 时密钥派生置换."""
    base = (np.arange(n_h * n_w) % msglen).astype(np.int64)
    if not keyed_perm:
        return base
    seed = int.from_bytes(hkdf_sha256(key, b'\x00' * 32, b'wm/perm/v1', 4), 'big')
    perm = np.random.RandomState(seed).permutation(n_h * n_w)
    return base[perm]


# ---------------------------------------------------------------- QIM
def qim_bit(c: float, delta: float = DELTA_DEFAULT) -> int:
    """QIM 判决, 与 run_signal_ra._demodulate_bit / r3-r6 一致."""
    r = c % delta
    return 1 if abs(r - delta / 2) < abs(r) else 0


def block_grid(shape) -> tuple:
    H, W = shape[0] // 8 * 8, shape[1] // 8 * 8
    return H, W, H // 8, W // 8


def embed_keyed(img: np.ndarray, key: bytes, delta: float = DELTA_DEFAULT,
                pay_coef=PAY_COEF_LOW, check_coef=CHECK_COEF_LOW,
                pattern: str = 'coord', msglen: int = MSGLEN_DEFAULT,
                msg: np.ndarray = None, keyed_perm: bool = False) -> np.ndarray:
    """双系数 QIM 嵌入 (payload + 校验), 期望序列来自 per-image 密钥."""
    H, W, n_h, n_w = block_grid(img.shape)
    y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)[:H, :W]
    ck = check_pattern(key, n_h, n_w, pattern)
    msg = payload_bits(key, msglen) if msg is None else np.asarray(msg, np.uint8)
    idx = _msg_index(key, n_h, n_w, msglen, keyed_perm).reshape(n_h, n_w)

    for i in range(n_h):
        for j in range(n_w):
            d = cv2.dct(y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8].astype(np.float32))
            c = d[pay_coef]
            d[pay_coef] = (np.floor(c / delta) * delta + delta / 2) if msg[idx[i, j]] == 1 \
                else np.round(c / delta) * delta
            c = d[check_coef]
            d[check_coef] = (np.floor(c / delta) * delta + delta / 2) if ck[i, j] == 1 \
                else np.round(c / delta) * delta
            y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = cv2.idct(d)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    out = ycrcb.copy()
    out[:H, :W, 0] = np.clip(y, 0, 255)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB).astype(np.uint8)


def fail_map_exp(img: np.ndarray, exp: np.ndarray, delta: float = DELTA_DEFAULT,
                 check_coef=CHECK_COEF_LOW) -> np.ndarray:
    """给定期望校验位矩阵, 返回逐块失败图 (1 = 校验失败 = 水印死亡).

    与旧实现 (r3–r6) 的判决逻辑逐字一致; 供旧方案对照与偏移搜索复用.
    """
    H, W, n_h, n_w = block_grid(img.shape)
    exp = np.asarray(exp)[:n_h, :n_w]
    y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)[:H, :W]
    fail = np.zeros((n_h, n_w), np.float32)
    for i in range(n_h):
        for j in range(n_w):
            c = cv2.dct(y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8])[check_coef]
            fail[i, j] = 1.0 if qim_bit(c, delta) != exp[i, j] else 0.0
    return fail


def fail_map(img: np.ndarray, key: bytes, delta: float = DELTA_DEFAULT,
             check_coef=CHECK_COEF_LOW, pattern: str = 'coord', off=(0, 0)) -> np.ndarray:
    """逐块校验失败图 (n_h, n_w) float32: 1 = 该块校验失败 (= 水印死亡)."""
    H, W, n_h, n_w = block_grid(img.shape)
    return fail_map_exp(img, check_pattern(key, n_h, n_w, pattern, off=off), delta, check_coef)


def check_fail_rate(img: np.ndarray, key: bytes, **kw) -> float:
    return float(fail_map(img, key, **kw).mean())


# ---------------------------------------------------------------- 指标
def auc(neg, pos) -> float:
    """AUC = P(pos > neg) (Mann-Whitney); 无 sklearn 依赖."""
    neg = np.asarray(neg, float).ravel()
    pos = np.asarray(pos, float).ravel()
    if len(neg) == 0 or len(pos) == 0:
        return float('nan')
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def smooth_fail_map(fail: np.ndarray, k: int = 5) -> np.ndarray:
    k = k if k % 2 == 1 else k + 1
    return cv2.blur(fail.astype(np.float32), (k, k))


def iou_at(pred_blocks: np.ndarray, gt_blocks: np.ndarray) -> float:
    """块级 IoU; pred/gt 均为 bool 块矩阵 (同尺寸)."""
    inter = float((pred_blocks & gt_blocks).sum())
    union = float((pred_blocks | gt_blocks).sum())
    return inter / union if union else 0.0
