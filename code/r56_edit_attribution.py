#!/usr/bin/env python3
"""r56: 编辑类型归因 (edit-type attribution) — 回答"改了什么".

任务: 编辑图 -> {remove, replace, add} 三类
对照: RGB-only vs RGB+failure-map (检验失效图对归因的贡献)
数据: bigset 4000 (英文指令解析) + attackset 1200 (中文指令解析), 标签自动解析零人工
输出: results/r56_edit_attribution.json
"""
import sys, os, json, re, time, collections
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{BASE}/code')
import numpy as np, cv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import keyed_scheme as ks
import r8_geometric as r8
from r14b_multiscale import layer_key
from r7_keyed_scheme import paste_edit
from r34_distill_matrix import ConvBlock, DELTA, CHK, MASTER

AS = ('/media/oyp/工作/Projects/042_image_forensic/watermark-mllm-robustness'
      '/results/attackset_20261001_222447')
BIG = f'{BASE}/data/bigset'
CLS = ['remove', 'replace', 'add']


def label_of(v):
    if not isinstance(v, str):
        return None
    v0 = v.strip()
    if v0.startswith('{'):
        m = re.search(r'"instruction"\s*:\s*"([^"]+)"', v0)
        v0 = m.group(1) if m else ''
    lo = v0.lower()
    if re.search(r'删除|移除|去掉|抹掉', v0) or lo.startswith('remove'):
        return 'remove'
    if re.search(r'替换|换成|变成|改为', v0) or lo.startswith(('replace', 'change')):
        return 'replace'
    if re.search(r'添加|加上|增加|放入', v0) or lo.startswith('add'):
        return 'add'
    return None


def build(root, limit=None):
    """-> list of (sid, edited_img, fail, label)"""
    meta = json.load(open(f'{root}/meta.json'))
    out = []
    t0 = time.time()
    for s_ in meta['samples']:
        lab = label_of(s_.get('instruction'))
        if lab is None:
            continue
        sid = s_['id']
        fo, fa = f'{root}/images/orig_{sid}.png', f'{root}/images/att_{sid}.png'
        if not (os.path.exists(fo) and os.path.exists(fa)):
            continue
        o = r8.crop8(np.array(Image.open(fo).convert('RGB')))
        a = r8.crop8(np.array(Image.open(fa).convert('RGB')))
        key = ks.image_key(MASTER, sid)
        k4 = layer_key(key, 'bs4')
        from r14b_multiscale import embed_bs
        wm = ks.embed_keyed(o, key, DELTA, (1, 0), CHK, pattern='coord')
        wm = embed_bs(wm, k4, 4, DELTA, (1, 1))
        edited, _ = paste_edit(wm, a, s_['bbox'])
        f8 = ks.fail_map(edited, key, DELTA, CHK, 'coord')
        fail = np.kron(ks.smooth_fail_map(f8, 5), np.ones((8, 8), np.float32))
        out.append((sid, edited, np.clip(fail, 0, 1), lab))
        if limit and len(out) >= limit:
            break
    print(f'{root.split("/")[-1]}: {len(out)} ({time.time()-t0:.0f}s) '
          f'{dict(collections.Counter(x[3] for x in out))}', flush=True)
    return out


class AttrNet(nn.Module):
    """4 通道 (RGB + fail) -> 3 类; rgb_only 时 3 通道."""
    def __init__(self, cin=4, base=24):
        super().__init__()
        self.b1 = ConvBlock(cin, base); self.b2 = ConvBlock(base, base*2)
        self.b3 = ConvBlock(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(base*4, 128), nn.ReLU(inplace=True),
                                  nn.Dropout(0.2), nn.Linear(128, 3))
    def forward(self, x):
        x = self.b1(x); x = self.b2(self.pool(x)); x = self.b3(self.pool(x))
        return self.head(x)


class DS(Dataset):
    def __init__(self, items, use_fail=True):
        self.items, self.use_fail = items, use_fail
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        sid, im, fl, lab = self.items[i]
        ch = [im.astype(np.float32) / 255.]
        if self.use_fail:
            ch.append(fl[..., None])
        x = np.concatenate(ch, -1).transpose(2, 0, 1)
        return torch.from_numpy(x).float(), torch.tensor(CLS.index(lab), dtype=torch.long)


def run(tr, te, use_fail, epochs=25, tag=''):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cin = 4 if use_fail else 3
    dl = DataLoader(DS(tr, use_fail), batch_size=32, shuffle=True, drop_last=True)
    # 类权重 (不均衡)
    cnt = collections.Counter(x[3] for x in tr)
    w = torch.tensor([len(tr) / (3 * cnt[c]) for c in CLS], dtype=torch.float, device=dev)
    m = AttrNet(cin=cin).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    lossf = nn.CrossEntropyLoss(weight=w)
    for ep in range(epochs):
        m.train(); tot = 0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); lo = lossf(m(xb), yb); lo.backward(); opt.step()
            tot += lo.item()
        if (ep + 1) % 5 == 0:
            print(f'  [{tag}] ep{ep+1} loss={tot/len(dl):.3f}', flush=True)
    m.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(te), 64):
            xb = torch.stack([DS(te, use_fail)[j][0] for j in range(i, min(i+64, len(te)))]).to(dev)
            preds.append(m(xb).argmax(1).cpu().numpy())
    p = np.concatenate(preds)
    gt = np.array([CLS.index(x[3]) for x in te])
    acc = float((p == gt).mean())
    M = np.zeros((3, 3), int)
    for a_, b_ in zip(gt, p):
        M[a_, b_] += 1
    per = {c: round(float(M[i, i] / max(1, M[i].sum())), 3) for i, c in enumerate(CLS)}
    print(f'  [{tag}] acc={acc:.3f} per-class={per}', flush=True)
    return acc, per, M.tolist()


def main():
    big = build(BIG)
    att = build(AS)
    # 三分: bigset 前 3200 训练, 后 800 测试; attackset 作跨分布测试
    tr = big[:3200]
    te = big[3200:]
    out = {'meta': dict(n_train=len(tr), n_test_in=len(te), n_test_cross=len(att),
                        classes=CLS, epochs=25,
                        labels='parsed from instruction text (zero manual annotation)'),
           'in_domain': {}, 'cross_domain': {}}
    for tag, uf in (('rgb_only', False), ('rgb_plus_fail', True)):
        acc, per, M = run(tr, te, uf, tag=tag)
        out['in_domain'][tag] = {'acc': round(acc, 4), 'per_class': per, 'matrix': M}
        acc2, per2, M2 = run(tr, att, uf, tag=tag + '_cross')
        out['cross_domain'][tag] = {'acc': round(acc2, 4), 'per_class': per2, 'matrix': M2}
    json.dump(out, open(f'{BASE}/results/r56_edit_attribution.json', 'w'), indent=1)
    print('-> results/r56_edit_attribution.json')


if __name__ == '__main__':
    main()
