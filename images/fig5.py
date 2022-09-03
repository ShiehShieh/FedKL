# coding: utf-8
import pandas as pd
df = pd.read_csv('fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics0.4-ec-seed0-da.csv', sep=',', header=None)
da = df.values
df = pd.read_csv('fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics0.4-ec-seed0-b.csv', sep=',', header=None)
b = df.values
df = pd.read_csv('fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-6e-1-1e-4-dynamics0.4-ec-seed0-evalh.csv', sep=',', header=None)
evalh = df.values
da_b = da-b
selected = []
for round_id in range(0, 200):
    np.random.seed(round_id + 0)
    indices = np.random.choice(64, 3, replace=False)
    selected.append(indices)
selected = np.array(selected)
out = []
for i in range(da_b.shape[0]):
    o = []
    for j in selected[i]:
        o.append(da_b[i, j])
    out.append(o)
out = np.array(out)
neg = []
for i, o in enumerate(out):
    if np.sum(o) < 0.0:
        neg.append(i)
    continue
    for oo in o:
        if oo < -0.0 and i not in neg:
            neg.append(i)
print(neg)
print(len(neg))
