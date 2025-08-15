import numpy as np, os
os.makedirs("data", exist_ok=True)
N, C, F, T, K = 200, 32, 4, 24, 5
rng = np.random.default_rng(0)
X = rng.normal(size=(N,C,F,T)).astype(np.float32)
y = rng.integers(0, K, size=N, endpoint=False)
# add a weak signal: band (y%F) gets a positive bias in the last third of time
for i in range(N):
    X[i, :, (y[i] % F), T//3:] += 0.35
np.save("data/X.npy", X)
np.save("data/y.npy", y)
print("Wrote data/X.npy", X.shape, "and data/y.npy", y.shape)