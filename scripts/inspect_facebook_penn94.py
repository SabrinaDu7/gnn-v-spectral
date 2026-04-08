from pathlib import Path
import numpy as np
import scipy.io

path = Path("data/cache/realworld/facebook_penn94/Penn94.mat")
mat = scipy.io.loadmat(path)

print("keys:", sorted(mat.keys()))

A = mat["A"]
local_info = mat["local_info"]

print("A type:", type(A))
print("A shape:", A.shape)
print("local_info shape:", local_info.shape)
print("local_info dtype:", local_info.dtype)

print("\nFirst 5 rows of local_info:")
print(local_info[:5])

print("\nUnique counts per column (first up to 10 unique values shown):")
for j in range(local_info.shape[1]):
    vals = np.unique(local_info[:, j])
    print(f"col {j}: n_unique={len(vals)}, first_vals={vals[:10]}")