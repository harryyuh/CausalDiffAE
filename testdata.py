import numpy as np
from improved_diffusion.image_datasets import load_morphomnist_like

root = "./datasets/morphomnist"
images, labels, metrics = load_morphomnist_like(
    root_dir=root,
    train=True,
    columns=["thickness", "slant"]
)

for col in ["thickness", "slant"]:
    arr = metrics[col].to_numpy()
    print(
        col,
        "min =", arr.min(),
        "max =", arr.max(),
        "mean =", arr.mean(),
        "std =", arr.std(),
        "q05 =", np.quantile(arr, 0.05),
        "q50 =", np.quantile(arr, 0.50),
        "q95 =", np.quantile(arr, 0.95),
    )