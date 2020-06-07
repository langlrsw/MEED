# MEED for explaining 2D CNN on Fashion-MNIST

## Download data

Download `fashion_mnist_aligned.npz` in https://drive.google.com/open?id=1zoaFEf8AVZUD0zc-g2CZl8Er3YwYuqjI

And put `fashion_mnist_aligned.npz` in the `data/` folder

## Run the codes

If use the wasserstein distance as L_u, run the following commands in shell:

```shell
python train_with_wasserstein_distance.py
```

If use the cross entropy as L_u, run the following commands in shell:

```shell
python train_with_cross_entropy.py
```


See `train_with_wasserstein_distance.py` or `train_with_cross_entropy.py` for details. 
