# **SiamFC** Implementation

Personal implementation of the **fully-convolutional Siamese network**
architecture for **visual object tracking** known as **SiamFC**.

## Useful Links

[SiamFC using GOT10K dataset](https://github.com/got-10k/siamfc)

[SiamFC in PyTorch](https://github.com/huanglianghua/siamfc-pytorch/tree/master/siamfc)

[SiamFC in PyTorch using VOT](https://github.com/GengZ/siameseFC-pytorch-vot)

***

## To-Do List

A **list of tasks** to **implement** from the
original [SiamFC](https://arxiv.org/pdf/1606.09549.pdf) paper. It also contains
various **implementation details** to consider.

### Training

- [ ] Use **Xavier** weight initialization.
- [ ] Optimize using the **Stochastic Gradient Descent** algorithm.
- [ ] **Train** for $50$ epochs.
- [ ] Each **epoch** consists of $50\ 000$ **samples** (pairs).
- [ ] Use **mini-bathes** of size $8$.
- [ ] Anneal the **learning rate** geometrically at **every epoch** from
  $10^{-2}$ to $10^{-5}$.

### Tracking

- [ ] **Compute** the **initial embedding** $\varphi \left( z \right)$ only **
  once** at the beginning.
- [ ] **Compare** the $\varphi \left( z \right)$ with the subsequent frames **
  convolutionally**.
- [ ] **Upsample** the **score map** using **bicubic interpolation** from $17
  \times 17$ to $272 \times 272$ (upsampling factor is $\frac{272}{17} = 16$)
  for **improved localization accuracy** since the **score map** is
  relatively **coarse**.