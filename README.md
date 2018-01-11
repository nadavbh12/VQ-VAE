# Basic VQ-VAE Example

This is a minimalistic implementation of the VQ-VAE (Vector Quantized Variational AutoEncoder)
 from [Neural Discrete representation learning](https://arxiv.org/pdf/1711.00937.pdf) for compressing MNIST and Cifar10.
The code is base upon [pytorch/examples/vae](https://github.com/pytorch/examples/tree/master/vae).

```bash
pip install -r requirements.txt
python main.py
```

# TODO: 
- [ ] Imporve results on cifar
- [ ] Merge VAE and VQ-VAE for MNIST and Cifar to one script

# Acknowledgement
[tf-vaevae](https://github.com/hiwonjoon/tf-vqvae) as a good reference.
