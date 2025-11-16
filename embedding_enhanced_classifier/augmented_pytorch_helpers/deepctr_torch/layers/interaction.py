"""Cross Network layer for Deep & Cross Network (DCN) architecture.

Modified from DeepCTR-Torch:
https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/layers/interaction.py

Licensed under Apache License 2.0.
"""
import torch
import torch.nn as nn


class CrossNet(nn.Module):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **in_features** : Positive integer, dimensionality of input features.
        - **layer_num**: Positive integer, the cross layer number
        - **parameterization**: string, ``"vector"``  or ``"matrix"`` ,  way to parameterize the cross network.
      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """

    def __init__(self, in_features, layer_num=2, parameterization='vector', device='cpu'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == 'vector':
            # weight in DCN.  (in_features, 1)
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))
        elif self.parameterization == 'matrix':
            # weight matrix in DCN-M.  (in_features, in_features)
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, in_features))
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")

        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias.shape[0]):
            nn.init.zeros_(self.bias[i])

        self.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Cross Network.

        Args:
            inputs: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, in_features) with learned
            feature crossings applied
        """
        # Add dimension for matrix operations: (batch, features) -> (batch, features, 1)
        x_0 = inputs.unsqueeze(2)
        x_l = x_0

        # Apply cross layers iteratively
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                # Vector parameterization (DCN): x_{l+1} = x_0 * (w_l^T * x_l) + b_l + x_l
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0])) # type: ignore
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == 'matrix':
                # Matrix parameterization (DCN-M): x_{l+1} = x_0 ⊙ (W_l * x_l + b_l) + x_l
                xl_w = torch.matmul(self.kernels[i], x_l)  # W * xi  (bs, in_features, 1)
                dot_ = xl_w + self.bias[i]  # W * xi + b
                x_l = x_0 * dot_ + x_l  # x0 ⊙ (W * xi + b) + xl  (Hadamard product)
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")

        # Remove extra dimension: (batch, features, 1) -> (batch, features)
        x_l = torch.squeeze(x_l, dim=2)
        return x_l

