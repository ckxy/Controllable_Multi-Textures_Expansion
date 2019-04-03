import torch
import torch.nn as nn


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, activation=nn.ReLU):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias,activation)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, activation):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        if activation == nn.LeakyReLU:
            conv_block += [activation(0.2, True)]
        else:
            conv_block += [activation(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class GramMatrix(nn.Module):
    def forward(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram


# class Inspiration(nn.Module):
#     """ Inspiration Layer (from MSG-Net paper)
#     tuning the featuremap with target Gram Matrix
#     ref https://arxiv.org/abs/1703.06953
#     """
#     def __init__(self, C, B=1):
#         super(Inspiration, self).__init__()
#         # B is equal to 1 or input mini_batch
#         self.weight = nn.Parameter(torch.Tensor(1, C, C), requires_grad=True)
#         # non-parameter buffer
#         self.G = Variable(torch.Tensor(B, C, C), requires_grad=True)
#         print('G', self.G.size())
#         self.C = C
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.weight.data.uniform_(0.0, 0.02)
#
#     def setTarget(self, target):
#         self.G = target
#
#     def forward(self, X):
#         # input X is a 3D feature map
#         self.P = torch.bmm(self.weight.expand_as(self.G), self.G)
#         print(self.G.size(), self.P.size(), X.size())
#         return torch.bmm(self.P.transpose(1, 2).expand(X.size(0), self.C, self.C),
#                          X.view(X.size(0), X.size(1), -1)).view_as(X)
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(' \
#             + 'N x ' + str(self.C) + ')'

class Inspiration(nn.Module):
    """ Inspiration Layer (from MSG-Net paper)
    tuning the featuremap with target Gram Matrix
    ref https://arxiv.org/abs/1703.06953
    """
    def __init__(self, C):
        super(Inspiration, self).__init__()
        # B is equal to 1 or input mini_batch
        self.weight = nn.Parameter(torch.Tensor(1, C, C), requires_grad=True)
        # non-parameter buffer
        self.C = C
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(0.0, 0.02)

    def forward(self, X, G):
        # input X is a 3D feature map
        self.P = torch.bmm(self.weight.expand_as(G), G)
        return torch.bmm(self.P.transpose(1, 2).expand(X.size(0), self.C, self.C),
                         X.view(X.size(0), X.size(1), -1)).view_as(X)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.C) + ')'
