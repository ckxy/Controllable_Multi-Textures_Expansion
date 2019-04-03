import torch
import torch.nn as nn
import functools
from .block import ResnetBlock, Inspiration, GramMatrix


###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(opt):
    which_model_netG = opt.which_model_netG
    gpu_ids = opt.gpu_ids

    netG = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netG == 'resnet_2x' or which_model_netG == 'resnet_2x_6blocks':
        netG = MultiResnet2XGenerator(opt)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)

    if use_gpu:
        netG.to(gpu_ids[0])
        netG = nn.DataParallel(netG, gpu_ids)
    netG.apply(weights_init)
    return netG

def define_D(opt):
    which_model_netD = opt.which_model_netD
    gpu_ids = opt.gpu_ids

    netD = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(opt)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)

    if len(gpu_ids) > 0:
        netD.to(gpu_ids[0])
        netD = nn.DataParallel(netD, gpu_ids)
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class MultiResnet2XGenerator(nn.Module):
    def __init__(self, opt):
        input_nc = opt.input_nc
        output_nc = opt.output_nc
        ngf = opt.ngf
        norm_layer = get_norm_layer(norm_type=opt.norm)
        use_dropout = not opt.no_dropout
        padding_type = opt.padding_type
        n_blocks = opt.nrb

        super(MultiResnet2XGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # image
        if padding_type == 'reflect':
            model = [nn.ReflectionPad2d(3)]
        elif padding_type == 'replicate':
            model = [nn.ReplicationPad2d(3)]
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        model += [
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        self.encoder = nn.Sequential(*model)

        self.gram = GramMatrix()
        self.MVC = Inspiration(ngf * 2 ** n_downsampling)

        # decoder
        model = []

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, padding=1, bias=use_bias),
                  norm_layer(ngf * mult * 2),
                  nn.ReLU(True)]

        n_upsampling = 3

        for i in range(n_upsampling):
            mult = 2 ** (n_upsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        if padding_type == 'reflect':
            model += [nn.ReflectionPad2d(3)]
        elif padding_type == 'replicate':
            model += [nn.ReplicationPad2d(3)]
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.decoder = nn.Sequential(*model)

    def forward(self, style, content):
        es = self.encoder(style)
        ec = self.encoder(content)
        # self.MVC.setTarget(self.gram(es))
        # e = self.MVC(ec)
        e = self.MVC(ec, self.gram(es))
        return self.decoder(e)


class NLayerDiscriminator(nn.Module):
    def __init__(self, opt):
        super(NLayerDiscriminator, self).__init__()
        input_nc = opt.input_nc
        self.n_pic = opt.n_pic
        ndf = opt.ndf
        n_layers = opt.n_layers_D
        norm_layer = get_norm_layer(norm_type=opt.norm)
        use_sigmoid = opt.no_lsgan

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                # here we change original stride=2 to stride=1, we think high resolution helps for detailed decision
                # nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                #           kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        self.model = nn.Sequential(*sequence)

        sequence = [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.dis = nn.Sequential(*sequence)

        amp = [nn.AdaptiveMaxPool2d(1)]
        self.amp = nn.Sequential(*amp)

        cla = [nn.Linear(ndf * nf_mult, self.n_pic), nn.Softmax(1)]
        self.cla = nn.Sequential(*cla)

    def forward(self, input):
        z = self.model(input)
        pz = self.amp(z)
        pz = pz.view(pz.size(0), -1)
        return self.dis(z), self.cla(pz)
