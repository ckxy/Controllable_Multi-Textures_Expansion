import numpy as np
import torch
import os
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
from .vgg import VGG, GramMatrix, GramMSELoss


class MultiHalfGanModel(BaseModel):
    def name(self):
        return 'MultiHalfGanModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.n_pic = opt.n_pic

        if self.opt.lambda_style != 0:
            self.style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
            self.content_layers = ['r42']
            self.loss_fns = [GramMSELoss()] * len(self.style_layers)
            if torch.cuda.is_available():
                self.loss_fns = [loss_fn.cuda() for loss_fn in self.loss_fns]
            self.vgg = VGG()
            self.vgg.load_state_dict(torch.load(os.getcwd() + '/Models/' + 'vgg_conv.pth'))
            self.vgg.to(self.device)
            for param in self.vgg.parameters():
                param.requires_grad = False

            print(self.vgg.state_dict().keys())

            self.style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
            self.content_weights = [1e0]

        # load/define networks
        self.netG = networks.define_G(opt)
        if self.isTrain:
            self.netD = networks.define_D(opt)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionBCE = torch.nn.BCELoss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_A_star = input['A_star'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.input_BL = input['B_label'].to(self.device)
        self.input_BL = torch.squeeze(self.input_BL)

    def forward(self):
        self.fake_B = self.netG(self.real_A_star, self.real_A)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # TODO here we use real image to create fake_AB
        fake_AB = self.fake_B.clone()
        self.pred_fake, _ = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        real_AB = self.real_B.clone()
        self.pred_real, cT = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        self.loss_D_C = self.criterionBCE(cT, self.input_BL)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 + self.loss_D_C

        self.loss_D.backward()

    def backward_G(self):
        if self.opt.lambda_style != 0:
            style_targets = [GramMatrix()(A).detach() for A in self.vgg(self.real_B, self.style_layers)]
            out = self.vgg(self.fake_B, self.style_layers)
            layer_losses = [self.style_weights[a] * self.loss_fns[a](A, style_targets[a]) for a, A in enumerate(out)]
            # print(layer_losses)
            self.style_loss = sum(layer_losses) * self.opt.lambda_style
            self.style_loss.backward(retain_graph=True)
            self.style_loss_value = self.style_loss.item() / self.opt.lambda_style

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)

        # First, G(A) should fake the discriminator
        fake_AB = self.fake_B.clone()
        pred_fake, cF = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_C = self.criterionBCE(cF, self.input_BL)
        self.loss_G = self.loss_G_GAN + self.loss_G_C + self.loss_G_L1 * self.opt.lambda_l1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        od = OrderedDict([])
        od['D_real'] = self.loss_D_real.item()
        od['D_fake'] = self.loss_D_fake.item()
        od['G_GAN'] = self.loss_G_GAN.item()
        od['D_C'] = self.loss_D_C.item()
        od['G_C'] = self.loss_G_C.item()
        od['G_L1'] = self.loss_G_L1.item()
        if self.opt.lambda_style != 0:
            od['Style'] = self.style_loss_value
        return od

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A)
        fake_B = util.tensor2im(self.fake_B)
        real_B = util.tensor2im(self.real_B)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        # print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
