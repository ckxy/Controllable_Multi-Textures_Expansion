from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
import random
from util.opt_reader import opt_reader
import os


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        self.opt = opt
        BaseModel.initialize(self, opt)

        if opt.test_mode == 'v_train':
            opt.dataset_mode = 'sn2t'
            opt.phase = 'test_trainset'
        elif opt.test_mode == 'v_test':
            opt.dataset_mode = 'snt'
            opt.phase = 'test_testset'
        elif opt.test_mode == 'v_st':
            assert opt.times > 0
            opt.dataset_mode = 'n2t'
            opt.phase = 'stress_test_' + str(opt.fineSize)
        elif opt.test_mode == 'h_test':
            assert opt.which_content != ''
            assert opt.which_style != ''
            opt.dataset_mode = 'single'
            opt.phase = 'test_testset'
        else:
            raise ValueError("Test Mode [%s] not recognized." % opt.test_mode)

        opt_dir = os.path.join(opt.checkpoints_dir, opt.name)
        opt_dir = os.path.join(opt_dir, 'opt.txt')
        para_name = ['input_nc', 'n_pic', 'ngf', 'niter', 'niter_decay', 'no_dropout', 'norm', 'output_nc',
                     'padding_type', 'save_epoch_freq', 'which_model_netG']
        para_type = ['int', 'int', 'int', 'int', 'int', 'bool', 'str', 'int', 'str', 'int', 'str']
        train_opt = opt_reader(opt_dir, para_name, para_type)

        opt.nThreads = 1  # test code only supports nThreads = 1
        opt.batchSize = 1  # test code only supports batchSize = 1
        opt.serial_batches = True  # no shuffle
        opt.no_flip = True  # no flip

        # from train options
        opt.input_nc = train_opt['input_nc']
        opt.n_pic = train_opt['n_pic']
        opt.ngf = train_opt['ngf']
        niter = train_opt['niter']
        niter_decay = train_opt['niter_decay']
        opt.no_dropout = train_opt['no_dropout']
        opt.norm = train_opt['norm']
        opt.output_nc = train_opt['output_nc']
        opt.padding_type = train_opt['padding_type']
        save_epoch_freq = train_opt['save_epoch_freq']
        opt.which_model_netG = train_opt['which_model_netG']

        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        self.netG = networks.define_G(opt)
        self.netG.eval()

        self.results = OrderedDict([])
        self.h_count = ''

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

        if opt.test_mode == 'h_test':
            self.epochs = [str(i) for i in range(save_epoch_freq, niter + niter_decay + 1, save_epoch_freq)]
            print(self.epochs)
        else:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)

    def get_titles(self):
        if self.opt.test_mode == 'h_test':
            path = os.path.splitext(self.opt.which_content)[0].split('/')[-1] + \
                   '->' + os.path.splitext(self.opt.which_style)[0]
            t1 = '%s_%s' % (self.opt.phase, path)
            t2 = 'Experiment = %s, Phase = %s, Image = %s' % (self.opt.name, self.opt.phase, path)
        else:
            t1 = '%s_%s' % (self.opt.phase, self.opt.which_epoch)
            t2 = 'Experiment = %s, Phase = %s, Epoch = %s' % (self.opt.name, self.opt.phase, self.opt.which_epoch)
        return t1, t2

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        input_B = input['B']
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_path']

    def load_which_epoch(self, epoch):
        self.load_network(self.netG, 'G', epoch)

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        if self.opt.test_mode == 'v_train':
            print('process image... %s' % self.image_paths)
            self.results.clear()
            self.single_test()
        elif self.opt.test_mode == 'v_test':
            print('process image... %s' % self.image_paths)
            self.results.clear()
            self.single_test()
        elif self.opt.test_mode == 'v_st':
            print('process image... %s' % self.image_paths)
            self.results.clear()
            self.stress_test_up(self.opt.times)
        elif self.opt.test_mode == 'h_test':
            for i in range(len(self.epochs)):
                print('process epoch... %s' % self.epochs[i])
                self.load_which_epoch(self.epochs[i])
                self.set_h_count(self.epochs[i])
                self.single_test()

    def single_test(self):
        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_B = self.netG.forward(self.real_B, self.real_A)

        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        if self.opt.test_mode == 'h_test':
            if self.h_count == self.epochs[0]:
                self.results['real_A'] = real_A
            self.results['fake_B_' + self.h_count] = fake_B
        else:
            self.results['real_A'] = real_A
            self.results['fake_B'] = fake_B

    def set_h_count(self, hc):
        self.h_count = hc

    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        return self.results

    def stress_test_up(self, step=3):
        results = []
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_B = self.netG.forward(self.real_B, self.real_A)

        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        results.append(('real_{}_A'.format(0), real_A))
        results.append(('fake_{}_B'.format(0), fake_B))
        for i in range(1, step):
            self.real_A = Variable(self.fake_B.data, volatile=True)
            print(self.real_A.size())
            self.real_B = self.real_B.repeat(1, 1, 2, 2)
            self.fake_B = self.netG.forward(self.real_B, self.real_A)
            fake_B = util.tensor2im(self.fake_B.data)
            results.append(('fake_{}_B'.format(i), fake_B))

        self.results = OrderedDict(results).copy()
