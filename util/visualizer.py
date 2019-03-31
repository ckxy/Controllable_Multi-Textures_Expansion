import os
import ntpath
import time
from . import util
from . import html
from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np


class Visualizer():
    def __init__(self, opt):
        self.use_visdom = False

        if opt.isTrain:
            assert opt.n_save_images <= opt.batchSize
            self.n_save = opt.n_save_images
            self.new_loss_lists = False
            self.loss_lists = OrderedDict([])
            self.axes = OrderedDict([])
            self.lines = OrderedDict([])
            self.top_y = OrderedDict([])
            self.fig = plt.figure()
            self.x = None
            self.global_count = 0
            self.last_count = 0

            if opt.use_visdom:
                import visdom
                self.use_visdom = True
                self.vis = visdom.Visdom()

        self.use_html = opt.isTrain and not opt.no_html
        self.name = opt.name
        self.fineSize = opt.fineSize
        self.win_size = 256
        self.channels = opt.output_nc

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            self.plot_dir = os.path.join(self.web_dir, 'plots')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir, self.plot_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def set_x(self, opt, len_of_dataset):
        if opt.isTrain:
            self.x = np.arange((opt.niter + opt.niter_decay - opt.epoch_count + 1) *
                               len_of_dataset // opt.print_freq + 1)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, outer_i):
        if self.use_html: # save images to a html file
            bh, bw, _ = visuals['fake_B'][0].shape
            newIm = Image.new("RGB", ((bw + 5) * len(visuals) - 5, (bh + 5) * self.n_save - 5))
            i = 0
            for label, image_numpy in visuals.items():
                for col in range(self.n_save):
                    image = image_numpy[col].astype(np.uint8)
                    img = Image.fromarray(image)
                    w, h = img.size
                    newIm.paste(img, ((w + 5) * i - 5, col * (self.fineSize + 5) - 5))
                i = i + 1
            newIm.save(os.path.join(self.img_dir, 'epoch%.3d_%.3d.png' % (epoch, outer_i)))

            if self.use_visdom:
                vis_images = np.transpose(np.array(newIm), (2, 0, 1))
                self.vis.images(vis_images, win='results', opts={'title': 'results'})

    def print_current_errors(self, epoch, i, errors):
        if not self.new_loss_lists:
            count = 1

        message = '(epoch: %d, i: %d) ' % (epoch, i)

        for k, v in errors.items():
            if not self.new_loss_lists:
                self.loss_lists[k] = [None] * self.x.shape[0]
                self.top_y[k] = 0
                self.axes[k] = self.fig.add_subplot(len(errors), 1, count)
                self.axes[k].set_xlim([self.x[0], self.x[-1]])
                self.axes[k].set_ylim([0, 0])
                self.axes[k].set_title(k)
                temp_line, = self.axes[k].plot(self.x, self.loss_lists[k])
                self.lines[k] = temp_line
                count += 1

            self.loss_lists[k][self.global_count] = v
            if v != None:
                message += '%s: %.3f ' % (k, v)
            else:
                message += '%s: none ' % k
        self.global_count += 1

        if not self.new_loss_lists:
            self.new_loss_lists = True

        # print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def display_errors_plot(self):
        for k, v in self.loss_lists.items():
            max_y = max(self.loss_lists[k][self.last_count:self.global_count])
            if self.top_y[k] < max_y:
                self.top_y[k] = 1.05 * max_y
                self.axes[k].set_ylim([0, self.top_y[k]])
            self.lines[k].set_ydata(self.loss_lists[k]) 
        self.last_count = self.global_count
        plt.draw() 
        # plt.pause(0.01)
        plt.savefig(os.path.join(self.plot_dir, 'loss_plot.png'))

        if self.use_visdom:
            for k, v in self.loss_lists.items():
                vis_item = dict(x=self.x.tolist(), y=self.loss_lists[k], mode="lines", type='custom', name=k)
                layout = dict(title=k, xaxis={'title': 'epochs'}, yaxis={'title': 'loss'})
                self.vis._send({'data': [vis_item], 'layout': layout, 'win': k})

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            # print(save_path)
            util.save_image(image_numpy[0].astype(np.uint8), save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)

        webpage.add_images(ims, txts, links, width=self.win_size)

    def save_images_epoch(self, webpage, visuals, image_path, epoch):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header('%s Epoch: %s' % (name, epoch))
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s_%s.png' % (name, label, epoch)
            save_path = os.path.join(image_dir, image_name)
            print(save_path)
            util.save_image(image_numpy[0].astype(np.uint8), save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

    # save image to the disk
    def save_recurrent_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
