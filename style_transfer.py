import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
from PIL import Image
import argparse
from data.image_folder import make_dataset
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--dir', required=True, help='path to images')
# n2:n expansion references + nA2 transfer references.2n:n expansion references + n transfer references
parser.add_argument('--mode', type=str, default='n2', help='mode,n2 or 2n')
parser.add_argument('--max_iters', type=int, default=1000, help='max iters')
parser.add_argument('--show_iters', type=int, default=200, help='print loss per N iters')
parser.add_argument('--gpu_id', type=int, default=0, help='e.g. 0, -1 for cpu')
opt = parser.parse_args()

dir = os.path.join(opt.dir, 'train')
paths = make_dataset(dir)
paths = sorted(paths)
size = len(paths)
print(paths)

l = []

if opt.mode == 'n2':
    for i in itertools.product([i for i in range(size)], [i for i in range(size)]):
        if i[0] != i[1]:
            l.append(i)
elif opt.mode == '2n':
    for i in range(size - 1):
        l.append((i, i + 1))
    l.append((size - 1, 0))
else:
    raise ValueError("Mode [%s] not recognized." % opt.mode)

print(l)


# vgg definition that conveniently let's you grab the outputs from any layer
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2))
        G.div_(h*w)
        return G


class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return (out)


def postp(tensor):  # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1
    t[t<0] = 0
    img = postpb(t)
    return img

# pre and post processing for images
prep = transforms.Compose([transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ])
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],  # add imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
                           ])
postpb = transforms.Compose([transforms.ToPILImage()])

# get network
vgg = VGG()
vgg.load_state_dict(torch.load('./Models/vgg_conv.pth'))
vgg.eval()
device = torch.device('cuda:' + str(opt.gpu_id) if opt.gpu_id != -1 and torch.cuda.is_available() else 'cpu')
vgg = vgg.to(device)

# define layers, loss functions, weights and compute optimization targets
style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
content_layers = ['r42']
loss_layers = style_layers + content_layers
loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
if torch.cuda.is_available():
    loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

# these are good weights settings:
style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
content_weights = [1e0]
weights = style_weights + content_weights

for i in range(len(l)):
    print(str(l[i][0]) + '->' + str(l[i][1]))
    print(paths[l[i][0]] + '->' + paths[l[i][1]])

    # load images, ordered as [style_image, content_image]
    img_dirs = [paths[l[i][1]], paths[l[i][0]]]

    imgs = [Image.open(img_dirs[j]) for j, name in enumerate(img_dirs)]
    imgs_torch = [prep(img) for img in imgs]
    imgs_torch = [img.unsqueeze(0).to(device) for img in imgs_torch]
    style_image, content_image = imgs_torch
    print(style_image.size(), content_image.size())

    # opt_img = Variable(torch.randn(content_image.size()).type_as(content_image.data), requires_grad=True) #random init
    opt_img = content_image.clone()

    # compute optimization targets
    style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
    content_targets = [A.detach() for A in vgg(content_image, content_layers)]
    targets = style_targets + content_targets

    # run style transfer
    optimizer = optim.LBFGS([opt_img.requires_grad_()])
    n_iter = [0]

    while n_iter[0] <= opt.max_iters:

        def closure():
            optimizer.zero_grad()
            out = vgg(opt_img, loss_layers)
            layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
            loss = sum(layer_losses)
            loss.backward()
            n_iter[0] += 1
            # print loss
            if n_iter[0] % opt.show_iters == (opt.show_iters - 1):
                print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.item()))
                # print([loss_layers[li] + ': ' + str(l.data[0]) for li, l in enumerate(layer_losses)]) #loss of each layer
            return loss

        optimizer.step(closure)

    # display result
    out_img = postp(opt_img[0].cpu().squeeze())

    save_path = os.path.join(opt.dir, opt.mode + '_trans')
    save_path = os.path.join(save_path, str(l[i][1]))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    out_img.save(os.path.join(save_path, str(l[i][0]) + '.jpg'))
