import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataset import TorchDataset
from crn import CRN
import os
from my_snip.config import MultiStageLearningRatePolicy
from my_snip.clock import TrainClock, AvgMeter, TorchCheckpoint
from pvgg import vgg19
import time
from tqdm import tqdm
#from tensorboardX import SummaryWriter
from my_snip.tensorboard import TensorBoard as SummaryWriter   #Using my wrapper funciton
import argparse
from encoding.parallel import ModelDataParallel, CriterionDataParallel
from torch.nn import DataParallel
parser = argparse.ArgumentParser()
parser.add_argument('prefix', type = str)
parser.add_argument('--resume',default = None, type = str, help = 'the model to load')
parser.add_argument('--record_step', default=50, type = int, help = 'record loss, etc per ?')

args = parser.parse_args()

record_step = args.record_step
epoch_num = 180
learning_rate_policy = [[30, 1e-3],
                        [40, 1e-4],
                        [50, 1e-5],
                        [50, 3e-6]
                        ]
get_learing_rate = MultiStageLearningRatePolicy(learning_rate_policy)

gpu_ids = [0, 1, 2, 3]


def adjust_learning_rate(optimzier, epoch):
    #global get_lea
    lr = get_learing_rate(epoch)
    for param_group in optimizer.param_groups:

        param_group['lr'] = lr


ds_train = TorchDataset('train', 256)
ds_val = TorchDataset('val', 256)

batch_size = 4
step_per_epoch = ds_train.instance_per_epoch / batch_size

dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=batch_size * 6)
dl_val = DataLoader(ds_val, batch_size=1, shuffle=True, num_workers=12)
print('Dataloader ready!')

base_dir = os.path.join('./data', args.prefix)
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

log_dir = os.path.join('./logs', args.prefix)
writer = SummaryWriter(log_dir)

make_checkpoint = TorchCheckpoint(base_dir, high = False)



net = CRN(256)

if args.resume != None:
    assert os.path.exists(args.resume), 'model does not exist!'
    net.load_state_dict(torch.load(args.resume))
    print('Model loaded from {}'.format(args.resume))

net.cuda()
net = DataParallel(net, gpu_ids)#, output_device = gpu_ids[1])

#optimizer = torch.optim.SGD(net.parameters(), lr = 1e-3, momentum=0.9, weight_decay=1e-4)
optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
vgg_perceptual_loss = vgg19(pretrained = True)
vgg_perceptual_loss.cuda()
vgg_perceptual_loss.eval()
vgg_perceptual_loss = DataParallel(vgg_perceptual_loss, gpu_ids)#, gpu_ids[-1])

clock = TrainClock()

epoch_loss = AvgMeter('loss')
data_time_m = AvgMeter('data time')
batch_time_m = AvgMeter('train time')

net.train()
for e_ in range(epoch_num):

    epoch_loss.reset()
    data_time_m.reset()
    batch_time_m.reset()

    clock.tock()
    adjust_learning_rate(optimizer, clock.epoch)

    epoch_time = time.time()

    start_time = time.time()
    for i, mn_batch in tqdm(enumerate(dl_train)):

        clock.tick()

        inp = mn_batch['label'].cuda()
        img = mn_batch['data'].cuda(gpu_ids[-1])

        '''
        testing
        '''
        #print(inp.size())

        data_time_m.update(time.time() - start_time)

        optimizer.zero_grad()
        out = net(inp)
        #print(out.type())

        #vgg_perceptual_loss(out, img)
        loss = vgg_perceptual_loss(out, img)

        #print(loss, loss.type())
        loss = loss.mean()


        loss.backward()
        optimizer.step()

        epoch_loss.update(loss.cpu().item())
        batch_time_m.update(time.time() - start_time)

        start_time = time.time()
        img.detach_()
        out.detach_()
        if clock.minibatch % record_step == 0:
            writer.add_scalar('Train/loss', loss.item(), clock.step // record_step)
            writer.add_image('Train/Raw_img', [img.cpu().numpy()[0]], clock.step // record_step)
            writer.add_image('Train/output', [out.cpu().numpy()[0]], clock.step // record_step)
        if clock.minibatch % 500 == 0:


            print('epoch-{}, step-{}'.format(clock.epoch, clock.minibatch))

            print('Loss: {}'.format(epoch_loss.mean))

            print('Time usage: data time-{:.3f}, batch time-{:.3f}'.format(data_time_m.mean, batch_time_m.mean))

            print('This epoch has lasted {:.3f} mins, expect {:.3f} mins to run'.format((start_time - epoch_time)/60,
                                                                                (batch_time_m.mean * (step_per_epoch - clock.minibatch) / 60)))

    make_checkpoint(net.module.state_dict(), epoch_loss.mean, clock.epoch)




writer.close()
print('Training Finished!')
