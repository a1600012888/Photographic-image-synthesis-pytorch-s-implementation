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

parser = argparse.ArgumentParser()
parser.add_argument('prefix', type = str)
parser.add_argument('--resume',default = None, type = str, help = 'the model to load')
parser.add_argument('--record_step', default=50, type = int, help = 'record loss, etc per ?')
parser.add_argument('--start_epoch', default = 0, type = int, help = 'start to train in epoch?')
args = parser.parse_args()
record_step = args.record_step
epoch_num = 200
learning_rate_policy = [[30, 1e-3],
                        [50, 1e-4],
                        [30, 1e-5],
                        [30, 3e-6]
                        ]
get_learing_rate = MultiStageLearningRatePolicy(learning_rate_policy)





def adjust_learning_rate(optimzier, epoch):
    #global get_lea
    lr = get_learing_rate(epoch)
    for param_group in optimizer.param_groups:

        param_group['lr'] = lr


ds_train = TorchDataset('train', 256)
ds_val = TorchDataset('val', 256)

step_per_epoch = ds_train.instance_per_epoch / 1

dl_train = DataLoader(ds_train, batch_size=1, shuffle=True, num_workers=6)
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
## init ??


#optimizer = torch.optim.SGD(net.parameters(), lr = 0.001, momentum=0.9, weight_decay=3e-4)
optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
vgg_perceptual_loss = vgg19(pretrained = True)

vgg_perceptual_loss.cuda()
vgg_perceptual_loss.eval()

clock = TrainClock()
clock.epoch = args.start_epoch
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

        inp = mn_batch['label']
        img = mn_batch['data']

        inp = inp.cuda()
        img = img.cuda()

        '''
        testing
        '''
        #print(inp.size())

        data_time_m.update(time.time() - start_time)

        optimizer.zero_grad()
        out = net(inp)

        loss = vgg_perceptual_loss(out, img)

        loss = loss.mean()
        epoch_loss.update(loss.item())

        loss.backward()

        optimizer.step()

        batch_time_m.update(time.time() - start_time)

        start_time = time.time()

        img.detach_()
        out.detach_()
        if clock.minibatch % record_step == 0:
            writer.add_scalar('Train/loss', loss.item(), clock.step // record_step)
            writer.add_image('Train/Raw_img', [img.cpu().numpy()[0]], clock.step // record_step)
            writer.add_image('Train/output', [out.cpu().numpy()[0]], clock.step // record_step)
        if clock.minibatch % 500 == 0:

            print(' ')
            print('epoch-{}, step-{}'.format(clock.epoch, clock.minibatch))

            print('Loss: {}'.format(epoch_loss.mean))

            print('Time usage: data time-{:.3f}, batch time-{:.3f}'.format(data_time_m.mean, batch_time_m.mean))

            print('This epoch has lasted {:.3f} mins, expect {:.3f} mins to run'.format((start_time - epoch_time)/60,
                                                                                (batch_time_m.mean * (step_per_epoch - clock.minibatch) / 60)))

    make_checkpoint(net.state_dict(), epoch_loss.mean, clock.epoch)




writer.close()
print('Training Finished!')
