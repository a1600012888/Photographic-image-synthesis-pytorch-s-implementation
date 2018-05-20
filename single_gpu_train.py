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
parser.add_argument('--record_step', default=200, type = int, help = 'record loss, etc per ?')
parser.add_argument('--start_epoch', default = 0, type = int, help = 'start to train in epoch?')
args = parser.parse_args()
record_step = args.record_step
epoch_num = 200
learning_rate_policy = [[30, 1e-2],
                        [70, 5e-4],
                        [20, 3e-5],
                        [30, 1e-5],
                        [20, 5e-6],
                        [20, 3e-6],
                        [10, 1e-6]
                        ]
get_learing_rate = MultiStageLearningRatePolicy(learning_rate_policy)


torch.backends.cudnn.benchmark = True


def adjust_learning_rate(optimzier, epoch):
    #global get_lea
    lr = get_learing_rate(epoch)
    for param_group in optimizer.param_groups:

        param_group['lr'] = lr


ds_train = TorchDataset('train', 256)
ds_val = TorchDataset('val', 256)

step_per_epoch = ds_train.instance_per_epoch / 1

dl_train = DataLoader(ds_train, batch_size=1, shuffle=True, num_workers=6)
dl_val = DataLoader(ds_val, batch_size=1, shuffle=True, num_workers=6)
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


optimizer = torch.optim.SGD(net.parameters(), lr = 0.001, momentum=0.9, weight_decay=0)
#optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
vgg_perceptual_loss = vgg19(pretrained = True)

vgg_perceptual_loss.cuda()
vgg_perceptual_loss.eval()
for p in vgg_perceptual_loss.parameters():
    p.requires_grad = False
clock = TrainClock()
clock.epoch = args.start_epoch
epoch_loss = AvgMeter('loss')
p0_loss = AvgMeter('conv_1_loss')
p1_loss = AvgMeter('conv_2_loss')
p2_loss = AvgMeter('conv_3_loss')
p3_loss = AvgMeter('conv_4_loss')
p4_loss = AvgMeter('conv_5_loss')
p5_loss = AvgMeter('conv_55_loss')
data_time_m = AvgMeter('data time')
batch_time_m = AvgMeter('train time')

for e_ in range(epoch_num):

    net.train()
    epoch_loss.reset()
    p0_loss.reset()
    p1_loss.reset()
    p2_loss.reset()
    p3_loss.reset()
    p4_loss.reset()
    p5_loss.reset()

    data_time_m.reset()
    batch_time_m.reset()

    clock.tock()
    adjust_learning_rate(optimizer, clock.epoch)

    epoch_time = time.time()

    start_time = time.time()

    for i, mn_batch in tqdm(enumerate(dl_train)):

        #if i > 20:
         #   break

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

        loss, perceptual = vgg_perceptual_loss(out, img, inp)
        #loss = vgg_perceptual_loss(out, img, inp)

        loss = loss.mean()
        #print(perceptual.size())
        loss.backward()

        #print(net.refine_block0.conv1.weight.grad)

        #print(net.refine_block5.conv1.weight.grad)
        optimizer.step()

        epoch_loss.update(float(loss.item()))

        #print(perceptual.size())
        perceptual = perceptual.mean(dim = 0)

        perceptual = perceptual.mean(dim = -1)
        perceptual = perceptual.mean(dim = 1) #
        p0_loss.update(float(perceptual[0].item()))
        p1_loss.update(float(perceptual[1].item()))
        p2_loss.update(float(perceptual[2].item()))
        p3_loss.update(float(perceptual[3].item()))
        p4_loss.update(float(perceptual[4].item()))
        p5_loss.update(float(perceptual[5].item()))

        batch_time_m.update(time.time() - start_time)

        start_time = time.time()
        loss.detach_()
        img.detach_()
        out.detach_()
        inp.detach_()
        if clock.minibatch % record_step == 0:
            writer.add_scalar('Train/loss', loss.item(), clock.step // record_step)
            writer.add_image('Train/Raw_img', [img.cpu().numpy()[0]], clock.step // record_step)
            writer.add_image('Train/output', [out.cpu().numpy()[0]], clock.step // record_step)
        if clock.minibatch % 200 == 0:

            print('epoch-{}, step-{}'.format(clock.epoch, clock.minibatch))

            print('Loss: {}'.format(epoch_loss.mean))

            print('Time usage: data time-{:.3f}, batch time-{:.3f}'.format(data_time_m.mean, batch_time_m.mean))

            print('This epoch has lasted {:.3f} mins, expect {:.3f} mins to run'.format((start_time - epoch_time)/60,
                                                                                (batch_time_m.mean * (step_per_epoch - clock.minibatch) / 60)))


    writer.add_scalar('Train/Epoch_loss', epoch_loss.mean, clock.epoch)
    writer.add_scalar('Train/Epoch_loss_p0', p0_loss.mean, clock.epoch)
    writer.add_scalar('Train/Epoch_loss_p1', p1_loss.mean, clock.epoch)
    writer.add_scalar('Train/Epoch_loss_p2', p2_loss.mean, clock.epoch)
    writer.add_scalar('Train/Epoch_loss_p3', p3_loss.mean, clock.epoch)
    writer.add_scalar('Train/Epoch_loss_p4', p4_loss.mean, clock.epoch)
    writer.add_scalar('Train/Epoch_loss_p5', p5_loss.mean, clock.epoch)
    make_checkpoint(net.state_dict(), epoch_loss.mean, clock.epoch)

    epoch_loss.reset()
    p0_loss.reset()
    p1_loss.reset()
    p2_loss.reset()
    p3_loss.reset()
    p4_loss.reset()
    p5_loss.reset()


    optimizer.zero_grad()

    net.eval()
    torch.cuda.empty_cache()
    print('Validation begin')
    val_time = time.time()
    for i, mn_batch in tqdm(enumerate(dl_val)):

        inp = mn_batch['label']
        img = mn_batch['data']

        inp = inp.cuda()
        img = img.cuda()
        out = net(inp)

        loss, perceptual = vgg_perceptual_loss(out, img, inp)
        #loss = vgg_perceptual_loss(out, img, inp)
        loss = loss.mean()
        epoch_loss.update(loss.item())

        #print(perceptual.size())
        perceptual = perceptual.mean(dim = 0)
        perceptual = perceptual.mean(dim = -1)
        perceptual = perceptual.mean(dim=1)  #
        p0_loss.update(perceptual[0].item())
        p1_loss.update(perceptual[1].item())
        p2_loss.update(perceptual[2].item())
        p3_loss.update(perceptual[3].item())
        p4_loss.update(perceptual[4].item())
        p5_loss.update(perceptual[4].item())

        loss.detach_()
        img.detach_()
        out.detach_()
        inp.detach_()
        if i % 50 == 0:
            writer.add_image('Val/Raw_img', [img.cpu().numpy()[0]], clock.epoch * 11 + i)
            writer.add_image('Val/output', [out.cpu().numpy()[0]], clock.epoch * 11 + i)

    writer.add_scalar('Val/Epoch_loss', epoch_loss.mean, clock.epoch)

    print('Validation finished.   Lasting {:.2f} mins'.format((time.time() - val_time) / 60))
    print('Validation loss: {:.3f}'.format(epoch_loss.mean))
    writer.add_scalar('Val/Epoch_loss_p0', p0_loss.mean, clock.epoch)
    writer.add_scalar('Val/Epoch_loss_p1', p1_loss.mean, clock.epoch)
    writer.add_scalar('Val/Epoch_loss_p2', p2_loss.mean, clock.epoch)
    writer.add_scalar('Val/Epoch_loss_p3', p3_loss.mean, clock.epoch)
    writer.add_scalar('Val/Epoch_loss_p4', p4_loss.mean, clock.epoch)
    writer.add_scalar('Val/Epoch_loss_p5', p5_loss.mean, clock.epoch)
writer.close()
print('Training Finished!')
