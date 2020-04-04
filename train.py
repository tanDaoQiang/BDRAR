import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import joint_transforms
from config import sbu_training_root
from dataset import ImageFolder
from misc import AvgMeter, check_mkdir
from model import BDRAR

cudnn.benchmark = True

torch.cuda.set_device(0)

ckpt_path = 'G:\\shadow-remove\\BDRAR\\ckpt'
exp_name = 'BDRAR'

# batch size of 8 with resolution of 416*416 is exactly OK for the GTX 1080Ti GPU
args = {
    'iter_num': 800,
    'train_batch_size': 2,
    'last_iter': 0,
    'lr': 5e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 416
}

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((args['scale'], args['scale']))
])

val_joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((args['scale'], args['scale']))
])
#归一化
img_transform = transforms.Compose([
    #ToTensor()将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor，其将每一个数值归一化到[0,1]，其归一化方法比较简单，直接除以255即可
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

target_transform = transforms.ToTensor()
#将张量的每个元素乘上转化为[0,255]
to_pil = transforms.ToPILImage()

#sbu_training_root 训练集
#joint_transform  转化的大小
#img_transform 归一化
#target_transform
train_set = ImageFolder(sbu_training_root, joint_transform, img_transform, target_transform)

# dataset：加载的数据集(Dataset对象)
# batch_size：batch size
# shuffle:：是否将数据打乱
# sampler： 样本抽样，后续会详细介绍
# num_workers：使用多进程加载的进程数，0代表不使用多进程
# collate_fn： 如何将多个样本数据拼接成一个batch，一般使用默认的拼接方式即可
# pin_memory：是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
# drop_last：dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=True)

#交叉熵函数
bce_logit = nn.BCEWithLogitsLoss().cuda()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


def main():
    net = BDRAR().cuda().train()

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('training resumes from \'%s\'' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    print("---")
    print(str(args))
    print("---")
#    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)


def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:
        #全部初始化为0
        train_loss_record, loss_fuse_record, loss1_h2l_record = AvgMeter(), AvgMeter(), AvgMeter()
        loss2_h2l_record, loss3_h2l_record, loss4_h2l_record = AvgMeter(), AvgMeter(), AvgMeter()
        loss1_l2h_record, loss2_l2h_record, loss3_l2h_record = AvgMeter(), AvgMeter(), AvgMeter()
        loss4_l2h_record = AvgMeter()
        #遍历测试集
        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            #把梯度设置为0
            optimizer.zero_grad()

            fuse_predict, predict1_h2l, predict2_h2l, predict3_h2l, predict4_h2l, \
            predict1_l2h, predict2_l2h, predict3_l2h, predict4_l2h = net(inputs)

            loss_fuse = bce_logit(fuse_predict, labels)
            loss1_h2l = bce_logit(predict1_h2l, labels)
            loss2_h2l = bce_logit(predict2_h2l, labels)
            loss3_h2l = bce_logit(predict3_h2l, labels)
            loss4_h2l = bce_logit(predict4_h2l, labels)
            loss1_l2h = bce_logit(predict1_l2h, labels)
            loss2_l2h = bce_logit(predict2_l2h, labels)
            loss3_l2h = bce_logit(predict3_l2h, labels)
            loss4_l2h = bce_logit(predict4_l2h, labels)

            loss = loss_fuse + loss1_h2l + loss2_h2l + loss3_h2l + loss4_h2l + loss1_l2h + \
                   loss2_l2h + loss3_l2h + loss4_l2h
            loss.backward()

            optimizer.step()

            train_loss_record.update(loss.data, batch_size)
            loss_fuse_record.update(loss_fuse.data, batch_size)
            loss1_h2l_record.update(loss1_h2l.data, batch_size)
            loss2_h2l_record.update(loss2_h2l.data, batch_size)
            loss3_h2l_record.update(loss3_h2l.data, batch_size)
            loss4_h2l_record.update(loss4_h2l.data, batch_size)
            loss1_l2h_record.update(loss1_l2h.data, batch_size)
            loss2_l2h_record.update(loss2_l2h.data, batch_size)
            loss3_l2h_record.update(loss3_l2h.data, batch_size)
            loss4_l2h_record.update(loss4_l2h.data, batch_size)

            curr_iter += 1

            log = '[iter %d], [train loss %.5f], [loss_fuse %.5f], [loss1_h2l %.5f], [loss2_h2l %.5f], ' \
                  '[loss3_h2l %.5f], [loss4_h2l %.5f], [loss1_l2h %.5f], [loss2_l2h %.5f], [loss3_l2h %.5f], ' \
                  '[loss4_l2h %.5f], [lr %.13f]' % \
                  (curr_iter, train_loss_record.avg, loss_fuse_record.avg, loss1_h2l_record.avg, loss2_h2l_record.avg,
                   loss3_h2l_record.avg, loss4_h2l_record.avg, loss1_l2h_record.avg, loss2_l2h_record.avg,
                   loss3_l2h_record.avg, loss4_l2h_record.avg, optimizer.param_groups[1]['lr'])
            #print log
#            open(log_path, 'a').write(log + '\n')


            print(log)

            if curr_iter > args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                return


if __name__ == '__main__':
    main()
