from __future__ import print_function
from math import log10

import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from monet import MoireCNN
from binet import BiCNN

from loadimg import ImageList
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from threshold import threshold_func

# Training settings
parser = argparse.ArgumentParser(description="PyTorch DemoireNet")
parser.add_argument("--batchSize", type=int, default=1, help="training batch size")
parser.add_argument('--testBatchSize', type=int, default=1, help='training batch size')
parser.add_argument("--nEpochs", type=int, default=1000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=40, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=250")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--seed", type=int, default=None, help="random seed")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.01, help="Clipping Gradients. Default=0.1")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=0.0001, type=float, help="weight decay, Default: 0")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def main():

    global opt, model, binet,training_data_loader, testing_data_loader ,testing_realdata_loader, writer
    opt = parser.parse_args()
    print(opt)    

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    if not opt.seed:
        opt.seed = random.randint(1, 10000)
        print("Random Seed: ", opt.seed)
        torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True
        
    print("===> Loading datasets")

    train_path_src = "../data/src/"
    train_path_tgt = "../data/tgt/"
    train_path_gt  = "../data/gt/"
    test_path_src  = "../data/src/"
    test_path_tgt  = "../data/tgt/"
    test_path_gt   = "../data/gt/"
    test_real_path_src  = "../data/src/"

    training_data_loader = torch.utils.data.DataLoader(
         ImageList(rootsour=train_path_src, roottar=train_path_tgt,
                   rootgt=train_path_gt,
                   transform=transforms.Compose([
                   transforms.CenterCrop(256),
                   transforms.ToTensor(),
                   ])),
         batch_size=opt.batchSize, shuffle=True)

    testing_data_loader = torch.utils.data.DataLoader(
        ImageList(rootsour=test_path_src, roottar=test_path_tgt,
                  rootgt=test_path_gt,
                  transform=transforms.Compose([
                      transforms.CenterCrop(256),
                      transforms.ToTensor(),
                  ])),
        batch_size=opt.testBatchSize, shuffle=False)

    testing_realdata_loader = torch.utils.data.DataLoader(
        ImageList(rootsour=test_real_path_src, roottar=None,
                  rootgt=None,
                  transform=transforms.Compose([
                      transforms.CenterCrop(256),
                      transforms.ToTensor(),
                  ]),
                  real_data = True),
        batch_size=opt.testBatchSize, shuffle=False)

    print("===> Building model")
    model = MoireCNN()
    binet = BiCNN()
    #binet._initialize_weights()
    # for layer, param in binet.state_dict().items():  # param is weight or bias(Tensor)
    #     print( layer, param)
    # for p in sub_module.parameters():
    #     p.requires_grad = False

    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        binet = binet.cuda()
        criterion = criterion.cuda()

    model.apply(weights_init)
    binet.apply(weights_init)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"])
            binet.load_state_dict(checkpoint["binet"])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    
    print("===> Setting Optimizer")
    #optimizer = optim.SGD(model.parameters(), lr=opt.lr,momentum=0.9, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    optimizer_bi = optim.Adam(binet.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        writer = SummaryWriter(log_dir='logs')
        train( optimizer,optimizer_bi, criterion, epoch)
        save_checkpoint(model, binet, epoch)
        if epoch%10==0:
           test( criterion, epoch)
           test_real(epoch)
    writer.close()


def adjust_learning_rate( epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr    

def train(optimizer,optimizer_bi, criterion, epoch):
    epoch_loss1, epoch_loss2= 0,0
    
    lr = adjust_learning_rate(epoch-1)
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr  

    print("epoch =", epoch,"lr =",optimizer.param_groups[0]["lr"])
    model.train()
    binet.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target,groundtruth= batch[0], batch[1],batch[2]
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
            groundtruth=groundtruth.cuda()


        output = model(input)
        output_detach = output.detach()
        binary = binet(output_detach)
        loss1 = criterion(output, target)
        epoch_loss1 += loss1.item()
        loss2=criterion(binary,groundtruth)
        epoch_loss2+=loss2.item()

        optimizer.zero_grad()
        optimizer_bi.zero_grad()
        loss1.backward()
        optimizer.step()
        loss2.backward()
        optimizer_bi.step()
        #nn.utils.clip_grad_norm_(model.parameters(),opt.clip)

        if iteration%50 == 0:
            writer.add_scalar('train_iterations_loss1', loss1.item(), epoch*iteration)
            writer.add_scalar('train_iterations_loss2', loss2.item(), epoch*iteration)
            loss_record="===> Epoch[{}]({}/{}): Loss1: {:.6f},loss2:{:.4f}".format(epoch, iteration, len(training_data_loader), loss1.item(),loss2.item())
            with open("train_loss_log.txt", "a") as train_log_file:
                train_log_file.write(loss_record + '\n')
            print(loss_record)
    writer.add_scalar('epoch_loss1', epoch_loss1, epoch)
    writer.add_scalar('epoch_loss2',epoch_loss2,epoch)
    epoch_loss_record="===>Training Epoch [{}] Complete: Avg. MSE Loss1: {:.6f},Loss2: {:.4f}".format(epoch, epoch_loss1 / len(training_data_loader),epoch_loss2/len(training_data_loader))
    with open("train_loss_log.txt", "a") as train_log_file:
        train_log_file.write(epoch_loss_record+ '\n')
    print(epoch_loss_record)
   
def test(criterion, epoch):
    avg_mse1, avg_mse2, avg_psnr =0,0,0
    #avg_ssim = 0
    print("===> Testing simulation images")
    if not os.path.exists("results/epoch_{}/".format(epoch)):
        os.makedirs("results/epoch_{}/".format(epoch))
    with torch.no_grad():
        for iteration, batch in enumerate(testing_data_loader, 1):
            input, target, groundtruth = batch[0], batch[1], batch[2]
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
                groundtruth=groundtruth.cuda()
            prediction = model(input)
            prediction_bi = binet(prediction)
            input_bi = binet(input)
            mse1 = criterion(prediction, target)
            mse2 =criterion(prediction_bi,groundtruth)
            psnr = 10 * log10(1 / mse1.item())
            #ssim = SSIM(prediction, target)
            avg_psnr += psnr
            #avg_ssim += ssim
            avg_mse1 += mse1.item()
            avg_mse2+=mse2.item()

            if epoch%1 == 0:
                vutils.save_image(prediction.data.cpu(),
                                  'results/epoch_{}/epoch_{}_img_{}_out.jpg'.format(epoch,epoch,iteration))
                vutils.save_image(input.data.cpu(),
                                 'results/epoch_{}/epoch_{}_img_{}_in.jpg'.format(epoch, epoch, iteration))
                vutils.save_image(target.data.cpu(),
                                 'results/epoch_{}/epoch_{}_img_{}_tgt.jpg'.format(epoch, epoch, iteration))
                vutils.save_image(input_bi.data.cpu(),
                                 'results/epoch_{}/epoch_{}_img_{}_bi_in.jpg'.format(epoch, epoch, iteration))
                vutils.save_image(prediction_bi.data.cpu(),
                                 'results/epoch_{}/epoch_{}_img_{}_bi_out.jpg'.format(epoch, epoch, iteration))
                vutils.save_image(groundtruth.data.cpu(),
                                 'results/epoch_{}/epoch_{}_img_{}_bi_gt.jpg'.format(epoch, epoch, iteration))

        test_loss_record="===>Testing Epoch[{}] Avg. PSNR: {:.4f} dB,  MSE1:{:.4f},MSE2:{:.4f}".format(epoch,
                                                                                            avg_psnr / len(testing_data_loader),
                                                                                         avg_mse1 / len(testing_data_loader),
                                                                                         avg_mse2/len(testing_data_loader))
        writer.add_scalar('test_epoch_psnr', avg_psnr, epoch)
        # writer.add_scalar('test_epoch_ssim', avg_ssim, epoch)
        writer.add_scalar('test_epoch_mse1', avg_mse1, epoch)
        writer.add_scalar('test_epoch_mse2',avg_mse2,epoch)
        print(test_loss_record)
        with open("test_loss_log.txt","a") as test_log_file:
            test_log_file.write(test_loss_record+ '\n')

def test_real(epoch):
    print("===> Testing Real images")
    if not os.path.exists("results_real/epoch_{}/".format(epoch)):
        os.makedirs("results_real/epoch_{}/".format(epoch))
    with torch.no_grad():
        for iteration, batch in enumerate(testing_realdata_loader, 1):
            input = batch
            if opt.cuda:
                input = input.cuda()
            prediction = model(input)
            input_bi = binet(input)
            prediction_bi = binet(prediction)
            if epoch%1 == 0:
                vutils.save_image(prediction.data.cpu(),
                                  'results_real/epoch_{}/epoch_{}_img_{}_out.jpg'.format(epoch,epoch,iteration))
                vutils.save_image(input.data.cpu(),
                                 'results_real/epoch_{}/epoch_{}_img_{}_in.jpg'.format(epoch, epoch, iteration))
                vutils.save_image(input_bi.data.cpu(),
                                 'results_real/epoch_{}/epoch_{}_img_{}_bi_in.jpg'.format(epoch, epoch, iteration))
                vutils.save_image(prediction_bi.data.cpu(),
                                 'results_real/epoch_{}/epoch_{}_img_{}_bi_out.jpg'.format(epoch, epoch, iteration))
    print("===> Done")

def save_checkpoint(model,binet, epoch):
    model_out_path = "model/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model.state_dict(),"binet":binet.state_dict()}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
