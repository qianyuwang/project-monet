from __future__ import print_function
from math import log10

import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy

import torch.utils.data
import torchvision.transforms as transforms
from monet import MoireCNN

from SaveImage import save_images
from metrics import SSIM
from loadimg import ImageList

from tensorboardX import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description="PyTorch DemoireNet")
parser.add_argument("--batchSize", type=int, default=1, help="training batch size")
parser.add_argument('--testBatchSize', type=int, default=1, help='training batch size')
parser.add_argument("--nEpochs", type=int, default=1000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=1000, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=250")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--is_debug", action="store_true", help="Use debug path?")
parser.add_argument("--seed", type=int, default=None, help="random seed")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.01, help="Clipping Gradients. Default=0.1")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=0, type=float, help="weight decay, Default: 0")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")


def main():

    global opt, model, training_data_loader, testing_data_loader ,writer
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

    train_path_src = "/home/diplab/Documents/demoire2/traindata/src/"
    train_path_tgt = "/home/diplab/Documents/demoire2/traindata/tgt/"
    train_path_gt="/home/diplab/Documents/demoire2/traindata/gt/"
    test_path_src = "/home/diplab/Documents/demoire2/testdata/src/"
    test_path_tgt = "/home/diplab/Documents/demoire2/testdata/tgt/"
    test_path_gt="/home/diplab/Documents/demoire2/testdata/gt/"
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


    print("===> Building model")
    model = MoireCNN()
    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()                         #这两句是把网络和loss放在cuda上
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):  #检验是否存在opt.resume
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    
    print("===> Setting Optimizer")
    #optimizer = optim.SGD(model.parameters(), lr=opt.lr,momentum=0.9, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        writer = SummaryWriter(log_dir='logs')
        train( optimizer, criterion, epoch)
        save_checkpoint(model, epoch)
        if epoch%10==0:
           test( criterion, epoch)
    writer.close()


def adjust_learning_rate( epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr    

def train(optimizer, criterion, epoch):
    epoch_loss1, epoch_loss2= 0,0
    
    lr = adjust_learning_rate(epoch-1)
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr  

    print("epoch =", epoch,"lr =",optimizer.param_groups[0]["lr"])
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target,groundtruth= batch[0], batch[1],batch[2]
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
            groundtruth=groundtruth.cuda()                         
        output = model(input)         
        convert=output.mul(255).byte()
        convert=convert.cpu().numpy().squeeze(0).transpose((1,2,0))    #将output这个tensor转换成numpy的数组并且transpose成opencv可以接收的格式          
        #convert=output.data.cpu().numpy()
        im_gray=cv2.cvtColor(convert,cv2.COLOR_BGR2GRAY)#转成灰度图
        binary=cv2.adaptiveThreshold(im_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,15)#二值化处理       
        binary = binary[numpy.newaxis,numpy.newaxis,:,:]
        binary=torch.from_numpy(binary).float().cuda()   #numpy转tensor
        
        #groundtruth进行和output一样的操作  
        #turn=groundtruth.mul(255).byte()
        #turn= turn.cpu().numpy().squeeze(0).transpose((1,2,0))  #将groundtruth从tensor转换成numpy
        #gray=cv2.cvtColor( turn,cv2.COLOR_BGR2GRAY)#转成灰度图
        #after=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,15)#二值化处理       
        #after = after[numpy.newaxis,numpy.newaxis,:,:]
        #after=torch.from_numpy(after).float()   #numpy转tensor     
      
        #output=nn.parallel.data_parallel(model,input,range(2))

        loss1 = criterion(output, target)
        epoch_loss1 += loss1.item()
        loss2=criterion(binary,groundtruth)
        epoch_loss2+=loss2.item()
        optimizer.zero_grad()  
        loss1.backward()
        nn.utils.clip_grad_norm_(model.parameters(),opt.clip)
        optimizer.step()

        if iteration%10 == 0:
            writer.add_scalar('train_iterations_loss1', loss1.item(), epoch*iteration)
            writer.add_scalar('train_iterations_loss2', loss2.item(), epoch*iteration)
            loss_record="===> Epoch[{}]({}/{}): Loss1: {:.4f},loss2:{:.4f}".format(epoch, iteration, len(training_data_loader), loss1.item(),loss2.item())
            with open("train_loss_log.txt", "a") as train_log_file:
                train_log_file.write(loss_record + '\n')
            print(loss_record)
    writer.add_scalar('epoch_loss1', epoch_loss1, epoch)
    writer.add_scalar('epoch_loss2',epoch_loss2,epoch)
    epoch_loss_record="===>Training Epoch [{}] Complete: Avg. MSE Loss1: {:.4f},Loss2: {:.4f}".format(epoch, epoch_loss1 / len(training_data_loader),epoch_loss2/len(training_data_loader))
    with open("train_loss_log.txt", "a") as train_log_file:
        train_log_file.write(epoch_loss_record+ '\n')
    print(epoch_loss_record)
   
def test(criterion, epoch):
    avg_mse1=0
    avg_mse2=0
    avg_psnr = 0
    #avg_ssim = 0
    print("===> Testing")
    with torch.no_grad():
        for iteration, batch in enumerate(testing_data_loader, 1):
            input, target, groundtruth = batch[0], batch[1], batch[2]
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
                groundtruth=groundtruth.cuda()
            prediction = model(input)
            predic=prediction.mul(255).byte()      #将prediction从tensor转换成一个numpy数组
            Num=predic.cpu().numpy().squeeze(0).transpose((1,2,0))
            IM_GRAY=cv2.cvtColor(Num,cv2.COLOR_BGR2GRAY)#转成灰度图
            BINARY=cv2.adaptiveThreshold(IM_GRAY,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,5)

            BINARY = BINARY[numpy.newaxis,numpy.newaxis,:,:]   #numpy转tensor
            BINARY=torch.from_numpy(BINARY).float().cuda() #将tensor从byte转成float
            #prediction=nn.parallel.data_parallel(model,input,range(2))
            Turn=input.mul(255).byte()
            Turn= Turn.cpu().numpy().squeeze(0).transpose((1,2,0)) 
            Gray=cv2.cvtColor( Turn,cv2.COLOR_BGR2GRAY)#转成灰度图
            After=cv2.adaptiveThreshold( Gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,15)#二值化处理       
            After = After[numpy.newaxis,numpy.newaxis,:,:]
            input_binary=torch.from_numpy(After).float().cuda()  #numpy转tensor
      
            mse1 = criterion(prediction, target)
            mse2 =criterion(BINARY,groundtruth)
            psnr = 10 * log10(1 / mse1.item())
            #ssim = SSIM(prediction, target)
            avg_psnr += psnr
            #avg_ssim += ssim
            avg_mse1 += mse1.item()
            avg_mse2+=mse2.item()

            if epoch%10 == 0:
                save_images(epoch,prediction,'epoch_{}_img_{}_out.jpg'.format(epoch,iteration),1)
                save_images(epoch,input,'epoch_{}_img_{}_in.jpg'.format(epoch,iteration),1)
                save_images(epoch,BINARY,'epoch_{}_img_{}_binary_out.jpg'.format(epoch,iteration),1)
                save_images(epoch,target,'epoch_{}_img_{}_tgt.jpg'.format(epoch,iteration),1)
                save_images(epoch,groundtruth,'epoch_{}_img_{}_binary_gt.jpg'.format(epoch,iteration),1)
                save_images(epoch, input_binary,'epoch_{}_img_{}_binary_in.jpg'.format(epoch,iteration),1)
                #prediction_output_filename= "result/prediction_{}.jpg".format(batch)
                #prediction.save(prediction_output_filename)
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

    
def save_checkpoint(model, epoch):
    model_out_path = "model/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model.state_dict()}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
