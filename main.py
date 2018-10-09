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

from SaveImage import save_images
from metrics import SSIM
from loadimg import ImageList

# Training settings
parser = argparse.ArgumentParser(description="PyTorch DemoireNet")
parser.add_argument("--batchSize", type=int, default=1, help="training batch size")
parser.add_argument('--testBatchSize', type=int, default=1, help='training batch size')
parser.add_argument("--nEpochs", type=int, default=1000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=1000, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=250")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--is_debug", action="store_true", help="Use debug path?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.01, help="Clipping Gradients. Default=0.1")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=0, type=float, help="weight decay, Default: 0")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")


def main():

    global opt, model, training_data_loader, testing_data_loader
    opt = parser.parse_args()
    print(opt)    

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True
        
    print("===> Loading datasets")
    if opt.is_debug:
        train_path_src = "/home/wqy/Documents/moire-test/source/"
        train_path_tgt = "/home/wqy/Documents/moire-test/target/"
        test_path_src = "/home/wqy/Documents/moire-test/source/"
        test_path_tgt = "/home/wqy/Documents/moire-test/target/"
        training_data_loader = torch.utils.data.DataLoader(
            ImageList(rootsour=train_path_src, roottar=train_path_tgt,
                      transform=transforms.Compose([
                          transforms.RandomCrop(64, pad_if_needed=True),
                          transforms.RandomHorizontalFlip(),
                          transforms.FiveCrop(64),
                          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                      ])),
            batch_size=opt.batchSize, shuffle=True)

        testing_data_loader = torch.utils.data.DataLoader(
            ImageList(rootsour=test_path_src, roottar=test_path_tgt,
                      transform=transforms.Compose([
                          transforms.CenterCrop(64),
                          transforms.ToTensor(),
                      ])),
            batch_size=opt.testBatchSize, shuffle=True)
    else:
        train_path_src = "/home/diplab/Documents/demoire/moire-data/strainData/source/"
        train_path_tgt = "/home/diplab/Documents/demoire/moire-data/strainData/target/"
        test_path_src = "/home/diplab/Documents/demoire/moire-data/stestData/source/"
        test_path_tgt = "/home/diplab/Documents/demoire/moire-data/stestData/target/"


        training_data_loader = torch.utils.data.DataLoader(
            ImageList(rootsour=train_path_src, roottar=train_path_tgt,
                      transform=transforms.Compose([
                transforms.CenterCrop(512),
                transforms.ToTensor(),
            ])),
            batch_size=opt.batchSize, shuffle=True)

        testing_data_loader = torch.utils.data.DataLoader(
            ImageList(rootsour=test_path_src, roottar=test_path_tgt,
                      transform=transforms.Compose([
                transforms.CenterCrop(512),
                transforms.ToTensor(),
            ])),
            batch_size=opt.testBatchSize, shuffle=False)

    print("===> Building model")
    model = MoireCNN()
    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
            
    print("===> Setting Optimizer")
    #optimizer = optim.SGD(model.parameters(), lr=opt.lr,momentum=0.9, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train( optimizer, criterion, epoch)
        save_checkpoint(model, epoch)
        if epoch%10==0:
           test( criterion, epoch)


def adjust_learning_rate( epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr    

def train(optimizer, criterion, epoch):
    epoch_loss = 0
    lr = adjust_learning_rate(epoch-1)
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr  

    print("epoch =", epoch,"lr =",optimizer.param_groups[0]["lr"])
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0], batch[1]

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
            
        output = model(input)
        #output=nn.parallel.data_parallel(model,input,range(2))

        loss = criterion(output, target)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),opt.clip)
        optimizer.step()

        if iteration%10 == 0:
            loss_record="===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.item())
            with open("train_loss_log.txt", "a") as train_log_file:
                train_log_file.write(loss_record + '\n')
            print(loss_record)
    epoch_loss_record="===>Training Epoch [{}] Complete: Avg. MSE Loss: {:.10f}".format(epoch, epoch_loss / len(training_data_loader))
    with open("train_loss_log.txt", "a") as train_log_file:
        train_log_file.write(epoch_loss_record+ '\n')
    print(epoch_loss_record)
    
def test(criterion, epoch):
    avg_mse=0
    avg_psnr = 0
    avg_ssim = 0
    print("===> Testing")
    with torch.no_grad():
        for iteration, batch in enumerate(testing_data_loader, 1):
            print('into test')
            input, target = batch[0], batch[1]
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
            prediction = model(input)
            #prediction=nn.parallel.data_parallel(model,input,range(2))
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            ssim = SSIM(prediction, target)
            avg_psnr += psnr
            avg_ssim += ssim
            avg_mse += mse.item()

            if epoch%10 == 0:
                save_images(epoch,prediction,'epoch_{}_img_{}.jpg'.format(epoch,iteration),1)
                #prediction_output_filename= "result/prediction_{}.jpg".format(batch)
                #prediction.save(prediction_output_filename)
        test_loss_record="===>Testing Epoch[{}] Avg. PSNR: {:.4f} dB, SSIM:{:.10f}  MSE:{:.10f}".format(epoch,
                                                                                            avg_psnr / len(testing_data_loader),
                                                                                            avg_ssim / len(testing_data_loader),
                                                                                         avg_mse / len(testing_data_loader))
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
