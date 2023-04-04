# 导入torch库
import torch.backends.cudnn as cudnn
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import argparse
# 导入自定义库
from models import EtEN
from datasets import Dataset

 
 
def main():
    """
    训练.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',
                        type=str,
                        default='./data',
                        help='specify the root path of dataset')        
    parser.add_argument('--batch-size',
                        type=int,
                        help='set the batch size for the train')  
    parser.add_argument('--epoch',
                        type=int,
                        help='set the epoch for the train')    
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help='set the learnig rate for the train')      
    parser.add_argument('--start-epoch',
                        type=int,
                        default=1,
                        help='set the start epoch for the train')                                                                                                                     
    args = parser.parse_args()
    # 数据集路径

 
    # set the parameters
    data_folder = args.data_path
    batch_size = args.batch_size  
    start_epoch = args.start_epoch  
    epochs = args.epochs
    lr = args.lr  
 
    # 设备参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ngpu = 1  # 用来运行的gpu数量
    cudnn.benchmark = True 
 
    # init model
    model = EtEN()
 
    # init optimizer
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad,
                                               model.parameters()),
                                 lr=lr)
 

    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    transformations = transforms.Compose([
        transforms.ToTensor()
    ])
 
    train_dataset = Dataset.EtEN_Datasets(data_folder,
                                     mode='train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)
 
    # start to train
    for epoch in range(start_epoch, epochs + 1):
 
        model.train()  
        n_iter = len(train_loader)
 

        for i, (imgs, labels) in enumerate(train_loader):
 
            imgs = imgs.to(device)
            labels = labels.to(device)
 
            pre_labels = model(imgs)
 
            loss = criterion(pre_labels, labels)
 
            optimizer.zero_grad()
            loss.backward()
 
            optimizer.step()
 
            print("No." + str(i) + " batch train stop")
 
        del imgs, labels, pre_labels
 
        torch.save(
            {
                'epoch': epoch,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict()
            }, 'results/checkpoint.pth')
 

 
if __name__ == '__main__':

    main()