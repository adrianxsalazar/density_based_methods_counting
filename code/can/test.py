import h5py
import PIL.Image as Image
import numpy as np
import os
import glob
import scipy
from image import *
from model import CANNet
import torch
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error,mean_absolute_error
from torchvision import transforms
import argparse
import json


transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

# the folder contains all the test images

parser = argparse.ArgumentParser(description='PyTorch CANNet')

parser.add_argument('test_json', metavar='test',
                    help='path to val json')

parser.add_argument('output', metavar='VAL',
                    help='path output')

args = parser.parse_args()


with open(args.test_json, 'r') as outfile:
    img_paths = json.load(outfile)

model = CANNet()

model = model.cuda()

checkpoint = torch.load(os.path.join(args.output,'model_best.pth.tar'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

pred= []
gt = []

for i in xrange(len(img_paths)):
    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    img = img.unsqueeze(0)
    h,w = img.shape[2:4]
    h_d = h/2
    w_d = w/2
    img_1 = Variable(img[:,:,:h_d,:w_d].cuda())
    img_2 = Variable(img[:,:,:h_d,w_d:].cuda())
    img_3 = Variable(img[:,:,h_d:,:w_d].cuda())
    img_4 = Variable(img[:,:,h_d:,w_d:].cuda())
    density_1 = model(img_1).data.cpu().numpy()
    density_2 = model(img_2).data.cpu().numpy()
    density_3 = model(img_3).data.cpu().numpy()
    density_4 = model(img_4).data.cpu().numpy()

    pure_name = os.path.splitext(os.path.basename(img_paths[i]))[0]
    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5'))
    groundtruth = np.asarray(gt_file['density'])

    pred_sum = density_1.sum()+density_2.sum()+density_3.sum()+density_4.sum()

    print (pred_sum)

    pred.append(pred_sum)

    gt.append(np.sum(groundtruth))

mae = mean_absolute_error(pred,gt)
rmse = np.sqrt(mean_squared_error(pred,gt))

print 'MAE: ',mae
print 'RMSE: ',rmse
results=np.array([mae,rmse])
np.savetxt(os.path.join(args.output,"restults.txt"),results,delimiter=',')
