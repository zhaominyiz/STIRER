import unicodedata
import lmdb
import six
import random
import numpy as np
# import cv2
import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils.augment import rand_augment_transform
from utils.label_converter import get_charset, strLabelConverter, str_filt
from torchvision import transforms as T
def buf2PIL(txn, key, type='RGB'):
    imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    im = Image.open(buf).convert(type)
    return im
class LRSTRDataset(Dataset):
    def __init__(self, root='/remote-home/myzhao/OCR_Syn_Train/MJ/MJ_train/', max_len=20, syn=False,train=False,args=None):
        super(LRSTRDataset, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)
        self.charset = args['charset']
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples

        print("We have ", self.nSamples, "samples... in",root)
        self.dataset_name = root
        self.max_len = max_len
        self.syn = syn
        self.train = train
        basic_transforms = rand_augment_transform()
        num_da = len(basic_transforms.ops)
        basic_transforms.ops = basic_transforms.ops[:num_da-3]
        basic_transforms.choice_weights = [1.0/(num_da-3)] * (num_da-3)
        quality_aware_transforms = rand_augment_transform()
        num_da = len(quality_aware_transforms.ops)
        quality_aware_transforms.ops = quality_aware_transforms.ops[num_da-3:]
        quality_aware_transforms.num_layers = 2
        quality_aware_transforms.choice_weights = [1.0/(3)] * 3
        self.basic_transforms = basic_transforms
        self.quality_transforms = quality_aware_transforms
        # print(self.basic_transforms, '|', self.quality_transforms)
        self.totensor = T.ToTensor()
        self.p_da = 0.5
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        cls_gt = 2
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        word = word.replace(' ','')
        if self.charset==37:
            word = str_filt(word,'lower')
        elif self.charset==95:
            # print("FROM",word)
            word = str_filt(word,'all')
            # print("TO",word)
        if len(word)==0:
            return self[index+1]
        if self.train and len(word)> self.max_len:
            return self[index + 1]
        if not self.syn:
            img_HR_key = b'image_hr-%09d' % index  # 128*32
            img_lr_key = b'image_lr-%09d' % index  # 64*16
            try:
                img_lr = buf2PIL(txn, img_lr_key, 'RGB')
                img_HR = buf2PIL(txn, img_HR_key, 'RGB')
                # print("??", img_lr.size, img_HR.size)
            except TypeError:
                img = buf2PIL(txn, b'image-%09d' % index, 'RGB')
                img_HR = img
                img_lr = img
            except IOError or len(word) > self.max_len:
                return self[index + 1]
            label_key = ''.join(word.split())
            label_key = unicodedata.normalize('NFKD', label_key).encode('ascii', 'ignore').decode().replace(" ","")
            return img_HR, img_lr, label_key, cls_gt
        else:
            img = buf2PIL(txn, b'image-%09d' % index, 'RGB')
            # img.save("RAW.jpg")
            label_key = ''.join(word.split())
            # Normalize unicode composites (if any) and convert to compatible ASCII characters
            label_key = unicodedata.normalize('NFKD', label_key).encode('ascii', 'ignore').decode().replace(" ","")
            if len(label_key) == 0:
                return self[index + 1]
            if random.random() < 0.0:#and self.charset!=37:#0.5
                img_HR = self.basic_transforms(img)
                img_LR = self.quality_transforms(img_HR)
                # will greatly break the pixel information
                cls_gt = 0
            else:
                # no DA performed
                img_HR = img
                if random.random()<0.8:
                    img_LR = self.quality_transforms(img)
                else:
                    img_LR = img
                cls_gt = 1
            # print(type(img_HR),type(img_LR))
            # img_HR.save("HR.jpg")
            # img_LR.save("LR.jpg")
            return img_HR, img_LR, label_key, cls_gt

class resizeNormalize(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = T.ToTensor()
        self.mask = mask

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img_tensor = self.toTensor(img)
        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)

        return img_tensor

class LRSTR_collect_fn(object):
    def __init__(self,
            imgH=32,
            imgW=128,
            down_sample_scale=2,
            keep_ratio=False,
            min_ratio=1,
            mask=True,
            train=True,
            upsample=False,
            stacklast=False,
            args=None
            ):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.down_sample_scale = down_sample_scale
        self.mask = mask
        self.train = train
        # use for encode & decode
        self.charset = args['charset']
        self.label_converter = strLabelConverter(get_charset(args['charset']))
        self.transform = resizeNormalize((imgW, imgH), self.mask)
        self.transformd2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)
        self.bicubic_factor = [2,3,4,5,6]
        self.upsample = upsample
        self.stacklast = stacklast
        self.args = args
    def __call__(self, batch):
        images_hr, raw_images_lr, label_strs, cls_gt = zip(*batch)
        label_ce = []
        images_hr = [self.transform(image) for image in images_hr]
        images_hr = torch.cat([t.unsqueeze(0) for t in images_hr], 0)

        images_lr = []
        for image in raw_images_lr:
            # if is evaluating, real datas are used, direct eval them without bicubic
            if min(image.size[0],image.size[1])<=10 or not self.train:
                image = image.resize((64,16),Image.BICUBIC)
                images_lr.append(image)
                continue
            # if min(image.size[0],image.size[1])<=10:
            #     images_lr.append(image.resize(
            #     (image.size[0]//2, image.size[1]//2),
            #     Image.BICUBIC))
            #     continue
            if min(image.size[0],image.size[1])<=28:
                d_factor = random.choice([1,2,3])
                images_lr.append(image.resize(
                    (image.size[0]//d_factor, image.size[1]//d_factor),
                    Image.BICUBIC))
                continue
            d_factor = random.choice([1,2,3,4])
            images_lr.append(image.resize(
                (image.size[0]//d_factor, image.size[1]//d_factor),
                Image.BICUBIC))
        if self.upsample:
            images_lr = [self.transform(image) for image in images_lr]
        else:
            images_lr = [self.transformd2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)
        
        label_tensors ,length_tensors = self.label_converter.encode(label_strs)
        if self.stacklast:
            # print(images_hr.shape,'vs',images_lr.shape)
            images_hr = torch.cat((images_hr,images_lr),0)
            label_tensors = torch.cat((label_tensors,label_tensors),0)
            length_tensors = torch.cat((length_tensors,length_tensors),0)
            label_strs = label_strs + label_strs
        # 30 + 1 plus eos
        # label_ce = None
        label_ce = torch.zeros((images_hr.size(0),31))
        # sum_v = 0
        # if self.train:
        #     for idx, v in enumerate(length_tensors.tolist()):
        #         try:
        #             label_ce[idx,0:v] = label_tensors[sum_v:(sum_v+v)]
        #         except:
        #             print('v=',v,label_strs[idx])
        #             os._exit(233)
        #         sum_v += v
        #         # add eos
        #         label_ce[idx,v] = self.args['charset']
        #         for j in range(15-v):
        #             label_ce[idx,v+j] = 0
        label_ce = label_ce.long()
        # print(label_ce)
        cls_gt = torch.LongTensor(cls_gt)
        # print(label_tensors.shape, length_tensors.shape, cls_gt.shape)
        return images_hr, images_lr, label_tensors, length_tensors, label_strs, label_ce#cls_gt
# d = LRSTRDataset(root='/remote-home/myzhao/VideoTextSpotting/TriFormer/dataset/mydata/test/hard/',syn=False)
# d.__getitem__(222)
# print(c)