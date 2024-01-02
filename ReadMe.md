# :sparkles: STIRER :sparkles:
Official Code for 'STIRER: A Unified Model for Low-Resolution Scene Text Image Recovery and Recognition'

ACMMM 2023 Accepted Paper 

## Step1. Dataset preparation

Syntheic datasets: 


Ankush Gupta, Andrea Vedaldi, and Andrew Zisserman. 2016. Synthetic data for text localisation in natural images. In Proceedings of the IEEE conference on computer vision and pattern recognition. 2315–2324.


Max Jaderberg, Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. 2014. Synthetic data and artificial neural networks for natural scene text recognition. arXiv preprint arXiv:1406.2227 (2014).


TextZoom: https://github.com/WenjiaWang0312/TextZoom

Make a dataset folder and put these LMDB datasets under that folder like this:

```
dataset/OCR_Syn_Train/ST/
dataset/OCR_Syn_Train/MJ/MJ_train/
dataset/textzoom/train1/
dataset/textzoom/train2/
```

## Step2. Model training
Download the CRNN checkpoint from https://github.com/meijieru/crnn.pytorch as put the checkpoint under 'ckpt'.

Launch the two-stage model training as follows:
```bash
python3 train.py
python3 train_ft.py
```

## Step3. Model evaluation
You can try our released checkpoint (STIRER_Final_Release.pth) and run the evaluation code:
```bash
python3 test.py
```
You will obtain a result as follows:
```bash
----------------<Evaluation>----------------
{1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9', 11: 'a', 12: 'b', 13: 'c', 14: 'd', 15: 'e', 16: 'f', 17: 'g', 18: 'h', 19: 'i', 20: 'j', 21: 'k', 22: 'l', 23: 'm', 24: 'n', 25: 'o', 26: 'p', 27: 'q', 28: 'r', 29: 's', 30: 't', 31: 'u', 32: 'v', 33: 'w', 34: 'x', 35: 'y', 36: 'z', 0: ''}
Evaluating dataset/mydata/test/easy/
STEP 0 Acc 78.44 PSNR 22.62 SSIM 78.66
STEP 1 Acc 82.09 PSNR 22.63 SSIM 79.60
STEP 2 Acc 80.05 PSNR 23.08 SSIM 79.97
STEP 3 Acc 87.89 PSNR 25.15 SSIM 87.41
CRNN Acc 52.56
Evaluating dataset/mydata/test/medium/
STEP 0 Acc 66.19 PSNR 19.51 SSIM 59.73
STEP 1 Acc 68.32 PSNR 19.42 SSIM 60.99
STEP 2 Acc 69.03 PSNR 19.81 SSIM 61.93
STEP 3 Acc 74.20 PSNR 20.66 SSIM 68.58
CRNN Acc 31.61
Evaluating dataset/mydata/test/hard/
STEP 0 Acc 48.62 PSNR 20.09 SSIM 65.53
STEP 1 Acc 51.53 PSNR 19.94 SSIM 66.57
STEP 2 Acc 51.23 PSNR 20.42 SSIM 68.12
STEP 3 Acc 59.64 PSNR 21.12 SSIM 73.50
CRNN Acc 27.48
Avg CRNN Acc 38.10
Final Acc 73.91 Final PSNR 22.31 SSIM 0.7650
```
## Citation
```
@inproceedings{zhao2023stirer,
  title={STIRER: A Unified Model for Low-Resolution Scene Text Image Recovery and Recognition},
  author={Zhao, Minyi and Xuyang, Shijie and Guan, Jihong and Zhou, Shuigeng},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={7530--7539},
  year={2023}
}
```

Feel free to contact me if you have any questions :)

My email is zhaomy20@fudan.edu.cn
