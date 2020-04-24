# coding:utf-8
# __author__ = yuan
# __time__ = 2020/4/13
# __file__ = exe
# __desc__ =
"python train.py --nb_epoch 600 --direction BtoA --data_root /data/soft/javad/local/pytorch-CycleGAN-and-pix2pix/datasets/facades"
"python action/test.py --direction BtoA --result_dir ./result --data_root /data/soft/javad/local/pytorch-CycleGAN-and-pix2pix/datasets/facades "

"python action/train.py --nb_epoch 200 \
--data_root /data/soft/javad/COCO/horse2zebra \
--netname cyclegan --data_mode unaligned --gan_mode lsgan --preview_dir ./preview"

"python action/test.py --result_dir ./result --data_root /data/soft/javad/COCO/horse2zebra --netname cyclegan --data_mode unaligned "