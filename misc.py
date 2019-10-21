import PIL.Image as Image
import PIL.Image as pil_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2
import torch
from torchvision import transforms

os.getcwd()

"""
Create mask to extract ROI from each image in dataset
"""
# image_path = './test-images/mask.jpg'
# input = pil_image.open(image_path)
# input = input.convert('L')  # convert to grayscale
# input = input.resize((28, 28))
# input_np = np.asarray(input.getdata(), dtype=np.float64).reshape((input.size[1], input.size[0]))
# input_np = np.asarray(input_np, dtype=np.uint8)
#
# mask = input_np
# mask[input_np > 0] = 255
#
# # plt.imshow(mask)
# # plt.show()
#
# np.save('mask_28x28.npy', mask)  # save the mask
# # mask = np.load('mask.npy')

"""
Extract ROI from each image in dataset using mask
"""
# mask = np.load('mask.npy')
# image_path = './dataset/45_training/Image/'
# save_path = './dataset/45_training/ROI/'
# for i, filename in enumerate(os.listdir(image_path)):
#     input = pil_image.open(image_path + filename)
#     input_np = np.asarray(input)
#     input_np = np.transpose(input_np, (2, 0, 1))  # HxWx3 -> 3xHxW
#     input_np.setflags(write=1)
#     input_np[0][mask == 0] = 0
#     input_np[1][mask == 0] = 0
#     input_np[2][mask == 0] = 0
#     output_np = np.transpose(input_np, (1, 2, 0))  # 3xHxW -> HxWx3
#     output = Image.fromarray(output_np)
#     output.save(save_path + filename)

"""
Read GroundTruth txt file and save numpy array to file
"""
# file_path = './dataset/45_training/ROI/train-gt.txt'
# destination = './dataset/45_training/ROI/'
# f = open(file_path, "r")
# out_total = np.empty((1, 60), dtype=np.uint8)
# out_total[0] = 0
# for line in f:
#     inp = np.asarray(line.split(' '))
#     inp = inp[0:-1]
#     out = np.asarray(list(map(int, inp)))
#     out = np.reshape(out, (-1, 60))
#     # print(out.shape)
#     out_total = np.append(out_total, out, axis=0)
#
# f.close()
# out_total = np.delete(out_total, 0, axis=0)
# print(out_total)
# print(out_total.shape)
# np.save(destination + 'train-gt.npy', out_total)
# train_gt = np.load('./dataset/45_training/ROI/train-gt.npy')
# print(train_gt)

"""
Extract polygon given vertices from an image
"""
# Let's define a white image that is 300x700.
# Then extract the region that's defined by that polygon and
# show what the output looks like.
# img = 255*np.ones((300, 700, 3), dtype=np.uint8)
#
# pts = np.array([[500, 100], [200, 250], [650, 100]], dtype=np.int32)
# mask = np.zeros((img.shape[0], img.shape[1]))
#
# cv2.fillConvexPoly(mask, pts, 1)
# mask = mask.astype(np.bool)
# print(mask)
#
# out = np.zeros_like(img)
# out[mask] = img[mask]
#
# cv2.imshow('Extracted image', out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


image_path = './dataset/45_training/ROI/gmm/'
save_path = './dataset/45_training/ROI/gmm_28x28/'
for i, filename in enumerate(os.listdir(image_path)):
    input = cv2.imread(image_path + filename, cv2.IMREAD_GRAYSCALE)
    input = cv2.resize(input, (28, 28))
    # input[input < 127] = 0
    print(filename, np.sum(input/255))
    cv2.imwrite(save_path + filename, input)

# image_path = './dataset/45_training/ROI/gmm_28x28/'
# test = cv2.imread(image_path + '1 (240).png', cv2.IMREAD_GRAYSCALE)/255
# print(np.sum(test))