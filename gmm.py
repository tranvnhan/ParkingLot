import PIL.Image as Image
import PIL.Image as pil_image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import os
from sklearn.mixture import GaussianMixture
import cv2
import torch
from torch.autograd import Variable

os.getcwd()

"""
Background modeling
"""
image_path = './dataset/45_training/Image/'
# image_path = './dataset/45_training/ROI/train/'
train_filename = '1 (84).png'
train_image = cv2.imread(image_path + train_filename)

""" Draw background selection polygons """
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(train_image)

pol_patch_1 = patches.Polygon(np.array([[170, 530], [829, 381], [870, 429], [230, 572]]), linewidth=1, edgecolor='r',
                              facecolor='none')
pol_patch_2 = patches.Polygon(np.array([[520, 746], [1116, 563], [1227, 616], [794, 785]]), linewidth=1, edgecolor='r',
                              facecolor='none')
pol_patch_3 = patches.Polygon(np.array([[116, 354], [668, 261], [726, 283], [136, 388]]), linewidth=1, edgecolor='r',
                              facecolor='none')
pol_patch_4 = patches.Polygon(np.array([[993, 358], [1087, 333], [1260, 414], [1235, 454]]), linewidth=1, edgecolor='r',
                              facecolor='none')
pol_patch_5 = patches.Polygon(np.array([[14, 554], [110, 530], [245, 665], [164, 701]]), linewidth=1, edgecolor='r',
                              facecolor='none')
pol_patch_6 = patches.Polygon(np.array([[14, 256], [261, 226], [267, 239], [21, 276]]), linewidth=1, edgecolor='r',
                              facecolor='none')
# pol_patch_7 = patches.Polygon(np.array([[345, 633], [590, 581], [802, 653], [505, 741]]), linewidth=1, edgecolor='r',
#                               facecolor='none')
ax.add_patch(pol_patch_1)
ax.add_patch(pol_patch_2)
ax.add_patch(pol_patch_3)
ax.add_patch(pol_patch_4)
ax.add_patch(pol_patch_5)
ax.add_patch(pol_patch_6)
# ax.add_patch(pol_patch_7)
ax.set_title('Background selection')
# plt.show()

""" Visualize the data in 3D """
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(train[:, 0], train[:, 1], train[:, 2])
# plt.show()

""" Background selection """
train_image = np.asarray(train_image)
pts_1 = np.array([[170, 530], [829, 381], [870, 429], [230, 572]], dtype=np.int32)
pts_2 = np.array([[520, 746], [1116, 563], [1194, 643], [794, 785]], dtype=np.int32)
pts_3 = np.array([[116, 354], [668, 261], [726, 283], [136, 388]], dtype=np.int32)
pts_4 = np.array([[993, 358], [1087, 333], [1260, 414], [1235, 454]], dtype=np.int32)
pts_5 = np.array([[14, 554], [110, 530], [245, 665], [164, 701]], dtype=np.int32)
pts_6 = np.array([[14, 253], [259, 217], [267, 239], [21, 276]], dtype=np.int32)
# pts_7 = np.array([[345, 633], [590, 581], [802, 653], [505, 741]], dtype=np.int32)

mask = np.zeros((train_image.shape[0], train_image.shape[1]))
cv2.fillConvexPoly(mask, pts_1, 1)
cv2.fillConvexPoly(mask, pts_2, 1)
cv2.fillConvexPoly(mask, pts_3, 1)
cv2.fillConvexPoly(mask, pts_4, 1)
cv2.fillConvexPoly(mask, pts_5, 1)
cv2.fillConvexPoly(mask, pts_6, 1)
# cv2.fillConvexPoly(mask, pts_7, 1)
mask = mask.astype(np.bool)

out = np.zeros_like(train_image)
out[mask] = train_image[mask]
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.set_title('Cropped background selection')
# ax.imshow(out)
# plt.show()

""" Prepare training and testing sets """
out = np.reshape(out, (-1, 3))
train = out[~np.all(out == 0, axis=1)]  # remove non-background elements, i.e. [0, 0, 0]
print('Training shape:', train.shape)

test_filename = '1 (84).png'
test_image = cv2.imread('./dataset/45_training/ROI/train/' + test_filename)
test_image = np.asarray(test_image)
test = np.reshape(test_image, (-1, 3))
print('Testing shape:', test.shape)

""" How many components? """
# n_components = np.arange(1, 21)
# models = [GaussianMixture(n, covariance_type='diag').fit(train)
#           for n in n_components]
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('How many components?')
# ax.plot(n_components, [m.bic(train) for m in models], label='BIC')
# ax.plot(n_components, [m.aic(train) for m in models], label='AIC')
# plt.legend(loc='best')
# plt.xlabel('n_components')
# plt.show()

""" Perform GMM fitting with selected n_components """
n_components = np.array([5, 10, 15, 20])
gmm = [GaussianMixture(n_components=n, covariance_type='diag').fit(train) for n in n_components]
thre = [3.5 * m.score(train) for m in gmm]
# print('GMM Threshold:', thre)
testScore = [m.score_samples(test) for m in gmm]
result = [testScore[i] < thre[i] for i in range(len(n_components))]
result = np.reshape(result, (len(n_components), test_image.shape[0], test_image.shape[1]))
mask_ROI = np.load('mask.npy')
for i in range(len(n_components)):
    result[i][mask_ROI == 0] = 0
result = np.uint8(result)
# print(result.shape)
# print(result)
fig = plt.figure()
fig.suptitle('Background prediction', fontsize=20)
ax = fig.add_subplot(2, 2, 1)
ax.set_title('(n_components = %d)' % n_components[0])
ax.imshow(result[0], cmap='binary')
ax = fig.add_subplot(2, 2, 2)
ax.set_title('(n_components = %d)' % n_components[1])
ax.imshow(result[1], cmap='binary')
ax = fig.add_subplot(2, 2, 3)
ax.set_title('(n_components = %d)' % n_components[2])
ax.imshow(result[2], cmap='binary')
ax = fig.add_subplot(2, 2, 4)
ax.set_title('(n_components = %d)' % n_components[3])
ax.imshow(result[3], cmap='binary')
# plt.tight_layout()
# plt.show()

selected_thre = thre[0]
selected_gmm = gmm[0]


def GMM_fit(gmm, threshold, input_imgs):
    n, c, h, w = input_imgs.shape  # eg: n, c, h, w = 4, 3, 224, 224
    input_imgs = input_imgs.view(n, h, w, c)  # eg: (4, 224, 224, 3)
    input_imgs = input_imgs.cpu().numpy()
    # input_imgs = [cv2.resize(input_imgs[i], (28, 28)) for i in range(n)]
    input_imgs = [np.reshape(input_imgs[i], (-1, 3)) for i in range(n)]  # eg: (4, 28*28, 3)
    testScore = [gmm.score_samples(input_imgs[i]) for i in range(n)]  # eg: (4, 28*28)
    result = [testScore[i] < threshold for i in range(n)]
    result = np.asarray(result)
    # result = np.reshape(result, (n, 28, 28))  # eg: (4, 28, 28)
    result = np.reshape(result, (n, h, w))  # eg: (4, h, w)
    # mask_ROI = np.load('mask_28x28.npy')
    mask_ROI = np.load('mask.npy')
    for i in range(n):
        result[i][mask_ROI == 0] = 0
    result = np.uint8(result)
    return result

image_path = './dataset/45_training/ROI/train/'
save_path = './dataset/45_training/ROI/gmm/'
for i, filename in enumerate(os.listdir(image_path)):
    input = cv2.imread(image_path + filename)
    h, w, c = input.shape
    input = np.reshape(input, (c, h, w))
    input = torch.from_numpy(input).float()
    input = Variable(input, requires_grad=False)
    input = input.unsqueeze(0)
    result = GMM_fit(selected_gmm, selected_thre, input)
    print(result.shape)
    # plt.imshow(result[0])
    # plt.show()
    # break
    cv2.imwrite(save_path + filename, result[0]*255)
