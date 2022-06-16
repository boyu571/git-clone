
from pkgutil import get_data
import cv2
import numpy as np
import json
import os
import glob

root = "D:/Dataset/object_harbor/"
'''
raw_root = "D:/Dataset/object_harbor/Training/seg/harbor/gantry_crane_raw"
val_root = "D:/Dataset/object_harbor/Validation/seg/harbor/gantry_crane_raw"
'''

# train = true: 훈련 데이터, train = false: 검증데이터
def get_data_idx(root, train = True, label = False, class_name = "gantry_crane"):
    file_name, file_root = [], []

    # for filename in os.istdir(root):
    root = os.path.expanduser(root)
    
    if label:
        root = root + "{}_label".format(class_name)
        file_root = glob.glob(root + "/*.json")
    else:
        root = root + "{}_raw".format(class_name)
        file_root = glob.glob(root + "/*.jpg")

    for idx in range(len(file_root)):
        file_name.append(file_root[idx].split('\\')[-1].split('.')[0])
    
    return file_root, file_name


def make_mask(root, train = True, class_name = 'gantry_crane'):
    # load polyfon data in json file
    if train:
        # D:/Dataset/object_harbor/{Training}/seg/harbor/
        root = root + "{}/seg/harbor/".format("Training")
    else:
        root = root + "{}/seg/harbor/".format("Validation")

    label_root, file_name = get_data_idx(root, train = train, label = True, class_name = class_name)
    img_root= get_data_idx(root,train = train, class_name = class_name)[0]

    for idx in range(len(label_root)):
        with open(label_root[idx], 'r') as f :
            json_data = json.load(f)

        points_data = json_data['shapes']
        points = points_data[0]["points"] # polygon points

        # read image
        print(img_root[idx])
        img = cv2.imread(img_root[idx])

        # points list to numpy array
        area = np.array(points)

        # make binary mask
        filled = np.zeros_like(img)
        filled = cv2.fillPoly(filled, pts = np.int32([area]), color =(255,255,255))

        # write binary mask
        cv2.imwrite(root + "{}_mask/{}.jpg".format(class_name, file_name[idx]),filled)

        # images are too big..
        img = cv2.resize(img, dsize=(800, 600), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(filled, dsize=(800, 600), interpolation=cv2.INTER_AREA)

        # image blend to check matching point of 2 images
        blend_img = cv2.addWeighted(img, 0.5, mask, 0.2, gamma=0)
        cv2.imwrite(root + "{}_blend/{}.jpg".format(class_name, file_name[idx]), blend_img)
    '''
        cv2.imshow("input", img)
        cv2.imshow("mask",mask)
        cv2.imshow('blend', blend_img)
        cv2.waitKey(0)
    '''

class_name = ['gantry_crane', 'ship', 'forklift_51', 'crane_58']

for classes in class_name:
    make_mask(root, train = True, class_name=classes) # train data mask 생성
    make_mask(root, train = False, class_name=classes) # validation data mask 생성
=======
import cv2
import numpy as np
import json


# load polyfon data in json file
with open('src/gantry_crane/gantry_crane_S_52_0023146.json', 'r') as f :
    json_data = json.load(f)

points_data = json_data['shapes']
points = points_data[0]["points"]

# read image
img = cv2.imread("src/gantry_crane/gantry_crane_S_52_0023146.jpg")

# points list to numpy array
area = np.array(points)

# make binary mask
filled = np.zeros_like(img)
filled = cv2.fillPoly(filled, pts = np.int32([area]), color =(255,255,255))

# write binary mask
cv2.imwrite("result/gantry_crane/gantry_crane_0023146.jpg",filled)

# images are too big..
img = cv2.resize(img, dsize=(800, 600), interpolation=cv2.INTER_AREA)
mask = cv2.resize(filled, dsize=(800, 600), interpolation=cv2.INTER_AREA)

# image blend to check matching point of 2 images
blend_img = cv2.addWeighted(img, 0.5, mask, 0.2, gamma=0)
cv2.imwrite("blend_img_gantry_crane.jpg", blend_img)

cv2.imshow("input", img)
cv2.imshow("mask",mask)
cv2.imshow('blend', blend_img)
cv2.waitKey(0)

