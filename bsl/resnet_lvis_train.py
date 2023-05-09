import torch
import torchvision

import os
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torchvision import transforms
from PIL import Image
import cv2

from detectron2.data.datasets import register_coco_instances, register_lvis_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from collections import Counter

# Specify paths to COCO dataset annotations and images
coco_annotations_path = '/media/admin1/mobileSSD/datasets/LVIS_v1.0/annotations/lvis_v1_train.json'
coco_images_dir = '/media/admin1/mobileSSD/datasets/LVIS_v1.0/images/'

# Register COCO dataset with detectron2
register_lvis_instances(
    name='lvis_datasets_v1_train',  # Change this string to a name of your choice
    metadata={},
    json_file=coco_annotations_path,
    image_root=coco_images_dir
)
dataset_train_name = 'lvis_datasets_v1_train'
# Get dataset metadata and split into training and validation sets (80/20 split)
metadata = MetadataCatalog.get('lvis_datasets_v1_train')
dataset = DatasetCatalog.get('lvis_datasets_v1_train')

dataset_ann_list = list(dataset)
# print(dataset_ann_list[1])

dataset_list = []
for data in dataset_ann_list:
    '''
    label 1049 annotations [{'bbox': [224.36, 134.25, 148.47, 148.6], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'category_id': 1049, 
    'segmentation': [[312.4, 230.69, 309.85, 260.46, 308.45, 282.47, 309.85, 282.85, 311.63, 281.96, 317.1, 247.35, 320.41, 213.89, 338.35, 
    '''
    data_json = {}
    data_json['file_name'] = data['file_name']
    data_json['image_id'] = data['image_id']

    if len(data['annotations']) > 0 and os.path.exists(data['file_name']) == True and \
            cv2.imread(data['file_name']).shape[0] > 0:
        for d in data['annotations']:
            data_json['bbox'] = d['bbox']
            data_json['height'] = int(d['bbox'][2])
            data_json['width'] = int(d['bbox'][3])
            data_json['category_id'] = d['category_id']
            data_json['segmentation'] = d['segmentation']
            data_json['bbox_mode'] = d['bbox_mode']
            data_json['annotations'] = d

            if data_json['height'] >= 100 and data_json['width'] >= 100 and \
                    data['annotations'] != None:
                print('data_json', data_json)
                dataset_list.append(data_json)
    # break
    # if len(dataset_list) == 20:
    #     break

# json_file_path = '/home/zxq/PycharmProjects/data/ciga_call/result.json'
# json_file = open(json_file_path, mode='w')
print(len(dataset_list))  # 1270141


# for i in range(len(dataset_list)):
#     print("")

class ImageDataset(Dataset):

    def __init__(self, dataset_list, transform=None):
        self.dataset_list = dataset_list
        self.transform = transform
        self.labels = [data['category_id'] for data in dataset_list]

    def __len__(self):
        return len(self.dataset_list)

    def plot_label_distribution(self):
        plt.bar(Counter(self.labels).keys(), Counter(self.labels).values())
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title('Label Distribution')
        plt.show()

    def __getitem__(self, index):
        data = self.dataset_list[index]

        # 读取原图像
        img = cv2.imread(data['file_name'])
        image_id = data['image_id']
        # print()
        # 裁剪原图，即截取bbox对应的图像
        bbox = data['bbox']
        x, y, w, h = list(map(int, bbox))
        img_cropped = img[y:y + h, x:x + w]

        # cv2.imwrite('/media/admin1/mobileSSD/datasets/lvis_clas_image/' + str(data['category_id']) + str(index) + str(data['image_id']), img_cropped)
        if img_cropped.shape[0] > 0 and img_cropped is not None and w >= 100 and h >= 100:
            # print('/media/admin1/mobileSSD/datasets/lvis_clas_image1/' + str(data['category_id']) +"_"+ str(index) + "_" + str(data['image_id']) + ".png")
            # cv2.imshow("img_cropped", img_cropped)
            # cv2.waitKey(0)
            # cv2.imwrite('/media/admin1/mobileSSD/datasets/lvis_clas_image4/' + str(data['category_id']) + "_" + str(
            #     index) + str(data['image_id']) + ".png", img_cropped)
            # print('/media/admin1/mobileSSD/datasets/lvis_clas_image1/' + str(data['category_id']) +"_"+ str(index) + "_" + str(data['image_id']) + ".png")
            # 将图像转换为PIL格式，并进行transform
            img_pil = Image.fromarray(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
            if self.transform is not None:
                img_pil = self.transform(img_pil)

            # 获取category id
            label = data['category_id']
            # img_pil.save('/media/admin1/mobileSSD/datasets/lvis_clas_image' + str(label) + str(index) + str(data['file_name']))
            return img_pil, label
        else:
            label = 0
            print('img_cropped is None')
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if self.transform is not None:
                img_pil = self.transform(img_pil)
            return img_pil, label


transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

split = int(0.9 * len(dataset_list))
train_list = dataset_list[:split]
val_list = dataset_list[split:]

## 加载数据集到DataLoader中
train_dataset = ImageDataset(train_list, transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)

# 绘制类别分布直方图
train_dataset.plot_label_distribution()

# val_dataset = ImageDataset(val_list, transform_train)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8)
#
# ## 创建分类模型
# import torchvision.models as models
#
# model = models.resnet18(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 1203)
#
# ## 定义优化器和损失函数
# import torch.optim as optim
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#
# ## 训练模型并保存效果最好的模型
# import time
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# num_epochs = 10
#
# best_acc = 0.0
# best_model_weights = model.state_dict()
#
# for epoch in range(num_epochs):
#     since = time.time()
#     train_loss = 0.0
#     train_acc = 0.0
#     val_loss = 0.0
#     val_acc = 0.0
#
#     model.train()
#     for inputs, labels in train_loader:
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#
#         optimizer.zero_grad()
#
#         with torch.set_grad_enabled(True):
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             _, preds = torch.max(outputs, 1)
#
#             loss.backward()
#             optimizer.step()
#
#             train_loss += loss.item() * inputs.size(0)
#             train_acc += torch.sum(preds == labels.data)
#
#     train_loss = train_loss / len(train_loader.dataset)
#     train_acc = train_acc.double() / len(train_loader.dataset)
#
#     model.eval()
#     for inputs, labels in val_loader:
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#
#         with torch.set_grad_enabled(False):
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             _, preds = torch.max(outputs, 1)
#
#             val_loss += loss.item() * inputs.size(0)
#             val_acc += torch.sum(preds == labels.data)
#
#     val_loss = val_loss / len(val_loader.dataset)
#     val_acc = val_acc.double() / len(val_loader.dataset)
#
#     if val_acc > best_acc:
#         best_acc = val_acc
#         best_model_weights = model.state_dict()
#         torch.save(best_model_weights, 'ckpts/resnet_best_model' + str(epoch) + '.pth')
#
#     time_elapsed = time.time() - since
#     print(
#         'Epoch {}/{} \t Training Loss: {:.4f} \t Training Acc: {:.4f} \t Validation Loss: {:.4f} \t Validation Acc: {:.4f} \t Time: {:.0f}m {:.0f}s'.format(
#             epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc, time_elapsed // 60, time_elapsed % 60))
