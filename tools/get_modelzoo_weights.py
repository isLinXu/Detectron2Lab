import torch

from detectron2 import model_zoo

'''
根据yaml和modelzoo下载相应的权重到指定位置
configs/
├── Base-RCNN-C4.yaml
├── Base-RCNN-DilatedC5.yaml
├── Base-RCNN-FPN.yaml
├── Base-RetinaNet.yaml
├── Cityscapes
│   └── mask_rcnn_R_50_FPN.yaml
├── COCO-Detection
│   ├── faster_rcnn_R_101_C4_3x.yaml
│   ├── faster_rcnn_R_101_DC5_3x.yaml
│   ├── faster_rcnn_R_101_FPN_3x.yaml
│   ├── faster_rcnn_R_50_C4_1x.yaml
│   ├── faster_rcnn_R_50_C4_3x.yaml
│   ├── faster_rcnn_R_50_DC5_1x.yaml
│   ├── faster_rcnn_R_50_DC5_3x.yaml
│   ├── faster_rcnn_R_50_FPN_1x.yaml
│   ├── faster_rcnn_R_50_FPN_3x.yaml
│   ├── faster_rcnn_X_101_32x8d_FPN_3x.yaml
│   ├── fast_rcnn_R_50_FPN_1x.yaml
│   ├── fcos_R_50_FPN_1x.py
│   ├── retinanet_R_101_FPN_3x.yaml
│   ├── retinanet_R_50_FPN_1x.py
│   ├── retinanet_R_50_FPN_1x.yaml
│   ├── retinanet_R_50_FPN_3x.yaml
│   ├── rpn_R_50_C4_1x.yaml
│   └── rpn_R_50_FPN_1x.yaml
├── COCO-InstanceSegmentation
│   ├── mask_rcnn_R_101_C4_3x.yaml
│   ├── mask_rcnn_R_101_DC5_3x.yaml
│   ├── mask_rcnn_R_101_FPN_3x.yaml
│   ├── mask_rcnn_R_50_C4_1x.py
│   ├── mask_rcnn_R_50_C4_1x.yaml
│   ├── mask_rcnn_R_50_C4_3x.yaml
│   ├── mask_rcnn_R_50_DC5_1x.yaml
│   ├── mask_rcnn_R_50_DC5_3x.yaml
│   ├── mask_rcnn_R_50_FPN_1x_giou.yaml
│   ├── mask_rcnn_R_50_FPN_1x.py
│   ├── mask_rcnn_R_50_FPN_1x.yaml
│   ├── mask_rcnn_R_50_FPN_3x.yaml
│   ├── mask_rcnn_regnetx_4gf_dds_fpn_1x.py
│   ├── mask_rcnn_regnety_4gf_dds_fpn_1x.py
│   └── mask_rcnn_X_101_32x8d_FPN_3x.yaml
├── COCO-Keypoints
│   ├── Base-Keypoint-RCNN-FPN.yaml
│   ├── keypoint_rcnn_R_101_FPN_3x.yaml
│   ├── keypoint_rcnn_R_50_FPN_1x.py
│   ├── keypoint_rcnn_R_50_FPN_1x.yaml
│   ├── keypoint_rcnn_R_50_FPN_3x.yaml
│   └── keypoint_rcnn_X_101_32x8d_FPN_3x.yaml
├── COCO-PanopticSegmentation
│   ├── Base-Panoptic-FPN.yaml
│   ├── panoptic_fpn_R_101_3x.yaml
│   ├── panoptic_fpn_R_50_1x.py
│   ├── panoptic_fpn_R_50_1x.yaml
│   └── panoptic_fpn_R_50_3x.yaml
├── common
│   ├── coco_schedule.py
│   ├── data
│   │   ├── coco_keypoint.py
│   │   ├── coco_panoptic_separated.py
│   │   ├── coco.py
│   │   └── constants.py
│   ├── models
│   │   ├── cascade_rcnn.py
│   │   ├── fcos.py
│   │   ├── keypoint_rcnn_fpn.py
│   │   ├── mask_rcnn_c4.py
│   │   ├── mask_rcnn_fpn.py
│   │   ├── mask_rcnn_vitdet.py
│   │   ├── panoptic_fpn.py
│   │   └── retinanet.py
│   ├── optim.py
│   ├── README.md
│   └── train.py
├── Detectron1-Comparisons
│   ├── faster_rcnn_R_50_FPN_noaug_1x.yaml
│   ├── keypoint_rcnn_R_50_FPN_1x.yaml
│   ├── mask_rcnn_R_50_FPN_noaug_1x.yaml
│   └── README.md
├── LVISv0.5-InstanceSegmentation
│   ├── mask_rcnn_R_101_FPN_1x.yaml
│   ├── mask_rcnn_R_50_FPN_1x.yaml
│   └── mask_rcnn_X_101_32x8d_FPN_1x.yaml
├── LVISv1-InstanceSegmentation
│   ├── mask_rcnn_R_101_FPN_1x.yaml
│   ├── mask_rcnn_R_50_FPN_1x.yaml
│   └── mask_rcnn_X_101_32x8d_FPN_1x.yaml
├── Misc
│   ├── cascade_mask_rcnn_R_50_FPN_1x.yaml
│   ├── cascade_mask_rcnn_R_50_FPN_3x.yaml
│   ├── cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml
│   ├── mask_rcnn_R_50_FPN_1x_cls_agnostic.yaml
│   ├── mask_rcnn_R_50_FPN_1x_dconv_c3-c5.yaml
│   ├── mask_rcnn_R_50_FPN_3x_dconv_c3-c5.yaml
│   ├── mask_rcnn_R_50_FPN_3x_gn.yaml
│   ├── mask_rcnn_R_50_FPN_3x_syncbn.yaml
│   ├── mmdet_mask_rcnn_R_50_FPN_1x.py
│   ├── panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml
│   ├── scratch_mask_rcnn_R_50_FPN_3x_gn.yaml
│   ├── scratch_mask_rcnn_R_50_FPN_9x_gn.yaml
│   ├── scratch_mask_rcnn_R_50_FPN_9x_syncbn.yaml
│   ├── semantic_R_50_FPN_1x.yaml
│   └── torchvision_imagenet_R_50.py
├── new_baselines
│   ├── mask_rcnn_R_101_FPN_100ep_LSJ.py
│   ├── mask_rcnn_R_101_FPN_200ep_LSJ.py
│   ├── mask_rcnn_R_101_FPN_400ep_LSJ.py
│   ├── mask_rcnn_R_50_FPN_100ep_LSJ.py
│   ├── mask_rcnn_R_50_FPN_200ep_LSJ.py
│   ├── mask_rcnn_R_50_FPN_400ep_LSJ.py
│   ├── mask_rcnn_R_50_FPN_50ep_LSJ.py
│   ├── mask_rcnn_regnetx_4gf_dds_FPN_100ep_LSJ.py
│   ├── mask_rcnn_regnetx_4gf_dds_FPN_200ep_LSJ.py
│   ├── mask_rcnn_regnetx_4gf_dds_FPN_400ep_LSJ.py
│   ├── mask_rcnn_regnety_4gf_dds_FPN_100ep_LSJ.py
│   ├── mask_rcnn_regnety_4gf_dds_FPN_200ep_LSJ.py
│   └── mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py
├── PascalVOC-Detection
│   ├── faster_rcnn_R_50_C4.yaml
│   └── faster_rcnn_R_50_FPN.yaml
└── quick_schedules
    ├── cascade_mask_rcnn_R_50_FPN_inference_acc_test.yaml
    ├── cascade_mask_rcnn_R_50_FPN_instant_test.yaml
    ├── fast_rcnn_R_50_FPN_inference_acc_test.yaml
    ├── fast_rcnn_R_50_FPN_instant_test.yaml
    ├── keypoint_rcnn_R_50_FPN_inference_acc_test.yaml
    ├── keypoint_rcnn_R_50_FPN_instant_test.yaml
    ├── keypoint_rcnn_R_50_FPN_normalized_training_acc_test.yaml
    ├── keypoint_rcnn_R_50_FPN_training_acc_test.yaml
    ├── mask_rcnn_R_50_C4_GCV_instant_test.yaml
    ├── mask_rcnn_R_50_C4_inference_acc_test.yaml
    ├── mask_rcnn_R_50_C4_instant_test.yaml
    ├── mask_rcnn_R_50_C4_training_acc_test.yaml
    ├── mask_rcnn_R_50_DC5_inference_acc_test.yaml
    ├── mask_rcnn_R_50_FPN_inference_acc_test.yaml
    ├── mask_rcnn_R_50_FPN_instant_test.yaml
    ├── mask_rcnn_R_50_FPN_pred_boxes_training_acc_test.yaml
    ├── mask_rcnn_R_50_FPN_training_acc_test.yaml
    ├── panoptic_fpn_R_50_inference_acc_test.yaml
    ├── panoptic_fpn_R_50_instant_test.yaml
    ├── panoptic_fpn_R_50_training_acc_test.yaml
    ├── README.md
    ├── retinanet_R_50_FPN_inference_acc_test.yaml
    ├── retinanet_R_50_FPN_instant_test.yaml
    ├── rpn_R_50_FPN_inference_acc_test.yaml
    ├── rpn_R_50_FPN_instant_test.yaml
    ├── semantic_R_50_FPN_inference_acc_test.yaml
    ├── semantic_R_50_FPN_instant_test.yaml
    └── semantic_R_50_FPN_training_acc_test.yaml

15 directories, 133 files
'''

def get_models_yaml(model_yaml, save_path, is_save=True):
    model = model_zoo.get(model_yaml, trained=True)
    #model = model_zoo.get("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml", trained=True)

    print('model:', model)
    print(model.state_dict().keys())

    if is_save:
        # torch.save(model.state_dict(), "../data/model_parameter.pkl")
        torch.save(model.state_dict(), save_path)
    return model

if __name__ == '__main__':
    model_yaml = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml'
    save_path = '../data/model_parameter.pkl'
    is_save = True
    get_models_yaml(model_yaml, save_path, is_save)