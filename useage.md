# Detectron2Lab usage Docs

本文档简要介绍了detectron2 中内置命令行工具的使用。

有关使用 API 进行实际编码的教程，请参阅我们的[Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5) ，其中介绍了如何使用现有模型运行推理，以及如何在自定义数据集上训练内置模型。

- https://detectron2.readthedocs.io/en/latest/tutorials/index.html
- https://detectron2.readthedocs.io/en/latest/modules/model_zoo.html

## 带有预训练模型的推理演示

1. [从model zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)中选择一个模型及其配置文件 ，例如`mask_rcnn_R_50_FPN_3x.yaml`.
2. 我们提供`demo.py`能够演示内置配置。运行它：

```
cd demo/
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
  --input input1.jpg input2.jpg \
  [--other-options]
  --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```

配置是为训练而制作的，因此我们需要`MODEL.WEIGHTS`从模型动物园中指定一个模型进行评估。此命令将运行推理并在 OpenCV 窗口中显示可视化。

有关命令行参数的详细信息，请参阅或查看其源代码以了解其行为。一些常见的论点是：`demo.py -h`

- 要**在您的网络摄像头上**运行，请替换为.`--input files``--webcam`
- 要**在视频上**运行，请替换为.`--input files``--video-input video.mp4`
- 要**在 cpu 上**运行，请在.`MODEL.DEVICE cpu``--opts`
- 要将输出保存到目录（用于图像）或文件（用于网络摄像头或视频），请使用`--output`.

## 命令行中的培训和评估

我们在“tools/plain_train_net.py”和“tools/train_net.py”中提供了两个脚本，用于训练detectron2中提供的所有配置。您可能希望将其用作编写自己的训练脚本的参考。

与“train_net.py”相比，“plain_train_net.py”支持的默认功能更少。它还包含更少的抽象，因此更容易添加自定义逻辑。

[要使用“train_net.py”训练模型，首先在datasets/README.md](https://github.com/facebookresearch/detectron2/blob/main/datasets/README.md)之后设置相应的数据 集，然后运行：

```
cd tools/
./train_net.py --num-gpus 8 \
  --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml
```

这些配置是为 8-GPU 训练而设计的。要在 1 个 GPU 上训练，您可能需要[更改一些参数](https://arxiv.org/abs/1706.02677)，例如：

```
./train_net.py \
  --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

要评估模型的性能，请使用

```
./train_net.py \
  --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

有关更多选项，请参阅。`./train_net.py -h`

## 在您的代码中使用 Detectron2 API

请参阅我们的[Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5) ，了解如何使用 detectron2 API 来：

1. 使用现有模型运行推理
2. 在自定义数据集上训练内置模型

有关在detectron2 上构建项目的更多方法，请参见[detectron2/projects](https://github.com/facebookresearch/detectron2/tree/main/projects) 。




```shell
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --input ../data/coco_test/000000000133.jpg    [--other-options]  --opts MODEL.WEIGHTS /home/hxzh02/PycharmProjects/detectron2Lab/weights/COCO-InstanceSegmentation/Mask_R_CNN/MaskRCNN_R50_FPN_3x.pkl
```