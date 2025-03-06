# Semantically-Aware Contrastive Learning for Multispectral Remote Sensing Images (SACo+)

This GitHub repository contains the oficial implementation of the SACo with experiments and results using a ResNet-18 and ResNet-50 encoder trained with contrastive methods on remote sensing images. The main contributions include leveraging semantic information and texture to construct a robust feature space in the contrastive learning framework. The repository showcases performance metrics across classification, semantic segmentation, and change detection tasks, highlighting the effectiveness of integrating semantic cues and textural features in remote sensing applications.

<p align="center">
  <img src="https://img.shields.io/badge/-pytorch-FA8072?logo=pytorch" alt="PyTorch Badge">
  <img src="https://img.shields.io/badge/-python-B0E0E6?logo=python" alt="Python Badge">
</p>

## Archicteure
The main idea behind semantic aware is to use the semantic combination of the bands from Sentinel-2 and the texture from the images in a temporal contrastive training. This involves processing augmented versions of the images at different time stamps to create a representative feature space.

### Semantic Bands

The Sentinel-2 bands are agrouped follow the logic bellow, this combination allows the image be more representative to some types of soil. Each group comprises three bands that can highlight and represent characteristics of the ground, such as vegetation, urban areas, and visible colors.

| **Groups**            | **R** | **G** | **B** |
|-----------------------|-------|-------|-------|
| Natural Colors        | B04   | B03   | B02   |
| Near-Infrared         | B08   | B04   | B03   |
| Urban                 | B12   | B11   | B04   |
| Agriculture           | B11   | B8A   | B02   |
| Atmospheric Pen.      | B12   | B11   | B8A   |
| Complementary 1       | B01   | B05   | B06   |
| Complementary 2       | B07   | B08   | B10   |

### Texture

To estimate the texture features we used a Local Binary Pattern for each band following the same groups of the semantic groups, some visual example of how these features look like.

<div style="text-align: center;">
    <img src="app/LBP_example_v2.svg" alt="LBP Representation" style="max-width: 40%; height: auto;">
    <p><i>On the top row, the original bands were used to estimate the LBP features, while the bottom displays matrices of LBP values with the same dimensions as the original image. It is possible to observe highlighted patterns in the images, such as edges and contours.</i></p>
</div>

### Semantic Aware

The texture and semantic band features act as a kind of "guide" for the model, which is trained to process each group of bands and textures individually. This means that the output should be similar for images of the same region with different magnifications and timestamps, while at the same time increasing the dissimilarity for the other regions in the memory bank.

<p align="center">
    <img src="app/space_representation_texture.png" alt="Model Workflow"/>
</p>


### Contrastive Training

<p><strong>Main architecture of the SACo training pipeline. It processes original, augmented, and temporally different images to produce and align feature representations.</strong></p>

<div align="center">

<img src="app/pipeline.svg" alt="SACo Training Pipeline" style="width:50%;"/>

</div>

## Paper Results
We tested the encoder on three different downstream tasks—change detection, land cover classification, and semantic segmentation—and the results are in the table below. We found that using Semantic Aware with texture features in the ResNet18 encoder gave us better results than similar approaches.

| Pre-training        | Classification Accuracy ↑ | Semantic Segmentation OA (PASTIS) ↑ | Semantic Segmentation mIoU (PASTIS) ↑ | Semantic Segmentation OA (GID) ↑ | Semantic Segmentation mIoU (GID) ↑ | Change Detection Precision ↑ | Change Detection Recall ↑ | Change Detection F1 ↑ |
|---------------------|---------------------------|-------------------------------------|---------------------------------------|----------------------------------|------------------------------------|------------------------------|----------------------------|-----------------------|
| MoCo V2             | 83.72                     | 45.23                               | 24.88                                 | -                                | -                                  | 62.21                        | 27.57                      | 38.21                 |
| **SA+MoCo V2**      | **85.79**                 | 45.70                               | 25.01                                 | -                                | -                                  | 37.63                        | 48.60                      | 42.42                 |
| SeCo                | 90.05                     | 49.23                               | 25.30                                 | -                                | -                                  | **64.15**                    | 38.89                      | 46.84                 |
| CACo                | 93.08                     | 49.20                               | 26.47                                 | -                                | -                                  | 60.68                        | 42.94                      | 50.29                 |
| **SACo (Ours)**     | **94.72**                 | **54.67**                           | **29.15**                             | -                                | -                                  | 53.51                        | **48.78**                  | **52.78**             |

## Qualitative Results
Bellow some examples of the results of the model in all tasks that we had tested.

### OSCD Change detection
Visual examples of change detection task on the OSCD dataset, where each of the rows shows the input images followed by the ground truth of the pixel that had changed, and the output of the model using different weights for a ResNet-18 network.

<table style="width: 100%; text-align: center;">
  <tr>
    <td>
      <img src="app/image 6/slice_1.png" alt="Image 1"  width="100" height="100">
      <p>Image 1</p>
    </td>
    <td>
      <img src="app/image 6/slice_2.png" alt="Image 2"  width="100" height="100">
      <p>Image 2</p>
    </td>
    <td>
      <img src="app/image 6/label.png" alt="Ground truth"  width="100" height="100">
      <p>Ground truth</p>
    </td>
    <td>
      <img src="app/image 6/imageNet_output.png" alt="ImageNet"  width="100" height="100">
      <p>ImageNet</p>
    </td>
    <td>
      <img src="app/image 6/moco_output.png" alt="MoCo"  width="100" height="100">
      <p>MoCo</p>
    </td>
    <td>
      <img src="app/image 6/sa+moco_output.png" alt="SA+MoCo"  width="100" height="100">
      <p>SA+MoCo</p>
    </td>
    <td>
      <img src="app/image 6/seco_output.png" alt="SeCo"  width="100" height="100">
      <p>SeCo</p>
    </td>
    <td>
      <img src="app/image 6/output.png" alt="SACo"  width="100" height="100">
      <p>SACo</p>
    </td>
  </tr>
  <tr>
    <td>
      <img src="app/image 7/slice_1.png" alt="Image 1"  width="100" height="100">
      <p>Image 1</p>
    </td>
    <td>
      <img src="app/image 7/slice_2.png" alt="Image 2"  width="100" height="100">
      <p>Image 2</p>
    </td>
    <td>
      <img src="app/image 7/label.png" alt="Ground truth"  width="100" height="100">
      <p>Ground truth</p>
    </td>
    <td>
      <img src="app/image 7/imageNet_output.png" alt="ImageNet"  width="100" height="100">
      <p>ImageNet</p>
    </td>
    <td>
      <img src="app/image 7/moco_output.png" alt="MoCo"  width="100" height="100">
      <p>MoCo</p>
    </td>
    <td>
      <img src="app/image 7/sa+moco_output.png" alt="SA+MoCo"  width="100" height="100">
      <p>SA+MoCo</p>
    </td>
    <td>
      <img src="app/image 7/seco_output.png" alt="SeCo"  width="100" height="100">
      <p>SeCo</p>
    </td>
    <td>
      <img src="app/image 7/output.png" alt="SACo"  width="100" height="100">
      <p>SACo</p>
    </td>
  </tr>
</table>

### EuroSAT classification

The performance of our model on the EuroSAT dataset is evaluated using a confusion matrix. This matrix provides a detailed view of how well the model predicts each land cover class from the dataset.

<div align="center">

<img src="app/cf/confusion_matrix_v2.svg" alt="SACo Training Pipeline" style="width:50%;"/>

</div>


### PASTIS semantic segmentation

Visual results of the semantic segmentation predictions on the PASTIS dataset are presented, showing the input image, the ground truth, and the predictions on the test set using the trained decoder with different encoder weights.

<table style="width: 100%; text-align: center;">
  <tr>
    <td>
      <img src="app/PASTIS/image_80846_PASTIS.png" alt="Image" width="100" height="100">
      <p>Image</p>
    </td>
    <td>
      <img src="app/PASTIS/SACo_best_label80846_output.png" alt="Ground truth" width="100" height="100">
      <p>Ground truth</p>
    </td>
    <td>
      <img src="app/PASTIS/imagenet_out80842_output.png" alt="ImageNet" width="100" height="100">
      <p>ImageNet</p>
    </td>
    <td>
      <img src="app/PASTIS/MoCo_out80862_output.png" alt="MoCo" width="100" height="100">
      <p>MoCo</p>
    </td>
    <td>
      <img src="app/PASTIS/SA+MoCo_out80825_output.png" alt="SA+MoCo" width="100" height="100">
      <p>SA+MoCo</p>
    </td>
    <td>
      <img src="app/PASTIS/SeCo_out80844_output.png" alt="SeCo" width="100" height="100">
      <p>SeCo</p>
    </td>
    <td>
      <img src="app/PASTIS/SACo_best_out80846_output.png" alt="SACo" width="100" height="100">
      <p>SACo</p>
    </td>
  </tr>
  <tr>
    <td>
      <img src="app/PASTIS/image_2387_PASTIS.png" alt="Image" width="100" height="100">
      <p>Image</p>
    </td>
    <td>
      <img src="app/PASTIS/label_2387_PASTIS.png" alt="Ground truth" width="100" height="100">
      <p>Ground truth</p>
    </td>
    <td>
      <img src="app/PASTIS/PASTIS_imagenet_out113501_output_2.png" alt="ImageNet" width="100" height="100">
      <p>ImageNet</p>
    </td>
    <td>
      <img src="app/PASTIS/PASTIS_SA+Moco_out113498_output_2.png" alt="MoCo" width="100" height="100">
      <p>MoCo</p>
    </td>
    <td>
      <img src="app/PASTIS/PASTIS_Moco_out113504_output_2.png" alt="SA+MoCo" width="100" height="100">
      <p>SA+MoCo</p>
    </td>
    <td>
      <img src="app/PASTIS/PASTIS_SeCo_out113499_output_2.png" alt="SeCo" width="100" height="100">
      <p>SeCo</p>
    </td>
    <td>
      <img src="app/PASTIS/PASTIS_SACo_out113506_output_2.png" alt="SACo" width="100" height="100">
      <p>SACo</p>
    </td>
  </tr>
</table>

# How to Use This Tool

## Requirement
Make sure you’ve got these bad boys installed before you dive in:
- `torch` >= 2.0.1
- `torchvision` >= 0.15.2
- `cuda` >= 12.1
- `rasterio` >= 1.3.0

## Trained model and weights
The table below provides download links for various models used in this repository:

### Model Downloads and Evaluation Scripts

The table below provides download links for various models and their corresponding evaluation scripts:

| Model                                      | Download Link ResNet18 | Download Link ResNet50                                                | Evaluation Script Link                  |
|--------------------------------------------|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|-----------------------------------------|
| Change Detection Model                     | [Download best_model.pth](https://zenodo.org/records/13736623/files/best_change_model.pth?download=1) | XXX | [Evaluate Model](change_detection_eval.py) |
| EuroSAT LBP Classification Model           | [Download best_model.pth](https://zenodo.org/records/13736623/files/best_classification_model.pth?download=1) | XXX | [Evaluate Model](eval_classification_lbp.py) |
| Semantic Segmentation PASTIS Model         | [Download best_pixel_segmentation.pth](https://zenodo.org/records/13736623/files/best_pixel_segmentation.pth?download=1) | XXX |  [Evaluate Model](segmentation_PASTIS_eval.py) |


## Citation
```

```
