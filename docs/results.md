---
layout: default
title: Results
description: Quantitative and qualitative results of SACo across classification, segmentation, and change detection tasks
---

<div class="container">
    <div class="hero">
        <h1>Results</h1>
        <p>Performance evaluation across multiple downstream tasks</p>
    </div>

    <div class="section">
        <h2>Quantitative Results</h2>
        <p>
            We tested the SACo encoder on three different downstream tasks—change detection, land cover classification, and semantic segmentation. The results demonstrate that using Semantic Aware with texture features in the ResNet-18 encoder provides better results than similar approaches.
        </p>
        
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Pre-training</th>
                        <th>Classification Accuracy ↑</th>
                        <th>Seg. OA (PASTIS) ↑</th>
                        <th>Seg. mIoU (PASTIS) ↑</th>
                        <th>Seg. OA (GID) ↑</th>
                        <th>Seg. mIoU (GID) ↑</th>
                        <th>CD Precision ↑</th>
                        <th>CD Recall ↑</th>
                        <th>CD F1 ↑</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>MoCo V2</td>
                        <td>83.72</td>
                        <td>45.23</td>
                        <td>24.88</td>
                        <td>-</td>
                        <td>-</td>
                        <td>62.21</td>
                        <td>27.57</td>
                        <td>38.21</td>
                    </tr>
                    <tr>
                        <td>SA+MoCo V2</td>
                        <td><strong>85.79</strong></td>
                        <td>45.70</td>
                        <td>25.01</td>
                        <td>-</td>
                        <td>-</td>
                        <td>37.63</td>
                        <td>48.60</td>
                        <td>42.42</td>
                    </tr>
                    <tr>
                        <td>SeCo</td>
                        <td>90.05</td>
                        <td>49.23</td>
                        <td>25.30</td>
                        <td>-</td>
                        <td>-</td>
                        <td><strong>64.15</strong></td>
                        <td>38.89</td>
                        <td>46.84</td>
                    </tr>
                    <tr>
                        <td>CACo</td>
                        <td>93.08</td>
                        <td>49.20</td>
                        <td>26.47</td>
                        <td>-</td>
                        <td>-</td>
                        <td>60.68</td>
                        <td>42.94</td>
                        <td>50.29</td>
                    </tr>
                    <tr style="background-color: var(--bg-tertiary);">
                        <td><strong>SACo (Ours)</strong></td>
                        <td><strong>94.72</strong></td>
                        <td><strong>54.67</strong></td>
                        <td><strong>29.15</strong></td>
                        <td>-</td>
                        <td>-</td>
                        <td>53.51</td>
                        <td><strong>48.78</strong></td>
                        <td><strong>52.78</strong></td>
                    </tr>
                </tbody>
            </table>
        </div>
        <p style="margin-top: 1rem; font-size: 0.9rem; color: var(--text-tertiary);">
            <strong>Note:</strong> Bold values indicate the best performance for each metric. CD = Change Detection, Seg. = Segmentation, OA = Overall Accuracy, mIoU = mean Intersection over Union.
        </p>
    </div>

    <div class="section">
        <h2>Qualitative Results</h2>
        <p>Below are visual examples demonstrating the performance of SACo across different tasks compared to baseline methods.</p>
    </div>

    <div class="section">
        <h3>OSCD Change Detection</h3>
        <p>Visual examples of change detection on the OSCD dataset. Each row shows the input images followed by the ground truth and model predictions using different pre-training methods.</p>
        
        <h4 style="margin-top: 2rem; font-size: 1.1rem;">Example 1</h4>
        <div class="gallery">
            <div>
                <img src="{{ '/assets/images/change_detection/image_6/slice_1.png' | relative_url }}" alt="Image 1">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">Image 1</p>
            </div>
            <div>
                <img src="{{ '/assets/images/change_detection/image_6/slice_2.png' | relative_url }}" alt="Image 2">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">Image 2</p>
            </div>
            <div>
                <img src="{{ '/assets/images/change_detection/image_6/label.png' | relative_url }}" alt="Ground Truth">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">Ground Truth</p>
            </div>
            <div>
                <img src="{{ '/assets/images/change_detection/image_6/imageNet_output.png' | relative_url }}" alt="ImageNet">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">ImageNet</p>
            </div>
            <div>
                <img src="{{ '/assets/images/change_detection/image_6/moco_output.png' | relative_url }}" alt="MoCo">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">MoCo</p>
            </div>
            <div>
                <img src="{{ '/assets/images/change_detection/image_6/sa+moco_output.png' | relative_url }}" alt="SA+MoCo">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">SA+MoCo</p>
            </div>
            <div>
                <img src="{{ '/assets/images/change_detection/image_6/seco_output.png' | relative_url }}" alt="SeCo">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">SeCo</p>
            </div>
            <div>
                <img src="{{ '/assets/images/change_detection/image_6/output.png' | relative_url }}" alt="SACo">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;"><strong>SACo</strong></p>
            </div>
        </div>

        <h4 style="margin-top: 2rem; font-size: 1.1rem;">Example 2</h4>
        <div class="gallery">
            <div>
                <img src="{{ '/assets/images/change_detection/image_7/slice_1.png' | relative_url }}" alt="Image 1">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">Image 1</p>
            </div>
            <div>
                <img src="{{ '/assets/images/change_detection/image_7/slice_2.png' | relative_url }}" alt="Image 2">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">Image 2</p>
            </div>
            <div>
                <img src="{{ '/assets/images/change_detection/image_7/label.png' | relative_url }}" alt="Ground Truth">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">Ground Truth</p>
            </div>
            <div>
                <img src="{{ '/assets/images/change_detection/image_7/imageNet_output.png' | relative_url }}" alt="ImageNet">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">ImageNet</p>
            </div>
            <div>
                <img src="{{ '/assets/images/change_detection/image_7/moco_output.png' | relative_url }}" alt="MoCo">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">MoCo</p>
            </div>
            <div>
                <img src="{{ '/assets/images/change_detection/image_7/sa+moco_output.png' | relative_url }}" alt="SA+MoCo">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">SA+MoCo</p>
            </div>
            <div>
                <img src="{{ '/assets/images/change_detection/image_7/seco_output.png' | relative_url }}" alt="SeCo">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">SeCo</p>
            </div>
            <div>
                <img src="{{ '/assets/images/change_detection/image_7/output.png' | relative_url }}" alt="SACo">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;"><strong>SACo</strong></p>
            </div>
        </div>
    </div>

    <div class="section">
        <h3>EuroSAT Classification</h3>
        <p>The performance of SACo on the EuroSAT dataset is evaluated using a confusion matrix, providing a detailed view of how well the model predicts each land cover class.</p>
        <div class="image-container">
            <img src="{{ '/assets/images/confusion_matrix_v2.svg' | relative_url }}" alt="EuroSAT Confusion Matrix" style="max-width: 70%;">
            <p class="image-caption">Confusion matrix showing SACo's classification performance across different land cover types in the EuroSAT dataset</p>
        </div>
    </div>

    <div class="section">
        <h3>PASTIS Semantic Segmentation</h3>
        <p>Visual results of semantic segmentation predictions on the PASTIS dataset, showing the input image, ground truth, and predictions using different encoder weights.</p>
        
        <h4 style="margin-top: 2rem; font-size: 1.1rem;">Example 1</h4>
        <div class="gallery">
            <div>
                <img src="{{ '/assets/images/segmentation/image_80846_PASTIS.png' | relative_url }}" alt="Input Image">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">Image</p>
            </div>
            <div>
                <img src="{{ '/assets/images/segmentation/SACo_best_label80846_output.png' | relative_url }}" alt="Ground Truth">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">Ground Truth</p>
            </div>
            <div>
                <img src="{{ '/assets/images/segmentation/imagenet_out80842_output.png' | relative_url }}" alt="ImageNet">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">ImageNet</p>
            </div>
            <div>
                <img src="{{ '/assets/images/segmentation/MoCo_out80862_output.png' | relative_url }}" alt="MoCo">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">MoCo</p>
            </div>
            <div>
                <img src="{{ '/assets/images/segmentation/SA+MoCo_out80825_output.png' | relative_url }}" alt="SA+MoCo">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">SA+MoCo</p>
            </div>
            <div>
                <img src="{{ '/assets/images/segmentation/SeCo_out80844_output.png' | relative_url }}" alt="SeCo">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">SeCo</p>
            </div>
            <div>
                <img src="{{ '/assets/images/segmentation/SACo_best_out80846_output.png' | relative_url }}" alt="SACo">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;"><strong>SACo</strong></p>
            </div>
        </div>

        <h4 style="margin-top: 2rem; font-size: 1.1rem;">Example 2</h4>
        <div class="gallery">
            <div>
                <img src="{{ '/assets/images/segmentation/image_2387_PASTIS.png' | relative_url }}" alt="Input Image">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">Image</p>
            </div>
            <div>
                <img src="{{ '/assets/images/segmentation/label_2387_PASTIS.png' | relative_url }}" alt="Ground Truth">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">Ground Truth</p>
            </div>
            <div>
                <img src="{{ '/assets/images/segmentation/PASTIS_imagenet_out113501_output_2.png' | relative_url }}" alt="ImageNet">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">ImageNet</p>
            </div>
            <div>
                <img src="{{ '/assets/images/segmentation/PASTIS_SA+Moco_out113498_output_2.png' | relative_url }}" alt="MoCo">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">MoCo</p>
            </div>
            <div>
                <img src="{{ '/assets/images/segmentation/PASTIS_Moco_out113504_output_2.png' | relative_url }}" alt="SA+MoCo">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">SA+MoCo</p>
            </div>
            <div>
                <img src="{{ '/assets/images/segmentation/PASTIS_SeCo_out113499_output_2.png' | relative_url }}" alt="SeCo">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;">SeCo</p>
            </div>
            <div>
                <img src="{{ '/assets/images/segmentation/PASTIS_SACo_out113506_output_2.png' | relative_url }}" alt="SACo">
                <p style="text-align: center; font-size: 0.85rem; margin-top: 0.5rem;"><strong>SACo</strong></p>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Citation</h2>
        <p>If you use SACo in your research, please cite our paper:</p>
        <div class="citation-box">
            <pre id="bibtex-results">@article{STIVAL2025173,
  title = {Semantically-Aware Contrastive Learning for multispectral remote sensing images},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume = {223},
  pages = {173-187},
  year = {2025},
  issn = {0924-2716},
  doi = {https://doi.org/10.1016/j.isprsjprs.2025.02.024},
  url = {https://www.sciencedirect.com/science/article/pii/S0924271625000826},
  author = {Leandro Stival and Ricardo {da Silva Torres} and Helio Pedrini}
}</pre>
            <button class="copy-btn" data-target="bibtex-results" aria-label="Copy BibTeX">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
                Copy
            </button>
        </div>
    </div>

    <div class="section">
        <h2>Paper Link</h2>
        <p>Read the full paper for detailed methodology, experiments, and analysis:</p>
        <a href="https://www.sciencedirect.com/science/article/pii/S0924271625000826" target="_blank" rel="noopener" class="btn">Read Paper on ScienceDirect</a>
    </div>
</div>
