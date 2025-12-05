---
layout: default
title: Downloads
description: Download pre-trained model weights and evaluation scripts for SACo
---

<div class="container">
    <div class="hero">
        <h1>Downloads</h1>
        <p>Pre-trained model weights and evaluation scripts</p>
    </div>

    <div class="section">
        <h2>Requirements</h2>
        <p>Make sure you have the following dependencies installed before using the models:</p>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Package</th>
                        <th>Version</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>torch</td>
                        <td>&gt;= 2.0.1</td>
                    </tr>
                    <tr>
                        <td>torchvision</td>
                        <td>&gt;= 0.15.2</td>
                    </tr>
                    <tr>
                        <td>cuda</td>
                        <td>&gt;= 12.1</td>
                    </tr>
                    <tr>
                        <td>rasterio</td>
                        <td>&gt;= 1.3.0</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>

    <div class="section">
        <h2>Pre-trained Models</h2>
        <p>Download the pre-trained SACo models for different downstream tasks. All models are trained with ResNet-18 backbone unless otherwise specified.</p>
        
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>ResNet-18</th>
                        <th>ResNet-50</th>
                        <th>Evaluation Script</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Change Detection Model</strong><br><small>Trained on OSCD dataset</small></td>
                        <td><a href="https://zenodo.org/records/13736623/files/best_change_model.pth?download=1" target="_blank" rel="noopener">Download</a></td>
                        <td><em>Coming soon</em></td>
                        <td><a href="https://github.com/lstival/SACo/blob/main/change_detection_eval.py" target="_blank" rel="noopener">View Script</a></td>
                    </tr>
                    <tr>
                        <td><strong>EuroSAT Classification Model</strong><br><small>With LBP texture features</small></td>
                        <td><a href="https://zenodo.org/records/13736623/files/best_classification_model.pth?download=1" target="_blank" rel="noopener">Download</a></td>
                        <td><em>Coming soon</em></td>
                        <td><a href="https://github.com/lstival/SACo/blob/main/eval_classification_lbp.py" target="_blank" rel="noopener">View Script</a></td>
                    </tr>
                    <tr>
                        <td><strong>PASTIS Segmentation Model</strong><br><small>Semantic segmentation model</small></td>
                        <td><a href="https://zenodo.org/records/13736623/files/best_pixel_segmentation.pth?download=1" target="_blank" rel="noopener">Download</a></td>
                        <td><em>Coming soon</em></td>
                        <td><a href="https://github.com/lstival/SACo/blob/main/segmentation_PASTIS_eval.py" target="_blank" rel="noopener">View Script</a></td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>

    <div class="section">
        <h2>Usage Instructions</h2>
        <h3>1. Download the Model</h3>
        <p>Click on the download link for the model you want to use. The models are hosted on Zenodo for long-term availability.</p>

        <h3>2. Set Up Your Environment</h3>
        <p>Install the required dependencies using conda or pip:</p>
        <pre><code># Using conda (recommended)
conda env create -f environment.yml
conda activate saco

# Or using pip
pip install torch>=2.0.1 torchvision>=0.15.2 rasterio>=1.3.0</code></pre>

        <h3>3. Run Evaluation</h3>
        <p>Use the corresponding evaluation script to test the model on your data:</p>
        <pre><code># Example: Change Detection
python change_detection_eval.py --model_path best_change_model.pth --data_path /path/to/oscd

# Example: Classification
python eval_classification_lbp.py --model_path best_classification_model.pth --data_path /path/to/eurosat

# Example: Segmentation
python segmentation_PASTIS_eval.py --model_path best_pixel_segmentation.pth --data_path /path/to/pastis</code></pre>
    </div>

    <div class="section">
        <h2>Training Your Own Models</h2>
        <p>To train your own models using the SACo framework, refer to the training scripts in the GitHub repository:</p>
        <ul style="color: var(--text-secondary); line-height: 2;">
            <li><a href="https://github.com/lstival/SACo/blob/main/change_detection_train.py" target="_blank" rel="noopener">change_detection_train.py</a> - Train change detection models</li>
            <li><a href="https://github.com/lstival/SACo/blob/main/classification.py" target="_blank" rel="noopener">classification.py</a> - Train classification models</li>
            <li><a href="https://github.com/lstival/SACo/blob/main/segmentation_PASTIS.py" target="_blank" rel="noopener">segmentation_PASTIS.py</a> - Train segmentation models</li>
        </ul>
    </div>

    <div class="section">
        <h2>Additional Resources</h2>
        <div class="download-grid">
            <div class="download-card">
                <h3>📦 Zenodo Repository</h3>
                <p>Access all model weights and supplementary materials</p>
                <a href="https://zenodo.org/records/13736623" target="_blank" rel="noopener" class="btn">Visit Zenodo</a>
            </div>
            <div class="download-card">
                <h3>💻 GitHub Repository</h3>
                <p>View source code, training scripts, and documentation</p>
                <a href="https://github.com/lstival/SACo" target="_blank" rel="noopener" class="btn">Visit GitHub</a>
            </div>
            <div class="download-card">
                <h3>📄 Paper</h3>
                <p>Read the full paper published in ISPRS Journal</p>
                <a href="https://www.sciencedirect.com/science/article/pii/S0924271625000826" target="_blank" rel="noopener" class="btn">Read Paper</a>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Support</h2>
        <p>If you encounter any issues or have questions about using the models, please:</p>
        <ul style="color: var(--text-secondary); line-height: 2;">
            <li>Check the <a href="https://github.com/lstival/SACo/issues" target="_blank" rel="noopener">GitHub Issues</a> for existing solutions</li>
            <li>Open a new issue on GitHub with detailed information about your problem</li>
            <li>Refer to the paper for methodological details</li>
        </ul>
    </div>
</div>
