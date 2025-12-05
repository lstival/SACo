---
layout: default
title: Home
description: Semantically-Aware Contrastive Learning for multispectral remote sensing images
---

<div class="container">
    <div class="hero">
        <h1>🛰️ SACo+</h1>
        <p>Semantically-Aware Contrastive Learning for Multispectral Remote Sensing Images</p>
    </div>

    <div class="section">
        <h2>Abstract</h2>
        <p>
            Satellites continuously capture vast amounts of data daily, including multispectral remote sensing images (MSRSI), which facilitate the analysis of planetary processes and changes. New machine-learning techniques are employed to develop models to identify regions with significant changes, predict land-use conditions, and segment areas of interest. However, these methods often require large volumes of labeled data for effective training, limiting the utilization of captured data in practice.
        </p>
        <p>
            According to current literature, self-supervised learning (SSL) can be effectively applied to learn how to represent MSRSI. This work introduces <strong>Semantically-Aware Contrastive Learning (SACo+)</strong>, a novel method for training a model using SSL for MSRSI. Relevant known band combinations are utilized to extract semantic information from the MSRSI and texture-based representations, serving as anchors for constructing a feature space.
        </p>
        <p>
            This approach is resilient against changes and yields semantically informative results using contrastive techniques based on sample visual properties, their categories, and their changes over time. This enables training the model using classic SSL contrastive frameworks, such as MoCo and its remote sensing version, SeCo, while also leveraging intrinsic semantic information.
        </p>
        <p>
            SACo+ generates features for each semantic group (band combination), highlighting regions in the images (such as vegetation, urban areas, and water bodies), and explores texture properties encoded based on Local Binary Pattern (LBP). To demonstrate the efficacy of our approach, we trained ResNet models with MSRSI using the semantic band combinations in SSL frameworks. Subsequently, we compared these models on three distinct tasks: land cover classification task using the EuroSAT dataset, change detection using the OSCD dataset, and semantic segmentation using the PASTIS and GID datasets.
        </p>
        <p>
            Our results demonstrate that <strong>leveraging semantic and texture features enhances the quality of the feature space, leading to improved performance in all benchmark tasks</strong>.
        </p>
    </div>

    <div class="section">
        <h2>Semantic Aware Architecture</h2>
        <p>
            The texture and semantic band features act as a kind of "guide" for the model, which is trained to process each group of bands and textures individually. This means that the output should be similar for images of the same region with different magnifications and timestamps, while at the same time increasing the dissimilarity for the other regions in the memory bank.
        </p>
        <div class="image-container">
            <img src="{{ '/assets/images/space_representation_texture.png' | relative_url }}" alt="Semantic Aware Feature Space Representation">
            <p class="image-caption">Semantic Aware feature space representation showing how texture and semantic bands guide the model to create meaningful representations</p>
        </div>
    </div>

    <div class="section">
        <h2>Semantic Band Combinations</h2>
        <p>
            The Sentinel-2 bands are grouped following a specific logic. This combination allows the image to be more representative of different types of soil. Each group comprises three bands that can highlight and represent characteristics of the ground, such as vegetation, urban areas, and visible colors.
        </p>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Groups</th>
                        <th>R</th>
                        <th>G</th>
                        <th>B</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Natural Colors</td>
                        <td>B04</td>
                        <td>B03</td>
                        <td>B02</td>
                    </tr>
                    <tr>
                        <td>Near-Infrared</td>
                        <td>B08</td>
                        <td>B04</td>
                        <td>B03</td>
                    </tr>
                    <tr>
                        <td>Urban</td>
                        <td>B12</td>
                        <td>B11</td>
                        <td>B04</td>
                    </tr>
                    <tr>
                        <td>Agriculture</td>
                        <td>B11</td>
                        <td>B8A</td>
                        <td>B02</td>
                    </tr>
                    <tr>
                        <td>Atmospheric Pen.</td>
                        <td>B12</td>
                        <td>B11</td>
                        <td>B8A</td>
                    </tr>
                    <tr>
                        <td>Complementary 1</td>
                        <td>B01</td>
                        <td>B05</td>
                        <td>B06</td>
                    </tr>
                    <tr>
                        <td>Complementary 2</td>
                        <td>B07</td>
                        <td>B08</td>
                        <td>B10</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>

    <div class="section">
        <h2>Texture Features</h2>
        <p>
            To estimate the texture features, we used a Local Binary Pattern for each band following the same groups of the semantic groups. The LBP features help capture local texture patterns that are invariant to monotonic gray-scale changes.
        </p>
        <div class="image-container">
            <img src="{{ '/assets/images/LBP_example_v2.svg' | relative_url }}" alt="LBP Texture Features Example" style="max-width: 60%;">
            <p class="image-caption">On the top row, the original bands were used to estimate the LBP features, while the bottom displays matrices of LBP values with the same dimensions as the original image. It is possible to observe highlighted patterns in the images, such as edges and contours.</p>
        </div>
    </div>

    <div class="section">
        <h2>Contrastive Training Pipeline</h2>
        <p>
            The main architecture of the SACo training pipeline processes original, augmented, and temporally different images to produce and align feature representations.
        </p>
        <div class="image-container">
            <img src="{{ '/assets/images/pipeline.svg' | relative_url }}" alt="SACo Training Pipeline" style="max-width: 70%;">
            <p class="image-caption">SACo training pipeline showing how semantic bands and texture features are processed through the contrastive learning framework</p>
        </div>
    </div>

    <div class="section">
        <h2>Key Features</h2>
        <ul style="color: var(--text-secondary); line-height: 2;">
            <li>🛰️ <strong>Semantic Band Combinations</strong>: Leverages known Sentinel-2 band combinations to extract meaningful semantic information</li>
            <li>🔍 <strong>Texture-Based Representations</strong>: Uses Local Binary Patterns (LBP) to capture texture features</li>
            <li>🎯 <strong>Contrastive Learning</strong>: Builds on MoCo and SeCo frameworks with semantic awareness</li>
            <li>📊 <strong>Multi-Task Performance</strong>: Achieves state-of-the-art results on classification, segmentation, and change detection</li>
            <li>🚀 <strong>Self-Supervised</strong>: Reduces dependency on large labeled datasets</li>
        </ul>
    </div>

    <div class="section">
        <h2>Quick Links</h2>
        <div class="download-grid">
            <div class="download-card">
                <h3>📥 Downloads</h3>
                <p>Access pre-trained model weights and evaluation scripts</p>
                <a href="{{ '/downloads' | relative_url }}" class="btn">View Downloads</a>
            </div>
            <div class="download-card">
                <h3>📊 Results</h3>
                <p>Explore quantitative and qualitative results across all tasks</p>
                <a href="{{ '/results' | relative_url }}" class="btn">View Results</a>
            </div>
            <div class="download-card">
                <h3>💻 GitHub</h3>
                <p>Access the source code and contribute to the project</p>
                <a href="https://github.com/lstival/SACo" target="_blank" rel="noopener" class="btn">View Repository</a>
            </div>
        </div>
    </div>
</div>
