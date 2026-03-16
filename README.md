# TAKNI: Temporal Attention-Driven Kernel Norm Integration Network for Fault Diagnosis

## Overview

This repository contains the source code and supplementary materials for the paper titled "TAKNI: Temporal Attention-Driven Kernel Norm Integration Network for Fault Diagnosis". 

TAKNI (Temporal Attention-Driven Kernel Norm Integration Network) is an advanced method for cross-working-condition fault diagnosis that combines two key innovations: a probabilistic sparsified temporal attention mechanism and Wasserstein distance-based distribution alignment optimization. The Temporal Attention Module (TAM) dynamically identifies long-term dependencies and important time points in time-series data, enhancing the discriminability of extracted features. The Multi-Scale Kernel Wasserstein Module (MSKWM) optimizes the alignment between source and target domain distributions, particularly improving classification accuracy for minority classes in imbalanced scenarios.

## Paper Abstract

In real industrial environments, changes in operating conditions such as load and speed significantly alter data distributions, adversely affecting fault diagnosis model performance. For example, a model trained under specific working conditions often performs poorly or even fails completely when applied to different conditions. This issue, known as the cross-domain problem, presents a major challenge in industrial fault diagnosis. Additionally, imbalanced class distributions severely degrade classification performance for minority classes, further limiting the practicality of these models. To address these challenges, this study proposes a discriminative model-based method for cross-working-condition fault diagnosis. The proposed approach introduces two key innovations: a probabilistic sparsified temporal attention mechanism and Wasserstein distance-based distribution alignment optimization. The Temporal Attention Module (TAM) dynamically identifies long-term dependencies and important time points in time-series data, enhancing the discriminability of extracted features. The Multi-Scale Kernel Wasserstein Module (MSKWM) optimizes the alignment between source and target domain distributions, particularly improving classification accuracy for minority classes in imbalanced scenarios. By combining these techniques, the proposed algorithm achieves exceptional adaptability and robustness across various industrial scenarios. Experiments on five publicly available datasets confirm its state-of-the-art performance, providing an effective solution for cross-working-condition fault diagnosis.

## Key Contributions

1. Introduction of a TAM that effectively captures long-term dependencies in time-series data, achieving better feature extraction and significantly improving the accuracy of cross-domain diagnosis.

2. Proposal of a MSKWM, which addresses the issue of majority class bias, enhances the discriminative ability for minority class samples, and reduces the risk of misclassification.

3. Design of the TAKNI algorithm by integrating TAM and MSKWM, achieving state-of-the-art performance on five publicly available fault diagnosis datasets and demonstrating its effectiveness and generalization capability in various cross-domain fault diagnosis tasks.

## Results Summary

The TAKNI model demonstrates exceptional performance across multiple datasets in transfer learning tasks. Its average performance on various datasets:

- **CWRU dataset**: Best time-domain model achieves 99.68%, best frequency-domain model reaches 100%
- **PHM2009 dataset**: Best time-domain model achieves 49.66%, best frequency-domain model reaches 61.29%
- **JNU dataset**: Best time-domain model achieves 96.88%, best frequency-domain model reaches 99.55%
- **PU dataset**: Best time-domain model achieves 64.17%, best frequency-domain model achieves 75.03%
- **SEU dataset**: Best time-domain model achieves 53.01%, best frequency-domain model reaches 83.40%

Detailed experimental results are available in the TAKNI_results.pdf file included in this repository.

## Repository Structure

```
TAKNI/
├── Diffusion/                 # Diffusion model components
├── DiffusionFreeGuidence/     # Conditional diffusion model components
├── bottlenecks/              # Bottleneck layers implementation
├── dataset/                  # Dataset loaders and utilities
├── dataset_diffusion/        # Diffusion-specific dataset processing
├── datasets/                 # Various dataset implementations
├── loss/                     # Loss function implementations
├── models/                   # Model architectures
├── transcal/                 # Transfer calibration components
├── utils/                    # Utility functions
├── optim/                    # Optimization algorithms
├── Main_Diffusion.py         # Main entry point for diffusion models
├── Scheduler.py              # Training schedulers
├── train_*.py                # Training scripts
├── *.sh                      # Shell scripts for running experiments
└── TAKNI_results.pdf         # Detailed experimental results
```

## Requirements

The project is built using Python and PyTorch. Specific dependencies are documented in the paper and implementation.

## Usage

See the individual script files and shell scripts for running the experiments described in the paper.

## License

This project is licensed under the terms provided in the LICENSE file.

## Citation

If you use this code or reference our results in your research, please cite our paper:

```
TAKNI: Temporal Attention-Driven Kernel Norm Integration Network for Fault Diagnosis
Peijian Zeng, Junhao Chen, Aimin Yang
```

## Acknowledgments

We thank the contributors and institutions that made the datasets used in our experiments publicly available.