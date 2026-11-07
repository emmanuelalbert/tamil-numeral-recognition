# Tamil Numeral Recognition Using Convolutional Neural Networks

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![Task](https://img.shields.io/badge/Task-Handwritten%20Tamil%20Digit%20Recognition-6f42c1.svg)]()

A deep-learning system for recognizing isolated handwritten Tamil
numerals (classes 1--10) from image inputs. The research evaluates
pretrained CNN architectures, classical machine-learning baselines, and
a Streamlit web interface for interactive prediction.

> **Research result:** MobileNetV2 achieved **99.84% test accuracy** and
> ResNet-18 achieved **99.68% test accuracy** on the reported 627-image
> test set. ResNet-18 is retained as the primary model in the paper;
> MobileNetV2 is the strongest accuracy/parameter-efficiency result.

------------------------------------------------------------------------

## Table of contents

-   [Overview](#overview)
-   [Key results](#key-results)
-   [Research contributions](#research-contributions)
-   [Dataset](#dataset)
-   [Methodology](#methodology)
-   [Model architectures](#model-architectures)
-   [Experimental setup](#experimental-setup)
-   [Web application](#web-application)
-   [Repository structure](#repository-structure)
-   [Getting started](#getting-started)
-   [Reproducing the experiments](#reproducing-the-experiments)
-   [Limitations and future work](#limitations-and-future-work)
-   [Research paper and citation](#research-paper-and-citation)
-   [Acknowledgements](#acknowledgements)

## Overview

Handwritten numeral recognition is a well-established computer-vision
task, but Tamil handwritten numerals have received comparatively limited
attention, particularly because suitable labelled datasets are scarce.
This project addresses that gap by creating a custom dataset of
handwritten Tamil numerals and evaluating deep-learning and classical
approaches for ten-class classification.

The system follows an end-to-end workflow:

1.  Collect handwritten samples from multiple contributors.
2.  Crop, label, and organize individual numeral images.
3.  Preprocess and augment the images.
4.  Train and evaluate CNN and classical machine-learning models.
5.  Deploy a trained model through a Streamlit interface.
6.  Accept an uploaded image and display the predicted numeral and
    confidence score.

**Scope:** The current system is designed for **isolated, single-numeral
image classification**. It is not a complete multi-digit sequence
recognizer or general-purpose Tamil OCR system.

## Key results

### CNN architecture comparison

  Metric                                      ResNet-18   MobileNetV2
  --------------------------------------- ------------- -------------
  Test accuracy                                  99.68%    **99.84%**
  Macro F1-score                                 0.9968    **0.9984**
  Parameters                                     11.18M     **2.24M**
  Misclassifications (627 test samples)               2         **1**
  GPU forward-pass time per image           **0.93 ms**       1.13 ms
  Best validation epoch                               3             8
  Final validation loss (epoch 20)               0.0131    **0.0055**
  Training time                               12.83 min     13.73 min

The reported test-accuracy difference is only 0.16 percentage points.
The paper notes that this difference is not statistically meaningful
based on the small number of test errors. MobileNetV2 has roughly
one-fifth the parameters, while ResNet-18 had the faster measured GPU
forward pass in the reported setup.

### Baseline comparison

  Model         Approach                             Test accuracy   Macro F1-score
  ------------- ---------------------------------- --------------- ----------------
  MobileNetV2   Pretrained deep CNN                     **99.84%**       **0.9984**
  ResNet-18     Pretrained deep CNN                         99.68%           0.9968
  HOG + SVM     Hand-crafted features + SVM                 97.13%           0.9714
  HOG + KNN     Hand-crafted features + KNN                 93.30%           0.9312
  Simple CNN    Shallow CNN trained from scratch            80.06%           0.7905

These results indicate that pretrained CNN feature representations
performed best in the reported experiments. HOG + SVM was a competitive
classical baseline, while the shallow CNN trained from scratch struggled
to generalize on the available dataset.

### ResNet-18 test-set performance

  Metric                       Score
  ------------------------- --------
  Training accuracy           99.69%
  Validation accuracy         99.84%
  Test accuracy               99.68%
  Macro precision             0.9968
  Macro recall                0.9968
  Macro F1-score              0.9968
  Test samples                   627
  Test misclassifications          2

The two reported ResNet-18 test errors were one instance of digit 4
predicted as digit 1 and one instance of digit 6 predicted as digit 9.

### Cross-validation and statistical comparison

The study also reports five-fold cross-validation over 4,179 images:

-   ResNet-18: 11 out-of-fold errors.
-   MobileNetV2: 10 out-of-fold errors.
-   McNemar's exact test: p = 1.00.
-   The paper reports no statistically significant difference between
    the two architectures.
-   The 6↔9 confusion was a recurring source of errors.

## Research contributions

-   **Custom dataset:** A balanced collection of handwritten Tamil
    numeral images covering classes 1--10.
-   **Architecture evaluation:** A controlled comparison of pretrained
    ResNet-18 and MobileNetV2 models.
-   **Classical baselines:** Evaluation of HOG-based SVM and KNN
    classifiers.
-   **From-scratch baseline:** Comparison with a shallow CNN trained
    without pretrained initialization.
-   **Error analysis:** Examination of class confusions, prediction
    confidence, cross-validation errors, and McNemar's test.
-   **Practical deployment:** A Streamlit application for uploading a
    numeral image and viewing a prediction.

The contribution is primarily the dataset, systematic empirical
evaluation, and deployable recognition pipeline---not the proposal of a
new CNN architecture.

## Dataset

The research dataset was collected specifically for this study.

  Property                          Description
  --------------------------------- ---------------------------------------------------
  Task                              Isolated handwritten Tamil numeral classification
  Number of classes                 10 (numerals 1--10)
  Approximate dataset size          4,000--4,300 images
  Samples per class                 Approximately 400--430
  Contributors                      More than 300 individuals
  Collection medium                 Handwritten entries on A4 paper
  Writing instruments               Black or blue ink
  Image capture device              Apple iPhone 14 rear camera
  Original image format             JPEG
  Original capture resolution       4032 × 3024 pixels
  Model image size                  224 × 224 pixels
  Train / validation / test split   70% / 15% / 15%

Participants came from varied age groups, educational backgrounds, and
handedness. Some participants provided samples using both hands.
Individual numerals were manually cropped, labelled, and stored in
class-specific folders.

**Dataset availability:** The research paper states that the data
supporting the findings are available within the article. The full image
dataset is not assumed to be included in this GitHub repository. Check
the paper or contact the authors for dataset access and reuse
conditions.

### Data preprocessing and augmentation

The paper describes the following preprocessing operations:

-   Manual image inspection, cropping, and class labelling.
-   Resizing to 224 × 224 pixels.
-   Grayscale conversion.
-   Pixel normalization.
-   Training/validation/test partitioning.
-   Training-time augmentation using random rotations of up to 10° and
    horizontal flipping.
-   Validation and test transformations without random flipping; the
    test set is processed without training-time augmentation.

The paper describes normalization in two ways: pixel scaling to \[0,
1\], and a torchvision normalization configuration with mean = 0.5 and
standard deviation = 0.5. It also specifies a three-channel model input
in the architecture tables. For reproducibility, the preprocessing used
at inference must match the exact preprocessing expected by the selected
trained checkpoint.

## Methodology

The overall pipeline is:

``` text
Handwritten Tamil Numeral Collection
                |
                v
       Image Cropping & Labelling
                |
                v
       Preprocessing & Augmentation
                |
                v
       70% Train | 15% Validation | 15% Test
                |
                v
       Model Training and Comparison
                |
       +--------+---------+----------------+
       |                  |                |
       v                  v                v
    ResNet-18          MobileNetV2       Baselines
       |                  |             HOG + SVM
       |                  |             HOG + KNN
       |                  |             Simple CNN
       +------------------+----------------+
                          |
                          v
                 Model Evaluation
                          |
                          v
                 Streamlit Web App
                          |
                          v
              Uploaded Image Prediction
```

### ResNet-18

ResNet-18 uses residual/shortcut connections to support gradient flow
through a deeper convolutional network. Its main stages are:

1.  Initial 7 × 7 convolution, batch normalization, ReLU, and max
    pooling.
2.  Four residual stages with 64, 128, 256, and 512 channels.
3.  Adaptive average pooling.
4.  Fully connected classification layer with 10 output classes.

### MobileNetV2

MobileNetV2 uses inverted residual blocks and depthwise separable
convolutions to reduce parameter count. Its classifier is adapted to the
same ten numeral classes.

### Classical and shallow baselines

-   **HOG + SVM:** Histogram of Oriented Gradients features followed by
    a Support Vector Machine classifier.
-   **HOG + KNN:** HOG features followed by a K-Nearest Neighbours
    classifier.
-   **Simple CNN:** A shallow convolutional neural network trained from
    scratch.

## Experimental setup

The paper reports the following training environment and configuration
for the ResNet-18 and MobileNetV2 comparison.

  Setting                     Value
  --------------------------- -----------------------------
  Deep-learning framework     PyTorch
  Computer                    Local machine
  GPU                         NVIDIA RTX 3050
  Reported CUDA version       13.0
  Optimizer                   Adam
  Batch size                  32
  Epoch budget                20
  Loss function               Cross-entropy loss
  Input resolution            224 × 224
  Number of output classes    10
  Pretrained initialization   ImageNet-pretrained weights
  Model selection             Best validation checkpoint

The reported best validation epochs were epoch 3 for ResNet-18 and epoch
8 for MobileNetV2. The paper does not specify every optimizer
hyperparameter or all implementation details in the manuscript; consult
the project source code and configuration files for the exact values
used in a particular run.

## Web application

A Streamlit application provides an interactive interface for
single-image prediction.

### User workflow

1.  Open the Streamlit application.
2.  Upload an image in PNG, JPG, or JPEG format.
3.  Inspect the uploaded image shown in the interface.
4.  Run the recognition workflow.
5.  View the predicted Tamil numeral and confidence score.

The manuscript describes image upload, image preview, real-time
prediction, and a minimal interface. The application is intended for
isolated numeral images rather than full-page document recognition.

## Repository structure

The following reflects the project layout at the time this README was
prepared. Some generated files, model weights, and dataset contents may
be excluded from version control.

``` text
tnr_model/
├── analysis/
│   ├── app/                   # Streamlit application
│   ├── checkpoints/           # Saved model checkpoints (not committed)
│   ├── data/
│   │   ├── raw/               # Original dataset (not committed)
│   │   └── splits/             # Dataset partitions (not committed)
│   ├── outputs/
│   │   ├── cv/
│   │   │   ├── mobilenet_v2_k5_e10/
│   │   │   └── resnet18_k5_e10/
│   │   ├── figures/            # Generated plots and figures
│   │   ├── logs/               # Experiment logs
│   │   └── reports/            # Generated evaluation reports
│   └── src/                    # Source code
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting started

### Prerequisites

-   Python 3.x
-   Git
-   pip
-   A compatible PyTorch installation
-   A trained model checkpoint for inference

A CUDA-capable NVIDIA GPU is useful for training, but the manuscript
does not establish a minimum hardware requirement for running the web
application.

### 1. Clone the repository

``` bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd <YOUR_REPOSITORY>
```

Replace the placeholders with the actual GitHub repository details.

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

``` cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux / macOS:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with the locally
installed NVIDIA driver and CUDA environment. Refer to the [official
PyTorch installation
selector](https://pytorch.org/get-started/locally/).

### 4. Add the required model checkpoint

Model weights are excluded from the repository by the project's ignore
rules. Obtain the appropriate checkpoint from the project maintainers or
the experiment output and place it in the location expected by the
application.

Do not rename a checkpoint or change its architecture without also
updating the application configuration.

### 5. Launch the Streamlit application

The manuscript confirms that the application uses Streamlit, but the
manuscript and repository overview do not identify the exact Python
entry-point filename. From the repository root, run:

``` bash
streamlit run analysis/app/<APP_ENTRYPOINT>.py
```

Replace `<APP_ENTRYPOINT>.py` with the actual Streamlit script in
`analysis/app/`.

## Reproducing the experiments

The research reports a 70/15/15 train/validation/test split and a
separate five-fold cross-validation analysis. To reproduce the published
results:

1.  Obtain the original labelled dataset and confirm the class-to-label
    mapping.
2.  Recreate the reported data partitions, keeping the test set isolated
    from training and model selection.
3.  Apply the same preprocessing and augmentation used by the original
    training pipeline.
4.  Initialize the ResNet-18 and MobileNetV2 backbones with the reported
    pretrained weights.
5.  Replace each classifier head with a ten-class output layer.
6.  Train using the reported 20-epoch budget, batch size of 32, Adam
    optimizer, and cross-entropy loss.
7.  Select checkpoints using validation performance.
8.  Evaluate once on the held-out test set.
9.  Report accuracy, precision, recall, F1-score, confusion matrix, and
    error cases.
10. For the architecture comparison, retain the same data split and
    training conditions across models.

The manuscript reports the experimental outcomes but does not provide a
complete executable command-line training interface or every
hyperparameter in the paper. Use the implementation in `analysis/src/`
and its actual configuration files as the source of truth for runnable
training commands.

## Limitations and future work

The study identifies several limitations and directions for further
development:

-   **Isolated numeral scope:** The current model recognizes one numeral
    per image; multi-digit sequence recognition remains future work.
-   **Real-world robustness:** The system has not yet been extensively
    evaluated on noisy, unconstrained real-world images.
-   **Visually similar classes:** The 6↔9 pair is a recurring source of
    confusion. Other observed confusions include 2↔3 and 1↔4.
-   **Confidence is not a guarantee:** Most errors were
    lower-confidence, but the study also observed a high-confidence
    error. Confidence scores should not be treated as proof that a
    prediction is correct.
-   **Dataset coverage:** Broader handwriting styles and capture
    conditions would help assess generalization beyond the collected
    samples.

Proposed future directions include multi-digit recognition, evaluation
on more varied real-world data, GAN-based data augmentation, and support
for numerals from additional Indian scripts.

## Notes for repository maintainers

Before publishing the repository, verify and update the following:

-   Replace the GitHub clone URL placeholders with the actual repository
    URL.
-   Confirm the exact Streamlit entry-point filename and checkpoint
    path.
-   Ensure `requirements.txt` contains the dependencies required by the
    current implementation.
-   Add a `LICENSE` file if the project is intended for reuse; no
    software licence is specified in the research manuscript.
-   Document the dataset access and usage terms before distributing any
    collected handwriting images.
-   Ensure the inference preprocessing exactly matches the preprocessing
    used to train the checkpoint.
-   Add screenshots of the application and representative predictions if
    they are available and approved for public release.
