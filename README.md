# Malaria Diagnosis with CNNs and Transfer Learning

Group 6 — Formative 2, Machine Learning

## About the project

Malaria is diagnosed by examining stained blood smears under a microscope, which is slow and depends on trained microscopists, both scarce in many of the regions most affected by the disease. This project applies deep learning to automate one step of that process: classifying a single red blood cell image as **parasitized** or **uninfected**.

Each group member designed, trained and evaluated one model, running at least seven documented experiments tracked in TensorBoard. The models were evaluated on held-out test data with accuracy, precision, recall (sensitivity), specificity, F1-score and ROC-AUC, and explained with Grad-CAM.

## Dataset

[NIH/NLM Malaria Cell Images](https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip): 27,558 thin-blood-smear cell images, 13,779 parasitized and 13,779 uninfected.

Each model uses a stratified 80 / 10 / 10 train / validation / test split (about 22,046 / 2,756 / 2,756 images). The dataset itself is not stored in this repository.

## Models

| Model | Approach | Folder | Author |
|---|---|---|---|
| Custom ResNet | Residual CNN built with the Keras Subclassing API, trained from scratch | [`custom_resnet/`](custom_resnet/) | Bode Murairi |
| DenseNet121 | Transfer learning from ImageNet, fine-tuning the last dense block | [`malaria_densenet/`](malaria_densenet/) | Faith Irakoze |
| ResNet50 | Transfer learning from ImageNet, fine-tuning the last 50 layers | [`ResNet50/`](ResNet50/) | Kevine Uwisanga |
| Deep plain CNN | Plain convolutional network (no skip connections), trained from scratch | [`deep-plain-CNN/`](deep-plain-CNN/) | Atete Mpeta Shina |

### Custom ResNet

A residual network following the ResNet-18 layout (He et al., 2016) with half the usual width: a 3×3 convolution stem, four stages of two residual blocks (32, 64, 128 and 256 filters), global average pooling and a single sigmoid output, for about 2.8 M parameters. The first block of each later stage halves the spatial size and uses a 1×1 projection shortcut. The final model adds a learning-rate plateau schedule and lossless augmentation (flips and 90° rotations).

## Results

Final model of each architecture, evaluated on its test set (threshold 0.5):

| Model | Final experiment | Accuracy | Precision | Sensitivity (recall) | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Custom ResNet | `custom_resnet_exp_05_augmentation_lossless` | 0.9648 | 0.9699 | 0.9594 | 0.9702 | 0.9646 | 0.9936 |
| DenseNet121 | unfreeze conv5 block<sup>1</sup> | 0.9680 | 0.9694 | 0.9666 | 0.9695 | 0.9680 | 0.9949 |
| ResNet50 | `resnet50_exp_06_unfreeze_last50_lr_5e-6` | 0.9641 | 0.9826 | 0.9448 | 0.9833 | 0.9634 | 0.9948 |
| Deep plain CNN | `plain_cnn_exp_07_two_convs_per_block` | 0.9637 | 0.9699 | 0.9572 | 0.9702 | 0.9635 | 0.9916 |

Targets from the brief: custom models ≥ 0.90 sensitivity and ≥ 0.90 specificity; transfer-learning models ≥ 0.95 sensitivity and ≥ 0.90 specificity. All models meet their specificity target; ResNet50's sensitivity (0.9448) falls just short of the 0.95 transfer-learning target.

Each member used their own random split, so the test sets differ. With about 1,378 images per class, differences of less than roughly one percentage point between models should be treated as ties.

<sup>1</sup> DenseNet121 values computed from the test confusion matrix in [`malaria_densenet/final_confmat_roc.png`](malaria_densenet/final_confmat_roc.png).

## Repository structure

```
├── custom_resnet/        Custom ResNet notebook, report charts, TensorBoard logs
├── malaria_densenet/     DenseNet121 notebook, final figures, results.json
├── ResNet50/             ResNet50 notebook
├── deep-plain-CNN/       Deep plain CNN notebook, charts, results tables
└── README.md
```

Trained model weights, the dataset and most TensorBoard logs are not stored in the repository because of their size; see each model's notebook for where its logs are shared.

## Running the notebooks

Each notebook is self-contained and was developed on Google Colab and Kaggle with a GPU. Open a notebook, select a GPU runtime, and run it from top to bottom; the setup cells describe how each one obtains the dataset. The custom ResNet notebook detects whether it runs on Colab or Kaggle and downloads the dataset automatically if it is missing.

## Contributors

| Name | GitHub | Model |
|---|---|---|
| Bode Murairi | [@bodemurairi](https://github.com/bodemurairi) | Custom ResNet |
| Faith Irakoze | [@Faithirakoze](https://github.com/Faithirakoze) | DenseNet121 |
| Kevine Uwisanga | [@u-kevine](https://github.com/u-kevine) | ResNet50 |
| Atete Mpeta Shina | [@shina227](https://github.com/shina227) | Deep plain CNN |

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
- Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. *CVPR*.
- Rajaraman, S., Antani, S. K., Poostchi, M., et al. (2018). Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images. *PeerJ*, 6, e4568.
- Selvaraju, R. R., Cogswell, M., Das, A., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.
