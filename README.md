## Deep Domain Adaptation by Joint Appearance Adaptation

This repository contains the original code for the methods proposed in [Wittich and Rottensteiner, 2021]( https://doi.org/10.1016/j.isprsjprs.2021.08.004) and [Wittich, 2023]().
The method addresses unsupervised domain adaptation with neural networks based on appearance adaptation.

This implementation is written in python and uses the [pytorch library](https://pytorch.org/).

### Contents:
This framework can be used to perform __supervised training__ and various variants of __unsupervised domain adaptation__.
On the one side, it can be used to apply joint appearance adaptation (cf. links above), but the framework can also be used 
to apply variants of instance transfer and adversarial representation transfer.

### Setup:
To use the code, follow these steps:
1. Clone the repository.
2. Install the [python conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), defined in `environment.yml` by running `conda env create -f environment.yml` from an anaconda prompt. Alternatively you can install the packages manually. There is no strict dependency on the python or pytorch version.
3. You may also need to install the [jpeg2000 decoder](https://itecnote.com/tecnote/python-pillow-and-jpeg2000-decoder-jpeg2k-not-available/) to run the examples.
4. Activate the environment by running `conda activate jda`
5. Navigate inside the `code/` folder and run an experiment `python main.py path/to/config.yaml`

### Configuration Files:
This framework uses configuration files to define any experiment.
A complete prototype serving as documentation can be found at `code/Documentation/_config_documentation.yaml`.
I recommend to have a look at the examplare configuration files provided in the ``runs/`` folder.

### Example: Running Domain Adaptation on Domains from the GeoNRW Dataset:
The following processing sequence serves as an example how to use the framework to perform deep domain adaptation.
In the example, source training is performed for the city _Bochum_. 
The classifier is then adapted to the city _Heinsberg_.
The two cities were selected because they were captured in different seasons which has a strong impact on the appearance of vegetation in the images.
Some examples are shown below.

|                   Bochum                   |                   Bochum                   |                   Heinsberg                   |                   Heinsberg                   |
|:------------------------------------------:|:------------------------------------------:|:---------------------------------------------:|:---------------------------------------------:|
| ![1.jpg](examples%2Fbochum_thumbs%2F1.jpg) | ![2.jpg](examples%2Fbochum_thumbs%2F2.jpg) | ![1.jpg](examples%2Fheinsberg_thumbs%2F1.jpg) | ![2.jpg](examples%2Fheinsberg_thumbs%2F2.jpg) |
| ![3.jpg](examples%2Fbochum_thumbs%2F3.jpg) | ![4.jpg](examples%2Fbochum_thumbs%2F4.jpg) | ![3.jpg](examples%2Fheinsberg_thumbs%2F3.jpg) | ![4.jpg](examples%2Fheinsberg_thumbs%2F4.jpg) |

To perform source training and domain adaptation, follow these steps:

1. __Download__ the GeoNRW dataset from [here](https://www.kaggle.com/datasets/javidtheimmortal/geonrw).
2. __Update__ the `PATHS.GeoNRW` attribute in the file `runs/local_paths.yaml`. The value should be set to the path of the root folder of the GeoNRW dataset (the next folders should correspond to the city names). You may use `~CONFIG` to start a relative path.
3. Navigate inside the code folder and activate the conda environment by running `conda activate jda`.
4. __Source training:__ 
   1. Run `python main.py ../runs/source_training/bochum_training.yaml` to train the classifier in the source domain.
   2. Run `python main.py ../runs/source_training/bochum_eval_heinsberg.yaml` to evaluate the classifier in the target domain without adaptation.
4. __Evaluating the classifier after source training:__ Evaluate using the test set of the source domain by running `python main.py ../runs/source_training/bochum_eval.yaml`. Evaluate the classifier without adaptation on the target domain by running `python main.py ../runs/source_training/bochum_eval_heinsberg.yaml`
5. __Domain adaptation:__
   1. Run `python main.py ../runs/domain_adaptation_b2h/[10... - 60...].yaml` to perform the variants of domain adaptation.
   1. Run `python main.py ../runs/domain_adaptation_b2h/[11... - 61...]_eval.yaml` to evaluate the variants of domain adaptation on the target domain.

#### Exemplary results of appearance adaptation:
The following images show the results of appearance adaptation using the variant with discriminator regularization.
The results can be reproduced by running `python main.py ../runs/domain_adaptation_b2h/20_appa_dis_reg.yaml` 
(exemplary appearance adaptations will be stored to `../runs/domain_adaptation_b2h/2_appearance_adaptation_dis_reg/images/4_adapted_images/`)..

|                             Input (left) / Adapted image (right)                             |
|:--------------------------------------------------------------------------------------------:|
| ![appearance_adaptation_1.jpg](examples%2Fbochum_to_heinsberg%2Fappearance_adaptation_1.jpg) |
| ![appearance_adaptation_2.jpg](examples%2Fbochum_to_heinsberg%2Fappearance_adaptation_2.jpg) |


#### Exemplary classification results (image / prediction / reference):

Respective results on the test sets.

|                                                     Source training on Bochum, evaluation on Bochum                                                     |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  ![dortmund-390_5703_rgb.jp2-img-prediction-reference.jpg](examples%2Fbochum_source_training%2Fdortmund-390_5703_rgb.jp2-img-prediction-reference.jpg)  |
|                                  **Source training on Bochum, evaluation on Heinsberg (Naive transfer before adaptation)**                                  |
| ![heinsberg-296_5663_rgb.jp2-img-prediction-reference.jpg](examples%2Fbochum_source_training%2Fheinsberg-296_5663_rgb.jp2-img-prediction-reference.jpg) |
|                 **Source training on Bochum, evaluation on Heinsberg after joint appearance adaptation with discriminator regularization**                 |
|  ![heinsberg-296_5663_rgb.jp2-img-prediction-reference.jpg](examples%2Fbochum_to_heinsberg%2Fheinsberg-296_5663_rgb.jp2-img-prediction-reference.jpg)   |


#### Quantitative results using various adaptation strategies:

|                    Startegy                    | mean F1-Score on target domain [%] |                                                                  Config                                                                  |
|:----------------------------------------------:|:----------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------:|
|                 Naive transfer                 |                53.5                |                      `runs/source_training/bochum_training.yaml`  `runs/source_training/bochum_eval_heinsberg.yaml`                      |
|      Joint adapt. without regularization       |            50.1 (-3.4)             |               `runs/domain_adaptation_b2h/10_appa_baseline.yaml`  `runs/domain_adaptation_b2h/11_appa_baseline_eval.yaml`                |
| Joint adapt. with discriminator regularization |            62.8 (+9.3)             |                `runs/domain_adaptation_b2h/20_appa_dis_reg.yaml`  `runs/domain_adaptation_b2h/21_appa_dis_reg_eval.yaml`                 |
|     Joint adapt. with auxiliary generator      |            49.0 (-4.5)             |                `runs/domain_adaptation_b2h/30_appa_aux_gen.yaml`  `runs/domain_adaptation_b2h/31_appa_aux_gen_eval.yaml`                 |
|          Adaptive batch normalization          |            45.6 (-7.9)             | `runs/domain_adaptation_b2h/40_adaptive_batch_normalization.yaml` `runs/domain_adaptation_b2h/41_adaptive_batch_normalization_eval.yaml` |
|               Instance transfer                |            50.9 (-3.6)             |            `runs/domain_adaptation_b2h/50_instance_transfer.yaml` `runs/domain_adaptation_b2h/51_instance_transfer_eval.yaml`            |
|            Representation transfer             |            59.7 (+6.2)             |      `runs/domain_adaptation_b2h/60_representation_transfer.yaml` `runs/domain_adaptation_b2h/61_representation_transfer_eval.yaml`      |

### Example 2: Running a Batch of Experiments
The framework allows to run a batch of experiments by specifying a list of configurations in YAML files and putting them to the same folder.
Using the script `code/experiment_scheduler.py` all configuration files in a specified folder are used sequentially.
this is useful, if you want to run a batch of experiments with different parameters, e.g. for hyperparameter tuning or to run domain adaptation between multiple domains.

An exemplary setup for hyperparameter tuning is provided in the folder `runs/source_training/tuning_example`.
To run the batch of experiments, simply run `python experiment_scheduler.py runs/source_training/tuning_example`.

In `eval.py` the results of the experiments are presented using the `seaborn` library. 
The output of the example is shown below.

![variant_comparison.png](examples%2Fvariant_comparison.png)

### Adding your own Datasets:
To use your own domains/datasets, the following steps are required:
1. Create a unique name for your domain.
2. Implement a training data loader in `datamanagement.py` and add it to the function `prepare_training_dataset`.
3. Implement a function that pre-loads a subset to (images,labels,names) and add it to the init function of the class ``EvalDataset`` in `datamanagement.py`.
4. Extend the functions ``idmap2color``, `color2idmap` and ``denorm4print`` in `tools.py`.
5. Create a configuration file and run it.

### Pre-trained Models:
I provide a bunch of pre-trained models for various tasks.
They are all based on the U-Net architecture with Xception backbone from [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch).
The parameter depth refers to the encoder stages (parameter SEG_MODEL.UNET.DEPTH in the config).
All checkpoints contain the weights of the model and the optimiser.
All models are trained with SGD with momentum, and it is highly recommended to use the same optimiser for fine-tuning.
For channel normalization cf. paper [1] (usually normalized to zero mean and unit std-dev).

|                                             Link                                              |        Input        |               Output               | Depth |  GSD  |                Task                 |
|:---------------------------------------------------------------------------------------------:|:-------------------:|:----------------------------------:|:-----:|:-----:|:-----------------------------------:|
| [LCC_IrRgGH_20cm_5cl](https://huggingface.co/wittich/JAA/resolve/main/LCC_IrRgGH_20cm_5cl.pt) | 4 Ch. Ir,R,G,Height |        5 Classes (cf. [1])         |   5   | 20 cm |      Land cover classification      |
|   [LCC_IrRG_16cm_6cl](https://huggingface.co/wittich/JAA/resolve/main/LCC_IrRG_16cm_6cl.pt)   |    3 Ch. Ir,R,G     |    6 Classes (cf. [1] + Water)     |   5   | 16 cm |      Land cover classification      |
|  [BDD_IrRGx2_10m_2cl](https://huggingface.co/wittich/JAA/resolve/main/BDD_IrRGx2_10m_2cl.pt)  |  6 Ch. Ir,R,G (x2)  |      2 Classes (Def./No Def.)      |   4   | 10 m  | Bi-temporal deforestation detection |
|   [VL_IrRGx2_10m_3cl](https://huggingface.co/wittich/JAA/resolve/main/VL_IrRGx2_10m_3cl.pt)   |  6 Ch. Ir,R,G (x2)  | 3 Classes (No Dmg./Dmg./Clear-cut) |   4   | 10 m  |    Vitality loss classification     |
|    [RLT_IrRG_10m_reg](https://huggingface.co/wittich/JAA/resolve/main/RLT_IrRG_10m_reg.pt)    |    3 Ch. Ir,R,G     |       1 Channel (regression)       |   4   | 10 m  |  Regression of remaining lifetime   |

### Changing the Config File:
Besides the documentation in `code/Documentation/_config_documentation.yaml` a code prototype is implemented in `config.py`. 
This copy allows to use auto-completion, to get type hints when coding or to perform refactoring. 
If you want to change the configuration file, it is suggestet to **first modify the yaml version** at `code/Documentation/_config_documentation.yaml`.
Afterward, run `python config.py` and copy the auto-generated code to the class `Config` in `config.py`.

### Citing
If you use this code for your research, please cite our paper
```
@article{Wittich2021,
  title = {Appearance based deep domain adaptation for the classification of aerial images},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume = {180},
  pages = {82-102},
  year = {2021},
  issn = {0924-2716},
  doi = {https://doi.org/10.1016/j.isprsjprs.2021.08.004},
  url = {https://www.sciencedirect.com/science/article/pii/S0924271621002045},
  author = {D. Wittich and F. Rottensteiner},
  keywords = {Domain Adaptation, Pixel-wise Classification, Deep Learning, Aerial Images, Remote Sensing, Appearance Adaptation},
}
```

and/or the GitHub repository

```
@misc{Wittich2023,
  author = {Dennis Wittich},
  title = {Joint Appearance Adaptation},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/denniswittich/JointAppearanceAdaptation}}
}
```


### License
This project is licensed under the terms of the MIT license.

### Contact
If you have any questions, please contact me via GitHub or [Research Gate](https://www.researchgate.net/profile/Dennis-Wittich).

### Acknowledgements
- This work uses the [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch).
- I thank my supervisor [Prof. Franz Rottensteiner](https://www.researchgate.net/profile/Franz-Rottensteiner) for his support and the [Institute of Photogrammetry and GeoInformation](https://www.ipi.uni-hannover.de/de/) lead by [Prof. Christian Heipke](https://www.researchgate.net/profile/Christian-Heipke) for providing the possibility to do this research.

### References
[1] Wittich, D., Rottensteiner, F. (2021): Appearance based deep domain adaptation for the classification of aerial images. In: ISPRS Journal of Photogrammetry and Remote Sensing (180), 82-102.
DOI: https://doi.org/10.1016/j.isprsjprs.2021.08.004

[2] Tsai, Y., Hung, W., Schulter, S., Sohn, K., Yang, M. and Chandraker, M., 2018. Learning to adapt
structured output space for semantic segmentation. In: IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), pp. 7472–7481.

[3] Vu, T.-H., Jain, H., Bucher, M., Cord, M. and Perez, P., 2019. ADVENT: Adversarial entropy minimization
for domain adaptation in semantic segmentation. In: IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), pp. 2512–2521.

