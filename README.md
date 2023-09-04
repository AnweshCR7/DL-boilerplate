# Focus distance prediction.
Deep neural network to predict the 'relative focus distance' from the focus plane based on a
single grayscale image containing cells resting on a glass.
## Setup
### Preparing the environment:
Poetry installation docs: https://python-poetry.org/docs/ 

To make sure poetry is installed and available in your system, run a version check in your terminal: 

```Poetry --version -> Poetry (version 1.x.x) ```
1. A poetry environment needs to be set up for running this repository locally.

```
poetry shell
```

A local env will get created (and activated) coupled with the python version that was selected for this repository
(in this case python 3.8):

```
(focus-distance-prediction-py3.8) anweshcr7@home:~/Dev/focus-distance-prediction$
```
2. Update poetry lock dependencies

```
poetry lock --no-update
```

3. Once the .lock file is resolved, we can install packages using the following command

```
poetry install
```

4. After the poetry packages have been correctly installed, the shell prompt should look like this:

```
(focus-distance-prediction-py3.8) anweshcr7@home:~/Dev/focus-distance-prediction$
```

5. (If Required) Missing packages may be installed using the following command:

```
poetry add <pkg-name>
```

or

```
poetry add <pkg-name>==<version>
```


## Dataset
Grayscale images in the format: 
```<unique_image_name>_<absolute_focus_height>_<relative_focus_distance>.png``` (Provided by Lumicks).
Each <unique_image_name> has been acquired at several <absolute_focus_height> and
<relative_focus_distance> combination. No train/test split was provided.

| Full Dataset stats                      |  #   |
|:----------------------------------------|:----:|
| Number of image instances (2048 x 2048) | 3117 |
| Number of unique_image_names            |  37  |

**Observations**:

* Each (unique) image_name has been recorded at multiple focus values.
* To prevent data leak, generate a clean split between train and validation sets (eventually also test).
* We need to make sure the split is done on an image_name level i.e. **All** the images from the same unique_image_name **should only belong to one set - train, validation or test!**
  (Check [data exploration notebook](./Data_exploration.ipynb) and [dataset split code](./src/dataset_split.py))

| Dataset stats after train/test split | # Unique image names |  # image instances  |
|:-------------------------------------|:--------------------:|:-------------------:|
| Train                                |          28          |        2568         |
| Validation                           |          7           |         507         |
| Test                                 |          2           |         42          |

The metadata for this split can be found in the file [dataset_split_metadata.json](data%2Fsplit_data%2Fdataset_split_metadata.json)

## Experiments
Shared parameters:
```
GPU: GeForce RTX 3060 (Mem: 12G)
Image (HxW): 1280x1280 
Transforms: Normalization
Models: Resnet34 and Wide Resnet
Initial learning rate: 1e2
Epochs: 120
Schedular: Reduce-LR-on-Plateau
```

Experiment details and logging can be found on Tensorboard. Run:

```tensorboad --logdir=./results```

### Results on the Test set

| Model                    | Batch-size | Image-size | LR  |  MAE  | RMSE  |
|:-------------------------|:----------:|:----------:|:---:|:-----:|:-----:|
| Resnet-34 (with L1-loss) |     8      |    1280    | 1e2 | 6.32  | 8.59  |
| Resnet-34 (with L2-loss) |     8      |    1280    | 1e2 | 26.01 | 29.28 |
| Wide-resnet-50_2         |     6      |    768     | 1e2 | 16.60 | 21.37 |


|          Model           |                  Scatter Plot                  |                 Error Histogram                  |
|:------------------------:|:----------------------------------------------:|:------------------------------------------------:|
| Resnet-34 (with L1-loss) |  ![](plots/resnet34_l1loss/scatter_plot.png)   |  ![](plots/resnet34_l1loss/histogram_plot.png)   |
| Resnet-34 (with L2-loss) |  ![](plots/resnet34_l2loss/scatter_plot.png)   |  ![](plots/resnet34_l2loss/histogram_plot.png)   |
|     Wide-resnet-50_2     | ![](plots/wide_resnet_l1loss/scatter_plot.png) | ![](plots/wide_resnet_l1loss/histogram_plot.png) |

## Quick Conclusions:
1. Used a ```resnet34``` to keep things simple without getting into the whole idea about trying to determine the "best architecture".
   1. Used normalization transform and then resize on the input grayscale images.
   2. The choice of loss function can be found in the [data exploration notebook](./Data_exploration.ipynb)
   3. Didn't have enough resources to run a lot of experiments but this was not the point of the assignment anyway.
   4. The resnet34 experiment with L2Loss got messed up - need to debug.
2. Since we had enough data (images), trained the resnet using pretrained = False. 
   1. Thought/intuition: Pretrained resnet isn't suited to this task of determining the focus distance imo.
   2. Empirical: Imagenet weights did start with a lower loss but did poorly and took longer in probably overwriting the weights.
3. Also tried ```wide resnets``` just to get another opinion (apparently they are quicker to train and use less memory - shallower than original Resnets)
   1. Didn't observe reasonable increase in speed and also had to reduce image size to run a wide resnet (needed more memory)
   2. I think both these issues along with not having correct hyperparameters meant that I had to drop this approach.
4. All in all, resnet34 with no pretrained weights was the best performing model (details provided above).
   1. Image size played a big role in the model performance: Higher image size resulted in better performance (seems intuitive) 
   2. Would've liked to experiment with a different scheduler or early stopping.
   2. Maybe needs a couple of FC layers at the end?
