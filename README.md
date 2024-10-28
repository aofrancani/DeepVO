# DeepVO - Unofficial PyTorch Implementation

This repository contains an unofficial PyTorch implementation of **DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks**. This code is tailored to train and evaluate the DeepVO model for visual odometry tasks, with configurations and scripts for easy experimentation.


<figure>
   <img src="deepvo_model.png" alt="DeepVO Model">
   <figcaption>[Source]: DeepVO model architecture. Extracted from the original DeepVO paper.</figcaption>
</figure>

---


## Implementation Details:
- **Input Resolution**
    - Due to the architecture, the input size must be a **multiple of 64**.
    - The paper does not specify the input size used, but in the figure, we see `(1280 x 384 x 3)`.
    - We are using half of this: `(640 x 192 x 3)`.

- **Optimiser**
    - The authors mention "*Adagrad optimiser is employed to train the network for up to 200 epochs with learning rate 0.001*"

- **Pre-trained FlowNet**
    - The authors mention "*the CNN is based on a pre-trained FlowNet model*" 
    - This is not yet implemented in this repository.

- **Normalization**
    - The authors do not specify normalization of the ground truth poses. Since they use a weighted MSE loss with a scale factor of $\kappa=100$ (high value), we assume they do not apply standard normalization.
    - Seeing the y-axis of the training/validation losses in Fig. 4 would provide more clarity.

- **Dropout**
    - The authors mention "*Dropout and early stopping techniques are introduced to prevent the models from overfitting*", but they provide no further details.
    - We tested with `lstm_dropout = 0.2` and `conv_dropout = 0.1`.

---

## 1. Dataset
To use this implementation, download the KITTI dataset directly from the [KITTI website](http://www.cvlibs.net/datasets/kitti/). Ensure the dataset is organized as specified for ease of data loading.

## 2. Pre-trained Models
Currently, pre-trained models are not available. A link will be provided for downloading trained checkpoints from Google Drive in the future.

## 3. Setup Environment

Create a virtual environment to install dependencies:

```bash
# Create a new conda environment
conda create -n deepvo python=3.8

# Activate the environment
conda activate deepvo

# Install dependencies
pip install -r requirements.txt
```

## 4. Usage
Configurations are read from `.json` files located in the `configs/` directory. Specify your experiment settings in `configs/exp.json` or create custom configuration files as needed.

---

### 4.1 Train 

To train the model, run in modular mode:

``` bash
python -m scripts.train --config <path_to_config>

#Example using the "exp1.json" configuration
python -m scripts.train --config configs/exp1.json
```

---

### 4.2 Test

Run model inference with:

```bash
python -m scripts.test <path_to_config> <checkpoint_name>

#Example for the "exp1.json" configuration and the "checkpoint_best"
python -m scripts.test configs/exp1.json checkpoint_best
```

---

### 4.3 Plot Results

After completing inferences, you can visualize the predicted trajectories by running:

```bash
python -m scripts.plot_results <path_to_config> <checkpoint_name>
```

---

### 4.4 Visualize Learning Curves

Launch TensorBoard to monitor training:

```bash
python -m utils.visualize_tensorboard --log_dir <path_to_desired_checkpoint>

#Example for the "exp1" experiment
python -m utils.visualize_tensorboard --log_dir checkpoints/exp1
```

## 5. Results

## 6. Reference

Original DeepVO paper: 

*Wang, Sen, et al. "Deepvo: Towards end-to-end visual odometry with deep recurrent convolutional neural networks." 2017 IEEE international conference on robotics and automation (ICRA). IEEE, 2017.*


