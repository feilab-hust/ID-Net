
# ID-Net

Example data and demo of ID-Net for 3-D super resolution of microscopy images. 

## Requirements

ID-Net（Isotropic Divide-stages-to-process neural network ）is built with Python and Tensorflow. Technically there are no limits to the operation system to run the code, but Windows system is recommended, on which the software has been tested.
The inference process of the ID-Net can run using the CPU only, but could be inefficiently. A powerful CUDA-enabled GPU device that can speed up the inference is highly recommended.

The inference process has been tested with:

 * Windows 10 pro (version 1903)
 * Python 3.6.7 (64 bit)
 * tensorflow 1.15.0
 * Intel Core i7-5930K CPU @3.50GHz
 * Nvidia GeForce RTX 2080 Ti


## Install

1. Install python 3.6 
2. (Optional) If your computer has a CUDA-enabled GPU, install the CUDA and CUDNN of the proper version.
3. Download the ID-Net_Demo.zip and unpack it. The directory tree should be: 

```  
ID-Net-Demo 
    . 
    .  
    ├── configs
    ├── checkpoint
    ├── crop_data.py
    ├── dataread.py
    ├── eval_all.py
    ├── generate_data.py
    ├── requirements.txt
    ├── train.py

```

4. Open the terminal in the ID-Net directory, install the dependencies using pip:

```
pip install -r requirements.txt
```

5. (Optional) If you have had the CUDA environment installed properly, run:

```
pip install tensorflow-gpu=1.15.0
```

The installation takes about 5 minutes in the tested platform. The time could be longer due to the network states.

*** 

## Data Generation

### 1. Open the terminal in the ID-Net Demo directory and run code ‘generate_data.py’; the ‘Crop data’ panel will appear..

### 2. Set parameters in ‘Global parameters’

 * HR path: Click the ’Choose’ button, select the raw hr data in your computer. (Full forms of italics are shown in the ‘Abbreviation Table’ of NOTE at the end, and the following also applies. )
 * Patch size: Set ‘Depth’ larger than 1 for sr/denoise net, and then the recommended values for depth, height, width are 32, 32, 32; Set ‘Depth’ equal to 1 for iso net, and then the recommended values for depth, height, width are 1, 32, 32.
 * Threshold: Default value is 0.9. For most data, the default value of 'Threshold' can be selected. However, you can determine the appropriate 'Threshold' according to the preview of the generated training data pair, ensuring that the non signal area of the data is as few as possible and the number of training data pairs is greater than 2K.
 * Gaussian noise: Set the standard deviation of gaussian noise added into lr data. Default value is 0, which means no gaussian noise.
 * Possion noise: Check means to add Poisson noise to lr data. Uncheck means not to add.

<div align=center>
<img width="480" height="360" src="/sample/fig/fig1.png"/>
</div>

### 3. Set parameters in ’3D data parameters’ or ‘ISO parameters’.

#### (1)For sr/denoise network: You need to set parameters in ‘3D data parameters’ to generate processed lr data.
* LR path: Click the ’Choose’ button, select raw lr data in your computer.
* Choose factor: Select the desired improvement factor, which needs to be set as 1 in denoise net and larger than 1 in sr net. 

#### (2)For iso net: You need to set parameters in ‘ISO parameters’ panel to generate synthetic anisotropic data.
* PSF file: Add optical blurring to the generated anisotropic x-y slice. First, you will need to prepare a 2D Gaussian function (.txt file) that can accurately simulate the axial PSF of your 3D datasets. Then click the ‘Choose’ button to load the PSF file in your computer. The program will generate the final synthetic x-y slices with axial resolution similar with that of the z-y slices. These degraded x-y slices with be paired with raw x-y slices for model training. 
* Axis subsample: Resample the isotropic x-y slices of the 3D datasets into anisotropic ones. For example, if you set a value of 2, the x-y slices will be down-sampled by a factor of 2 along x direction. Then the program will generate resampled x-y slices that simulate the anisotropic z-y slices of the 3D datasets.

NOTE: When generating the data for iso-net, which processes a 3D image stack slice by slice, the depth of patch size should be set as 1. In this case, the ‘3D data parameters’ panel will be deactivated by the program. As shown in Fig. 1, only the parameters in the global and iso parameters panels are available (red boxes, Fig. 1). Otherwise, when generating the data for denoising or sr net, which treat a 3D image stack as a whole volume, the depth of patch size should be set greater than 1. In this case, the ‘ISO parameters’ panel will be deactivated (Fig. 2). The datasets for sr or denoising network will be generated.

<div align=center>
<img width="960" height="360" src="/sample/fig/fig2_3.png"/>
</div>

### 4. Click the ‘Start running’ button. A window showing several data blocks will pop up. The first row is the hr data, and the second row is generated low-resolution or anisotropic data, which can be used to judge whether or not the simulated data is correct.

<div align=center>
<img width="960" height="400" src="/sample/fig/fig4.png"/>
</div>

NOTE: Only when patch_size’s depth value is equal to 1 will the isotropic data be generated, then the parameter setting of ‘3D data parameters’ will be ignored by the system. As is shown in figure 1 below, only the parameters in the box will be used; when the depth value of patch_size is greater than 1, the data of sr/denoise network will be generated, then the parameter setting of ‘ISO parameters’ will be ignored by the system. As is shown in figure 2 below, only the parameters in the box will be used. 

<div align=center>
<img width="960" height="360" src="/sample/fig/fig5_6.png"/>
</div>

*** 

## Training Process

### 1. We can start training models after the training pairs have been properly generated. To run the training procedure, open the terminal in the ID-Net Demo directory and run “python train.py ”; the panel of ‘Training GUI’ will appear..

<div align=center>
<img width="480" height="360" src="/sample/fig/fig7.png"/>
</div>

### 2. Set the parameters. 

 * Choose data type: Select the organelle structure type of your data.
 * Choose net type: Select the purpose of your net from SR, Denoise and ISO.
 * Choose factor: Select the factor that matches the training data.
 * Choose first net: Select the type of the first net. If your are going to train a denoise or iso network, select ‘None’.
 * Choose second net: Select the type of the second net.
 * Choose loss function: Select the appropriate loss function for the training net from ‘mae’ and ‘mse’.
 * LPIPS loss: Choose whether to add LPIPS loss function during the training. 
 * Label tag: Add specific tag to the currently trained network.
 * Patch size: Set its values according to the 'patch size' of ‘Global parameters' in ‘Crop data’ panel.
 
### 3. Click the ‘Start running’ button. 

net type      | Denoise         |SR    			 | ISO 				|
:-----:       |:-----:          |:-----:		 |:---------:		|
factor        | 1               | >1             | 1				| 
first net     | None            |dbpn3d/convnet3d| None 			|
second net    | drunet3d/unet3d |rdn3d           |drunet2d/unet2d   |
loss function | mae/mse         |mse/mae         |mse/mae 			|

*** 

## Inference Process

### Due to the limitation of the data size, the pre-trained model will be stored [here](https://drive.google.com/drive/folders/1fpMpI5DnDpU-eU5kB9H51hSrm326zR5G?usp=sharing), you need to download it and unzip it to the checkpoint folder of the original ID-Net demo. And Data used to test the model including various organelles can be downloaded [here](https://drive.google.com/drive/folders/1Ds2dBVO138aw0Chd40k17vOTVljV8ZOS?usp=sharing).Please note that if you want to restore the confocal or light slice microscope data collected by yourself, please match according to the pixel size of the corresponding cell structure listed in our excel table. 

### 1. Open the terminal in the ID-Net Demo directory and run the code ’eval.py’; the GUI interface of ‘Validation’ will appear.

### 2. Set parameter ‘data type’: Select the organelle structure type of the data to be reconstructed.

### 3. Check the type of network you want to apply at the bottom of GUI. For example, if ‘Denoise’ and ‘SR’ are checked synchronously, the denoised and denoised super-resolution image will be output.

<div align=center>
<img width="480" height="360" src="/sample/fig/fig8.png"/>
</div>

### 4. According to the parameter settings in Training Process, set the corresponding parameters of networks selected in step 3, namely ‘net type’, ‘factor’, ‘first net’, ‘second net’, ‘loss function’ and ‘label tag’. And our default network label tag is “lightsheet” and “confocal”. Note that "confocal" label is only used to convert "confocal" data to “lightsheet” data..

 * Normalize_threshold: The threshold of data normalization. Default value is 99.99.
 * ISO z sub factor: The threshold of validation data normalization. Default value is 99.99.
 * Validation_data_path：Click the button ’Choose’, select validation data in your computer

### 5. Click ‘Strat running’ button to start the inference. The model outputs will be saved at newly created subfolder of the “Validation_data_path”.

## Acknowledgements:
This program was built based on based on deep learning via TensorFlow and TensorLayer. We also acknowledge the selfless share from Martin Weigert’ groups[1]. You are welcome to use the code or program freely for research purpose. If you publish your work with the help of this program, please cite it at “Multi-color 4D superresolution light-sheet microscopy reveals organelle interactions at isotropic 100-nm resolution and sub-second timescales，Y Zhao, M Zhang, W Zhang, Q Liu, P Wang, R Chen, P Fei, YH Zhang, bioRxiv, 2021”. Please feel free to contact us at (feipeng@hust.edu.cn) if you have further question. 

## References:
[1]Weigert, M., Schmidt, U., Boothe, T. et al. Content-aware image restoration: pushing the limits of fluorescence microscopy. Nat Methods 15, 1090–1097 (2018).
 

