This repository contains the code to run several density-based counting approaches. In our implementation, we are focused on applications in agriculture. The goal of this implementation is to, in a few steps, run multiple density-based counting approaches on several datasets. So, practitioners can further develop these techniques.

What can you do with this implementation?
<ul>
 <li>Train density-based models with your custom datasets with just a few commands.</li>
 <li>Set up the characteristic of the model with few commands for exhaustive search.</li>
 <li>Test your model with just a few commands.</li>
</ul>

<p class="aligncenter">
<img src="https://github.com/adrianxsalazar/density_based_methods_counting/blob/master/readme_images/output.png" alt="detection sample">
</p>

<h4> Covered approaches. </h4>
Currently, we covered the following methodologies. We will add more density-based counting methods to this repo.

<ul>
 <li>Context aware networks (CAN). Based on the paper "Context-Aware Crowd Counting". </li>
 <li>CSRNet. Based on the paper "CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes".</li>
 <li>Bayesian-Crowd-Counting (Soon). Based on the paper "Bayesian Loss for Crowd Count Estimation with Point Supervision".</li>
 <li>Multi column crowd counting (MCNN) (Soon). Based on the paper "Single-Image Crowd Counting via Multi-Column Convolutional Neural Network".<\li>
</ul>

We try to cover representative approaches from each family. We have covered techniques from the single-column and multicolumn families in density based crowd counting. Multicolumn methods such as "can" extract information at multiple levels by using neural structures of different sizes, to then combine them for the sake of counting. Below, you can find an example of can.
<img src="https://github.com/adrianxsalazar/density_based_methods_counting/blob/master/readme_images/can.png"/>

Also, we have covered single column structures such as CSRNet. These approaches use long neural structures to extract the information required for counting.
<img src="https://github.com/adrianxsalazar/density_based_methods_counting/blob/master/readme_images/crsnet.png" />


<h4> Directory structure. </h4>

To run the presented implementations, we have to set up the following directory structure in our project directory.
It is easy to do so. You only need to download this repository and place your dataset in the correct directory.
Then, running our approaches with custom datasets will be as simple as using a couple of short commands.

```

project                                           #Project folder. Typically we run our code from this folder.
│   README.md                                     #Readme of the project.
│
└───code                                          #Folder where we store the code.
│   │
|   └───models
|   │   └───can                                   #Folder that contains the code to run can.
|   │   |   │   train.py                          #File with the code to train a can model.
|   │   |   │   test.py                           #File with the code to test a can model.
|   |   |   |   dataset.py                        #File with the code with the data loader. We do not use this file directly.
|   |   |   |   image.py                          #This code contains is in charge of modifying the images we use. We do not use this file directly.
|   |   |   |   utils.py                          #Several tools we use in the training process. We do not use this file directly.
|   |   |   |   model.py                          #The code contains the model. We do not use this file directly.
|   |   |   
|   |   └───CSRNet                                #Folder that contains the code to run CSRNet.
|   │   |   │   train.py                          #File with the code to train a CSRNet model.
|   │   |   │   test.py                           #File with the code to test a CSRNet model.
|   |   |   |   dataset.py                        #File with the code with the data loader. We do not use this file directly.
|   |   |   |   image.py                          #This code contains is in charge of modifying the images we use. We do not use this file directly.
|   |   |   |   utils.py                          #Several tools we use in the training process. We do not use this file directly.
|   |   |   |   model.py                          #The code contains the model. We do not use this file directly.
|   |   |   
|   |   └───more models                           #We will place future models under this directory.
|   |
|   └───utils                                     #This folder contains tools to train density-based models.
|       |   creation_density_maps.py              #Code to create the ground truth density maps.
|       |   json_files_creator.py                 #Code to create the .json file with the path of the images we want to use for training, testing, and validation.
|
└───datasets                                      #Folder where we save the datasets.
|   │   ...
|   └───dataset_A                                 #Dataset folder. Each dataset has to have a folder.
|       |   density_test_list.json                #.json file that contains a list of the paths of the testing images.
|       |   density_train_list.json               #.json file that contains a list of the paths of the training images.
|       |   density_val_list.json                 #.json file that contains a list of the paths of the validation images.
|       └───all                                   #Folder where we place the images and the ground truth density maps.
|           | img_1.png                           #Image we are using
|           | img_1.h5                            #ground truth density map
|           | img_2.png                           #""
|           | img_2.h5                            #""
|           | ...
|   
└───saved_models                                  #Folder where we save the models.
    |   ...
    └───can                                       #Folder where we save the models as a result the can training process.
    |   |   ...
    |   └───dataset_A                             #Folder where we save the models we trained using dataset A.
    |       └───best_model.pth                    #Model we get from training for can.
    |
    └───CSRNet
        |   ...
        └───dataset_A                             #Folder where we save the models we trained using dataset A.
            └───best_model.pth                    #Model we get from training for CSRNet.


```


To run the presented implementations, it is nece

```

$

```

<h3> Code to prepare the .json file to train your models </h3>

Below, you can find an example of the ".json" file that we need. This file contains the paths of the images we will use for training, testing, and validation. There is a .json for each set.

```

$

```

We included the file "json_creator.py" under the "code/utils/" directory. This code takes all the images you want to use to create the validation, training, and testing files. The following command runs this code.

```

$

```

The parameter "-"  


<h3> Testing the model </h3>
