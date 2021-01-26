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
 <li>Context aware networks (CAN). Based on the paper </li>
 <li>CSRNet. Based on the paper </li>
 <li>Bayesian-Crowd-Counting (Soon). Based on the paper</li>
 <li>Multi column crowd counting (MCNN) (Soon). Based on the paper <\li>
</ul>

<h4> Directory structure. </h4>

To run the presented implementations, we have to set up the following directory structure in our project directory. Then, to run our approaches with custom datasets will be as simple as writing a short command.

```

project                                           #Project folder. Typically we run our code from this folder.
│   README.md                                     #Readme of the project.
│
└───code                                          #Folder where we store the code.
│   │   ...
│   └───can                                       #Folder that contains the code to run can.
│   |   │   train.py                              #File with the code to train a can model.
│   |   │   test.py                               #File with the code to test a can model.
|   |   |   dataset.py                            #File with the code with the data loader. We do not use this file directly.
|   |   |   image.py                              #This code contains is in charge of modifying the images we use. We do not use this file directly.
|   |   |   utils.py                              #Several tools we use in the training process. We do not use this file directly.
|   |   |   model.py                              #The code contains the model. We do not use this file directly.
|   |   
|   └───CSRNet                                    #Folder that contains the code to run CSRNet.
│   |   │   train.py                              #faster rcnn training code.
│   |   │   test.py                               #faster rcnn testing code.
|   |   |   dataset.py                            #K-means approach to choose anchor sizes.
|   |   |   image.py                              #K-means approach to choose anchor sizes.
|   |   |   utils.py                              #K-means approach to choose anchor sizes.
|   |   |   model.py                              #Plot the results of the trained models.
|   |   
|   └───more models                               #
|   |
|   └───utils                                     #
|       |   creation_density_maps.py              #
|       |   json_files_creator.py                 #
|
└───datasets                                      #Folder where we save the datasets.
|   │   ...
|   └───dataset_A                                 #Dataset folder. Each dataset has to have a folder.
|       |   json_test_set.json                    #COCO JSON annotation file of the testing images.
|       |   json_train_set.json                   #COCO JSON annotation file of the training images.
|       |   json_val_set.json                     #COCO JSON annotation file of the validation images.
|       └───all                                   #Folder where we place the images.
|           | img_1.png
|           | img_2.png
|           | ...
|   
└───saved_models                                  #Folder where we save the models.
    |   ...
    └───faster_cnn                                
        |   ...
        └───dataset_A                             #Folder where we save the models we trained using dataset A.
            └───best_model.pth                    #Model we get from training.

```


To run the presented implementations, it is nece
<img src="https://github.com/adrianxsalazar/density_based_methods_counting/blob/master/readme_images/can.png"/>

<img src="https://github.com/adrianxsalazar/density_based_methods_counting/blob/master/readme_images/crsnet.png" />
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
