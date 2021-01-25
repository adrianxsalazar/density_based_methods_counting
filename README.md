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

<h4> Covered approaches </h4>
Currently, we covered the following methodologies, and we will add more density-based counting methods to this repo.

<ul>
 <li>Context aware networks.</li>
 <li>Set up the characteristic of the model with few commands. We cover elements such as the rpn anchor size, stopping criteria, base-model used for training, etc.</li>
 <li>Test your model with just a few commands.</li>
</ul>



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
