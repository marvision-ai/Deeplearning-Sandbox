# Deep Learning Sandbox
A deep learning utilities library for my experiments and development. Free to use by all. 


# Table of contents
1. [Utility functions](#utility_functions)
2. [Experiments](#experiments)
    1. [k-NN: A Simple Classifier](#knn)
    2. [Parameterized Learning (Linear Regression)](#linear_regression)
    3. [Gradient Descent, SGD, Mini-batch SGD, Regularization](#gradient)



## Utility functions <a name="utility_functions"></a>
1. **Preliminary Dataset loader:** 
   - This dataset loader is simple by design; however, it affords us the ability to apply any number of image preprocessors to every image in our dataset with ease. The only caveat of this dataset loader is that it assumes that all images in the dataset can fit into main memory at once. (To be expanded upon soon)


```
 simpleDatasetloader.py
 simplePreprocessor.py
```


## Experiments: <a name="experiments"></a>
### **1. **k-NN: A Simple Classifier**** <a name="knn"></a>
```
$ python knn.py --dataset ../datasets/animals
```

- [ ] Future Enhancement: Utilizing Nvidia Rapids cuML

### **2. Parameterized Learning** <a name="linear_regression"></a>

```
$ python linear_classifier.py
```

### **3. Gradient Descent, SGD, Mini-batch SGD, Regularization** <a name="gradient"></a>

``` 
$ python gradient_descent.py
$ python sgd.py
$ python regularization.py
```
