# Deep learning Sandbox
A deep learning utilities library for my experiments and development. Free to use by all. 


## Utilities
1. **Preliminary Dataset loader:** 
   - This dataset loader is simple by design; however, it affords us the ability to apply any number of image preprocessors to every image in our dataset with ease. The only caveat of this dataset loader is that it assumes that all images in the dataset can fit into main memory at once. (To be expanded upon soon)


```
 simpleDatasetloader.py
 simplePreprocessor.py
```


## Use Cases: 
1. **k-NN: A Simple Classifier**
```
python knn.py --dataset ../datasets/animals
```

- [ ] Future Enhancement: Utilizing Nvidia Rapids cuML

2. 
