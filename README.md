# SNF2
*This package is a Python implementation of similarity network fusion 2 (SNF2), an improvement method for the original SNF*


## Requirements and installation
This package requires Python version 3.6 or greater. Assuming you have the correct version of Python, you can install this package by opening a command terminal and running the following:
```bash

```

## Purpose
Similarity network fusion is a technique originally proposed by [Wang et al., 2014, Nature Methods](https://www.ncbi.nlm.nih.gov/pubmed/24464287), Nature Methods to combine data from different sources for a shared group of samples. However, the SNF can only perform integration on the intersecting samples from different sources, which results in information loss by disgarding the unique samples in each sources. Here we present SNF2, which has the power of integrating the union of all the samples. And we have also proved that those unique samples do encode useful information which can contribute to the analysis of the intersecting samples.

![Similarity network fusion 2](https://github.com/rexxxx1234/SNF2/blob/master/image/snf2_figure.png)


## Sample Usage
To illustrate the power of SNF2, we include an example of applying SNF2 to simulated butterfly datasets originally coming from butterfly images. We use two different common encoding methods (Fisher Vector (FV) and Vector of Linearly Aggregated Descriptors (VLAD) with dense SIFT) to generate two different vectorizations of each image. These two encoding methods describe the content of the images differently and therefore capture different information about the images. Each method can generate a similarity network. 

The first dataset has 1032 samples and the second has 1132 samples. There are 832 common samples from the 2 datasets, which means 200 samples are unique for dataset1 and 300 samples are unique for dataset2. The SNF2 will perform integration for all the samples and product a final similarity network of union 1332 samples. Please feel free to try it in `test/test_butterfly.py`

Sample Output:
```bash
Original SNF for clustering intersecting 832 samples NMI score:  0.6226555202570552
Before diffusion for full 1032 p1 NMI score:  0.5795249338097694
Before diffusion for full 1132 p2 NMI score: 0.5893063921861074
Start indexing input expression matrices!
Start applying diffusion!
Diffusion ends! Times: 4.561135768890381s
After diffusion for full 1032 p1 NMI score:  0.685142665678969
After diffusion for full 1132 p2 NMI score: 0.687926828180068
Start iterative final matching!
Matching ends! Times: 0.250882625579834s
SNF2 + kernel matching for clustering union 1332 samples NMI score: 0.7420092262408033
```



## Acknowledgments

