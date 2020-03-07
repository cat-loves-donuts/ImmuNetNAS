# ImmuNetNAS

### ImmuNetNAS is a Neural Architecture Search (NAS) method which inspired by the Immune Network Theory.

The core of ImmuNetNAS is built on the original Immune Network Algorithm, which iteratively updates the population through hypermutation
and selection, and eliminates the self-generation individuals that do not meet the requirements through comparing antibody affinity and interspecific similarity.

#### We used Google Cloud Platform to build the basic environment. The specific configuration and hardware selection are showing below:

* Configuration requirements:
  * `python` 3.7.1
  * `torch` 1.1.0
  * `torchvision` 0.4.0
  
* Hardware Selection:
  * `CPU` Intel(R) Xeon(R) CPU @ 2.30GHz
  * `GPU` NVIDIA Tesla T4
  
### By the way:

We have already finished a paper which we put it on arxiv right now. The title of the paper is [ImmuNetNAS: An Immune-network approach for searching Convolutional Neural Network Architectures](https://arxiv.org/abs/2002.12704). I have to admit that some 'aspects' in this research are not perfect. We will keep researching in the future. 

### By the way again:

This is version 3.0, we will keep updating the code and adding functions.


 
