# Complete solutions for all the assignments in the Stanford CS231n course.

## Assignment 1 - [Solutions Directory](./Assignments/assignment1/)

* Q1 - [k-Nearest Neighboor Classifier](./Assignments/assignment1/knn.ipynb)
* Q2 - [Training a Support Vector Machine](./Assignments/assignment1/svm.ipynb)
* Q3 - [Train a Softmax Classifier](./Assignments/assignment1/softmax.ipynb)
* Q4 - [Train a Two Layer Fully Connected Neural Network](./Assignments/assignment1/two_layer_net.ipynb)
* Q5 - [Higher Level Representations: Image Features](./Assignments/assignment1/features.ipynb)

## Assignment 2 - [Solutions Directory](./Assignments/assignment2/)

* Q1 - [Multi Layer Fully Connected Neural Networks](./Assignments/assignment2/FullyConnectedNets.ipynb)
* Q2 - [Batch Normalization](./Assignments/assignment2/BatchNormalization.ipynb)
* Q3 - [Dropout](./Assignments/assignment2/Dropout.ipynb)
* Q4 - [Convolutional Neural Networks](./Assignments/assignment2/ConvolutionalNetworks.ipynb)
* Q5 - [Pytorch on CIFAR10](./Assignments/assignment2/PyTorch.ipynb)

## Assignment 3 - [Solutions Directory](./Assignments/assignment3/)

* Q1 - [Network Visualization](./Assignments/assignment3/Network_Visualization.ipynb)
* Q2 - [Image Captioning with Vanilla RNNs (Recurrent Neural Networks)](./Assignments/assignment3/RNN_Captioning.ipynb)
* Q3 - [Image Captioning with LSTMs (Long Short Term Memory Networls)](./Assignments/assignment3/LSTM_Captioning.ipynb)
* Q4 - [Image Captioning with Transformers](./Assignments/assignment3/Transformer_Captioning.ipynb)
* Q5 - [GANs (Generative Adversial Networks)](./Assignments/assignment3/Generative_Adversarial_Networks.ipynb)
* Q6 - [Self Supervised (Contrastive) Learning](./Assignments/assignment3/Self_Supervised_Learning.ipynb)

## Topics Index
| Topic           | Relevant Reading                                         | Brief Intuitive Explanation | Implemented                                                                                 |
| :-------------- | :------------------------------------------------------- | :-------------------------- | :------------------------------------------------------------------------------------------ |
| kNN  Classifier | + [cs231n notes](https://cs231n.github.io/classification/) <br>+ [A Few Useful Things to Know About Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) <br>+ [Recognizing and Learning Object Categories](https://people.csail.mit.edu/torralba/shortCourseRLOC/index.html) <br>+[kNN classification using Neighbourhood Components Analysis](https://kevinzakka.github.io/2020/02/10/nca/) <br>+[PCA (Principal Component Analysis) and SVD(Singular Value Decomposition)](https://web.archive.org/web/20150503165118/http://www.bigdataexaminer.com:80/understanding-dimensionality-reduction-principal-component-analysis-and-singular-value-decomposition/) <br>+[Random Projections](https://scikit-learn.org/stable/modules/random_projection.html) <br>+[Approximate Nearest Neighboor Classifier - FLANN](https://github.com/mariusmuja/flann) | **Training** - This step can be replaced with storing all the available training data. k is hyperparameter that needs to be tuned on the validation data. <br> <br> **Inference** - The sample to classify is compared to all the available training samples by computing L1 norm ($d_1 (I_1, I_2) = \sum_{p} \left\lVert I^p_1 - I^p_2 \right\rVert$) or L2 norm ($d_2 (I_1, I_2) = \sqrt{\sum_{p} \left( I^p_1 - I^p_2 \right)^2}$). The classifier assigns to the inference sample the most common class of the k training samples with the lowest distance to the inference sample. | [k_nearest_neighbor.py](./Assignments/assignment1/cs231n/classifiers/k_nearest_neighbor.py) |
| Stuff           | Stuff                                                    | Stuff                       | Stuff                                                                                       |
