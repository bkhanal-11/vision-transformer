# Vision Transformers (ViT)

This is an implementation of Vision Transformers (ViT) from scrtach for image classification as described in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).

Traditionally, Convolutional Neural Networks (CNNs) have been the go-to model for image classification tasks. However, ViT introduces a new approach by using a purely attention-based architecture. The ViT architecture is composed of a transformer encoder, which consists of multiple layers of self-attention mechanisms, and a linear classifier at the end.

In ViT, the input image is first divided into small patches, which are then flattened into a sequence of vectors. These vectors are then processed by the transformer encoder, which attends to each vector and computes their relationships to all other vectors in the sequence. The output of the transformer encoder is then passed through a linear classifier to produce the final classification output.

One key advantage of ViT is its ability to handle varying input sizes, as it does not rely on a fixed input size like CNNs. However, this also means that ViT requires a large amount of training data to generalize well to various input sizes.

![Vision Transformer](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fbroutonlab.com%2Fghost%2Fcontent%2Fimages%2F2021%2F04%2Ffig16.png&f=1&nofb=1&ipt=8fb707287a40779ff1e5f34cae20a0ac95ac41ed2dc48a7092f03c616ec4a5f7&ipo=images)

The input image is first divided into patches, which are then flattened into a sequence of vectors. The sequence of vectors is then processed by the transformer encoder, which attends to each vector and computes their relationships to all other vectors in the sequence. Finally, the output of the transformer encoder is passed through a linear classifier to produce the final classification output.

## Inductive Biases in Machine Learning

This is something I came across while studying about why Transformers are replacing traditional CNNs and LSTMs. I was unaware of this before. I have put it here for mine as well as readers reference.

Inductive bias refers to the set of assumptions that a machine learning algorithm makes about the relationship between input data and output labels, before it sees any training examples. It is the set of assumptions that are built into the algorithm, based on its design and the problem it is intended to solve. The inductive bias guides the learning process, and can influence the accuracy and generalization performance of the learned model.

In other words, the inductive bias of a machine learning algorithm shapes its learning behavior and determines the types of hypotheses that it can generate. It reflects the prior knowledge that the algorithm has about the structure of the data and the problem it is solving. For example, a linear regression model has an inductive bias that assumes the relationship between the input variables and the output variable is linear.

Inductive bias can be either explicit or implicit. Explicit inductive bias refers to the assumptions that are explicitly programmed into the algorithm, such as the choice of features or the choice of algorithmic model. Implicit inductive bias, on the other hand, refers to the assumptions that are built into the algorithm's learning process, such as the regularization method or the optimization algorithm used.

There are several types of inductive biases that are commonly used in machine learning. One of the most common types is the bias towards simplicity or Occam's Razor, which favors simpler hypotheses over more complex ones. Another type is the bias towards locality, which assumes that data points that are close to each other in input space are likely to have similar output labels. Other types of inductive biases include the bias towards smoothness, sparsity, and modularity, among others.

The choice of inductive bias can have a significant impact on the performance and generalization of a machine learning algorithm. A good inductive bias can help the algorithm learn more efficiently and generalize better to new data, while a bad inductive bias can lead to overfitting or underfitting, and poor generalization performance.

[1] Inductive Biases in ML: [The Inductive Bias of ML Models, and Why You Should Care About It](https://towardsdatascience.com/the-inductive-bias-of-ml-models-and-why-you-should-care-about-it-979fe02a1a56)
