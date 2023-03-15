# Vision Transformers (ViT)

This is an implementation of Vision Transformers (ViT) from scrtach for image classification as described in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).

Traditionally, Convolutional Neural Networks (CNNs) have been the go-to model for image classification tasks. However, ViT introduces a new approach by using a purely attention-based architecture. The ViT architecture is composed of a transformer encoder, which consists of multiple layers of self-attention mechanisms, and a linear classifier at the end.

In ViT, the input image is first divided into small patches, which are then flattened into a sequence of vectors. These vectors are then processed by the transformer encoder, which attends to each vector and computes their relationships to all other vectors in the sequence. The output of the transformer encoder is then passed through a linear classifier to produce the final classification output.

One key advantage of ViT is its ability to handle varying input sizes, as it does not rely on a fixed input size like CNNs. However, this also means that ViT requires a large amount of training data to generalize well to various input sizes.

![Vision Transformer](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fbroutonlab.com%2Fghost%2Fcontent%2Fimages%2F2021%2F04%2Ffig16.png&f=1&nofb=1&ipt=8fb707287a40779ff1e5f34cae20a0ac95ac41ed2dc48a7092f03c616ec4a5f7&ipo=images)

The input image is first divided into patches, which are then flattened into a sequence of vectors. The sequence of vectors is then processed by the transformer encoder, which attends to each vector and computes their relationships to all other vectors in the sequence. Finally, the output of the transformer encoder is passed through a linear classifier to produce the final classification output.
