# Generalized-Pooling
Tensorflow implementation of the model presented in [Enhancing Sentence Embedding with Generalized Pooling](https://arxiv.org/abs/1806.09828) .

Note that this code is not the offical implement, you can find it [here](https://github.com/lukecq1231/generalized-pooling).

## Implement details

**mask**: The calculation of attention weights should mask the padding tokens.

**num_classes**: Specificily, we choose the Natural Language Inference(NLI) as the downstream task with the number of label classes as 2. Of cause, you can change it to satisfy your own data and task.

**penalty_type**: Author proposed three types of penalization terms, i.e. Parameter Matrices, Attention Matrices and Sentence Embeddings. We choose the first type in our implementation.

## Reporting issues

Please let me know, if you encounter any problems.