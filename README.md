# MaSSL
Official PyTroch implementation for *Learning from Memory: A Non-Parametric Memory Augmented Self-Supervised Learning of Visual Features*

![Alt Text](https://github.com/sthalles/MaSSL/blob/main/assets/method_video.gif)

Project webpage: https://sthalles.github.io/MaSSL/

## Abstract

*We present Consistent Assignment of Views over Random Partitions (CARP), a self-supervised clustering method for representation learning of visual features. CARP learns prototypes in an end-to-end online fashion using gradient descent without additional non-differentiable modules to solve the cluster assignment problem. CARP optimizes a new pretext task based on random partitions of prototypes that regularizes the model and enforces consistency between views’ assignments. Additionally, our method improves training stability and prevents collapsed solutions in joint-embedding training. Through an extensive evaluation, we demonstrate that CARP’s representations are suitable for learning downstream tasks. We evaluate CARP’s representations capabilities in 17 datasets across many standard protocols, including linear evaluation, few-shot classification, k-NN, kmeans, image retrieval, and copy detection. We compare CARP performance to 11 existing self-supervised methods. We extensively ablate our method and demonstrate that our proposed random partition pretext task improves the quality of the learned representations by devising multiple random classification tasks. In transfer learning tasks, CARP achieves the best performance on average against many SSL methods trained for a longer time.*