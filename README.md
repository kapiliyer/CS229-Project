# CS 229 Project

We implement a set of transformer-MLP models to perform the paraphrase detection and semantic textual similarity tasks.
We created models in the regime of classical training, with a separate model for each task, as well as models using multitask training.
We drew on recent research on how to best resolve the conflicting gradients problem under the regime of multitask training.

### Conflicting Gradients Solutions

- https://github.com/LucasBoTang/GradNorm (GradNorm)
- https://github.com/WeiChengTseng/Pytorch-PCGrad (PCGrad)
- https://github.com/Cranial-XIX/CAGrad (CAGrad)

### Datasets
- https://www.kaggle.com/c/quora-question-pairs (PARA)
- https://huggingface.co/datasets/mteb/stsbenchmark-sts (STS)
