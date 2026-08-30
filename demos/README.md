# Deep Learning Demo Notebooks

This folder contains five demonstration notebooks covering:

- Image classification
- Object detection
- Image segmentation
- Image generation
- Vision-language understanding

The notebooks are intended to provide inspiration for course projects and to illustrate what modern deep-learning models can do.

Most examples use pretrained models from libraries such as Hugging Face. This makes it possible to demonstrate advanced capabilities with relatively little code. However, running a pretrained model with default settings is **not sufficient for a course project**.

## From a demo to a course project

A demo typically looks like this:

1. Load a pretrained model.
2. provide an image or prompt.
3. inspect the output.

A course project should go substantially further. It should formulate and investigate a question through training, evaluation, and controlled experiments.

A typical project should include:

1. **A clear task and research question**  
   Explain what you want to predict or generate and what you want to investigate.

2. **A suitable dataset**  
   Describe its size, annotations, class distribution, splitting strategy, and potential limitations.

3. **A baseline**  
   Establish a simple, credible starting point against which later experiments can be compared.

4. **Training or fine-tuning**  
   Train a model or meaningfully adapt a pretrained model to your task and data.

5. **Motivated experiments**  
   Investigate a specific limitation or hypothesis. Examples include:

   - Comparing fine-tuning strategies
   - Studying data augmentation or regularization
   - Addressing class imbalance
   - Comparing loss functions or architectures
   - Testing robustness to limited data or domain shift
   - Improving sampling or preprocessing
   - Adding metadata, auxiliary tasks, or uncertainty estimates

6. **Appropriate evaluation**  
   Use metrics suited to the task and include relevant qualitative analysis.

7. **Analysis and discussion**  
   Explain what worked, what did not work, and what the results allow you to conclude.

## Using pretrained models is allowed

You are welcome to use pretrained models, Hugging Face, PyTorch libraries, and existing open-source implementations. Reusing established components is often sensible and allows you to focus on the question you want to investigate.

The important distinction is that your project must not outsource all meaningful decisions to a ready-made pipeline. You should demonstrate ownership of:

- The research question
- Dataset preparation and splitting
- Training or adaptation strategy
- Experimental design
- Evaluation
- Interpretation of the results

For example, simply running a pretrained object detector and displaying its predictions would be a demo. A project might instead fine-tune the detector on a new dataset and investigate how image resolution, data augmentation, class imbalance, or hard-negative sampling affects small-object recall.

## Possible starting points

The notebooks can inspire questions such as:

- How well does a pretrained model transfer to a specialized dataset?
- How much labeled data is required to obtain useful performance?
- Which fine-tuning strategy works best?
- How does class imbalance affect the model?
- Does data augmentation improve generalization?
- How robust is the model to a different data source or domain?
- Which failure cases remain after fine-tuning?
- Does a more complex model actually outperform a simpler baseline?
- Are generated outputs only visually plausible, or do they preserve important semantic information?
- Does a vision-language model understand domain-specific concepts without additional adaptation?

## Final reminder

Use these notebooks to explore possibilities — not as complete project templates.

A strong course project is not defined by using the newest or most complicated model. It is defined by a clear question, a credible baseline, controlled experiments, valid evaluation, and thoughtful analysis.