# Generative Adversarial Networks (GANs) Specialization

[Generative Adversarial Networks (GANs) | Coursera](https://www.coursera.org/specializations/generative-adversarial-networks-gans)

## About this Specialization

The **DeepLearning.AI Generative Adversarial Networks (GANs) Specialization** provides an exciting introduction to image generation with GANs, charting a path from foundational concepts to advanced techniques through an easy-to-understand approach. It also covers social implications, including bias in ML and the ways to detect it, privacy preservation, and more.

Build a comprehensive knowledge base and gain hands-on experience in GANs. Train your own model using PyTorch, use it to create images, and evaluate a variety of advanced GANs. 

This Specialization includes three courses:

[Course 1: Build Basic Generative Adversarial Networks](https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans)

[Course 2: Build Better Generative Adversarial Networks](https://www.coursera.org/learn/build-better-generative-adversarial-networks-gans)

[Course 3: Apply Generative Adversarial Networks (GANs)](https://www.coursera.org/learn/apply-generative-adversarial-networks-gans)

## About this Repo

In this repo you'll find all  programming assignments I finished from this Specialization.

## Programming Assignments
### Course 1: Build Basic Generative Adversarial Networks (GANs)

In this course, you will:

\- Learn about GANs and their applications

\- Understand the intuition behind the fundamental components of GANs 

\- Explore and implement multiple GAN architectures 

\- Build conditional GANs capable of generating examples from determined categories

#### Week 1: Intro to GANs

See some real-world applications of GANs, learn about their fundamental components, and build your very own GAN using PyTorch!

**Learning Targets:** 

- Construct your first GAN.
- Develop intuition behind GANs and their components.
- Examine real life applications of GANs.

**Programming Assignment:** 

- [Your First GAN](./C1W1_Your_First_GAN.ipynb)
- In this notebook, you're going to create your first generative adversarial network (GAN) for this course! Specifically, you will build and train a GAN that can generate hand-written images of digits (0-9). You will be using PyTorch in this specialization, so if you're not familiar with this framework, you may find the [PyTorch documentation](https://pytorch.org/docs/stable/index.html) useful. The hints will also often include links to relevant documentation.

#### Week 2: Deep Convolutional GANs

Learn about different activation functions, batch normalization, and transposed convolutions to tune your GAN architecture and apply them to build an advanced DCGAN specifically for processing images!

**Learning Targets:** 

- Be able to explain the components of a Deep Convolutional GAN.
- Compose a Deep Convolutional GAN using these components.
- Examine the difference between upsampling and transposed convolutions.

**Programming Assignment:** 

- [Deep Convolutional GAN (DCGAN)](./C1W2_DCGAN.ipynb)
- In this notebook, you're going to create another GAN using the MNIST dataset. You will implement a Deep Convolutional GAN (DCGAN), a very successful and influential GAN model developed in 2015. 

#### Week 3: Wasserstein GANs with Gradient Penalty

Learn advanced techniques to reduce instances of GAN failure due to imbalances between the generator and discriminator! Implement a WGAN to mitigate unstable training and mode collapse using W-Loss and Lipschitz Continuity enforcement.

**Learning Targets:** 

- Examine the cause and effect of an issue in GAN training known as mode collapse.
- Implement a Wasserstein GAN with Gradient Penalty to remedy mode collapse.
- Understand the motivation and condition needed for Wasserstein-Loss.

**Programming Assignment:** 

- [WGAN](./C1W3_WGAN_GP_small.ipynb)
- In this notebook, you're going to build a Wasserstein GAN with Gradient Penalty (WGAN-GP) that solves some of the stability issues with the GANs that you have been using up until this point. Specifically, you'll use a special kind of loss function known as the W-loss, where W stands for Wasserstein, and gradient penalties to prevent mode collapse.

#### Week 4: Conditional GAN & Controllable Generation

Understand how to effectively control your GAN, modify the features in a generated image, and build conditional GANs capable of generating examples from determined categories!

**Learning Targets:** 

- Control GAN generated outputs by adding conditional inputs.
- Control GAN generated outputs by manipulating z-vectors.
- Be able to explain disentanglement in a GAN.

**Programming Assignment:** 

- [Conditional GAN](./C1W4A_Build_a_Conditional_GAN.ipynb)
- In this notebook, you're going to make a conditional GAN in order to generate hand-written images of digits, conditioned on the digit to be generated (the class vector). This will let you choose what digit you want to generate.  You'll then do some exploration of the generated images to visualize what the noise and class vectors mean. 
- [Controllable Generation](./C1W4B_Controllable_Generation.ipynb)
- In this notebook, you're going to implement a GAN controllability method using gradients from a classifier. By training a classifier to recognize a relevant feature, you can use it to change the generator's inputs (z-vectors) to make it generate images with more or less of that feature. This will also be the first notebook where you generate faces, as we work our way up to StyleGAN in the next course!

### Course 2: Build Better Generative Adversarial Networks (GANs)

In this course, you will:

\- Assess the challenges of evaluating GANs and compare different generative models

\- Use the Fréchet Inception Distance (FID) method to evaluate the fidelity and diversity of GANs

\- Identify sources of bias and the ways to detect it in GANs

\- Learn and implement the techniques associated with the state-of-the-art StyleGANs

#### Week 1: Evaluation of GANs

Understand the challenges of evaluating GANs, learn about the advantages and disadvantages of different GAN performance measures, and implement the Fréchet Inception Distance (FID) method using embeddings to assess the accuracy of GANs!

**Learning Targets:** 

- Differentiate across different evaluation metrics and their pros/cons.
- Justify the use of feature embeddings in GAN evaluation.
- Evaluate your GANs by implementing Fréchet Inception Distance (FID) and Inception Score.

**Programming Assignment:** 

- Fréchet Inception Distance