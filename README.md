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

**Assignment:** 

- [Your First GAN](./C1W1_Your_First_GAN.ipynb)

#### Week 2: Deep Convolutional GANs

Learn about different activation functions, batch normalization, and transposed convolutions to tune your GAN architecture and apply them to build an advanced DCGAN specifically for processing images!

**Learning Targets:** 

- Be able to explain the components of a Deep Convolutional GAN.
- Compose a Deep Convolutional GAN using these components.
- Examine the difference between upsampling and transposed convolutions.

**Assignment:** 

- [Deep Convolutional GAN (DCGAN)](./C1W2_DCGAN.ipynb)

#### Week 3: Wasserstein GANs with Gradient Penalty

Learn advanced techniques to reduce instances of GAN failure due to imbalances between the generator and discriminator! Implement a WGAN to mitigate unstable training and mode collapse using W-Loss and Lipschitz Continuity enforcement.

**Learning Targets:** 

- Examine the cause and effect of an issue in GAN training known as mode collapse.
- Implement a Wasserstein GAN with Gradient Penalty to remedy mode collapse.
- Understand the motivation and condition needed for Wasserstein-Loss.

**Assignment:** 

- [WGAN](./C1W3_WGAN_GP_small.ipynb)

#### Week 4: Conditional GAN & Controllable Generation

Understand how to effectively control your GAN, modify the features in a generated image, and build conditional GANs capable of generating examples from determined categories!

**Learning Targets:** 

- Control GAN generated outputs by adding conditional inputs.
- Control GAN generated outputs by manipulating z-vectors.
- Be able to explain disentanglement in a GAN.

**Assignment:** 

- [Conditional GAN](./C1W4A_Build_a_Conditional_GAN.ipynb)
- [Controllable Generation](./C1W4B_Controllable_Generation.ipynb)

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

**Assignment:** 

- Fréchet Inception Distance