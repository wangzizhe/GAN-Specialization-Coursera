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

This repo includes an introduction of this Specialization and all programming assignments I finished.

## Programming Assignments
### Course 1: Build Basic Generative Adversarial Networks (GANs)

\- Learn about GANs and their applications

\- Understand the intuition behind the fundamental components of GANs 

\- Explore and implement multiple GAN architectures 

\- Build conditional GANs capable of generating examples from determined categories

#### Week 1: Intro to GANs

See some real-world applications of GANs, learn about their fundamental components, and build your very own GAN using PyTorch!

**Learning Objectives:** 

- Construct your first GAN.
- Develop intuition behind GANs and their components.
- Examine real life applications of GANs.

**Programming Assignment:** 

- [Your First GAN](./C1W1_Your_First_GAN.ipynb)
- In this notebook, you're going to create your first generative adversarial network (GAN) for this course! Specifically, you will build and train a GAN that can generate hand-written images of digits (0-9). You will be using PyTorch in this specialization, so if you're not familiar with this framework, you may find the [PyTorch documentation](https://pytorch.org/docs/stable/index.html) useful. The hints will also often include links to relevant documentation.

#### Week 2: Deep Convolutional GANs

Learn about different activation functions, batch normalization, and transposed convolutions to tune your GAN architecture and apply them to build an advanced DCGAN specifically for processing images!

**Learning Objectives:** 

- Be able to explain the components of a Deep Convolutional GAN.
- Compose a Deep Convolutional GAN using these components.
- Examine the difference between upsampling and transposed convolutions.

**Programming Assignment:** 

- [Deep Convolutional GAN (DCGAN)](./C1W2_DCGAN.ipynb)
- In this notebook, you're going to create another GAN using the MNIST dataset. You will implement a Deep Convolutional GAN (DCGAN), a very successful and influential GAN model developed in 2015. 

#### Week 3: Wasserstein GANs with Gradient Penalty

Learn advanced techniques to reduce instances of GAN failure due to imbalances between the generator and discriminator! Implement a WGAN to mitigate unstable training and mode collapse using W-Loss and Lipschitz Continuity enforcement.

**Learning Objectives:** 

- Examine the cause and effect of an issue in GAN training known as mode collapse.
- Implement a Wasserstein GAN with Gradient Penalty to remedy mode collapse.
- Understand the motivation and condition needed for Wasserstein-Loss.

**Programming Assignment:** 

- [WGAN](./C1W3_WGAN_GP_small.ipynb)
- In this notebook, you're going to build a Wasserstein GAN with Gradient Penalty (WGAN-GP) that solves some of the stability issues with the GANs that you have been using up until this point. Specifically, you'll use a special kind of loss function known as the W-loss, where W stands for Wasserstein, and gradient penalties to prevent mode collapse.

#### Week 4: Conditional GAN & Controllable Generation

Understand how to effectively control your GAN, modify the features in a generated image, and build conditional GANs capable of generating examples from determined categories!

**Learning Objectives:** 

- Control GAN generated outputs by adding conditional inputs.
- Control GAN generated outputs by manipulating z-vectors.
- Be able to explain disentanglement in a GAN.

**Programming Assignment:** 

- [Conditional GAN](./C1W4A_Build_a_Conditional_GAN.ipynb)
- In this notebook, you're going to make a conditional GAN in order to generate hand-written images of digits, conditioned on the digit to be generated (the class vector). This will let you choose what digit you want to generate.  You'll then do some exploration of the generated images to visualize what the noise and class vectors mean. 
- [Controllable Generation](./C1W4B_Controllable_Generation.ipynb)
- In this notebook, you're going to implement a GAN controllability method using gradients from a classifier. By training a classifier to recognize a relevant feature, you can use it to change the generator's inputs (z-vectors) to make it generate images with more or less of that feature. This will also be the first notebook where you generate faces, as we work our way up to StyleGAN in the next course!

### Course 2: Build Better Generative Adversarial Networks (GANs)

\- Assess the challenges of evaluating GANs and compare different generative models

\- Use the Fréchet Inception Distance (FID) method to evaluate the fidelity and diversity of GANs

\- Identify sources of bias and the ways to detect it in GANs

\- Learn and implement the techniques associated with the state-of-the-art StyleGANs

#### Week 1: Evaluation of GANs

Understand the challenges of evaluating GANs, learn about the advantages and disadvantages of different GAN performance measures, and implement the Fréchet Inception Distance (FID) method using embeddings to assess the accuracy of GANs!

**Learning Objectives:** 

- Differentiate across different evaluation metrics and their pros/cons.
- Justify the use of feature embeddings in GAN evaluation.
- Evaluate your GANs by implementing Fréchet Inception Distance (FID) and Inception Score.

**Programming Assignment:** 

- [Fréchet Inception Distance](./C2W1_FID.ipynb)
- In this notebook, you're going to gain a better understanding of some of the challenges that come with evaluating GANs and a response you can take to alleviate some of them called Fréchet Inception Distance (FID).

#### Week 2: GAN Disadvantages and Bias

Learn the disadvantages of GANs when compared to other generative models, discover the pros/cons of these models—plus, learn about the many places where bias in machine learning can come from, why it’s important, and an approach to identify it in GANs!

**Learning Objectives:** 

- Propose generative model alternatives to GANs and their pros/cons.
- Scrutinize bias in machine learning and examine its various sources.
- Describe an application of GANs that demonstrates bias.
- Explain several definitions of fairness in machine learning.

**Programming Assignment:** 

- [Bias](./C2W2_Bias.ipynb)
- In this notebook, you're going to explore a way to identify some biases of a GAN using a classifier, in a way that's well-suited for attempting to make a model independent of an input. Note that not all biases are as obvious as the ones you will see here.

#### Week 3: StyleGAN and Advancements

Learn how StyleGAN improves upon previous models and implement the components and the techniques associated with StyleGAN, currently the most state-of-the-art GAN with powerful capabilities!

**Learning Objectives:** 

- Analyze key advancements of GANs.
- Build and compose the components of StyleGAN.
- Investigate the controllability, fidelity, and diversity of StyleGAN outputs.

**Programming Assignment:** 

- [Components of StyleGAN](./C2W3_Components_of_StyleGAN.ipynb)
- In this notebook, you're going to implement various components of StyleGAN, including the truncation trick, the mapping layer, noise injection, adaptive instance normalization (AdaIN), and progressive growing.

### Course 3: Apply Generative Adversarial Networks (GANs)

\- Explore the applications of GANs and examine them wrt data augmentation, privacy, and anonymity

\- Leverage the image-to-image translation for framework and identify applications to modalities beyond images

\- Implement Pix2Pix, a paired image-to-image translation GAN, to adapt satellite images into map route (and vice versa)

\- Compare paired image-to-image translation to unpaired image-to-image translation and identify how their key difference necessitates different GAN architectures

\- Implement CycleGAN, an unpaired image-to-image translation model, to adapt horses to zebras (and vice versa) with two GANs in one

#### Week 1: GANs for Data Augmentation and Privacy

Learn different applications of GANs, understand the pros/cons of using them for data augmentation, and see how they can improve downstream AI models!

**Learning Objectives:** 

- Explore different applications of GANs and examine their specific applications in data augmentation, privacy, and anonymity.
- Improve your downstream AI models with GAN-generated data.

**Programming Assignment:** 

- [Data Augmentation](./C3W1_Data_Augmentation.ipynb)
- In this notebook, you're going to build a generator that can be used to help create data to train a classifier. You'll learn to understand some use cases for data augmentation and why GANs suit this task.

#### Week 2: Image-to-Image Translation with Pix2Pix

Understand image-to-image translation, learn about different applications of this framework, and implement a U-Net generator and Pix2Pix, a paired image-to-image translation GAN!

**Learning Objectives:** 

- Learn and leverage the image-to-image translation framework.
- Identify extensions, generalizations, and applications of this framework to modalities beyond images.
- Implement a paired image-to-image translation GAN, called Pix2Pix, to turn satellite images into map routes (and vice versa), with advanced U-Net generator and PatchGAN discriminator architectures.

**Programming Assignment:** 

- [U-Net](./C3W2A_U-Net.ipynb)
- In this notebook, you're going to implement a U-Net for a biomedical imaging segmentation task. Specifically, you're going to be labeling neurons, so one might call this a neural neural network! ;) Note that this is not a GAN, generative model, or unsupervised learning task. This is a supervised learning task, so there's only one correct answer (like a classifier!) You will see how this component underlies the Generator component of Pix2Pix in the next notebook this week.
- [Pix2Pix](./C3W2B_Pix2Pix.ipynb)
- In this notebook, you will write a generative model based on the paper Image-to-Image Translation with Conditional Adversarial Networks by Isola et al. 2017, also known as Pix2Pix.You will be training a model that can convert aerial satellite imagery ("input") into map routes ("output"), as was done in the original paper. Since the architecture for the generator is a U-Net, which you've already implemented (with minor changes), the emphasis of the assignment will be on the loss function. So that you can see outputs more quickly, you'll be able to see your model train starting from a pre-trained checkpoint - but feel free to train it from scratch on your own too.

#### Week 3: Unpaired Translation with CycleGAN

Understand how unpaired image-to-image translation differs from paired translation, learn how CycleGAN implements this model using two GANs, and implement a CycleGAN to transform between horses and zebras!

**Learning Objectives:** 

- Compare paired image-to-image translation to unpaired image-to-image translation.
- Identify how their key difference necessitates a different GAN architecture.
- Implement unpaired image-to-image translation model, called CycleGAN, to adapt horses to zebras (and vice versa) with two GANs in one.

**Programming Assignment:** 

- [CycleGAN](./C3W3_CycleGAN.ipynb)
- In this notebook, you will write a generative model based on the paper [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) by Zhu et al. 2017, commonly referred to as CycleGAN. You will be training a model that can convert horses into zebras, and vice versa. Once again, the emphasis of the assignment will be on the loss functions. In order for you to see good outputs more quickly, you'll be training your model starting from a pre-trained checkpoint. You are also welcome to train it from scratch on your own, if you so choose.

