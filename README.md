# Paper Review:  DVD-GANS

* **Title**: Adversarial Video Generation on complex datasets
* **Authors**: Aidan Clark, Jeff Donahue, Karen Simonyan
* **[Link](https://arxiv.org/pdf/1907.06571.pdf)**
* **Tags**: Video Generation, Dual Video Discriminator
* **Year**: 2019

# Summary

* **What:**

  * The authors present us that large Generative Adversarial Networks trained on the complex dataset are able to produce video samples of substantially high complexity and fidelity
  * Dual Video Discriminator GAN (DVD-GAN) scales to longer and higher resolution videos.
  * Evaluation of video synthesis and video prediction
  
* **How:**

  * DVD-GAN was build upon BigGAN(large state-of-the-art image generation model) with the intent of extending it for video generation by including numerous modification specifically for this. Mainly these are the dual discriminator architecture and applied seperable self-atention and RNN
    * Spatial Discriminator critiques single frame content and structure by randomly sampling k full-resolution frames and judging them individually
    * Temporal Discriminator provides learning signal to generate movement, in other words it forces the generator to generate movements of objects within several frame
    * _SelfAttention(X) = softmax[XQ(XK)<sup>T</sup>]XV_
  * It was trained on Kinetics-600. Kinetics is a large dataset of 10-second high-resolution YouTube clips(around 500,000 videos, due to its size and diversity, 600 classes, overfitting is avoided). Another dataset used for video synthesis, prediction and testing is UCF-101 which consists of 13,320 video and 101 classes
  * The generator in DVD-GAN contains no explicit priors for foreground, background or optical flow, which otherwise would give out the difference between frames by considering pixel intensities and other such attributes. Instead it relies on learning of neural network.
  * Each DVD-GAN was trained on TPU pods using between 32 and 512 replicas with an Adam optimizer. Video Synthesis models are trained for around 300,000 learning steps, whilst Video Prediction models are trained for up to 1,000,000 steps. Most models took between 12 and 96 hours to train
  
* **Results:**
  **<br>DVD-GAN is able to produce new realistically looking videos with resolutio 256x256 and length up to 48 frames(4 seconds). It achieved new state-of-the-art Fréchet Inception Distance for prediction for Kinetics600, as well as state-of-the-art Inception Score for synthesis on the UCF-101 dataset.**
  
  ![Example_Results](assets/dvd.gif)
  
  * **Class-Conditional video synthesis**
 
   1. **Training on KINETICS-600 results(Inception Score and Fréchet Inception Distance) with different resolutions**
 
     ![KINETICS_Results](assets/res_kinetics.jpg?raw=true "KINETICS Results")
     
   2. **Video Synthesis on UCF-101 produces samples with Inception Score of 32.97, which sagnificantly overperforms state-of-the-art(Table 2). When testing prediction on single-class dataset BAIR the results where better comparing to other adverserial models trained on it but slighly worse than autoregressive model Video Transformer**
     
     ![UCF-101_Results](assets/res_ucf101.jpg?raw=true "UCF-101 Results")

  * **Video prediction**
  
   1. **Frame-conditional Kinetics**
   
     ![Frame-conditional_Kinetics_Results](assets/res_fcKinetics.jpg?raw=true "Frame-conditional Kinetics Results")
   
   2. **BAIR Robot Pushing**
   
     ![BAIR_Results](assets/res_BAIR.jpg?raw=true "BAIR Robot Pushing Results")

# Dual Video Discriminator GAN (DVD_GAN)
## Simplified architecture
![Architecture](assets/architecture.jpg?raw=true "DVD GAN")
