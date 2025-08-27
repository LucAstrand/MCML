# A learnable random number generator for Monte Carlo simulation frameworks

Lucas Åstrand 

## Introduction 

Monte Carlo (MC) methods are an essential tool in computational science, widely used in physics to obtain numerical estimates through the process of random sampling. At the heart of MC methods and simulations is a random number generator, or RNG, which is designed to produce uniform samples, without specific regard for the underlying characteristics of the system which is being studied. While this general purpose approach is generally effective, it can limit the performance of MC simulations, specifically in the task of reproducing experimental data.  

This report presents the idea of a learnable random number generator, a machine learning model dedicated to the transfomation of uniform random inputs, such that, when used in MC simulation frameworks, the resulting outputs are more faithfull to a considered experimental dataset. Instead of treating randomness as something static and context independent, this approach makes use of data driven learning to produce a controlled bias in the sampling process, ultimately improving agreement between simulation results and empirical observations. 

### Usecase and plan for project 

## Physics (toy) model description 

The physics model implemented in this project is a Monte Carlo framework designed to simulate the nucleon–nucleus reaction $^{32}S(n,p)^{32}P$, and extract the observables of angular and energy differential cross sections. The inputs to the simulation are the incident neutron kinetic energy and three random variables that sample the available phase space of proton energy and scattering angles $(\theta, \phi)$. The model ensures conservation laws by considering the kinematic constraints brought by the separation energies and ultimately generates the final-state proton 4-momentum in the centre-of-mass frame. To aid with the comparison with experimental data, the proton 4-momentum is then Lorentz boosted to the laboratory frame.  

In order to turn the generated event samples into the cross sections mentioned above, the model employs a differentiable histogramming procedure. Instead of hard binning, each simulated event is softly assigned to neighboring bins via a sigmoid kernel with a learnable width. This approach smooths out the statistical noise inherent in Monte Carlo methods and allows gradients to propagate backward through the histogram, enabling the entire reaction simulation to be embedded in a machine-learning optimization pipeline. The resulting histograms can be normalized to approximate probability densities, which ultimately correspond to the differential cross sections.

## Network 

The machine learning model used for this project is built around a residual network (ResNet) like multilayer perceptron (MLP) designed to map inputs $[E_k, u]$ into outputs, with a uniform-preserving transformation, unique for each input neutron kinetic energy or $E_k$.


### Input layer 

The inputs to the model consist of two parts: random numbers $u$, which represent the stochastic sampling variables required by the physics simulation, and external parameters, in this case the neutron kinetic energy $E_k$. Together, these define the input vector $[E_k, u]$, where the random numbers encode the Monte Carlo phase-space sampling and $E_k$ acts as the conditioning parameter that modulates the transformation learned by the network.

### Architecture 

In contrast to a standard MLP, which would directly predict new values, this model learns a residual correction $\Delta ([E_k, u])$ which is then added to the logit of the input $u$. At the initialization phase, this residual is exactly zero, ensuring that the model acts like an identity transformation, preserving the uniform distribution of the inputs. As training occurs, the network learns to apply specific shifts in logit space, achieving non-trivial transformations, yet still maintaining the uniformity constraint. Architecturally, the model employs the stacked linear, layer normalization, ReLU blocks typically found in ResNet-style models.

The hidden layers of the model transform the input vector into a latent representation, while a final “head” layer generates the residual correction discussed above. A careful initialization scheme is employed: hidden layers make use of a Kaiming initialization to ensure stability during the gradient propagation, while the output head layer is initialized with small random weights to encourage exploration around the identity mapping. An optional L1 penalty scheme on the head layer weights is also included, penalizing variations too far from uniformity for the outputs.
### Output layer and Loss Function 

The output of the network is constrained to the interval $[0,1]$, obtained by applying a sigmoid to the corrected logits. This design combines the stability and identity-preserving advantages of ResNets with the bounded output structure required for probabilistic modeling. By construction, the outputs remain in the proper range throughout training, while the model remains flexible enough to learn complex conditional transformations as required by the project.

Training is carried out using a custom implementation of the mean squared error (MSE) loss, optimised for stability and batched training. Furthermore, an interpolation based compatibility layer is employed, to ensure the simulated data matches the shape of the considered experimental dataset. The L1 regularization penalty on the head weights can also be added to the overall loss to control deviations from uniformity.

## Results & Discussion