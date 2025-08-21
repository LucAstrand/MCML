# A learnable random number generator for Monte Carlo simulation frameworks

Lucas Åstrand 

## Introduction 

Monte Carlo (MC) methods are an essential tool in computational science, widely used in physics to approximate complex processes through random sampling. At the heart of MC methods and simulations is a random number generator, or RNG, which is designed to produce uniform samples, without specific regard for the underlying characteristics of the system which is being studied. While this general purpose approach is generally effective, it can limit the performance of MC simulations, specifically in the task of reproducing experimental data.  

This report presents the idea of a learnable random number generator, a machine learning model dedicated to the transfomation of uniform random inputs, such that, when used in MC simulation frameworks, the resulting outputs are more faithfull to the experimental data. Instead of treating randomness as something static and context-independent, this approach makes use of data driven learning to produce a controlled bias in the sampling process, ultimately improving agreement between simulation results and empirical observations. 

### Usecase and plan for project 

## Physics (toy) model description 

## Network 

### Input layer 

Random numbers + External parameters (and what they would represent in the physics model)

### Architecture 
### Output layer and Loss Function 

## Results & Discussion