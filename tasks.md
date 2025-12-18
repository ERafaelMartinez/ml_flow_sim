1. Build a CNN training pipeline and train/overfit it to process a random input image into the /img/output.png output image.

- confirm traning loop is working
- confirm model setup is correct
- confirm that the dimensions of the input-output are correct 

2. Overfit the model to a single simulation output (one data point):

- data generation pipeline
- data preparation 
- model training

- confirm model overfits to a single data point. Confirm netweork is capable of learning the output image from a single data point.

3. Overfit model to 10 simulation outputs (ten data points):

- data generation pipeline
- data preparation 
- model training

- confirm outputs contain enough information for the network to differentiate between the 10 data points and that no additional inputs are required.

4. Train the model on 100 simulation outputs (one hundred data points):

- data generation pipeline
- data preparation 
- model training
- split data into training and validation sets

- confirm training and validation losses are decreasing to avoid overfitting
