# deep-learning-challange


# Harnessing Deep Learning to Refine a Dataset


The objective of this analytical endeavor is to apply our burgeoning knowledge in the realm of deep learning to enhance and refine a pre-existing dataset.

# Origin of the Data
The dataset we are employing for this project is courtesy of a non-profit entity, Alphabet Soup. Comprising information culled from 34,000 funding applications submitted by various companies, this dataset provides an extensive array of metadata including the name, type of application, affiliation, classification, status, and the income amount of each funding applicant.

# Analytical Procedures


# Data Refinement
Our first step towards building an effective model was to identify the X and Y targets. In this case, the target variables hinge on the success or failure of a funding application. Subsequently, we narrowed down the dataset's variables to enhance the model's accuracy. Key variables, such as 'Application_type' and 'Classification', were selected to be 'T3' or 'Other', and 'C1000', 'C1200', 'C2000', and 'Other', respectively. We discarded all data that did not conform to these specific categories.

# Model Compilation, Training, and Assessment
With the refined data at our disposal, the next step involved compiling, training, and assessing our chosen model.

## Compilation
Our first task in this phase was to define the model we'd be using, which in this case was the Keras Sequential Model. We then proceeded to establish the first and second hidden layers and the output layer. These layers are crucial in determining the model's number of neurons and activation function. I opted for 80 neurons in the first hidden layer and 30 in the second, both with RELU activation, while the output layer utilized a Sigmoid activation function.

## Training
Once the model was fully defined, I proceeded to train it using the test data.  set the number of epochs, or iterations, at 100, running the model to fit the test data and then calculating the accuracy, loss, and recall for each epoch.

## Evaluation
Following the training phase, I plotted the model's accuracy, loss, and recall to gauge its performance. The results indicated an accuracy rate of 76%.

## Model Evaluation

# Refinements

## First Adjustment
In an attempt to further optimize the model, I increased the neuron count. This adjustment led to a slightly lower accuracy of 75.6%, but the resultant linear progression in the graph suggested a correlation between increased neurons and calculated loss.


## Second Adjustment
The second adjustment involved doubling the number of neurons from the first adjustment. However, this seemed to negatively impact the model's accuracy, which fell to 73%, while the chart seemed to revert to a pattern similar to that of the original model.



# Summation
Reflecting on the various models and their corresponding accuracy levels, I would propose that the company commence with the first optimized model and gradually refine it further for superior results.

As potential alterations for the future, I would contemplate adding more hidden layers, experimenting with different activation functions, and introducing a multi-category loss function during compilation. These changes could substantially alter the outcomes of our analytical efforts.


