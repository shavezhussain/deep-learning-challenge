# <ins>Deep Machine Learning Challenge</ins>
Module 21 Neural Networks and deep learning

## Overview

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With my knowledge of machine learning and neural networks, I’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, I have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years.

Please see report below.

## Preprocess the Data

Using my knowledge of Pandas and scikit-learn’s StandardScaler(), I preprocessed the dataset. The charity_data.csv was read to a Pandas DataFrame, using Google Colab notebook. Once this was done, I identified the target and feature variables for the model. In my first attempt, I dropped the EIN and NAME columns, and determined the number of unique values for each column. These were used to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful. `pd.get_dummies()` was used to encode categorical variables. I then split the preprocessed data into a features array, X, and a target array, y. Using these arrays and the train_test_split function, I split the data into training and testing datasets.


## Compile, Train, and Evaluate the Model

Using my knowledge of TensorFlow and Keras, I designed a neural network model to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. Once I completed this step, I compiled, trained, and evaluated my binary classification model to calculate the model’s loss and accuracy.

* First attempt

For the first attempt, I used the `relu` activation function. There were two layers - layer 1 had 70 hidden nodes and layer 2  had 30 hidden nodes.  Epochs was 100. For the outer layer, I used `sigmoid`.


![Screen Shot 2023-04-09 at 00 15 10](https://user-images.githubusercontent.com/116304118/230746444-ba25feab-8f80-46c3-b345-6a7d20c53931.png)


![first_attempt](https://user-images.githubusercontent.com/116304118/230739546-9a2eb2fb-a366-4227-bfc1-7dc870af502d.png)


![first_attempt1](https://user-images.githubusercontent.com/116304118/230739532-2bdd70d1-d104-4f32-bff5-a945c1e1de49.png)


To view the results, click here [`AlphabetSoupCharity.h5`](https://github.com/HJandu/deep-learning-challenge/blob/main/h5_files/AlphabetSoupCharity.h5)

Click here to view [code](https://github.com/HJandu/deep-learning-challenge/blob/main/starter_code_TestData.ipynb)

## Optimize the Model

Using my knowledge of TensorFlow, I optimized my model to achieve a target predictive accuracy higher than 75%. 
This was achieved by dropping only the 'NAME' column of the database, adding more neurons to the hidden layers and more epochs. 

* Second attempt

For the second attempt, I used the same activation function `relu`, however increased the number of layers and the number of nodes for each layer. I also increased the number of epochs from 100 to 150, as I felt that it would help with achieving a higher than 75% accuracy. See image below. 

![Screen Shot 2023-04-09 at 00 14 18](https://user-images.githubusercontent.com/116304118/230746590-2c9298c0-9268-4eb4-a7cc-bf7173ac96f0.png)


![second_attempt_1](https://user-images.githubusercontent.com/116304118/230743136-cec2f187-398a-481f-87a1-9dfb087ee51e.png)

![second_attemp](https://user-images.githubusercontent.com/116304118/230743148-a1270395-33ed-4e54-a6da-592ef4448b10.png)

This model achieved a predictive accuracy of 76.5%, which is higher than 75%. 

To view the results, click here [`AlphabetSoupCharity_Optimization.h5`](https://github.com/HJandu/deep-learning-challenge/blob/main/h5_files/AlphabetSoupCharity_Optimization.h5)

Click here to view [code](https://github.com/HJandu/deep-learning-challenge/blob/main/AlphabetSoupCharity_Optimization.ipynb)

## Final try

Although the second attempt achieved 76.5% predictive accuracy, which is higher than 75%, I thought I would try changing the activation function from `relu` to `LeakyReLU`, to see if a even higher predictive accuracy score could be achieved. I  kept everything the same, including the number of layers. 
With this model, the predictive accuracy was 77.2%. 

![Screen Shot 2023-04-08 at 23 31 15](https://user-images.githubusercontent.com/116304118/230745389-ad048d20-7259-450b-86a3-22e708fdb660.png)


![Screen Shot 2023-04-08 at 23 31 06](https://user-images.githubusercontent.com/116304118/230745414-4f8403ff-0973-4a33-a668-106782aafc50.png)

![Screen Shot 2023-04-08 at 23 30 55](https://user-images.githubusercontent.com/116304118/230745454-dce0dfe6-a0c9-488f-817d-2040d50e2434.png)

To view the results, click here[`AlphabetSoupCharity_Optimization_final`](https://github.com/HJandu/deep-learning-challenge/blob/main/h5_files/AlphabetSoupCharity_Optimization_final.h5 ) 

Click here to view [code](https://github.com/HJandu/deep-learning-challenge/blob/main/AlphabetSoupCharity_Optimization_final.ipynb)

## Summary

In conclusion, the final model performed better with a predictive accuracy of 77.2% and loss of 0.465, which is less than the loss of the first and second model. Trying different activation functions, removing unnecessary columns, adding more hidden layers and increasing the epochs has helped with with the predictive accuracy score. 
In the future, I could continue with the 'tanh' function and adjust some of the binning thresholds for the `NAME`, `CLASSIFICATION`, and `APPLICATION_TYPE` value counts, to see if that has an impact on the accuracy, or even increase the number of epochs to 200, to ensure a better chance of finding an optimum model.



