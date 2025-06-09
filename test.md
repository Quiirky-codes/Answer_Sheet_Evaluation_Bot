## CIE SCHEME

**Global Academy of Technology, Bengaluru**

**Department of Artificial Intelligence & Machine Learning**

**Internal Test No: 2**

**Semester** 5th

**Subject Name** Deep Learning Principles & Practices

**Subject Code** 2 1 A M L 5 3

**Time: 90 Mins.**  

**Max. Marks: 40**

| Q. No. | Questions | Marks |
|---|---|---|
| 1 | Examine the implications of initializing weights with excessively large or small values in a deep neural network. Discuss the potential challenges associated with weight initialization and the strategies employed to address these challenges. | 10 |

**Initializing weight in Neural Network**

Let the initial weight value be 'Zero'.

If the weight on the first layer is:

Given a1 
(Image: a matrix equation W(i) = [W11, W12, W13; W21, W22, W23; W31, W32, W33; W41, W42, W43] = [0,0,0; 0,0,0; 0,0,0; 0,0,0] )
2M

Z(i) = [X1 X2 X3 X4] *
(Image: a matrix equation W(i) = [W11, W12, W13; W21, W22, W23; W31, W32, W33; W41, W42, W43] )

Z(i) = [X1 X2 X3 X4] * [0,0,0; 0,0,0; 0,0,0; 0,0,0] 
2M

Individually, for each neuron in the hidden layer 1, we have: 
```
^(1)
Z_1 = [X_1.0 + X_2.0 + X_3.0 + X_4.0]
= 0.
^(1)
Z_2 = [X_1.0 + X_2.0 + X_3.0 + X_4.0]
= 0
^(1)
Z_3 = [X_1.0 + X_2.0 + X_3.0 + X_4.0]
^(1)
Z = [Z_1^(1) Z_2^(1) Z_3^(1)]
= [0 0 0]
Assume Same activation function is applied to all
hidden layers.
^(1)
A_1 = A_2 = A_3
⇒ Neuron are Said to be 'Symmetric'
⇒ using 'Zero' as the Value to initiate the weight
is practically going to make the use of multiple
neuron in hidden layers Redundant. That is why
the weight should not be initialized as 0.
However, the Bias can be initialized as 0.
⇒ The option of initializing is with random Values.
2 ways
by wrinormal
distribution
→ If the weight are initialized with large value,
the Value of Z will be relatively large.
If observed in Sigmoid and tanh function, the slope
Very large Value of Z
SIGMOID
0.5
tanh
Very Small Value of Z
Effect of large or small Value of weight
2M
2M
```

The image contains handwritten text and diagrams. The text discusses the initialization of weights and biases in a neural network. The main takeaway is that while it is not recommended to initialize weights as zero due to redundancy, initializing biases as zero is acceptable. The text highlights the option of using a random normal distribution for weight initialization. 

The diagrams illustrate the effect of large and small values of Z (the weighted sum of inputs to a neuron) on the activation functions Sigmoid and tanh. 

- For Sigmoid: A very small Z results in a near-zero activation, while a very large Z results in a near-one activation.
- For tanh: A very small Z results in a near-zero activation, while a very large Z results in a near-one activation.

These diagrams demonstrate how different values of Z influence the output of these activation functions.Leads to Vanishing Gradient and Exploding gradients. These challenges are overcome by using some optimum solutions. To strike the balance between these extremes, 2 types of weight initializations are used.
1) Kaiming He initialization
2) Xavier initialization
2M

Explore the challenges posed by local minima and saddle points in the context of batch gradient descent. Elaborate on the difficulties these points present and discuss how stochastic gradient descent (SGD) offers solutions to mitigate these challenges.
10

The image shows a non-convex curve with a global minima and two local minima. The global minima is the point on the curve with the lowest value and the local minima are the points with the lower value in their neighborhood. There is a saddle point on the curve which is characterized by being the highest value in its neighborhood. 
Fig (c) Batch Gradient Descent for Non-Convex Loss Curve
2M

The three figures illustrate how batch gradient descent, mini-batch gradient descent and stochastic gradient descent minimize the loss function and find the global minima. The figure for batch gradient descent shows that the loss function decreases with each step in the direction of the gradient. The figure for mini-batch gradient descent shows that the loss function decreases in a slightly more irregular manner with each step, but it is still able to find the global minima. The figure for stochastic gradient descent shows that the loss function decreases in a much more irregular manner with each step, but it is still able to find the global minima. 
Batch Gradient Descent

An image of a contour plot with a concentric circle pattern. The minimum of the loss function is represented by the innermost circle. A red arrow starts from the left side of the plot and takes 6 steps towards the minima in the innermost circle following the contours. 

Mini-Batch Gradient Descent

An image of a contour plot with a concentric circle pattern. The minimum of the loss function is represented by the innermost circle. A red arrow starts from the left side of the plot and takes 5 steps towards the minima in the innermost circle following the contours. The arrow changes direction at each step.

Stochastic Gradient Descent 

An image of a contour plot with a concentric circle pattern. The minimum of the loss function is represented by the innermost circle. A red arrow starts from the left side of the plot and takes 10 steps towards the minima in the innermost circle following the contours. The arrow changes direction at each step.

3MA mini-batch gradient descent follow a middle path. It is not as smooth as batch gradient descent and as oscillating as Stochastic gradient descent.
Although, Batch gradient descent seem to give the smoothest paths to reach the point of lowest loss, that are 2 main challenges.
1) loading the entire training data to memory can be a big challenge.
2) They may stick to a local minima.
In all practical situations, the loss curve in a Simple Convex Curve (L1) as show in fig(C) not a more complex, non-Convex Curve (L2).
For L2, thus is a possibility to get trapped but a few minima.
In local minima:
A diagram of a non-convex curve with local minima and global minima is shown. The local minima are labeled L1, L2, L3, L4, L5, L6, and L7. The global minima is labeled L9. The saddle point is labeled L8. An arrow is shown going from the global minima to the saddle point. Arrows are also shown moving between the local minima.
Fig(d): Stochastic Gradient Descent for Non-Convex loss curve
In Stochastic gradient descent, the possibility to get trapped in local minima is much low.
In fig(d), each training record t1, t2, t3 etc., thus is a separate loss curve L1, L2, L3 etc, respectively. 
2M-> This lead to a higher poubility to escape local minima and reach the global minima for the curve.

-> Therefore , the possibility of minning the global optimal point is much leu in case of Stochastic gradient descent.

In certain points of lou curve , the gradient may also be 0. Such points known as Saddle Points.

-> An SGD can overcome Saddle points too , whereas the batch gradient descent face a serious challenge.

-> disadvantage of SGD is that the model parameters update is so frequent that it is computationally expensive & also take a Significantly high training time for large training datasets.

**Image 1**:
The image shows two diagrams representing the effect of learning rate on gradient descent.
- **Top diagram:** depicts a large learning rate and shows the gradient descent jumping over the global minima multiple times, eventually ending up at a local minima.
- **Bottom diagram:** shows a small learning rate. The gradient descent slowly descends towards the global minima and avoids getting stuck in local minima.

**Text 2**
3) a) Interpret the challenges faced by traditional ANN to deal with image and what are the building blocks of CNN.

**Image 2:** The image is blank.
4. a) What will be the dimensions of the output feature map for an input feature map of size 380x270x64 which passes through a pooling layer having filter size 2x2 and a stride size of 2?

(n<sub>h</sub> - f  + 1)/ s) * ((n<sub>w</sub> - f  + 1)/ s) * n<sub>c</sub>

n<sub>h</sub>=380, n<sub>w</sub>=270, n<sub>c</sub>=64, f=2 and s=2

therefore, the dimension of the output feature map will be 190x135x64

10
3M
2M

b) Enumerate the various types of pooling employed in CNN architectures and delve into the details of one specific pooling type, supported by a practical example by highlighting the significance of the pooling layer in Convolutional Neural Networks (CNNs) and its diverse applications.

=> **Pooling**
=> An effective technique to reduce the number of trainable parameters in by using pooling layers.
=> Pooling layers help to successfully downsample an image, thus reducing its dimension and reducing the number of trainable parameters.
=> Pooling layer helps to progressively reduce the spatial size of the representation by it maximum or average or sum Value; So it is independent of the Spatial coordinate of the value within a specific window.
=> For an input feature map having dimensions n<sub>h</sub>x n<sub>w</sub>x n<sub>c</sub>, if a fxf pooling filter is applied with a stride size 's', the dimension of the output map will be 
(n<sub>h</sub> - f  + 1)/ s) * ((n<sub>w</sub> - f  + 1)/ s) * n<sub>c</sub>

3-type of pooling 
1. Max pooling
2. Average pooling
3. Sum pooling

The image shows 3x4 matrix with values. To the right are the results of different pooling methods:
- **Max pooling**: 
    - A 2x2 sliding window is applied to the original matrix.
    - The maximum value within each window is selected as the output.
    - The result is a 2x2 matrix with values 5, 7, 13, and 24.
- **Average pooling**:
    - A 2x2 sliding window is applied to the original matrix.
    - The average value within each window is calculated as the output.
    - The result is a 2x2 matrix with values 1, 3, 6, and 11.
- **Sum pooling**:
    - A 2x2 sliding window is applied to the original matrix.
    - The sum of the values within each window is calculated as the output.
    - The result is a 2x2 matrix with values 4, 12, 24, and 44.

Original image:
| 1 | 5 | 2 | 4 |
|---|---|---|---|
| 5 | -7 | 7 | -1 |
| 13 | -3 | 4 | 7 |
| 5 | 9 | 24 | 9 |

=> | 5 | 7 |
|---|---|---|
| 13 | 24 |
Max pooling

=> | 1 | 3 |
|---|---|---|
| 6 | 11 |
Avg pooling

=> | 4 | 12 |
|---|---|---|
| 24 | 44 |
Sum pooling

3M
## Handwritten Text and Image Description:

**Image 1:  Handwritten notes on pooling types**

-  "for 2x2 pooling filter and a stride size of 2"
-  "For max pooling - maximum value of the Subset is considered."
-  "For average pooling - average of the pixel value of the different subset are taken."
-  "i.e., 1+5+5+(-7) = 1
    ----------------------
             4 "
-  "||  3, 6 & 11 respectively"
-  "For sum pooling - Sum of all the values in the Subset are done."

**Image 2: Table comparing different gradient descent types**

**Table:** 
The table has three columns. The first column is labeled "5" at the top and "10" at the bottom. The second column contains the text, and the third column is labeled "3M" at the top and bottom.

**Text:**
What are the trade-offs between batch, minibatch, and stochastic gradient descent in terms of convergence speed, computational efficiency, and generalization? Explain with a neat diagram how gradient descent differs from other gradient descent variants. 

1. Batch Gradient Descent:
   * Convergence Speed:  Slower compared to other variants, especially on large datasets, as it processes the entire dataset before updating weights.
   * Computational Efficiency: Computationally expensive, as it requires storing and processing the entire dataset in each iteration.
   * Generalization: May converge to a more accurate minimum due to the use of the entire dataset, but might be prone to getting stuck in local minima.

2. Mini-Batch Gradient Descent: 
    * Convergence Speed:  Faster than batch gradient descent due to processing smaller subsets (batches) of the dataset in each iteration.
    * Computational Efficiency: More computationally efficient than batch gradient descent, especially for large datasets. Utilizes parallelism better.
    * Generalization: Balances between batch and stochastic; tends to generalize better than batch but may still find a good minimum.

3. Stochastic Gradient Descent (SGD):
    * Convergence Speed: Fastest convergence, as it updates weights based on individual data points. 
    * Computational Efficiency: Highly efficient due to processing one data point at a time.
    * Suitable for online learning scenarios.
    * Generalization: May oscillate around the minimum, but the noisy updates can help escape local minima, leading to better generalization. 

Trade-Offs: 
   * Convergence Speed: 
     * Batch: Slow 
     * Mini-Batch: Moderate
     * Stochastic: Fast
   * Computational Efficiency:
     * Batch: Low
     * Mini-Batch: Moderate to High
     * Stochastic: High 
## Text from image:

- Global Average Pooling (GAP): Utilized in later layers for spatial information reduction and feature summarization. 
4. Normalization and Regularization:
    - Batch Normalization: Enhances convergence and stability during training.
    - Dropout: Regularizes the network by randomly dropping neurons to prevent overfitting. 
5. Skip Connections:
    - Residual Connections: Employed to facilitate the flow of gradients and ease the training of deeper networks.
    - Feature Concatenation: Incorporates skip connections between different layers to enhance feature reuse.
6. Fully Connected Layers:
    - Flatten Layer: Precedes fully connected layers to transition from convolutional layers to enhance features.
    - Dense Layers: Multiple dense layers with gradual reduction in neurons to extract high-level features.
7. Output Layer:
    - Softmax Activation: Applied to produce class probabilities for multi-class object recognition.
    - Sigmoid Activation: Used for binary classification tasks. 
Additional Considerations:
    - Transfer Learning:
        - Pre-training on large datasets like ImageNet and fine-tuning for the target task to leverage learned features.
        - Potential use of architecture variants inspired by successful models like Efficient Net or Mobile Net for efficiency.
    - Data Augmentation:
        - Implementation of data augmentation techniques (rotation, scaling, flipping) to increase model robustness and handle diverse scenarios.
    - Optimization and Regularization:
        - Adaptive learning rate strategies (e.g., Adam optimizer) for efficient convergence.
        - L2 regularization for weight decay.
Evaluation Metrics:
    - Performance Metrics: Precision, recall, F1-score, and accuracy for comprehensive evaluation.
    - Latency: Assessment of inference time for real-time deployment on autonomous vehicles.

**8. Apply the principles of transfer learning to explain how pre-trained CNN models, such as ResNet, can be fine-tuned for a new image classification task. Provide specific steps in the process.**

**Steps for Fine-Tuning a Pre-trained ResNet Model:**
1. **Select a Pre-trained ResNet Model:**
    - Choose a ResNet model that was pre-trained on a large and diverse dataset, such as ImageNet. The pre-trained model will have learned generic features that can be beneficial for a variety of tasks.
2. **Remove the Fully Connected Layers:**
    - The last few layers of a pre-trained ResNet model typically include fully connected layers for the specific task it was trained on (e.g., ImageNet classification). Remove these layers to retain the convolutional base.
3. **Add New Fully Connected Layers:**
    - Append new fully connected layers at the end of the pre-trained ResNet model. These layers will be specific to the new image classification task. Adjust the number of neurons in the output layer based on the number of classes in the new task.
4. **Freeze Convolutional Layers:**
## Text from Image:

**Freeze the weights of the convolutional layers in the pre-trained ResNet model.**
This prevents these layers from being updated during the initial training on the new task.

**5. Compile the Model:**
• Compile the fine-tuned model using an appropriate loss function (e.g., categorical crossentropy for multi-class classification) and an optimizer (e.g., Adam). 

**6. Data Preprocessing:**
• Preprocess the new dataset using the same preprocessing steps applied to the original dataset used to train the pre-trained ResNet model. This may include normalization, resizing, and data augmentation. 

**7. Fine-Tuning:**
• Train the model on the new dataset using the compiled model and frozen convolutional layers. This step allows the new fully connected layers to learn task-specific features while preserving the knowledge embedded in the pretrained convolutional base.

**8. Unfreeze Convolutional Layers (Optional):**
• Optionally, unfreeze some of the top layers of the convolutional base and continue training. This allows the model to adapt to more task-specific features present in the new dataset.

**9. Hyperparameter Tuning:**
• Fine-tune hyperparameters such as learning rate, batch size, and regularization strength to optimize model performance on the new task.

**10. Evaluate and Validate:**
• Evaluate the fine-tuned model on a validation set to ensure it generalizes well to unseen data. Adjust the model architecture or hyperparameters if necessary.

**11. Inference on Test Set:**
• Once satisfied with the model’s performance, use it for inference on the test set to assess its effectiveness in real-world scenarios.

**Benefits of Fine-Tuning with Transfer Learning:**
• **Faster Convergence:** Utilizes pre-learned features, accelerating convergence on the new task.
• **Data Efficiency:** Effective even with limited labeled data for the new task.
• **Generalization:** Transfers knowledge from one domain to another, improving model generalization.

**ResNet50 Model Architecture**

A diagram showing a ResNet50 model architecture. The model consists of 5 stages. Stage 1 has the following layers:
* Zero Padding
* Conv
* Batch Norm
* ReLu
* Max Pool

Stages 2, 3, 4, and 5 consist of the following layers:
* Conv Block
* ID Block
* Conv Block
* ID Block
* Conv Block
* ID Block
* Conv Block
* ID Block
* Conv Block
* ID Block

Stage 6 consists of the following layers:
* Avg Pool
* Flattening
* FC

Input and output are at the left and right ends of the diagram, respectively.

**COs Addressed and Cognitive Level**

| CO No. | Course Outcomes | RBT Level |
|---|---|---|
| 21CO53.1 | Understand and Analyse the fundamentals that drive deep learning networks | L2 |
| 21CO53.2 | Build, train and apply fully connected neural networks | L3 |
| 21CO53.3 | Analyze convolutional networks and their role in image processing. | L3 |
| 21CO53.4 | Implementation of deep learning techniques to solve real-world problems. | L3 |

**Signature of Course Coordinator**  **Signature of Module Coordinator**  **Signature of HOD**
