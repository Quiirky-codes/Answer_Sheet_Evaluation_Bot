## Extracted Text 

**CIE SCHEME**                                                                                                                                                                                                                   **UG**

**[Global Academy of Technology Logo]**  **ಗ್ಲೋಬಲ್ ಅಕಾಡೆಮಿ ಆಫ್ ಟೆಕ್ನಾಲಜಿ, ಬೆಂಗಳೂರು**    **[NAAC Logo]**
**Global Academy of Technology**, Bengaluru
**Department of Artificial Intelligence & Machine Learning**            Internal Test No: 2

Semester _______ 5<sup>th</sup> _______  Subject Code  _2_  _1_  **A**  _M_  _L_  _5_  _3_  _______  _______  _______
Subject Name: Deep Learning Principles & Practices                                                                                                 **Max. Marks: 40**

**Time: 90 Mins.**

| Q. No. | Questions | Marks |
|---|---|---|
| 1 |  Examine the implications of initializing weights with excessively large or small values in a deep neural network. Discuss the potential challenges associated with weight initialization and the strategies employed to address these challenges.
<br> <br>
$\implies$  _Initializing_  _weight_  _in_  _Neural_  _Network_ 
<br>
Let the initial weight value be 'Zero'. 
<br>
If the weight on the first layer is
<br>
Given as
<br>
$W^{(1)} =  \begin{bmatrix} W_{11} & W_{12} & W_{13} \\ W_{21} & W_{22} & W_{23} \\ W_{31} & W_{32} & W_{33} \\ W_{41} & W_{42} & W_{43}  \end{bmatrix}  = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0  \end{bmatrix}$           **2M**
<br> <br>
$Z^{(1)} =  \begin{bmatrix} X_1 & X_2 & X_3 & X_4 \end{bmatrix} * \begin{bmatrix} W_{11} & W_{12} & W_{13} \\ W_{21} & W_{22} & W_{23} \\ W_{31} & W_{32} & W_{33} \\ W_{41} & W_{42} & W_{43} \end{bmatrix}$
<br> <br>
$ \therefore Z^{(1)} =  \begin{bmatrix} X_1 & X_2 & X_3 & X_4 \end{bmatrix} * \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}$         **2M**
<br>
Individually, for each neuron in the hidden layer 1, we have
 | 10 |


## Image Description:

The image is a scanned document of a university exam paper. The paper is in English, with a handwritten answer in black ink. 

**Key visual elements:**

* **Header:** Features a logo of the Global Academy of Technology on the left and a NAAC logo on the right. The name and location of the institution are prominently displayed in both English and Kannada.
* **Exam Details:** Clearly outlines the department, semester, subject name, subject code, internal test number, maximum marks, and time allotted. 
* **Question Paper Format:**  The table neatly presents question numbers along with their corresponding marks.
* **Handwritten Answer:**  The answer to the question is written in a slightly messy cursive style, indicating it was likely written in a time-sensitive exam environment. 
* **Mathematical Notation:** The answer utilizes mathematical symbols and matrices to explain the concept of weight initialization in a neural network.

**Overall Impression:** 

The document gives a glimpse into the academic environment of an Indian university, specifically focusing on a Deep Learning course. The handwritten response, while slightly messy, demonstrates the student's attempt to apply their knowledge in a structured manner during the exam. 
## Extracted Text

**(1)**
Z<sub>1</sub> = [X<sub>1</sub>.0 + X<sub>2</sub>.0 + X<sub>3</sub>.0 + X<sub>0</sub>.0]
     = 0
 
**(2)**
Z<sub>2</sub> = [X<sub>1</sub>.0 + X<sub>2</sub>.0 + X<sub>3</sub>.0 + X<sub>0</sub>.0] 
    = 0

**(3)**
Z<sub>3</sub> = [X<sub>1</sub>.0 + X<sub>2</sub>.0 + X<sub>3</sub>.0 + X<sub>0</sub>.0]

**(n)**
Z = [ Z<sub>1</sub><sup>(1)</sup>   Z<sub>2</sub><sup>(2)</sup>  Z<sub>3</sub><sup>(3)</sup> ] 
   = [ 0  0  0]

*Assume* Same activation function is applied to all neuron in a hidden layer.
neuron = [ ]
A<sub>1</sub><sup>(1)</sup> =  A<sub>2</sub><sup>(2)</sup> = A<sub>3</sub><sup>(3)</sup>

=> Neuron are Said to be 'Symmetric'. 

=> using 'Zero' as the Value to initiate the weight, is practically going to make the use of multiple neuron in hidden layers redundant. That is why the weight should not be initialized as 0.

However, the Bias Can be initialized with random Values.

=> the option of initializing is with random Values. 

                                    2 ways
                     |-----------------T------------------|
                     |                                        |    
                     V                                       |
 by unimodal                                             by using a 
distribution                                               uniform distribution. 

=> If the weight are initialized with large Values, the Value of Z will be relatively large. 

=> If observed on Sigmoid and tanh functions, the Slope

<br>
<br>

**Effect of large or Small Value of weight**


## Image Description

The image presents handwritten notes and diagrams explaining the impact of weight initialization in neural networks. 

**Top Section:**

* Mathematical equations demonstrate how initializing weights to zero leads to symmetric neurons and redundancy, rendering multiple neurons in hidden layers ineffective. 
* The notes emphasize that while weights should not be initialized to zero, bias can be initialized with random values.
* Two primary methods for weight initialization are presented: using a unimodal distribution and using a uniform distribution.

**Bottom Section:**

* **Hand-drawn graphs** illustrate the effect of large weight initialization on Sigmoid and Tanh activation functions.
* **Left Graph (Sigmoid):** The x-axis represents the net input 'Z', and the y-axis represents the activation output. A steep slope is observed near the activation function's threshold (0.5), indicating that a large 'Z' value results in the neuron being pushed into the saturated region, leading to slow learning.
* **Right Graph (Tanh):**  Similar to the Sigmoid graph, a large 'Z' value pushes the neuron toward the saturated regions of the Tanh function (both positive and negative extremes), leading to vanishing gradients and hindered learning.

**Overall:**

The image effectively conveys the importance of proper weight initialization in neural networks. Initializing weights with large values, especially when using Sigmoid or Tanh activations, can lead to saturation and slow down the learning process. 
**Text Transcription:**

Leads to Vanishing Gradient and Exploding gradients. These challenges are overcome by using some optimum solutions. To strike the balance between these extremes, 2 types of weight initializations are used. 
1) Kaiming He initialization
2) Xavier initialization                                                                                    **2M**

Explore the challenges posed by local minima and saddle points in the context of batch gradient descent. Elaborate on the difficulties these points present and discuss how stochastic gradient descent (SGD) offers solutions to mitigate these challenges.                                      **10**


_local_  
       minima           _Global_ 
                              minima                                                                                               **Saddle** 
                                                                                                                                       L2
                                                                                                                                       / \
                                                                                                                                      /   \
                                                                                                                                      L1
                                                                                                                                              **2M**


_Fig.(c) Batch Gradient Descent for_
        _Non-Convex loss Curve_

**Batch Gradient Descent**                    **Mini-Batch Gradient Descent** 

(Image 1)                                                     (Image 2)

**Stochastic Gradient Descent**                                                                                                    **3M**
 
(Image 3)

**Image Description:**

The image depicts different types of gradient descent optimization algorithms, illustrating their behaviour in navigating a non-convex loss curve. 

**Top Diagram:**
The top diagram showcases a non-convex loss curve, marked with a global minimum, a local minimum, and a saddle point. Arrows illustrate the path a hypothetical optimization algorithm might take.

* **Global Minimum:** This is the lowest point on the loss curve, representing the optimal solution.
* **Local Minimum:**  A low point on the curve, but not the absolute lowest. Optimization algorithms can get stuck in local minima.
* **Saddle Point:** A point on the curve with a flat gradient, making it difficult for algorithms to escape.

**Bottom Diagrams:**

Three diagrams illustrate different gradient descent approaches:

* **Batch Gradient Descent (Image 1):** Uses the entire dataset for each step, resulting in a smooth but potentially slow descent towards the minimum (marked with a '+').
* **Mini-Batch Gradient Descent (Image 2):** Employs a subset (mini-batch) of the data for each step, creating a less smooth but faster descent path.
* **Stochastic Gradient Descent (Image 3):** Utilizes a single data point for each step, leading to the most erratic but potentially fastest descent. The path oscillates significantly while approaching the minimum.

**Overall:**

The image aims to visually compare and contrast how these gradient descent methods operate within the challenging landscape of a non-convex loss curve, highlighting their strengths and weaknesses in finding the global minimum.
## Text:

A mini-batch gradient descent follows a middle path. It is not as smooth as batch gradient descent and as oscillating a Stochastic gradient descent.

Although, Batch Gradient descent Seem to give the Smoothest path to reach the point of lowest loss, there are 2 main Challenges.
(1) loading the entire training data to memory Can be a big challenge.
(2) They may Stick to a local minima.
- In all practical Situations, the loss curve is not a Simple Convex Curve (L1) as Shown in fig(c) but a more Complex, non-Convex Curve (L2).
- For L2, there is a possibility to get trapped in local minima.

*Figure with hand-drawn axes and a curve representing a loss function. The curve is non-convex with multiple local minima and a single global minimum. Several dotted lines (Lt1, Lt2, Lt3, Lt4, Lt5, Lt6, Lt7, Lt8, Lt9) originating from different points on the curve converge towards different minima.  A saddle point is marked on the curve. The vertical axis is unlabeled. The horizontal axis arrow points towards 'global minima'.  Below the axis: "Fig(d), Stochastic Gradient Descent for _non-convex_ loss curve".* 

-> In Stochastic gradient descent, the possibility to get trapped in local minima is much less.
-> In fig(d), each training record t1, t2, t3, etc., thus is a Separate loss curve Lt1, Lt2, Lt3 etc., respectively.


## Description:

The image shows a handwritten page of notes discussing mini-batch gradient descent and its comparison to batch gradient descent and stochastic gradient descent. The notes highlight two main challenges of batch gradient descent: memory requirements for loading large datasets and the tendency to get stuck in local minima. 

The central part of the image is a hand-drawn graph illustrating the concept of local minima in a non-convex loss curve. The graph depicts:

* **Non-convex Loss Curve:** The curve represents a complex loss function with multiple dips and rises, indicating the presence of local minima. 
* **Local Minima:** Several points on the curve are labeled as Lt1, Lt2, Lt3, etc., representing local minima where the gradient descent algorithm might converge prematurely.
* **Global Minimum:** The lowest point on the curve is labeled as "global minima," representing the ideal solution.
* **Saddle Point:** A point on the curve is marked as a "saddle point," indicating a region where the gradient is flat in some directions.
* **Dotted Lines:** The dotted lines labeled Lt1, Lt2, etc., starting from different points on the curve and converging towards different minima, visually demonstrate how the gradient descent algorithm might take different paths and get trapped in local minima.

The handwritten text below the graph emphasizes that in stochastic gradient descent, the possibility of getting trapped in local minima is significantly reduced because each training record essentially forms a separate loss curve, leading to a more diverse exploration of the loss landscape.
**Text Transcription:**

--> This leads to a higher probability to escape local minima and reach the global minima for the curve 
--> Therefore, the possibility of missing the global optimal point is much less in case of Stochastic gradient descent. 
In Certain points of low curve, the gradient may also be 0. Such points known as Saddle points.
--> An SGD can overcome Saddle points too, whereas the batch gradient descent faces a serious challenge.
↳disadvantage of SGD is that the model parameter update is so frequent that it is computationally expensive & also takes a significantly high training time for large training datasets. 

**Large learning rate** 

**(Figle) Effect of learning rate on Gradient Descent Small learning rate**

3 a) Interpret the challenges faced by traditional ANN to deal with image and what are the building blocks of CNN. 10 

**Image Description:**

The image consists of handwritten text and two diagrams that illustrate the concept of learning rate in gradient descent optimization. 

**Diagrams:**

Both diagrams depict contour plots representing a loss function with a global minimum. The contour lines represent points of equal loss, with the centermost line indicating the minimum.  Arrows within the diagrams symbolize the path of gradient descent.

* **Top Diagram (Large Learning Rate):** This diagram illustrates a scenario with a large learning rate. The arrows take large, erratic steps across the contour plot, potentially overshooting the minimum and oscillating around it. This indicates difficulty in converging to the optimal solution. The text "Large learning rate" is underlined beneath the diagram.

* **Bottom Diagram (Small Learning Rate):** This diagram illustrates gradient descent with a small learning rate.  The arrows take smaller, more measured steps down the loss function. While this leads to slower progress, it's more likely to converge directly towards the minimum.  The text "Small learning rate" appears below this diagram.


**Overall:** 

The handwritten text, combined with the diagrams, explains the trade-off between large and small learning rates in gradient descent optimization. It highlights that while a large learning rate might escape local minima, it can lead to instability. Conversely, a small learning rate offers more stable convergence but may require more time to reach the global minimum. 
## Extracted Text

**4**  
a) What will be the dimensions of the output feature map for an input feature map of size 380x270x64 which passes through a pooling layer having filter size 2x2 and a stride size of 2?

 ((n<sub>h</sub> - f)/s + 1) x ((n<sub>w</sub> - f)/s + 1) x n<sub>c</sub>

n<sub>h</sub>=380, n<sub>w</sub>=270, n<sub>c</sub> =64, f=2 and s=2

therefore, the dimension of the output feature map will be **190x135x64** 

b) Enumerate the various types of pooling employed in CNN architectures and delve into the details of one specific pooling type, supported by a practical example by highlighting the significance of the pooling layer in Convolutional Neural Networks (CNNs) and its diverse applications. 

**Pooling**
-> An effective technique to reduce the number of trainable parameters is by using pooling layers. 
-> Pooling layers help to successfully downsample an image, thus reducing its dimension and reducing the number of trainable parameters.
-> Pooling layer helps to progressively reduce the spatial size of the representation 
-> A pooling window is represented by its maximum or average or sum value; so it is independent of the Spatial Coordinate of the value within a specific window.
-> For an input feature map having dimensions n<sub>h</sub> x n<sub>w</sub> x n<sub>c</sub>, if a f x f pooling filter is applied with a stride size 's', the dimension of the o/p map will be
((n<sub>h</sub> - f)/s + 1) x ((n<sub>w</sub> - f)/s + 1) x n<sub>c</sub>

**3-types of Pooling**
1. Max Pooling
2. Average pooling
3. Sum Pooling 

**Original Image**
| 1 | 5 | 2 | 4 |
|---|---|---|---|
| 5 | -7 | 7 | -1 |
| 13 | -3 | 4 | 7 | 
| 5 | 9 | 24 | 9 |
 **->**
**Max pooling**
| 5 | 7 |
|---|---|
| 13 | 24 |

**Avg pooling**
| 1 | 3 |
|---|---|
| 6 | 11 |

**Sum Pooling**
| 4 | 12 |
|---|---|
| 24 | 44 |

## Image Description:

The image displays handwritten notes and calculations related to pooling in Convolutional Neural Networks (CNNs). 

**Key elements**:

* **Formula:** A formula for calculating the output feature map dimensions after pooling is written at the top of the page:  ((n<sub>h</sub> - f)/s + 1) x ((n<sub>w</sub> - f)/s + 1) x n<sub>c</sub>. Variables are defined (nh, nw, nc, f, s)
* **Explanation of Pooling:**  Below the formula, the notes define pooling, listing its benefits like reducing the number of trainable parameters, downsampling images, and reducing spatial size. 
* **Types of Pooling:**  Three common types of pooling (Max, Average, Sum) are listed.
* **Example:** A practical example visually demonstrates the different pooling types using a sample image (a 4x4 grid of numbers) and 2x2 pooling: 
    * **Original Image:**  The original 4x4 matrix is shown.
    * **Max Pooling:** The resulting 2x2 matrix after max pooling is displayed.
    * **Average Pooling:** The 2x2 matrix resulting from average pooling is shown.
    * **Sum Pooling:**  The output 2x2 matrix after sum pooling is presented.

**Overall Impression**:  The image effectively combines text and visuals to explain the concept and types of pooling in CNNs, making it easy to understand for someone learning about this topic. 
## Extracted Text: 

**Handwritten Text (Top Image):**

for 2x2 pooling filter and a stride size of 2 
For max pooling -> maximum Value of the 
                                          Subset in Considered.

for average pooling -> average of the pixel value
                                     of the different Subset are 
                                     taken.
                              i.e;       (1+5+5+ (-7))  = 1
                                                    -------- 
                                                      4    
                    ||| lads         3,6, $11 respectively.
for sum pooling ->   Sum of the Value in
                                   the Subset are done.

**Digital Text:**

5 What are the trade-offs between batch, minibatch, and stochastic gradient descent in terms of convergence speed, computational efficiency, and generalization? Explain with neat diagram how gradient descent differs from other gradient descent variants.

1. **Batch Gradient Descent:**
* **Convergence Speed:** Slower compared to other variants, especially on large datasets, as it processes the entire dataset before updating weights.
* **Computational Efficiency:** Computationally expensive, as it requires storing and processing the entire dataset in each iteration.
* **Generalization:** May converge to a more accurate minimum due to the use of the entire dataset, but might be prone to getting stuck in local minima. 

2. **Mini-Batch Gradient Descent:**
* **Convergence Speed:** Faster than batch gradient descent due to processing smaller subsets (batches) of the dataset in each iteration.
* **Computational Efficiency:** More computationally efficient than batch gradient descent, especially for large datasets. Utilizes parallelism better.
* **Generalization:** Balances between batch and stochastic; tends to generalize better than batch but may still find a good minimum.

3. **Stochastic Gradient Descent (SGD):**
* **Convergence Speed:** Fastest convergence, as it updates weights based on individual data points.
* **Computational Efficiency:** Highly efficient due to processing one data point at a time. 
* Suitable for online learning scenarios.
* **Generalization:** May oscillate around the minimum, but the noisy updates can help escape local minima, leading to better generalization.

**Trade-offs:**
* **Convergence Speed:**
    * Batch: Slow
    * Mini-Batch: Moderate
    * Stochastic: Fast
* **Computational Efficiency:**
    * Batch: Low
    * Mini-Batch: Moderate to High
    * Stochastic: High

3M
2M
10
3M




## Image Description: 

The image contains handwritten notes about different types of pooling operations used in convolutional neural networks (CNNs) alongside printed text discussing variations of gradient descent algorithms. 

**Handwritten Notes:**

The notes explain three types of pooling using a 2x2 filter and a stride of 2:

1. **Max Pooling:** The maximum value within the 2x2 subset of the input is selected as the output.
2. **Average Pooling:** The average of all values within the 2x2 subset is calculated and used as the output. An example calculation is shown with values 1, 5, 5, and -7, resulting in an average of 1.
3. **Sum Pooling:** The sum of all values within the 2x2 subset is computed and used as the output.

**Digital Text:**

This section delves into the trade-offs between three gradient descent variants: Batch, Mini-Batch, and Stochastic Gradient Descent (SGD). The text explains each variant's convergence speed, computational efficiency, and generalization capabilities, highlighting their strengths and weaknesses. A summary of the trade-offs is presented at the end, comparing the three variants across the mentioned aspects. 
## Extracted Text:

- **Global Average Pooling (GAP):** Utilized in later layers for spatial information reduction and feature summarization. 
4. **Normalization and Regularization:**
   - **Batch Normalization:** Enhances convergence and stability during training.
   - **Dropout:** Regularizes the network by randomly dropping neurons to prevent overfitting. 
5. **Skip Connections:**
   - **Residual Connections:** Employed to facilitate the flow of gradients and ease the training of deeper networks. 
   - **Feature Concatenation:** Incorporates skip connections between different layers to enhance feature reuse. 
6. **Fully Connected Layers:**
   - **Flatten Layer:** Precedes fully connected layers to transition from convolutional layers. 
   - **Dense Layers:** Multiple dense layers with gradual reduction in neurons to extract high-level features. 
7. **Output Layer:**
   - **Softmax Activation:** Applied to produce class probabilities for multi-class object recognition. 
   - **Sigmoid Activation:** Used for binary classification tasks. 
**Additional Considerations:** 
- **Transfer Learning:** 
   - **Pre-training:** on large datasets like ImageNet and fine-tuning for the target task to leverage learned features.
   - **Potential use of architecture variants** inspired by successful models like **Efficient Net** or **Mobile Net** for efficiency. 
- **Data Augmentation:** 
   - Implementation of data augmentation techniques (rotation, scaling, flipping) to increase model robustness and handle diverse scenarios. 
- **Optimization and Regularization:**
   - **Adaptive learning rate** strategies (e.g., Adam optimizer) for efficient convergence.
   - **L2 regularization** for weight decay. 
**Evaluation Metrics:**
- **Performance Metrics:** Precision, recall, F1-score, and accuracy for comprehensive evaluation.
- **Latency:** Assessment of inference time for real-time deployment on autonomous vehicles. 

8. **Apply the principles of transfer learning to explain how pre-trained CNN models, such as ResNet, can be fine-tuned for a new image classification task. Provide specific steps in the process.**

**Steps for Fine-Tuning a Pre-trained ResNet Model:**
1. **Select a Pre-trained ResNet Model:**
   - Choose a ResNet model that was pre-trained on a large and diverse dataset, such as ImageNet. The pre-trained model will have learned generic features that can be beneficial for a variety of tasks.
2. **Remove the Fully Connected Layers:**
  - The last few layers of a pre-trained ResNet model typically include fully connected layers for the specific task it was trained on (e.g., ImageNet classification). Remove these layers to retain the convolutional base.
3. **Add New Fully Connected Layers:**
  - Append new fully connected layers at the end of the pre-trained ResNet model. These layers will be specific to the new image classification task. Adjust the number of neurons in the output layer based on the number of classes in the new task. 
4. **Freeze Convolutional Layers:**

## Image Description:

The image provided is a text document, potentially an educational material or notes on Convolutional Neural Networks (CNNs). It lacks any graphical elements or diagrams. The document is structured with bullet points and numbered lists to present information systematically. 
## Text Extraction:

**Fine-Tuning a Pre-trained ResNet Model**

1. **Load Pre-trained Model:**
   * Load a pre-trained ResNet model (e.g., ResNet50) trained on a large dataset like ImageNet.
2. **Remove Original Classifier:**
   * Discard the fully connected layers at the top of the pre-trained model, which are specific to the original task.
3. **Add New Classifier:**
   * Add new fully connected layers on top of the convolutional base to match the number of classes in the new dataset.
4. **Freeze Convolutional Base:**
   * Freeze the weights of the convolutional layers in the pre-trained ResNet model. This prevents these layers from being updated during the initial training on the new task.
5. **Compile the Model:**
   * Compile the fine-tuned model using an appropriate loss function (e.g., categorical crossentropy for multi-class classification) and an optimizer (e.g., Adam).
6. **Data Preprocessing:**
   * Preprocess the new dataset using the same preprocessing steps applied to the original dataset used to train the pre-trained ResNet model. This may include normalization, resizing, and data augmentation.
7. **Fine-Tuning:**
   * Train the model on the new dataset using the compiled model and frozen convolutional layers. This step allows the new fully connected layers to learn task-specific features while preserving the knowledge embedded in the pre-trained convolutional base. 
8. **Unfreeze Convolutional Layers (Optional):**
   * Optionally, unfreeze some of the top layers of the convolutional base and continue training. This allows the model to adapt to more task-specific features present in the new dataset.
9. **Hyperparameter Tuning:**
   * Fine-tune hyperparameters such as learning rate, batch size, and regularization strength to optimize model performance on the new task.
10. **Evaluate and Validate:**
   * Evaluate the fine-tuned model on a validation set to ensure it generalizes well to unseen data. Adjust the model architecture or hyperparameters if necessary.
11. **Inference on Test Set:**
   * Once satisfied with the model's performance, use it for inference on the test set to assess its effectiveness in real-world scenarios.

**Benefits of Fine-Tuning with Transfer Learning:**

* **Faster Convergence:** Utilizes pre-learned features, accelerating convergence on the new task.
* **Data Efficiency:** Effective even with limited labeled data for the new task.
* **Generalization:** Transfers knowledge from one domain to another, improving model generalization.

**ResNet50 Model Architecture**

[Image of ResNet50 Model Architecture]

**COs Addressed and Cognitive Level**

| CO No. | Course Outcomes | RBT Level |
|---|---|---|
| 21CO53.1 | Understand and Analyse the fundamentals that drive deep learning networks | L2 |
| 21CO53.2 | Build, train and apply fully connected neural networks | L3 |
| 21CO53.3 | Analyze convolutional networks and their role in image processing. | L3 |
| 21CO53.4 | Implementation of deep learning techniques to solve real-world problems. | L3 |

**Signature of Course Coordinator**

**Signature of Module Coordinator**

**Signature of HOD**

## Image Description:

The image depicts a simplified architecture of the ResNet50 model, a popular Convolutional Neural Network (CNN) used in image processing tasks. The architecture is laid out horizontally, illustrating the flow of data through various layers. 

Key elements of the architecture include:

* **Input:** The image data to be processed.
* **Zero Padding:** Adds padding around the input image.
* **CONV:** Convolutional layers, the core building blocks of the network, extracting features from the input.
* **Batch Norm:** Batch normalization layers, improving training stability and speed.
* **ReLU:** Rectified Linear Unit activation functions, introducing non-linearity.
* **Max Pool:** Max pooling layers, downsampling feature maps to reduce computational complexity.
* **ID Block:** Identity blocks, repeating convolutional and normalization layers.
* **Conv Block:** Convolutional blocks, similar to identity blocks but with additional convolutional operations.
* **Avg Pool:** Average pooling layer, performing final downsampling.
* **Flattening:** Flattens the multi-dimensional output into a single vector.
* **FC:** Fully connected layer, classifying the extracted features.
* **Output:** The final prediction or classification result.

The ResNet50 architecture is divided into five stages, each marked below the corresponding layers. This staged structure highlights the hierarchical feature extraction process of the network. 
