## Internal Assessment Answer Booklet Text Transcription and Image Description

**Image Text Transcription:**

**Header:**

* INTERNAL ASSESSMENT ANSWER BOOKLET UG/PG
*  ಶ್ರೀಲೀಲಾ ಆರ್ಟ್ಸ್ ಸನ್ಸ್ ®  ಟೆಕ್ನಾಲಜಿ, ಬೆಂಗಳೂರು
* Global Academy of Technology, Bengaluru
* (NAAC logo with 'A' grade)

**Student Information:**

* TEST - I  **(O marked)** 
* TEST - II  **(filled circle marked)**
* TEST - III  **(O marked)**
* Script No. **310224**
* Date **21 02 24** 
* Semester **5th** 
* USN **1GA21AI006**
* Subject Name **Deep learning principles** 
* Subject Code **21AML53** 

**Candidate Statement:**

*  I here with abide by the rules and regulations of the institute. 
* **AISHWARYA. H.M** 
* **Aishwarya.H.M** (signature)
* Name and Signature of the Candidate

**Invigilator Statement:**

* *practices* (handwritten)
* I have verified the data filled by the candidate
* **_pp_** (signature) 
* Room Superintendent's Signature
* # of Graph Sheets / Drawing Sheets Attached  (blank space below) 

**Internal Assessment Marks Table:**

|        |  a  |  b  |  c  | Total |  a  |  b  |  c  | Total |
|--------|-----|-----|-----|-------|-----|-----|-----|-------|
|    1   | 10  |     |     |  10   |  5  |  5  |     |  10   |
|    2   |     |     |     |       | 10  |     |     |  10   |
|    3   |  5  |  5  |     |  10   |     |     |     |       |
|    4   |     |     |     |       |     |     |     |       |
|    5   |     |     |     |       |     |     |     |       |
|--------|-----|-----|-----|-------|-----|-----|-----|-------|
| Total Marks obtained |         | Out of 40 Marks | *40 Very good* (handwritten) |

**Faculty and Student Signatures:**

* *Signature of Faculty* 
* 21/2/24
* Name and Signature of Faculty with Date
* **AISHWARYA.H.M** (signature) 
* Name and Signature of the Student with Date 
* *Aishwarya.H.M* (signature)
* 23/2/24.. 

**Image Description:**

The image depicts an internal assessment answer booklet from the Global Academy of Technology, Bengaluru. It's a standard form used for evaluating student performance. The document is mostly filled in with black ink, except for the marks, which are written in red ink.  Handwritten signatures in blue ink confirm the authenticity of student and faculty submissions. The document has a clean and organized layout, designed for easy readability and processing. The presence of the NAAC logo with an 'A' grade signifies the institution's accreditation and academic excellence. Overall, the image provides a snapshot of the academic assessment process within the institution. 
## Extracted Text:

Q. Nos.

1. The weight initializations can be made through by two techniques using random values:

a) Normal Distribution
b) Uniform Distribution 

=> The weights when large values are initialized, the 'z' value increases as it is large. 
=> The learning rate also takes huge/more amount of time.
=> We can observe these changes on the Sigmoid/tanh graphs, where the slope of the graph gradient descent decreases and learning rate is high. 

[Image Description: Two graphs are drawn on the same axes. 

* **Left Graph:** Represents a sigmoid function. It starts with a small value of 'z' close to 0 on the x-axis and gradually increases as 'z' increases. The slope is steeper in the middle and flattens out at higher values of 'z'. It's labeled "sigmoid" at the bottom.  The y-axis is marked at 0 and 0.5. An arrow indicates a "small value of 'z'" near the origin of the curve. An arrow at the top right points to a "large value of 'z'."

* **Right Graph:** Represents a tanh (hyperbolic tangent) function. It also starts with a small 'z' value but has a steeper initial slope compared to the sigmoid. It then flattens out similarly to the sigmoid at higher 'z' values. An arrow on the right indicates a "large value of 'z'." An arrow below the x-axis points to a "Small value of 'z'." The x-axis is simply labeled "x". An arrow next to the curve indicates "tanh 'z'."]

=> The above diagram depicts the sigmoid/ tanh graphs.
=> As in the same case if the value of weights initialized as small, in this case also we get 'z' value as small 
## Extracted Text: 

Q. Nos. 

and learning rate also takes large amount of time.

→ As we propagate back from output layer, the weights are multiplied across each neuron in the hidden layers.
→ Let us Assume 15 Hidden layers, the weight matrix is multiplied across all layers. And we get very large number in the early layers. This creates a **exploding gradient descent.**

→ If the small values of 'z' are feed the slope of the graph decreases progressively and weight matrix is updated slowly. This leads to the **vanishing gradient descent** 

→ We came across these challenges and got two methods of initializing namely:

1) He 
2) Xavier Glorot.

→ He's Initialization methods takes random initialization methods using normal distribution.

## Text Transcription:

> We have the formula: √2/(Size[l-1])

> To the layers of Network, He can be written as: 
W[l] = random_matrix * √2/(Size[l-1])

> Xavier initialization takes the weights for each neurons in the layers same as he, but slightly differs as:
Formula: √1/(Size[l-1]), ?t is bit change in initialization as: √2/(Size[l-1]+Size[l])
> The Xavier initialization for layers of Network can be written as:
W[l] = random_matrix * √2/(Size[l-1]+Size[l])

> The potential challenges were overcome due to some of the techniques.

> The initialization of cost function also played crucial role in weight initialization where if cost value of z is 0, the cost function would not 

## Image Description:

The image displays handwritten notes on lined paper. The notes appear to discuss mathematical formulas related to neural networks, specifically focusing on weight initialization techniques. 

Here's a breakdown:

- **Formulas:** The notes prominently feature mathematical formulas involving square roots and fractions. These formulas include terms like "Size[l-1]" and "Size[l]," likely referring to the size of different layers in a neural network. 
- **Terminology:**  Terms like "Xavier initialization," "random_matrix," "layers of Network," and "cost function" suggest a context of neural network design and training.
- **Explanations:**  The handwritten text provides explanations and observations related to the formulas, mentioning concepts like slight differences in initialization, potential challenges, and the role of the cost function.
- **Numbering:**  There's a circled "18" at the left margin, possibly indicating a page or section number.
- **Handwriting:**  The handwriting is generally legible but shows some inconsistencies in letter formation and spacing.

Overall, the image conveys a snippet of someone's notes on neural network weight initialization, likely part of a larger study or learning material. 
**Text:**

Q. Nos. 

converge.
-> If 2 was very small and the activation converged in beginning itself, it would converge and diverge.
-> If 2 was very large the cost function would converge and diverge. 
-> If 2 was probable estimated the learning rate was at pace and cost function converges!
-> Also due to multiple neurons redundancy values, it cannot initialize weight to '0', but bias can be '0'. 
-> So the neurons could be symmetric. 

**Description:**

The image showcases a handwritten note on a lined notebook page.  The note discusses the impact of a variable, represented as "2," on the convergence of a system, likely within the context of machine learning, based on terms like "neurons," "activation," "cost function," and "learning rate." It explores scenarios with different values of "2,"  and their effects on the system's behavior. 

The page number "6" is circled at the bottom, suggesting it's part of a larger set of notes or a workbook. 
## Traditional ANN to deal with Image Dataset

3) a) Traditional ANN to deal with image Dataset.

* The very popular MNIST dataset was used in Deep Neural Network. 
* The traditional ANN could classify the MNIST dataset pretty good.
* The MNIST dataset is very specific where images are uniformly spaced and aligned properly.
* In real-world we do not get image dataset as those specific.
* The traditional ANN could not highlight the most trainable features.
* The ANN took very long time and memory space to calculate these features.
* It did not consider the spatial features. 
* ANN could not deeply identify the image location features dimensions.
* CNN architecture has the core components to address the spatial features.
* The above challenges were faced with also respect to
    * Overfitting
    * Spatial features 
    * Dimensionality reduction.

---

## Description

The image presents handwritten notes on lined paper. The notes discuss the limitations of using traditional Artificial Neural Networks (ANN) for image datasets, specifically referencing the MNIST dataset.  Key points highlighted include:

* **MNIST limitations:** While traditional ANNs perform well on the MNIST dataset, its uniform and simplistic nature doesn't reflect real-world image data.
* **Feature extraction:** Traditional ANNs struggle to efficiently extract meaningful features from images, particularly spatial relationships between elements.
* **Computational cost:** Processing images with traditional ANNs requires significant time and memory resources.
* **CNN advantages:** The notes introduce Convolutional Neural Networks (CNN) as a solution. CNNs are designed to address the highlighted limitations by incorporating mechanisms to handle spatial features and improve efficiency.
* **Additional challenges:** The notes briefly mention challenges related to "overfitting," "spatial features," and "dimensionality reduction," hinting at broader issues in image processing. 

The handwriting is legible, and the content is structured as bullet points for clarity. Overall, the image effectively communicates the drawbacks of traditional ANNs for image data and introduces CNNs as a more suitable alternative. 
**Text:**

Q. No.

The Building Blocks of CNN are: 
1) kernel/filter
2) convolution layer
3) Pooling layer.

*) Kernel : It is a matrix of special type applied to image to extract features and underlying patterns of object in images.

-> Kernel/filter can be of many varieties so we can identify different types of patterns. 
Eg:* Sobel filter which identifies vertical and horizontal edges in images..

* Outline filter which identifies or outlined the prominent features.

*) convolution layer: 
-> The convolution layer is identified as the important layer to classify the different challenges.

-> CNN is a special type of Neural Networks which uses convolution instead of 

**Description:**

The image presents handwritten notes on a lined paper, detailing the fundamental building blocks of Convolutional Neural Networks (CNNs). 

The notes are structured using arrows and numbering to highlight key concepts and their definitions. The handwriting is legible and employs common abbreviations for technical terms.
## Simple matrix multiplication. 
  [Image] <-(arg 1)
  
## Two arguments in CNN  
                     \(arg_2\)
                                     [kernel/ filter]
  
Ex: | 1 | 1 | 1 | 1 | 0 |        Filter              | 2 | 3 | 2 |
     |---|---|---|---|---|     | 1 | 0 | 0 |  *  -->     |---|---|---| 
     | 0 | 1 | 1 | 0 | 0 |     | 0 | 1 | 0 |           | 2 | 3 | 2 |
     |---|---|---|---|---|     |---|---|---|           |---|---|---|
     | 0 | 1 | 1 | 1 | 0 |     | 0 | 0 | 1 |           | 1 | 2 | 2 |
     |---|---|---|---|---|                               |---|---|---|
     | 0 | 0 | 1 | 0 | 1 |                              3x3        Activation 3x3
                                                                       Map
 Input image 5x5

Formula:  ((n - f + 1) x (n - f + 1)) 

=> In convolution we also have two important factors stride and padding. 

=> Stride is the pixels moved by kernel around image subset. 
=> If it is moved by 1 pixel then stride is = 1. 

=> Padding is used to reduce the input image dimensions when it is processed to output.

=> Padding is nothing adding dummy pixels around the image subset of value 0. 

=>  ((n - f + 2p + 1) x (n - f + 2p + 1))
                 s                      s

## Description:
The image showcases a handwritten explanation of a fundamental concept in Convolutional Neural Networks (CNNs) - **matrix multiplication using a kernel (or filter)**.

**1. Title and Context:**
- The title "Simple matrix multiplication" sets the stage, highlighting the core operation.
- The note "Two arguments in CNN" emphasizes the specific application within CNNs.

**2. Arguments Visualized:**
- Two boxes labeled "[Image]" and "[kernel/filter]" represent the input image and the kernel, respectively.
- Arrows labeled "arg 1" and "arg 2" point towards the respective boxes, signifying their roles as inputs.

**3. Numerical Example:**
- A 5x5 matrix labeled "Input image" represents the input data.
- A 3x3 matrix labeled "Filter" illustrates the kernel used for convolution.
- An asterisk (*) indicates the multiplication operation between the input and kernel.
- A 3x3 matrix labeled "Activation Map" depicts the output after convolution.
- The numbers within the matrices exemplify the result of the element-wise multiplication and summation process involved in convolution.

**4. Formula and Key Factors:**
- The handwritten formula "((n - f + 1) x (n - f + 1))" calculates the output dimensions (activation map size), where:
    - 'n' represents the input dimension (assuming a square matrix).
    - 'f' represents the kernel dimension.
- The text further explains two crucial factors:
    - **Stride:**  Described as the movement of the kernel across the image, affecting the overlap and output size. 
    - **Padding:** Defined as adding dummy pixels (value 0) around the input image, helping to control the output dimensions and retain edge information.

**5.  Padding Formula:**
- The final formula "((n - f + 2p + 1) x (n - f + 2p + 1)) / s" incorporates padding ('p') and stride ('s') for a more comprehensive output dimension calculation.

**Overall, the image effectively uses a combination of text, arrows, boxes, and a numerical example to deliver a clear and concise explanation of matrix multiplication with a kernel in the context of CNNs.** 
## Extracted Text:

Q. No. 

→ Padding is of two types → Valid
→ Valid padding is no padding is done. 
→ Same padding is the zero padding where input image data dimension and output image data are same.
→ If filter is odd, padding is asymmetric.
→ If filter is even, padding should be asymmetric.

3) Pooling Layer:
→ Pooling layer is mainly used to deal with trainable parameters. 
→ This layer also downsamples the input image dimensions.
→ The pooling layer also has the important towards spatial feature representation.
→ There are three types of padding.
* Max pooling. 

10 

## Image Description:

The image showcases a lined sheet of paper with handwritten notes about padding and pooling layers, likely in the context of convolutional neural networks. The writing is in blue ink, with certain words underlined and arrows used to connect related concepts. 

Key elements of the notes:

* **Top Section:** Explains padding with "Valid" (no padding) and "Same" (zero padding) types. It further details how padding works with odd and even filter sizes.
* **Middle Section:** Introduces the "Pooling Layer" and lists its key functions like handling trainable parameters, downsampling image dimensions, and spatial feature representation. 
* **Bottom Section:**  Mentions three types of padding, starting with "Max pooling" as the first.

The notes appear to be part of a larger set, indicated by the "Q. No." at the top and the page number "10" at the bottom. The use of underlines and arrows suggests an attempt to organize and highlight key information for study purposes. 
## Text transcription

* Average pooling.
* sum Uppooling.

-> All the above specified building blocks overall form the CNN networks.
-> where at last @at we have fully connected layer that provides the output to output layer.

b) Image of size 50X50.
Filter size 3X3
Assumed same padding,

P = f-1 / 2

<span style="text-decoration: line-through;">P = 3-1 / 2 = 2 / 2</span> = 1
with padding 
(n + 2P+1) x (n + 2P+1) <span style="text-decoration: line-through;">/s</span>
-> (50 - 3 + 2 + 1) x (50 - 3 + 2 + 1) / 1
-> (50 X 50)
-> As we assumed same pad, where padding = 1 -> (51X51) 

## Description of non-textual elements

The image shows a page from a notebook with handwritten text and mathematical equations in blue ink. 

- There are two bullet points at the top with the terms "Average pooling" and "sum Uppooling".  
- An arrow follows, pointing to a paragraph discussing building blocks of CNN networks.
- Another arrow leads to the next paragraph about a fully connected layer providing output.
- Section 'b' presents a problem about calculating output size after applying a filter with "same padding".
- Calculation for padding ('P') is shown. 
- An equation with padding applied is written, with some parts struck through, likely indicating corrections.
- The final calculation results in an output size of 50x50.
- A concluding statement reaffirms the "same padding" assumption with padding = 1, resulting in a 51x51 output.


The writing is somewhat messy and includes several strikethroughs, suggesting an evolving thought process as the writer works through the problem. 
## Extracted Text:

6) a) VGG 16:
-> The VGG 16 is the one of the most CNN model built across various types of layers.
-> VGG 16 has 3x3 convolutional filters, where the model was used to train number of pixel images.
-> VGG16 has the image with max layers and the 'soft max' activation function.
-> VGG16 was very accurate and converges to the input computations, features and resolutions.
-> This Model was Applied by the small trainable tasks, for computations.
-> The Model was so accurate with accuracy and characteristics local features.
-> This model was used due its simplicity and its higher memory computations.

## Image Description:

The image shows a page from a notebook with handwritten text in blue ink. The text is neatly written and discusses the VGG16 model, a Convolutional Neural Network (CNN) architecture. 

Key visual elements:

* **Handwritten Text:** The primary content is handwritten, indicating personal notes or study material.
* **Numbered List:** The information is structured as a numbered list (starting with "6)"), suggesting a series of topics or questions.
* **Sub-points:** Each numbered point has several sub-points marked with "->", indicating a breakdown of information or key characteristics.
* **Red Circle with "S":** A red circle with an "S" inside is present on the left margin near the fifth sub-point, possibly indicating an important point or a section marked for review. 


The image suggests a learning environment where someone is summarizing or studying the characteristics and applications of the VGG16 model in the field of deep learning. 
## Extracted Text:

**Left Side:**
Flow Diagram

*Image*
Conv 64
Conv 64
Maxpool
 
Conv 128
Conv 128
Maxpool

Conv 256
Conv 256
Conv 256
Maxpool

Conv 512
Conv 512
Conv 512
Maxpool

Conv 512
Conv 512
Conv 512
Maxpool

FC 4096
FC 4096
FC 1000
Softmax

VGG 16. Architectural Diagram 

**Right Side:**
Applications:
-> Healthcare
-> Manufacturing units
-> Traffic signs 


## Image Description:

The image depicts a hand-drawn architectural diagram of the VGG16 convolutional neural network. 

* **Structure:** The diagram is structured vertically, showing the flow of data through the network from the input image at the top to the output at the bottom. Each layer of the network is represented by a box.
* **Layers:** The diagram shows the different types of layers in VGG16:
    * **Convolutional Layers (Conv):**  These layers are represented by boxes labeled "Conv" followed by a number (e.g., "Conv 64"). The number indicates the number of filters in the convolutional layer.
    * **Max-Pooling Layers (Maxpool):** Represented by boxes labeled "Maxpool."
    * **Fully Connected Layers (FC):** Represented by boxes labeled "FC" followed by a number (e.g., "FC 4096"). The number indicates the number of neurons in the fully connected layer.
    * **Softmax Layer:** The final layer is labeled "Softmax."
* **Flow:** Lines connecting the boxes indicate the flow of data through the network.  The data flows from the input image through a series of convolutional and max-pooling layers, then through fully connected layers, and finally to the Softmax layer for classification. 

**Additional Notes:**

* A line drawn with red ink partially covers the last Maxpool layer and the first FC layer. This might indicate a connection or relationship between these layers, possibly flattening the output of the convolutional layers before feeding them into the fully connected layers.
* A list of potential applications for VGG16 is written to the right of the diagram. 

The diagram effectively visualizes the structure and flow of information within the VGG16 network. 
**Text:**

Q. Nos.

(b) GoogleNet:
-> The GoogleNet has advanced fully connected layers to deeply generalize the model of CNNs.
-> It was introduced in 2014 and developed the Google company to enhance the enhancement of image classifications.
-> The images daily 3 billion are out there through various media.
-> As this model has Input layers, fully many hidden layers to embark the features. 
-> The DeepNN model is very prominent in GoogleNet.
-> The dimensionality Reduction and Spatial features are mainly focused here. 
-> The computations of the computer vision on images are more accurate compared to earlier LeNet, ResNet. 
-> The GoogleNet Architectural

14

**Description:**

The image showcases a page from a notebook or document, featuring handwritten text in blue ink. The content revolves around the topic of "GoogleNet," a specific type of convolutional neural network (CNN) architecture.  The text is structured as a series of bullet points, each marked with a right-pointing arrow.  

There are no other visual elements besides the handwritten text and the printed page numbers "13" and "14" at the bottom of the image. 
**Text:**

pattern use the softmax and Relu activation function. 

-> The googleNet has fundamental deep learning approaches and a high level model to solve real-world problems.

Applications:
-> Traffic Recognition
-> Sentiment Analysis
-> Language translation using deep learning techniques.

GoogleNet <-

Input layer

Maxpool layer

Architectural design

**Description:**

The image presents a hand-drawn diagram outlining a basic architectural design labeled "GoogleNet." It depicts a simple flow:

1. **GoogleNet:**  This box, written in a larger font and partially underlined for emphasis, represents the overall system or model.  An arrow points back to it from the right side of the diagram, indicating a feedback or recurrent process.
2. **Input Layer:**  Positioned below "GoogleNet," this box signifies the initial stage where data is fed into the model.
3. **Maxpool Layer:**  Located below "Input Layer," this box denotes another processing stage.  An arrow points downward from "Input Layer" to this layer, illustrating the flow of data.
4. **Architectural Design:** This handwritten caption at the bottom clarifies that the diagram represents the structural layout of the GoogleNet model.

The diagram is drawn on lined paper, and a red circle with a faint "15" is visible at the bottom, potentially indicating a page number or note unrelated to the diagram itself. 
## Text Transcription:

**Q. No.**

**7. CNN architecture tailored for a specific computer vision applications are:**

[Image of a basic CNN architecture with arrows indicating the flow of data]

* **Input Layer**
* **Convolutional Layer**
* **Pooling Layer**
* **Fully connected layers**

-> The important area of the deep learning networks in the computer vision application. 

-> The deep neural network has tailored feature among the computer vision. 

-> The computer vision is yielding better results than human vision.

-> The alignment of Human phaletopy is analyzed better than Human ophthalmologist.

16


## Image Description:

The image depicts a simple Convolutional Neural Network (CNN) architecture. It consists of four stages represented by rectangular boxes, connected sequentially by arrows to indicate the flow of data:

1. **Input Layer:** The first box on the left, labeled "Input Layer," represents the raw image data fed into the network.
2. **Convolutional Layer:**  The second box, labeled "Convolutional Layer," symbolizes the layer where the network learns to extract features from the input image using convolutional filters.
3. **Pooling Layer:** The third box, labeled "Pooling Layer," represents the stage where the network down-samples the extracted features, reducing their dimensionality while retaining essential information. 
4. **Fully Connected Layers:** The final stage consists of two boxes labeled "Fully connected layers." These layers process the extracted features and perform the final classification or regression tasks depending on the application. 

The entire diagram provides a high-level overview of a basic CNN architecture commonly used in computer vision applications. 
## Extracted Text:

The huge amount of data generated are analyzed and computer vision are leveraged its importance. 

The Computer Vision has the major Application:

* Healthcare:
    * ECG
    * PET scan
    * CT Scan 
    * MRI's

* Manufacturing:
    * Reading barcodes 
    * Manufacturing the items in correct specific range.
    * Placing order of items in correct place with suggest quantity.

* Agriculture:
    * The first computer vision that was applied in Agriculture was Israel that was developed company.
    * It based on the crop growth and diseases.
    * Sorting of the right crop produced 

## Image Description:

The image displays a page from a notebook with handwritten text in blue ink. The handwriting is legible but exhibits some inconsistencies in letter formation and spacing, suggesting it was written by hand. The notes appear to outline key applications and a historical fact related to "Computer Vision."  

The page is lightly lined, providing a structure for the handwritten content. There are three bullet points marked by asterisks (*) indicating different sectors where "Computer Vision" is applied. Each sector has further sub-bullets, marked with right arrows, detailing specific applications. 
## The Design Choices and Layers: 

* **Input layer:** 
    -> Where the image of pixels like 5x5, 7x7 - are given to a CNN deep neural network. 

* **Convolutional Layers:**
    * This layer convolutes the input image by adding kernel or filter.
    -> This layer commonly has a pixels called as window in the smaller numbers. 
    -> So the padding is finished to all the pixels, pixel cannot determine the blurred image matrix. 
    -> The CNN convolutes the image pixels rather simply doing matrix multiplication. 

-> **Pooling layers:** 
    * This layer downsamples the image dimensions also the loss of information is connected.

-> The pooling layers extract the major spatial features of representation. 
## Extracted Text:

> The pooling layer has the input given, 
> nh x nw x nc , with stride 's'.
> The fully connected layer, processing the del input kernel, gives the output matrix of image of pixels of same dimensions by input or downsampling.
> The computer vision has the range of pixels varying 0 to 255. 
> 0 is black, 255 is white.
> We also have RGB pixels which takes change between 0 to 255.
> The computer vision has: 

**(drawing of a rectangular prism)**

where 'n' is width, 'y' is length, '2' a channel of colors. 

## Image Description: 

The image features a hand-drawn rectangular prism. The prism is depicted in a slightly distorted perspective, suggesting depth.  The lines are somewhat shaky, indicating a quick sketch rather than a precise drawing. There are no markings or labels on the prism itself. 

The text beside the drawing  explains that the 'n' represents the width of the prism, 'y' stands for its length, and '2' signifies a channel of colors. This suggests that the drawing likely represents a visual element within the context of computer vision and image processing, where dimensions and color channels are crucial components. 
