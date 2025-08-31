# EMOTION-DETECTION-FROM-TEXT
🎯 Objective
The goal is to classify emotions (e.g., happy, sad, angry, fear, love, surprise) from text messages using a Convolutional Neural Network (CNN).
⚙️ Steps to Build the Model
🔹 Step 1 — Import Necessary Libraries
📌 Why?
We need Python libraries for data handling, text preprocessing, model building, and visualization.
Pandas, NumPy → Handle dataset and numerical operations.
Matplotlib, Seaborn → Plot graphs and visualize accuracy.
Scikit-learn → Split data, encode labels, and evaluate metrics.
TensorFlow / Keras → Build the CNN model.
🔹 Step 2 — Load the Dataset
📌 What happens here?
We load a dataset that contains text messages and their corresponding emotion labels.
For example, your dataset (tweet_emotions.csv) contains:
Text	Emotion
I love this movie!	happiness
I feel so lonely	sadness
Why did you do this?	anger
✅ Goal: Make the data available for further processing.
🔹 Step 3 — Data Preprocessing
📌 Why do we need preprocessing?
Real-world text data is noisy and needs cleaning.
a) Handle Missing Values
If there are any empty rows, remove them.
b) Text Cleaning
Convert everything to lowercase.
Remove special characters, numbers, and punctuations.
Example:
Original: "I'm soooo HAPPY!!! #blessed 😍"
Cleaned: "im soooo happy blessed"
c) Label Encoding
Since CNN works with numbers, we convert emotion labels into integers.
Example:
happiness → 0  
sadness → 1  
anger → 2  
🔹 Step 4 — Split Data into Training & Testing Sets
📌 Why?
Training set → Used to train the model.
Testing set → Used to evaluate the model.
🔹 Example Split:
80% → Training data
20% → Testing data
This helps prevent overfitting and ensures the model performs well on unseen data.
🔹 Step 5 — Tokenization & Padding
📌 Why needed?
CNNs work with numbers, not raw text.
We convert each word into an integer using Tokenization.
a) Tokenization
Example:
Sentence: "I am happy"
Tokens: {"i": 1, "am": 2, "happy": 3}
b) Padding
CNN models expect fixed-length input sequences.
If a sentence is short, we pad it with zeros.
Example:
Max length = 5  
"I am happy" → [1, 2, 3, 0, 0]
🔹 Step 6 — Build the CNN Model
📌 Why CNN?
CNNs are excellent for pattern recognition and work well for NLP tasks.
CNN Architecture:
Embedding Layer → Converts words into dense vectors of fixed size.
Conv1D Layer → Detects important word patterns related to emotions.
Global Max Pooling → Extracts the most important features from convolution results.
Dense Layer + Dropout → Fully connected layers for classification, with dropout to avoid overfitting.
Output Layer → Uses softmax activation to predict the emotion class.
🔹 Step 7 — Train the Model
📌 What happens here?
The model learns from the training data.
We specify:
Epochs → Number of times the entire dataset is trained.
Batch Size → Number of samples processed at a time.
Validation Split → Reserve some training data to validate performance.
🔹 Step 8 — Test & Evaluate the Model
📌 Metrics Used:
Accuracy → Overall correctness.
Precision → How many predicted emotions are correct.
Recall → How many actual emotions are detected
F1-Score → Balance between precision and recall.
📊 Confusion Matrix → Visualizes where the model is predicting wrong emotions.
🔹 Step 9 — Predict Emotion for New Messages
📌 Goal:
Allow users to input text and see the predicted emotion.
Example:
Input: "I am so happy today!"
Output: happiness 🎉
🔹 Step 10 — Dataset Links
Emotions Dataset for NLP (Kaggle)
 ✅ (Recommended)
Twitter Emotion Dataset
DailyDialog Emotion Dataset
📊 Expected Accuracy
With the current CNN model:
Accuracy: ~86% to 89% ✅
For even better accuracy (>90%), you can:
Use pre-trained embeddings like GloVe.
Try LSTM or BERT models.
🎯 Expected Outcome
A working Emotion Detection Model ✅
Useful for:
Student feedback analysis 📘
Mental health monitoring 🧠
