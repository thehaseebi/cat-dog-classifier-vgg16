🐱🐶 Cat vs Dog Image Classifier using Transfer Learning (VGG16)

This project implements a Cat vs Dog Image Classifier using Transfer Learning with the VGG16 convolutional neural network pre-trained on ImageNet.
By leveraging VGG16 as a feature extractor and fine-tuning custom layers, the model achieves efficient and accurate binary image classification.

📘 Project Overview

This notebook demonstrates how to use VGG16 for transfer learning on a custom dataset of cat and dog images.
The pre-trained VGG16 convolutional base is used to extract meaningful features from images, while new dense layers are trained for classification.

Key Objectives:

Apply Transfer Learning with VGG16.

Train a classifier to distinguish between cats and dogs.

Utilize Keras image generators for dataset loading and preprocessing.

Evaluate model performance on a separate test set.

🧠 Model Architecture

Base Model: VGG16 (pre-trained on ImageNet, include_top=False)

Added Layers:

Flatten()

Dense(256, activation='relu')

Dense(1, activation='sigmoid')

Training Strategy:

The VGG16 base is frozen (not trainable).

Only the new dense layers are trained.

🗂️ Dataset

The dataset is expected to be organized in a directory structure like:

dataset/
│
├── training_set/
│   ├── cats/
│   └── dogs/
│
└── test_set/
    ├── cats/
    └── dogs/


Each subfolder should contain the respective class images.

⚙️ Data Preprocessing

Images are:

Loaded using keras.utils.image_dataset_from_directory()

Resized to 150×150 pixels

Normalized to a range of [0, 1]

🚀 Training
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=10
)


Uses binary cross-entropy loss.

Tracks accuracy on training and validation sets.

📊 Evaluation

After training, model performance can be evaluated using:

model.evaluate(validation_ds)


and predictions can be made on unseen data with:

model.predict(image_batch)

💾 Saving the Model

You can save the trained model using:

model.save('vgg16_cat_dog_classifier.h5')