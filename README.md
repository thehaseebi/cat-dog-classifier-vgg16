# Cat vs Dog Image Classifier (Transfer Learning with VGG16)

A binary image classification project using **Transfer Learning** with the **VGG16** convolutional neural network pre-trained on **ImageNet**. The model leverages VGG16 as a fixed feature extractor and trains custom dense layers to classify images as **cat** or **dog** with high accuracy.

---

## Project Overview

This project demonstrates how to apply **transfer learning** using the VGG16 architecture for a custom image classification task.  
The workflow includes dataset loading, preprocessing, model construction, training, evaluation, and model saving.

**Key Objectives:**
- Utilize **VGG16** as a feature extractor  
- Train a binary classifier (cats vs dogs)  
- Apply Keras data generators for image loading and preprocessing  
- Evaluate model performance on a separate validation/test set  

---

## Model Architecture

**Base Model:**  
- `VGG16(include_top=False, weights='imagenet')`  
- All convolutional layers **frozen**

**Added Classification Layers:**  
- `Flatten()`  
- `Dense(256, activation='relu')`  
- `Dense(1, activation='sigmoid')`

**Training Strategy:**  
- Only custom dense layers are trainable  
- VGG16 base remains frozen to retain learned ImageNet features  

---

## Dataset Structure

The dataset should follow this directory format:
dataset/
│
├── training_set/
│ ├── cats/
│ └── dogs/
│
└── test_set/
├── cats/
└── dogs/

Each subfolder contains images corresponding to each class.

---

## Data Preprocessing

- Loaded using `keras.utils.image_dataset_from_directory()`  
- Resized to **150×150**  
- Normalized to **[0, 1]**  
- Batched and shuffled automatically  

---

## Training

python
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

## Training uses:
- Adam optimizer

- Binary cross-entropy loss

- Accuracy as the primary metric

## Evaluate model performance:
- model.evaluate(validation_ds)

## Generate predictions:
- model.predict(image_batch)

## Save the trained model:
- model.save('vgg16_cat_dog_classifier.h5')


📎 Summary

This project highlights how transfer learning with VGG16 can be used to efficiently train a high-performing binary image classifier with minimal data and compute resources.

