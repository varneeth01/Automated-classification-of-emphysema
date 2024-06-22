import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from google.colab import drive
from skimage.transform import resize
from tensorflow.keras.applications.vgg16 import preprocess_input

drive.mount('/content/drive')

data_dir = "/content/drive/My Drive/disease/slicesjpeg"
csv_file_path = '/content/drive/My Drive/disease/image_cluster_labels.csv'
labels_df = pd.read_csv(csv_file_path)

image_paths = []
labels = []
for image_name in os.listdir(data_dir):
    if image_name.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(data_dir, image_name)
        label = labels_df.loc[labels_df['Image Path'].str.contains(image_name), 'Category'].values
        if len(label) > 0:
            image_paths.append(image_path)
            labels.append(label[0])
        else:
            print(f"Warning: No label found for image {image_name}")

df = pd.DataFrame({"Image Path": image_paths, "Label": labels})

train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["Label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["Label"])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col="Image Path", y_col="Label", target_size=(224, 224), batch_size=32, class_mode="categorical"
)
val_generator = val_datagen.flow_from_dataframe(
    val_df, x_col="Image Path", y_col="Label", target_size=(224, 224), batch_size=32, class_mode="categorical"
)

def build_vdsnet():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

vdsnet_model = build_vdsnet()
vdsnet_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


vdsnet_history = vdsnet_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    epochs=10,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

def build_object_detection_model():
    pass

object_detection_model = build_object_detection_model()

def detect_empyema(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    features = vdsnet_model.predict(image)

    empyema_coordinates = object_detection_model.detect(image)

    return empyema_coordinates

while True:
    user_input = input("Enter the path to an image (or 'q' to quit): ")
    if user_input.lower() == 'q':
        break

    if os.path.exists(user_input):
        try:
            empyema_coordinates = detect_empyema(user_input)
            display_image_with_dots(user_input, empyema_coordinates)
        except Exception as e:
            print(f"An error occurred while processing the image: {e}")
    else:
        print("Invalid image path. Please try again.")
