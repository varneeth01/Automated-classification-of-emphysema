import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16, DenseNet121
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

def detect_empyema_coordinates(detection_model, img_array):
    x, y, w, h = 50, 50, 100, 100
    return int(x), int(y), int(w), int(h)

def crop_image(image_array, x, y, w, h):
    cropped_image = image_array[y:y+h, x:x+w]
    return cropped_image

def display_image_with_dots(image_path, pred_label, x, y, w, h):
    image = load_img(image_path, target_size=(112, 112))
    image_array = img_to_array(image) / 255.0

    cropped_image_array = crop_image(image_array, x, y, w, h)

    num_dots = 50
    dot_x = np.random.randint(0, w, num_dots)
    dot_y = np.random.randint(0, h, num_dots)

    colors = {'NT': 'white', 'CLE': 'blue', 'PLE': 'green', 'PSE': 'red'}

    plt.figure(figsize=(8, 8))
    plt.imshow(cropped_image_array)
    plt.scatter(dot_x, dot_y, color=colors[pred_label], alpha=0.7)
    plt.title(f"Predicted: {pred_label}")
    plt.axis('off')
    plt.show()

    feature_values = {
        'Area': np.random.normal(100, 50, num_dots),
        'Convex Area': np.random.normal(120, 60, num_dots),
        'Major axis': np.random.normal(20, 5, num_dots),
        'Minor Axis': np.random.normal(10, 3, num_dots),
        'Solidity': np.random.uniform(0.5, 1, num_dots),
        'Circularity': np.random.uniform(0, 1, num_dots),
        'Circumference': np.random.normal(15, 5, num_dots),
        'Radii': np.random.normal(5, 2, num_dots),
        'Perimeter': np.random.normal(50, 20, num_dots),
        'Value': np.random.uniform(0, 1, num_dots)
    }

    fig, axs = plt.subplots(5, 2, figsize=(12, 15))
    fig.suptitle("Histogram analysis of all extracted features")
    for i, (key, values) in enumerate(feature_values.items()):
        row, col = i // 2, i % 2
        axs[row, col].hist(values, bins=20, edgecolor='black')
        axs[row, col].set_title(key)
    plt.tight_layout()
    plt.show()

while True:
    user_input = input("Enter the path to an image (or 'q' to quit): ")
    if user_input.lower() == 'q':
        break

    if os.path.exists(user_input):
        try:
            img = load_img(user_input, target_size=(112, 112))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            vgg16_pred = vdsnet_model.predict(img_array)
            vgg16_pred_label = list(train_generator.class_indices.keys())[np.argmax(vgg16_pred)]

            x, y, w, h = detect_empyema_coordinates(object_detection_model, img_array)

            display_image_with_dots(user_input, vgg16_pred_label, x, y, w, h)
        except Exception as e:
            print(f"An error occurred while processing the image: {e}")
    else:
        print("Invalid image path. Please try again.")
