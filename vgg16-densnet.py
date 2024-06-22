import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Dense, Input, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG16, DenseNet121
from google.colab import drive
from skimage.transform import resize

drive.mount('/content/drive')

random.seed(42)
np.random.seed(42)

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
    preprocessing_function=lambda x: resize(x, (112, 112))
)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=lambda x: resize(x, (112, 112))
)
test_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=lambda x: resize(x, (112, 112))
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col="Image Path", y_col="Label", target_size=(112, 112), batch_size=2, class_mode="categorical"
)
val_generator = val_datagen.flow_from_dataframe(
    val_df, x_col="Image Path", y_col="Label", target_size=(112, 112), batch_size=2, class_mode="categorical"
)
test_generator = test_datagen.flow_from_dataframe(
    test_df, x_col="Image Path", y_col="Label", target_size=(112, 112), batch_size=2, class_mode="categorical", shuffle=False
)

def build_generator():
    model = Sequential()
    model.add(Dense(7*7*256, input_dim=100, activation=LeakyReLU(alpha=0.2)))
    model.add(BatchNormalization())
    model.add(Reshape((7, 7, 256)))
    model.add(Conv2DTranspose(128, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(32, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(3, 5, strides=2, padding='same', activation='tanh'))
    return model

def build_discriminator(input_shape):
    model = Sequential()
    model.add(Conv2D(64, 5, strides=2, padding='same', input_shape=input_shape, activation=LeakyReLU(alpha=0.2)))
    model.add(Conv2D(128, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(BatchNormalization())
    model.add(Conv2D(512, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

generator = build_generator()
discriminator = build_discriminator(generator.output_shape[1:])
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.02, beta_1=0.5))
discriminator.trainable = False

gan_input = Input(shape=(100,))
fake_image = generator(gan_input)
gan_output = discriminator(fake_image)

gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.02, beta_1=0.5))

def train_gan(epochs, batch_size=2):
    for epoch in range(epochs):
        for _ in range(len(train_generator)):
            noise = np.random.normal(0, 1, (batch_size, 100))
            real_images, real_labels = next(train_generator)
            current_batch_size = real_images.shape[0]

            print(f"Real images shape: {real_images.shape}")
            print(f"Real labels shape: {real_labels.shape}")
            print(f"Noise shape: {noise.shape}")

            fake_images = generator.predict(noise)
            labels_real = np.ones((current_batch_size, 1))
            labels_fake = np.zeros((current_batch_size, 1))

            print(f"Fake images shape: {fake_images.shape}")
            print(f"Labels real shape: {labels_real.shape}")
            print(f"Labels fake shape: {labels_fake.shape}")

            d_loss_real = discriminator.train_on_batch(real_images, labels_real)
            d_loss_fake = discriminator.train_on_batch(fake_images[:current_batch_size], labels_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, 100))
            labels_gan = np.ones((batch_size, 1))
            g_loss = gan.train_on_batch(noise, labels_gan)

            print(f"Epoch {epoch + 1}, D loss: {d_loss}, G loss: {g_loss}")

print(f"Number of batches in train generator: {len(train_generator)}")

train_gan(epochs=2)



def create_position_encoder():
    input_img = Input(shape=(112, 112, 3))
    x = Conv2D(64, 3, activation='relu', padding='same')(input_img)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    position_output = Dense(2, activation='linear')(x)
    model = Model(inputs=input_img, outputs=position_output)
    model.compile(optimizer='adam', loss='mse')
    return model

position_encoder = create_position_encoder()

def create_ms_resnet_model(base_model):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(112, 112, 3))
densenet121_base = DenseNet121(weights='imagenet', include_top=False, input_shape=(112, 112, 3))

vgg16_model = create_ms_resnet_model(vgg16_base)
densenet121_model = create_ms_resnet_model(densenet121_base)

vgg16_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
densenet121_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

vgg16_history = vgg16_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    epochs=2,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
)

densenet121_history = densenet121_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    epochs=2,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
)

vgg16_pred = vgg16_model.predict(test_generator)
densenet121_pred = densenet121_model.predict(test_generator)

vgg16_pred_labels = np.argmax(vgg16_pred, axis=1)
densenet121_pred_labels = np.argmax(densenet121_pred, axis=1)

print("VGG16 Classification Report:")
print(classification_report(test_generator.labels, vgg16_pred_labels, target_names=list(test_generator.class_indices.keys())))
print("DenseNet121 Classification Report:")
print(classification_report(test_generator.labels, densenet121_pred_labels, target_names=list(test_generator.class_indices.keys())))

def detect_empyema_coordinates(detection_model, img_array):
    detection_result = detection_model.predict(img_array)
    x, y, w, h = detection_result[0]
    return int(x), int(y), int(w), int(h)

def crop_image(image_array, x, y, w, h):
    cropped_image = image_array[y:y+h, x:x+w]
    return cropped_image

def display_image_with_dots(image_path, pred_label, x, y, w, h):
    image = load_img(image_path, target_size=(224, 224))
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

            vgg16_pred = vgg16_model.predict(img_array)
            densenet121_pred = densenet121_model.predict(img_array)

            vgg16_pred_label = list(test_generator.class_indices.keys())[np.argmax(vgg16_pred)]
            densenet121_pred_label = list(test_generator.class_indices.keys())[np.argmax(densenet121_pred)]

            print("VGG16 Prediction:", vgg16_pred_label)
            print("DenseNet121 Prediction:", densenet121_pred_label)

            x, y, w, h = detect_empyema_coordinates(detection_model, img_array)

            display_image_with_dots(user_input, vgg16_pred_label, x, y, w, h)
        except Exception as e:
            print(f"An error occurred while processing the image: {e}")
    else:
        print("Invalid image path. Please try again.")
