import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 1. Define Constants
IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 32
DATA_DIR = 'data' # This is the folder we created
NUM_CLASSES = 6 # buildings, forest, glacier, mountain, sea, street

# 2. Prepare the Data
# Use ImageDataGenerator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Rescale pixel values from [0, 255] to [0, 1]
    rotation_range=40,       # Randomly rotate images
    width_shift_range=0.2,   # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.2,         # Shear transformations
    zoom_range=0.2,          # Randomly zoom in on images
    horizontal_flip=True,    # Randomly flip images horizontally
    fill_mode='nearest',     # Strategy for filling in newly created pixels
    validation_split=0.2     # Set aside 20% of data for validation
)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training' # Specify this is the training set
)

# Validation data generator
validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation' # Specify this is the validation set
)

# 3. Build the Model using Transfer Learning
# Load the MobileNetV2 model, pre-trained on ImageNet, without the top classification layer
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False, # Do not include the final Dense layer
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
)

# Freeze the base model layers so they are not re-trained
base_model.trainable = False

# Add our custom classification layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x) # A pooling layer to reduce dimensions
x = Dense(1024, activation='relu')(x) # A fully-connected layer
predictions = Dense(NUM_CLASSES, activation='softmax')(x) # The final output layer

# Combine the base model and our custom layers
model = Model(inputs=base_model.input, outputs=predictions)

# 4. Compile the Model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train the Model
# We'll train for a few epochs as an example. For better results, increase the number of epochs.
EPOCHS = 5
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# 6. Save the Trained Model
# The model will be saved in the H5 format
model.save('scene_classifier_model.h5')

print("Model training complete and saved as scene_classifier_model.h5")