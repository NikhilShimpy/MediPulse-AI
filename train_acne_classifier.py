import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Paths and Configs
DATA_DIR = 'dataset'
MODEL_PATH = 'model/acne_classifier.h5'
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10

# Ensure model folder exists
os.makedirs('model', exist_ok=True)

# Data loading (no validation split)
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes
])

# Compile model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_gen, epochs=EPOCHS)

# Save model
model.save(MODEL_PATH)
print(f"✅ Model trained and saved at {MODEL_PATH}")
