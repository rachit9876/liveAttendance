"""
Custom Face Recognition Model Training
Uses transfer learning on pre-trained FaceNet with custom dataset
"""
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

class CustomFaceRecognitionModel:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = self.build_model()
    
    def build_model(self):
        """Build model using transfer learning from InceptionResNetV2"""
        # Load pre-trained base model
        base_model = InceptionResNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(160, 160, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu', name='embedding_layer')(x)
        x = Dropout(0.3)(x)
        output = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_dir, epochs=10, batch_size=32):
        """Train model on custom dataset"""
        # Data augmentation for better generalization
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(160, 160),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        val_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(160, 160),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        # Train model
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            verbose=1
        )
        
        return history
    
    def save_model(self, path='models/custom_face_model.h5'):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def get_embedding_model(self):
        """Extract embedding layer for face recognition"""
        return Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('embedding_layer').output
        )

# Example usage
if __name__ == '__main__':
    # Organize your faces folder like:
    # dataset/
    #   person1/
    #     img1.jpg
    #     img2.jpg
    #   person2/
    #     img1.jpg
    
    dataset_path = 'dataset'  # Create this structure
    
    if os.path.exists(dataset_path):
        num_classes = len(os.listdir(dataset_path))
        
        print(f"Training custom model with {num_classes} classes...")
        model = CustomFaceRecognitionModel(num_classes)
        
        history = model.train(dataset_path, epochs=10)
        model.save_model()
        
        print("Training completed!")
    else:
        print(f"Dataset not found. Create '{dataset_path}' folder with subfolders for each person.")
