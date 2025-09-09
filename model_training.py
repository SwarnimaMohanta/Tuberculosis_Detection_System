import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import json

class TBModelTrainer:
    def __init__(self, data_path="processed_data", img_size=224, batch_size=32):
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.base_model = None
        self.save_dir = 'D:\\tuberculosis-detection\\models'
        
    def create_data_generators(self, augment=True):
        """Create data generators for training, validation, and testing"""
        
        if augment:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # FIXED: Use class_mode='binary' for simple binary classification
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_path, 'train'),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True
        )
        
        validation_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.data_path, 'validation'),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.data_path, 'test'),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.test_generator = test_generator
        
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {validation_generator.samples}")
        print(f"Test samples: {test_generator.samples}")
        print(f"Class indices: {train_generator.class_indices}")
        
        return train_generator, validation_generator, test_generator
    
    def create_model(self, learning_rate=0.001):
        """Create VGG16-based model for tuberculosis detection - SIMPLIFIED VERSION"""
        
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        self.base_model = base_model
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu', name='fc2')(x)
        x = Dropout(0.3)(x)
        
        # FIXED: Single output for simple binary classification
        output = Dense(1, activation='sigmoid', name='tb_detection')(x)
        
        # FIXED: Single output model
        model = Model(inputs=base_model.input, outputs=output)
        
        # FIXED: Simple compilation for binary classification
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        print("Model created successfully!")
        print(f"Total parameters: {model.count_params():,}")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        return model
    
    def train_model(self, epochs=30, fine_tune_epochs=10, fine_tune_lr=0.0001):
        """Train the model with transfer learning approach"""
        
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Created directory: {self.save_dir}")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.save_dir, 'best_tb_model.h5'),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Calculate steps per epoch
        steps_per_epoch = self.train_generator.samples // self.batch_size
        validation_steps = self.validation_generator.samples // self.batch_size
        
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        print("Starting initial training (frozen base model)...")
        
        # Phase 1: Train with frozen base model
        history1 = self.model.fit(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self.validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tuning (unfreeze some layers)
        print("\nStarting fine-tuning phase...")
        
        if self.base_model is None:
            print("Error: Base model reference not found. Skipping fine-tuning.")
            return history1.history
        
        # Enable fine-tuning
        self.base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = 100
        
        # Freeze all layers except the top ones for fine-tuning
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        print(f"Fine-tuning from layer {fine_tune_at} onwards...")
        print(f"Trainable layers: {sum(1 for layer in self.base_model.layers if layer.trainable)}")
        
        # Recompile with a lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=fine_tune_lr),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Fine-tune training
        history2 = self.model.fit(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=fine_tune_epochs,
            validation_data=self.validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        self.history = {
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy']
        }
        
        print("Training completed!")
        
        # Save final model
        final_model_path = os.path.join(self.save_dir, 'tb_detection_model.h5')
        self.model.save(final_model_path)
        print(f"Model saved as '{final_model_path}'")
        
        return self.history
    
    def evaluate_model(self):
        """Evaluate the trained model on test data"""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print("Evaluating model on test data...")
        
        test_steps = self.test_generator.samples // self.batch_size
        
        # Evaluate
        test_results = self.model.evaluate(self.test_generator, steps=test_steps, verbose=1)
        
        print("\nTest Results:")
        for i, metric_name in enumerate(self.model.metrics_names):
            print(f"{metric_name}: {test_results[i]:.4f}")
        
        # Get predictions for detailed analysis
        print("Generating predictions...")
        self.test_generator.reset()
        predictions = self.model.predict(self.test_generator, verbose=1)
        
        # Convert probabilities to binary predictions
        tb_pred_binary = (predictions > 0.5).astype(int).flatten()
        tb_true = self.test_generator.classes
        
        # Ensure same length
        min_length = min(len(tb_pred_binary), len(tb_true))
        tb_pred_binary = tb_pred_binary[:min_length]
        tb_true = tb_true[:min_length]
        
        # Classification report
        print("\nTuberculosis Detection Classification Report:")
        class_names = ['Normal', 'Tuberculosis']
        print(classification_report(tb_true, tb_pred_binary, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(tb_true, tb_pred_binary)
        
        print("Creating confusion matrix...")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - TB Detection')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        confusion_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {confusion_path}")
        plt.close()
        
        print("Evaluation completed!")
        return test_results, tb_pred_binary, tb_true
    
    def plot_training_history(self):
        """Plot training history"""
        
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
        
        print("Creating training history plots...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(self.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate placeholder
        axes[1, 0].text(0.5, 0.5, 'Learning Rate\n(Dynamic via ReduceLROnPlateau)', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Learning Rate Schedule')
        
        # Validation accuracy vs loss
        axes[1, 1].scatter(self.history['val_loss'], self.history['val_accuracy'])
        axes[1, 1].set_title('Validation Accuracy vs Loss')
        axes[1, 1].set_xlabel('Validation Loss')
        axes[1, 1].set_ylabel('Validation Accuracy')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        history_path = os.path.join(self.save_dir, 'training_history.png')
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        print(f"Training history plots saved to: {history_path}")
        plt.close()
        
        print("Training history plotting completed!")
    
    def save_model_info(self):
        """Save model information and training details"""
        
        model_info = {
            'architecture': 'VGG16 + Custom Head (Binary Classification)',
            'input_shape': [self.img_size, self.img_size, 3],
            'output_shape': 'Single sigmoid output for binary classification',
            'batch_size': self.batch_size,
            'total_parameters': self.model.count_params() if self.model else 0,
            'classes': ['Normal', 'Tuberculosis'],
            'preprocessing': {
                'rescale': '1./255',
                'augmentation': True,
                'target_size': [self.img_size, self.img_size]
            },
            'training_info': {
                'base_model_frozen_initially': True,
                'fine_tuning_enabled': True,
                'fine_tune_from_layer': 100,
                'loss_function': 'binary_crossentropy',
                'optimizer': 'Adam'
            }
        }
        
        info_path = os.path.join(self.save_dir, 'model_info.json')
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model information saved to '{info_path}'")


def check_data_structure():
    """Check if data structure is correct"""
    data_path = "processed_data"
    required_folders = ['train', 'validation', 'test']
    required_classes = ['Normal', 'Tuberculosis']
    
    print("Checking data structure...")
    
    if not os.path.exists(data_path):
        print(f"❌ Data path '{data_path}' not found!")
        return False
    
    all_good = True
    for folder in required_folders:
        folder_path = os.path.join(data_path, folder)
        if not os.path.exists(folder_path):
            print(f"❌ Missing folder: {folder_path}")
            all_good = False
            continue
        
        print(f"✅ Found: {folder}")
        for class_name in required_classes:
            class_path = os.path.join(folder_path, class_name)
            if os.path.exists(class_path):
                count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"   - {class_name}: {count} images")
                if count == 0:
                    print(f"   ⚠️  Warning: No images found in {class_name}")
            else:
                print(f"❌ Missing class folder: {class_path}")
                all_good = False
    
    return all_good


def main():
    """Main training function"""
    
    print("=== Tuberculosis Detection Model Training (Fixed) ===\n")
    
    # Check data structure
    if not check_data_structure():
        print("❌ Please fix the data structure before training.")
        print("Required structure:")
        print("processed_data/")
        print("├── train/")
        print("│   ├── Normal/")
        print("│   └── Tuberculosis/")
        print("├── validation/")
        print("│   ├── Normal/")
        print("│   └── Tuberculosis/")
        print("└── test/")
        print("    ├── Normal/")
        print("    └── Tuberculosis/")
        return
    
    print("✅ Data structure looks good!")
    
    try:
        # Initialize trainer
        trainer = TBModelTrainer(
            data_path="processed_data",
            img_size=224,
            batch_size=32
        )
        
        # Create data generators
        print("\nCreating data generators...")
        trainer.create_data_generators(augment=True)
        
        # Create model
        print("\nCreating model...")
        trainer.create_model(learning_rate=0.001)
        
        # Train model
        print("\nStarting training...")
        trainer.train_model(epochs=25, fine_tune_epochs=10, fine_tune_lr=0.0001)
        
        # Evaluate model
        print("\nEvaluating model...")
        trainer.evaluate_model()
        
        # Plot training history
        print("\nPlotting training history...")
        trainer.plot_training_history()
        
        # Save model information
        trainer.save_model_info()
        
        print("\n✅ Training completed successfully!")
        print(f"Files created in {trainer.save_dir}:")
        print("- tb_detection_model.h5 (trained model)")
        print("- best_tb_model.h5 (best model checkpoint)")
        print("- model_info.json (model information)")
        print("- training_history.png (training plots)")
        print("- confusion_matrix.png (evaluation results)")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check data structure above")
        print("2. Ensure sufficient disk space")
        print("3. Verify GPU/memory availability")
        print("4. Check image file formats")


if __name__ == "__main__":
    main()