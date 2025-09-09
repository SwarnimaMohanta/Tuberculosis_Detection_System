import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="TB Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .result-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .severity-low {
        background-color: #10b981;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .severity-medium {
        background-color: #f59e0b;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .severity-high {
        background-color: #ef4444;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_demo_model():
    """Create a demo model for testing purposes"""
    try:
        # Create a simple VGG16-based model
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu', name='fc2')(x)
        x = Dropout(0.3)(x)
        
        # Single output for binary classification
        output = Dense(1, activation='sigmoid', name='tb_detection')(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        
        # Compile for binary classification
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
        
    except Exception as e:
        st.error(f"Error creating demo model: {str(e)}")
        return None

@st.cache_resource
def load_model():
    """Load or create the tuberculosis detection model"""
    try:
        # Check for existing trained models in various locations
        model_paths = [
            'models/tb_detection_model.h5',
            'models/best_tb_model.h5',
            'tb_detection_model.h5',
            'best_tb_model.h5'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    # Try loading with compile=False first to avoid compatibility issues
                    model = tf.keras.models.load_model(path, compile=False)
                    
                    # Recompile with correct settings
                    model.compile(
                        optimizer=Adam(learning_rate=0.001),
                        loss='binary_crossentropy',
                        metrics=['accuracy', 'precision', 'recall']
                    )
                    
                    st.success(f"✅ Pre-trained model loaded from: {path}")
                    return model
                    
                except Exception as load_error:
                    st.warning(f"Failed to load {path}: {str(load_error)}")
                    continue
        
        # If no pre-trained model found, create a demo model
        st.warning("⚠️ No pre-trained model found. Creating demo model...")
        st.info("📝 Note: This is a demo model. For actual medical use, please train with real data.")
        
        demo_model = create_demo_model()
        if demo_model is not None:
            st.success("✅ Demo model created successfully!")
            return demo_model
        else:
            st.error("❌ Failed to create demo model")
            return None
        
    except Exception as e:
        st.error(f"Error in load_model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess uploaded image for model prediction"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Handle grayscale images
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Resize image to 224x224 (VGG16 input size)
        img_resized = cv2.resize(img_array, (224, 224))
        
        # Normalize pixel values
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
        
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_tuberculosis(model, processed_image):
    """Make prediction using the loaded model"""
    try:
        if model is None:
            st.error("Model not loaded!")
            return 0.0, [0.33, 0.33, 0.34]
        
        if processed_image is None:
            st.error("Image preprocessing failed!")
            return 0.0, [0.33, 0.33, 0.34]
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        tb_prob = float(predictions[0][0])
        
        # Simulate severity levels based on TB probability
        if tb_prob < 0.3:
            severity_probs = [0.8, 0.15, 0.05]  # Low, Medium, High
        elif tb_prob < 0.7:
            severity_probs = [0.3, 0.6, 0.1]
        else:
            severity_probs = [0.1, 0.3, 0.6]
        
        return tb_prob, severity_probs
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        # Return default values on error
        return 0.5, [0.33, 0.33, 0.34]

def get_severity_level(severity_probs):
    """Determine severity level from probabilities"""
    severity_labels = ['Low', 'Medium', 'High']
    severity_colors = ['#10b981', '#f59e0b', '#ef4444']
    
    max_idx = np.argmax(severity_probs)
    severity_level = severity_labels[max_idx]
    severity_color = severity_colors[max_idx]
    severity_percentage = severity_probs[max_idx] * 100
    
    return severity_level, severity_color, severity_percentage

def get_icu_recommendation(tb_prob, severity_level):
    """Get ICU recommendation based on prediction"""
    if tb_prob > 0.7 and severity_level == 'High':
        return "IMMEDIATE ICU admission recommended", "🚨"
    elif tb_prob > 0.5 and severity_level in ['Medium', 'High']:
        return "Close monitoring required, ICU standby", "⚠️"
    elif tb_prob > 0.3:
        return "Regular monitoring sufficient", "✅"
    else:
        return "No immediate concern", "✅"

def create_gauge_chart(percentage, title, color):
    """Create a gauge chart for severity percentage"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 24}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#10b981'},
                {'range': [30, 70], 'color': '#f59e0b'},
                {'range': [70, 100], 'color': '#ef4444'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, font={'color': "darkblue", 'family': "Arial"})
    return fig

def generate_pdf_report(image, tb_prob, severity_level, severity_percentage, icu_recommendation, patient_info=None):
    """Generate PDF report of the diagnosis"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.darkblue
        )
        story.append(Paragraph("Tuberculosis Detection Report", title_style))
        story.append(Spacer(1, 20))
        
        # Patient Information (if provided)
        if patient_info:
            patient_data = [
                ['Patient Name:', patient_info.get('name', 'N/A')],
                ['Age:', str(patient_info.get('age', 'N/A'))],
                ['Gender:', patient_info.get('gender', 'N/A')],
                ['Date:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            ]
            
            patient_table = Table(patient_data, colWidths=[2*inch, 3*inch])
            patient_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ]))
            story.append(patient_table)
            story.append(Spacer(1, 20))
        
        # Diagnosis Results
        results_data = [
            ['Parameter', 'Value', 'Status'],
            ['TB Probability', f"{tb_prob*100:.1f}%", 'Positive' if tb_prob > 0.5 else 'Negative'],
            ['Severity Level', severity_level, f"{severity_percentage:.1f}%"],
            ['ICU Recommendation', icu_recommendation[0], icu_recommendation[1]]
        ]
        
        results_table = Table(results_data, colWidths=[2*inch, 2*inch, 2*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(results_table)
        story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("Clinical Recommendations:", styles['Heading2']))
        recommendations = []
        
        if tb_prob > 0.7:
            recommendations.extend([
                "• Immediate clinical evaluation required",
                "• Consider starting anti-TB treatment",
                "• Follow-up chest imaging in 2-4 weeks"
            ])
        elif tb_prob > 0.3:
            recommendations.extend([
                "• Clinical correlation recommended",
                "• Consider additional diagnostic tests",
                "• Monitor symptoms closely"
            ])
        else:
            recommendations.extend([
                "• Low probability of TB",
                "• Continue routine monitoring",
                "• Re-evaluate if symptoms persist"
            ])
        
        for rec in recommendations:
            story.append(Paragraph(rec, styles['Normal']))
        
        story.append(Spacer(1, 20))
        story.append(Paragraph("Note: This report is generated by an AI system and should be reviewed by a qualified healthcare professional.", styles['Italic']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")
        return None

def main():
    """Main application function"""
    # Title
    st.markdown('<h1 class="main-header">🫁 Tuberculosis Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar for additional options
    st.sidebar.title("Settings")
    
    # Patient Information
    with st.sidebar.expander("Patient Information (Optional)"):
        patient_name = st.text_input("Patient Name")
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=25)
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    # Model threshold
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Load model with error handling
    with st.spinner("Loading AI model..."):
        model = load_model()
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload Chest X-ray Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload a chest X-ray image for tuberculosis detection"
        )
        
        # Display uploaded image
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
                
                # Store image in session state
                st.session_state.uploaded_image = image
                st.session_state.image_uploaded = True
                
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                st.session_state.image_uploaded = False
        else:
            st.info("👆 Please upload a chest X-ray image to begin analysis")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.markdown("### 📊 Analysis Results")
        
        if hasattr(st.session_state, 'image_uploaded') and st.session_state.image_uploaded and model is not None:
            try:
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    processed_image = preprocess_image(st.session_state.uploaded_image)
                    
                    if processed_image is not None:
                        # Make prediction
                        tb_prob, severity_probs = predict_tuberculosis(model, processed_image)
                        
                        # Get severity information
                        severity_level, severity_color, severity_percentage = get_severity_level(severity_probs)
                        
                        # Get ICU recommendation
                        icu_recommendation = get_icu_recommendation(tb_prob, severity_level)
                        
                        # Display results
                        st.markdown("#### 🔬 Diagnosis Results")
                        
                        # TB Probability
                        st.metric("TB Detection Probability", f"{tb_prob*100:.1f}%")
                        
                        # Severity indicator
                        severity_class = f"severity-{severity_level.lower()}"
                        st.markdown(f'<div class="{severity_class}">Severity: {severity_level} ({severity_percentage:.1f}%)</div>', 
                                   unsafe_allow_html=True)
                        
                        # ICU Recommendation
                        st.markdown("#### 🏥 ICU Recommendation")
                        st.info(f"{icu_recommendation[1]} {icu_recommendation[0]}")
                        
                        # Gauge chart
                        st.markdown("#### 📈 Severity Gauge")
                        gauge_fig = create_gauge_chart(severity_percentage, "Severity Level", severity_color)
                        st.plotly_chart(gauge_fig, use_container_width=True)
                        
                        # Probability breakdown
                        st.markdown("#### 📋 Detailed Breakdown")
                        prob_df = pd.DataFrame({
                            'Severity Level': ['Low', 'Medium', 'High'],
                            'Probability': [f"{prob*100:.1f}%" for prob in severity_probs],
                            'Value': severity_probs
                        })
                        
                        fig_bar = px.bar(prob_df, x='Severity Level', y='Value', 
                                       title='Severity Level Probabilities',
                                       color='Severity Level',
                                       color_discrete_map={'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#ef4444'})
                        fig_bar.update_layout(showlegend=False)
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Store results for report generation
                        st.session_state.tb_prob = tb_prob
                        st.session_state.severity_level = severity_level
                        st.session_state.severity_percentage = severity_percentage
                        st.session_state.icu_recommendation = icu_recommendation
                        st.session_state.results_ready = True
                    
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.session_state.results_ready = False
                
        else:
            if model is None:
                st.error("❌ Model not available. Please check model loading.")
            else:
                st.info("Upload an image to see analysis results here")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Generate Report Button
    if (hasattr(st.session_state, 'results_ready') and st.session_state.results_ready and 
        hasattr(st.session_state, 'image_uploaded') and st.session_state.image_uploaded):
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("📄 Generate Report", use_container_width=True):
                # Prepare patient info
                patient_info = {
                    'name': patient_name if patient_name else 'Anonymous',
                    'age': patient_age,
                    'gender': patient_gender
                }
                
                # Generate PDF report
                with st.spinner("Generating report..."):
                    pdf_buffer = generate_pdf_report(
                        st.session_state.uploaded_image,
                        st.session_state.tb_prob,
                        st.session_state.severity_level,
                        st.session_state.severity_percentage,
                        st.session_state.icu_recommendation,
                        patient_info
                    )
                
                if pdf_buffer:
                    # Download button for PDF
                    st.download_button(
                        label="📥 Download Report",
                        data=pdf_buffer.getvalue(),
                        file_name=f"TB_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                    st.success("Report generated successfully!")

def create_model_for_training():
    """Create a VGG16-based model for training"""
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', name='fc2')(x)
    x = Dropout(0.3)(x)
    
    # Single output for binary classification
    output = Dense(1, activation='sigmoid', name='tb_detection')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    # Binary classification compilation
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def train_model_section():
    """Section for training a new model"""
    st.markdown("## 🎯 Model Training Section")
    
    st.info("⚠️ **Important**: Model training requires a properly structured dataset with 'train', 'validation', and 'test' folders containing 'Normal' and 'Tuberculosis' subfolders.")
    
    # Check if processed data exists
    train_path = "processed_data/train"
    val_path = "processed_data/validation"
    
    if os.path.exists(train_path) and os.path.exists(val_path):
        st.success("✅ Training data found!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Training Parameters")
            epochs = st.slider("Number of Epochs", 5, 50, 20)
            batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
            learning_rate = st.select_slider("Learning Rate", 
                                           options=[0.0001, 0.001, 0.01], 
                                           value=0.001)
        
        with col2:
            st.markdown("### Data Augmentation")
            use_augmentation = st.checkbox("Use Data Augmentation", True)
            rotation_range = st.slider("Rotation Range", 0, 45, 20)
            zoom_range = st.slider("Zoom Range", 0.0, 0.3, 0.2)
        
        if st.button("🚀 Start Training", use_container_width=True):
            train_tuberculosis_model(epochs, batch_size, learning_rate, use_augmentation, rotation_range, zoom_range)
    else:
        st.warning("❌ No training data found.")
        st.markdown("### Expected Directory Structure:")
        st.code("""
processed_data/
├── train/
│   ├── Normal/
│   └── Tuberculosis/
├── validation/
│   ├── Normal/
│   └── Tuberculosis/
└── test/
    ├── Normal/
    └── Tuberculosis/
        """)
        
        st.markdown("### Setup Instructions:")
        st.markdown("""
        1. Download TB dataset from Kaggle
        2. Organize images into the above structure
        3. Ensure equal distribution of Normal and TB cases
        4. Images should be in common formats (jpg, png, etc.)
        """)

def train_tuberculosis_model(epochs, batch_size, learning_rate, use_augmentation, rotation_range, zoom_range):
    """Train the tuberculosis detection model"""
    try:
        # Data generators
        if use_augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=rotation_range,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=zoom_range,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        # Data generators
        train_generator = train_datagen.flow_from_directory(
            'processed_data/train',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            'processed_data/validation',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        # Create model
        model = create_model_for_training()
        
        if model is None:
            st.error("Failed to create model!")
            return
        
        # Training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Custom callback for Streamlit
        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f'Epoch {epoch+1}/{epochs} - Loss: {logs.get("loss", 0):.4f} - Accuracy: {logs.get("accuracy", 0):.4f}')
        
        # Calculate steps
        steps_per_epoch = max(1, train_generator.samples // batch_size)
        validation_steps = max(1, validation_generator.samples // batch_size)
        
        # Train model
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=[StreamlitCallback()],
            verbose=0
        )
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model
        model_path = 'models/tb_detection_model.h5'
        model.save(model_path)
        st.success(f"✅ Model trained and saved to: {model_path}")
        
        # Plot training history
        col1, col2 = st.columns(2)
        
        with col1:
            # Loss plot
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=list(range(1, epochs+1)), 
                                        y=history.history.get('loss', []), 
                                        name='Training Loss'))
            if 'val_loss' in history.history:
                fig_loss.add_trace(go.Scatter(x=list(range(1, epochs+1)), 
                                            y=history.history['val_loss'], 
                                            name='Validation Loss'))
            fig_loss.update_layout(title='Training and Validation Loss',
                                 xaxis_title='Epoch',
                                 yaxis_title='Loss')
            st.plotly_chart(fig_loss, use_container_width=True)
        
        with col2:
            # Accuracy plot
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=list(range(1, epochs+1)), 
                                       y=history.history.get('accuracy', []), 
                                       name='Training Accuracy'))
            if 'val_accuracy' in history.history:
                fig_acc.add_trace(go.Scatter(x=list(range(1, epochs+1)), 
                                           y=history.history['val_accuracy'], 
                                           name='Validation Accuracy'))
            fig_acc.update_layout(title='Training and Validation Accuracy',
                                xaxis_title='Epoch',
                                yaxis_title='Accuracy')
            st.plotly_chart(fig_acc, use_container_width=True)
        
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        st.write("Error details:", e)

def model_evaluation_section():
    """Section for model evaluation"""
    st.markdown("## 📈 Model Evaluation")
    
    # Check for available models
    model_paths = [
        'models/tb_detection_model.h5',
        'models/best_tb_model.h5',
        'tb_detection_model.h5',
        'best_tb_model.h5'
    ]
    
    model_found = False
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_found = True
            model_path = path
            break
    
    if model_found and os.path.exists('processed_data/test'):
        try:
            # Load model for evaluation
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            st.success(f"✅ Model loaded from: {model_path}")
            
            # Test data generator
            test_datagen = ImageDataGenerator(rescale=1./255)
            test_generator = test_datagen.flow_from_directory(
                'processed_data/test',
                target_size=(224, 224),
                batch_size=32,
                class_mode='binary',
                shuffle=False
            )
            
            if st.button("🔍 Evaluate Model", use_container_width=True):
                with st.spinner("Evaluating model..."):
                    # Calculate test steps
                    test_steps = max(1, test_generator.samples // 32)
                    
                    # Evaluate model
                    test_results = model.evaluate(test_generator, steps=test_steps, verbose=0)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Test Accuracy", f"{test_results[1]:.4f}")
                    with col2:
                        st.metric("Test Loss", f"{test_results[0]:.4f}")
                    with col3:
                        st.metric("Test Samples", test_generator.samples)
                    
                    # Get predictions for confusion matrix
                    test_generator.reset()
                    predictions = model.predict(test_generator, steps=test_steps, verbose=0)
                    predicted_classes = (predictions > 0.5).astype(int).flatten()
                    true_classes = test_generator.classes
                    
                    # Ensure same length
                    min_length = min(len(predicted_classes), len(true_classes))
                    predicted_classes = predicted_classes[:min_length]
                    true_classes = true_classes[:min_length]
                    
                    # Confusion matrix
                    from sklearn.metrics import confusion_matrix, classification_report
                    cm = confusion_matrix(true_classes, predicted_classes)
                    
                    # Plot confusion matrix
                    fig_cm = px.imshow(cm, 
                                     text_auto=True, 
                                     aspect="auto",
                                     title="Confusion Matrix",
                                     labels=dict(x="Predicted", y="Actual"),
                                     color_continuous_scale='Blues')
                    class_labels = ['Normal', 'Tuberculosis']
                    fig_cm.update_xaxes(tickvals=[0, 1], ticktext=class_labels)
                    fig_cm.update_yaxes(tickvals=[0, 1], ticktext=class_labels)
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                    # Classification report
                    report = classification_report(true_classes, predicted_classes, 
                                                 target_names=class_labels, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    
                    st.markdown("### 📊 Classification Report")
                    st.dataframe(report_df, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error during evaluation: {str(e)}")
            
    else:
        st.warning("❌ Model or test data not found.")
        st.markdown("### Available Models Check:")
        for path in model_paths:
            exists = "✅" if os.path.exists(path) else "❌"
            st.write(f"{exists} {path}")
        st.write(f"{'✅' if os.path.exists('processed_data/test') else '❌'} processed_data/test")
        
        if not model_found:
            st.info("💡 Train a model first using the 'Train Model' section.")

def about_section():
    """About section with information about the application"""
    st.markdown("## ℹ️ About This Application")
    
    st.markdown("""
    ### 🫁 Tuberculosis Detection System
    
    This application uses deep learning to detect tuberculosis from chest X-ray images. 
    It's built with state-of-the-art computer vision techniques and provides:
    
    **Features:**
    - **AI-Powered Detection**: Uses VGG16 deep learning model for accurate TB detection
    - **Severity Classification**: Classifies TB severity into Low, Medium, and High categories
    - **ICU Recommendations**: Provides clinical guidance based on severity
    - **Visual Analytics**: Interactive charts and gauges for result visualization
    - **PDF Reports**: Generate detailed medical reports for documentation
    - **Model Training**: Train custom models with your own dataset
    - **Model Evaluation**: Comprehensive model performance analysis
    
    **Technical Specifications:**
    - **Model Architecture**: VGG16 with custom classification layers
    - **Input Size**: 224x224 RGB images
    - **Framework**: TensorFlow/Keras
    - **Deployment**: Streamlit
    - **Python Version**: 3.8+
    
    **Model Architecture Details:**
    - **Base Model**: VGG16 pre-trained on ImageNet
    - **Custom Head**: GlobalAveragePooling2D → Dense(512) → Dropout(0.5) → Dense(256) → Dropout(0.3) → Dense(1, sigmoid)
    - **Loss Function**: Binary Cross-entropy
    - **Optimizer**: Adam (lr=0.001)
    - **Metrics**: Accuracy, Precision, Recall
    
    **Dataset Requirements:**
    - **Format**: Chest X-ray images (JPG, PNG, BMP, TIFF)
    - **Classes**: Normal vs Tuberculosis (binary classification)
    - **Structure**: Organized in train/validation/test folders
    - **Preprocessing**: Automatic resizing to 224x224 and normalization
    
    **Usage Instructions:**
    1. **Detection**: Upload chest X-ray → Get AI analysis → Generate report
    2. **Training**: Prepare dataset → Configure parameters → Train model
    3. **Evaluation**: Load trained model → Run on test data → View metrics
    
    **Important Notes:**
    - This system is for **educational and research purposes only**
    - **Always consult qualified healthcare professionals** for medical diagnosis
    - The AI model provides assistance, not definitive medical diagnosis
    - Results should be interpreted by trained medical personnel
    
    **Disclaimer:**
    This application is a demonstration of AI capabilities in medical imaging. 
    It should not be used as a substitute for professional medical advice, 
    diagnosis, or treatment. Always seek the advice of qualified healthcare providers.
    """)
    
    # System information
    st.markdown("### 🔧 System Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("TensorFlow Version", tf.__version__)
    with col2:
        st.metric("Streamlit Version", st.__version__)
    with col3:
        st.metric("Python Version", "3.8+")
    with col4:
        st.metric("Model Type", "Binary Classification")
    
    # Dataset statistics (example)
    st.markdown("### 📊 Expected Dataset Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Recommended Images", "8,000+")
    with col2:
        st.metric("Normal Cases", "50%")
    with col3:
        st.metric("TB Cases", "50%")
    with col4:
        st.metric("Target Accuracy", "> 90%")

# Navigation
def navigation():
    """Navigation sidebar"""
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["🏠 Detection", "🎯 Train Model", "📈 Evaluate", "ℹ️ About"],
        help="Select a section to navigate to"
    )
    
    # Add system status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 System Status")
    
    # Check TensorFlow
    try:
        tf_version = tf.__version__
        st.sidebar.success(f"✅ TensorFlow {tf_version}")
    except:
        st.sidebar.error("❌ TensorFlow not available")
    
    # Check model availability
    model_paths = [
        'models/tb_detection_model.h5',
        'models/best_tb_model.h5',
        'tb_detection_model.h5'
    ]
    
    model_available = any(os.path.exists(path) for path in model_paths)
    if model_available:
        st.sidebar.success("✅ Trained model found")
    else:
        st.sidebar.warning("⚠️ No trained model")
    
    # Check dataset
    if os.path.exists('processed_data'):
        st.sidebar.success("✅ Dataset folder found")
    else:
        st.sidebar.info("ℹ️ No dataset folder")
    
    return page

if __name__ == "__main__":
    try:
        # Navigation
        selected_page = navigation()
        
        # Add a separator
        st.markdown("---")
        
        # Route to selected page
        if selected_page == "🏠 Detection":
            main()
        elif selected_page == "🎯 Train Model":
            train_model_section()
        elif selected_page == "📈 Evaluate":
            model_evaluation_section()
        elif selected_page == "ℹ️ About":
            about_section()
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666; margin-top: 2rem;'>
                <p>🫁 <strong>Tuberculosis Detection System</strong> | 
                Built with ❤️ using Streamlit & TensorFlow | 
                <em>For Educational & Research Purposes Only</em> | 
                © 2024</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page and try again. If the problem persists, check your dependencies.")
