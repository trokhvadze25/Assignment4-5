# Assignment4-5# DDoS Attack Detection Using CNN on Network Traffic Images

This project presents a Deep Learning–based DDoS detection system that converts network traffic data into images and classifies them using a Convolutional Neural Network (CNN).
The approach leverages CNNs’ strength in spatial pattern recognition to detect abnormal traffic behavior caused by Distributed Denial-of-Service (DDoS) attacks.

**System Workflow Overview**

The system follows these main stages:

Load network traffic dataset

Preprocess and clean data

Normalize numerical features

Convert network flows into images

Train a CNN model

Evaluate and save results

**Dataset Loading**

The dataset is stored in CSV format, where each row represents a single network flow, and each column represents a network feature.

```
CSV_FILE_PATH = 'Friday-WorkingHours-Afternoon-DDos.csv'

df = pd.read_csv(CSV_FILE_PATH)
print(f"Loaded {len(df)} network flows")
print(f"Dataset shape: {df.shape}")
```

pd.read_csv() loads the dataset into a Pandas DataFrame.

Basic statistics are printed to verify successful loading.

Early termination is used if loading fails to prevent further execution errors.

**Data Preprocessing**
Label Extraction and Binary Encoding
The original labels contain textual values such as BENIGN, DDoS, or DoS. These are converted into binary values.

```
label_col = 'Label' if 'Label' in df.columns else ' Label'

X = df.drop(columns=[label_col])
y = df[label_col]

y_binary = y.apply(
    lambda x: 1 if 'ddos' in str(x).lower()
    or 'dos' in str(x).lower()
    or 'attack' in str(x).lower()
    else 0
)
```
The label column is detected dynamically. Binary encoding simplifies the problem into binary classification:
0 → BENIGN
1 → ATTACK

**Handling Missing and Invalid Values**

Network datasets often contain invalid or infinite values.

```
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

numeric_cols = X.select_dtypes(include=[np.number]).columns
X_numeric = X[numeric_cols]
```

Infinite values are replaced with NaN.

Missing values are filled with zero.

Only numeric features are retained, which are required for CNN input.

Feature Normalization

All features are scaled into the range [0, 1].

```
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X_numeric)
```

Min–Max normalization ensures that all features contribute equally.

Normalization improves training stability and convergence speed.

**Converting Network Flows to Images** 

Each network flow (feature vector) is reshaped into a 2D grayscale image.

```
num_features = X_normalized.shape[1]
img_size = int(np.ceil(np.sqrt(num_features)))

def flow_to_image(flow_features, img_size):
    total_pixels = img_size * img_size
    padded = np.pad(flow_features, (0, total_pixels - len(flow_features)), 'constant')
    return padded.reshape(img_size, img_size)

```

The image size is determined dynamically based on the number of features.

Padding ensures a perfect square image.

Each pixel represents a normalized network feature.
```
Image Generation Loop
images = []
for i in range(len(X_normalized)):
    img = flow_to_image(X_normalized[i], img_size)
    images.append(img)

images = np.array(images).reshape(-1, img_size, img_size, 1)


Images are stored as 4D tensors (samples, height, width, channels).
```


**Dataset Splitting**

The dataset is split into training, validation, and testing sets.

```
from sklearn.model_selection import train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(
    images, y_binary, test_size=0.2, stratify=y_binary, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

```

Stratified splitting preserves class distribution.

Final ratio:

60% Training

20% Validation

20% Testing


CNN Model Architecture

```
model = models.Sequential([
    layers.Input(shape=(img_size, img_size, 1)),

    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

```

Convolution layers extract spatial features.

Batch normalization stabilizes learning.

Dropout prevents overfitting.

Sigmoid output provides probability of attack.

```
Model Compilation
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

9. Model Training (Step 8)
Code Snippet
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=128,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

```

Early stopping avoids overfitting.

Learning rate reduction improves convergence.

Best model is saved automatically.

**Model Evaluation**

```
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print(classification_report(y_test, y_pred))

```

Predictions are thresholded at 0.5.

Evaluation includes accuracy, precision, recall, and F1-score.

**Saving Model and Scaler**
```
model.save('ddos_cnn_model.h5')
joblib.dump(scaler, 'feature_scaler.pkl')

```

Saved model can be reused without retraining.

Scaler ensures consistent preprocessing during inference.
