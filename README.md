# Multiclass Pathological Classification in a Voice Signal Using Deep Learning

This project implements a deep learning-based approach for multiclass classification of pathological voice signals. It utilizes machine learning techniques like SVM, Random Forest, and Logistic Regression for comparison. The dataset includes extracted features such as MFCCs for model training.


## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/jvg-jeevan/Multiclass-Pathological-Classification.git
   ```
2. Navigate to the project directory:
   ```sh
   cd Multiclass-Pathological-Classification
   ```
3. Install requirements



## Project Structure
```
.
├── data/                   # Contains dataset files
├── results/                # Stores results and trained models
├── dataanalysis.py         # Data analysis and visualization
├── preprocess.py           # Data preprocessing steps
├── trainmodel.py           # Model training script
├── predictmodel.py         # Prediction script
├── LogisticRegression.py   # Logistic Regression model
├── RandomForest.py         # Random Forest model
├── SVMLinearKernel.py      # SVM with Linear Kernel
├── SVMGaussianKernel.py    # SVM with Gaussian Kernel
├── recordingaudio.py       # Script for recording audio samples
├── dataset.csv             # Main dataset
├── hypoMFCC.csv            # Hypothetical MFCC features
├── hyperMFCC.csv           # Hyper MFCC features
├── rlMFCC.csv              # Real-life MFCC features
└── data/
    ├── Healthy/                  # Voice samples from healthy individuals
    ├── hyperkinetic_dysphonia/    # Voice samples with hyperkinetic dysphonia
    ├── hypokinetic_dysphonia/     # Voice samples with hypokinetic dysphonia
    ├── reflux_laryngitis/         # Voice samples with reflux laryngitis
```

## Dataset
The dataset consists of extracted MFCC features from pathological and healthy voice recordings.
- `dataset.csv`: Main dataset for training and testing
- `Healthy/`: Healthy voice samples
- `hyperkinetic_dysphonia/`: Samples of hyperkinetic dysphonia
- `hypokinetic_dysphonia/`: Samples of hypokinetic dysphonia
- `reflux_laryngitis/`: Samples of reflux laryngitis
- `hypoMFCC.csv`, `hyperMFCC.csv`, `rlMFCC.csv`: Feature extraction results

## Models
The project implements and compares multiple classification models:
- Logistic Regression
- Random Forest
- Support Vector Machines (Linear & Gaussian Kernels)

## Results
The classification performance is evaluated using accuracy, precision, recall, and F1-score. Results are stored in the `results/` folder.

## License
This project is open-source under the MIT License.
