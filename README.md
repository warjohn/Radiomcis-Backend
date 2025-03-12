# Sklearn Report Generator

This Python library helps generate detailed reports for scikit-learn models. It can create reports in various formats such as HTML, PDF, and DOCX. The library supports model training, metric evaluation, and report generation for both classification and regression tasks.

## Usage

After installation, you can use the SklearnReportGenerator to train a model, generate predictions, and create reports.

## Project Structure
```
- reportGeneration - main class
- calculateMetrics - main class that calculates metrics
- DOCS_format/HTMP_format/PDF_format - classes that create output files of different formats
- drawgraph - class that draws graphs
- generatelogs - debug class (!in development!)
- LoaderConfig - laod file "config.yaml"
```

### Example
```python
from reportGeneration.reportGeneration import SklearnReportGenerator
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

import uvicorn
import yaml
import os

app = FastAPI()
UPLOAD_DIR = 'reports'

# report_generator.fit(X_train, y_train, X_test, y_test)
# report_generator.predict(X_test)

@app.post("/upload/")
async def upload_yaml_file(file: UploadFile = File(...)):
    """
    Принимает YAML-файл от клиента, проверяет его формат, сохраняет на сервере,
    вызывает функцию для обработки файла и возвращает результат.
    """
    try:
        # Читаем содержимое файла
        file_content = await file.read()

        # Декодируем содержимое файла в строку (YAML - текстовый формат)
        file_content_str = file_content.decode('utf-8')

        # Пытаемся разобрать YAML-файл в словарь Python
        try:
            yaml_data = yaml.safe_load(file_content_str)  # Парсим YAML в словарь
        except yaml.YAMLError as e:
            return JSONResponse(
                content={"error": f"Invalid YAML format: {str(e)}"},
                status_code=400
            )

        # Сохраняем файл на сервере
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_content_str)

        # Вызываем вашу функцию для обработки YAML-файла
        report_generator = SklearnReportGenerator(config_file='config.yaml', output_format="PDF")
        report_generator.extract(csv_file_path="tmp.csv",
                                 output_csv_path="output-scv.csv",
                                 n_jobs=40)

        # Возвращаем результат: имя файла, путь к сохраненному файлу,
        # содержимое как строка, распарсенный YAML и результат обработки
        return JSONResponse(
            content={
                "filename": file.filename,
                "file_path": file_path,
                "file_content": file_content_str,  # Содержимое файла в виде строки
                "parsed_yaml": yaml_data,  # Распарсенный YAML в виде словаря
                "processing_result": "processing_result"  # Результат вашей обработки
            },
            status_code=200
        )
    except Exception as e:
        # В случае ошибки возвращаем сообщение об ошибке
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    # Запускаем сервер Uvicorn напрямую из скрипта
    uvicorn.run(app, host="127.0.0.1", port=5555)
```

## Configuration Example

The configuration file (config.yaml) defines the transformers, model, selection parameters (optional), and metrics to be used in the report generation.
```yaml
data:
  pathan-path: "testDir/pathan.csv"  # Path to the main dataset file (e.g., patient data)
  norma-path: "path/to/your/norma.csv"  # Path to the normal (control) dataset file
  target:
    - name: "outcome"  # Name of the target variable (dependent variable)
    - type-variable: "binary"  # Type of the target variable: binary or continuous
      # binary: Used for classification tasks (e.g., healthy vs. diseased)
      # continuous: Used for regression tasks (e.g., predicting a numeric value like tumor size)

# Configuration for the Language Model (LLM)
llmModel:
  name: 'llama3.1'  # Name of the language model being used
  lang: 'RU'        # Language of the model (in this case, Russian)

# Radiomics feature extraction settings
radiomics:
  filters:
    - name: ["Original", "LoG"]  # Filters to apply: Original (unfiltered) and Laplacian of Gaussian (LoG)
    - settings: {
          "binWidth": 25,       # Bin width for discretization of image intensity values
          "wavelet": 'db2',     # Wavelet type for wavelet transformation (e.g., Daubechies 2)
          "sigma": [0.5, 2.5]   # Sigma values for the Laplacian of Gaussian filter
      }

# Scikit-learn pipeline configuration
sklearn:
  transformers:  # List of data transformers to apply before modeling
    - name: StandardScaler  # Standardizes features by removing the mean and scaling to unit variance
      params: {}            # No additional parameters for StandardScaler
    - name: OneHotEncoder   # Encodes categorical features as one-hot numeric arrays
      params:
        handle_unknown: ignore  # Ignore unknown categories during encoding

  model:  # Machine learning model configuration
    name: KNeighborsClassifier  # Algorithm to use: K-Nearest Neighbors Classifier
    params:
      weights: 'distance'       # Weight function used in prediction (distance-based weighting)
      leaf_size: 1              # Leaf size for the KDTree or BallTree algorithm

  selectionParams:  # Hyperparameter tuning settings
    enable: True                # Enable hyperparameter optimization
    name: GridSearchCV          # Use Grid Search for hyperparameter tuning
    params:
      cv: 5                     # Number of cross-validation folds
      verbose: 3                # Verbosity level (higher = more detailed output)
      n_jobs: 20                # Number of parallel jobs to run
    param_grid:                 # Parameter grid for Grid Search
      KNeighborsClassifier__leaf_size: [1, 2, 20]  # Range of leaf_size values to test

  metrics:  # Metrics to evaluate model performance
    - name: accuracy_score      # Accuracy metric
      params: {}
    - name: precision_score     # Precision metric
      params:
        average: weighted       # Weighted averaging for multi-class classification
        zero_division: 0        # Handle cases where there are no predicted samples
    - name: balanced_accuracy_score  # Balanced accuracy metric (handles imbalanced datasets)
      params: {}

  pictures:  # Visualization settings
    - name: roc_curve           # Plot the Receiver Operating Characteristic (ROC) curve
    - name: confusion_matrix    # Plot the Confusion Matrix
```

## Configuration Breakdown

1. transformers: List of data preprocessing steps. Examples include StandardScaler and OneHotEncoder. Parameters can be customized for each transformer.
2. model: Defines the model to be used (e.g., KNeighborsClassifier) and its hyperparameters.
3. selectionParams: Optional grid search (or other selection models) for hyperparameter tuning. GridSearchCV is enabled in this example with cross-validation (cv), verbosity, and parallelization (n_jobs).
4. metrics: List of metrics used to evaluate the model's performance. For example, accuracy_score and precision_score.

## Report Generation

Once the model is trained, a report will be generated in the specified format. The available formats are:

    HTML: Generates an HTML report with metrics and graphs.
    PDF: Generates a PDF report with metrics and graphs.
    DOCX: Generates a DOCX report with metrics and graphs.

The report will be saved in a directory named reports, with a timestamp in the file name to ensure uniqueness.

## Example Report Output

The report includes:

- Training time
- Model metrics (e.g., accuracy, precision)
- Visualizations such as confusion matrix (for classification tasks) - _there may be bugs in development_
- ROC curve (for binary classification tasks) - _there may be bugs in development_

#### _This version of the library is in testing and will be developed and improved in the future. In subsequent versions it is planned to add new functions, improve data processing and expand reporting capabilities_
