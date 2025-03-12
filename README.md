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
llmModel:
  name: 'llama3.1'
  lang: 'RU'
radiomics:
  filters: {
          "origin" : [1.0],
          "origin_1" : [2.0]
          }
sklearn:
  transformers:
    - name: StandardScaler
      params: {}
    - name: OneHotEncoder
      params:
        handle_unknown: ignore
  model:
    name: KNeighborsClassifier
    params:
      weights: 'distance'
      leaf_size: 1
  selectionParams:
    enable: True
    name: GridSearchCV
    params:
      cv: 5
      verbose: 3
      n_jobs: 20
    param_grid:
      KNeighborsClassifier__leaf_size: [1, 2, 20]
  metrics:
    - name: accuracy_score
      params: {}
    - name: precision_score
      params:
        average: weighted
        zero_division: 0
    - name: balanced_accuracy_score
      params: {}
  pictures:
    - name: roc_curve
    - name: confusion_matrix
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
