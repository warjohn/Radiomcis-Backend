from reportGeneration.reportGeneration import SklearnReportGenerator
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

import uvicorn
import yaml
import os
import traceback

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
        report_generator = SklearnReportGenerator(config_file=file_path, output_format="PDF")
        report_generator.extract(n_jobs=40)


        # Возвращаем результат: имя файла, путь к сохраненному файлу,
        # содержимое как строка, распарсенный YAML и результат обработки
        return JSONResponse(
            content={
                "filename": file.filename,
                "file_content": file_content_str,  # Содержимое файла в виде строки
                "parsed_yaml": yaml_data,  # Распарсенный YAML в виде словаря
                "processing_result": "processing_result"  # Результат вашей обработки
            },
            status_code=200
        )
    except Exception as e:
        # В случае ошибки возвращаем сообщение об ошибке
        error_traceback = traceback.format_exc()
        print(error_traceback)  # Выводим трассировку в консоль (для отладки)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    # Запускаем сервер Uvicorn напрямую из скрипта
    uvicorn.run(app, host="127.0.0.1", port=5555)