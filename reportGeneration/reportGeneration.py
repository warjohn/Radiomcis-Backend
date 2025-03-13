import os
import time
import shutil
import datetime
import sklearn as sk

from reportGeneration.loader.LoaderConfig import LoaderConfig
from reportGeneration.metrics.calculateMetrics import CalculateMetrics
from reportGeneration.graphs.drawgraph import Draw
from reportGeneration.formats.PDF_format import GeneratePdfReport
from reportGeneration.formats.DOCS_format import GenerateDocsReport
from reportGeneration.llm.llm import ChatModel
from reportGeneration.radiomics.radiomics import Radiomcis
from reportGeneration.dataProcess.dataProcess import Data


class SklearnReportGenerator(sk.base.BaseEstimator):
    """
    :param
        config_file - yaml configuration file, which contains full pipeline
        output_format - format fot output format

    """
    def __init__(self, config_file, output_format = "HTML"):
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.report_dir = "reports"
        self.testDir = f"{self.report_dir}/test_{self.timestamp}"
        self.config_file = config_file
        self.format = output_format # default - HTML
        self.model = None
        self.metrics = []
        self.pipeline = None

    def _load_config(self, config_file):
        loader = LoaderConfig(config_file)
        self.pipeline = loader.generateWay()
        self.model = loader.model
        self.metrics = loader.metrics
        self.pictures = loader.pictures

    def _loadRadiomics(self):
        loader = LoaderConfig(self.config_file)
        self.path2inputFileDict = loader.loadInputFile()
        self.filters = loader.loadRadiomics()
        self.settings = loader.loadRadiomicsSettings()

    def _loadChatModel(self, config_file):
        loader = LoaderConfig(config_file)
        self.llm_model, self.lang, self.base_url = loader.loadLLM()

    def __create_root_dir(self):
        dirs = [self.testDir, f"{self.testDir}/radiomics", f"{self.testDir}/features"]
        for directory in dirs:
            os.makedirs(directory, exist_ok=False)

    def extractFeatures(self, n_jobs, new_spacing=None):
        self.__create_root_dir()
        self._loadRadiomics()

        for key, value in self.path2inputFileDict.items():
            radiomics_input = os.path.join(self.testDir, "radiomics", os.path.basename(key))
            features_output = os.path.join(self.testDir, "features", f"{os.path.basename(key).split('.')[0]}-original.csv")
            data_pros = os.path.join(self.testDir, "features", f"{os.path.basename(key).split('.')[0]}-process.csv")
            try:
                shutil.copy(key, radiomics_input)
                shutil.move(self.config_file, self.testDir)
            except (shutil.Error, FileNotFoundError) as e:
                raise RuntimeError(f"Error: {e}")
            radiomics = Radiomcis(
                csv_file_path=radiomics_input,
                output_csv_path=features_output,
                filters=self.filters,
                settings=self.settings,
                n_jobs=n_jobs,
                new_spacing=new_spacing
            )
            radiomics.extract_features_from_csv()
            data = Data(features_output, data_pros)
            self.df = data.pipeline(value)

    def fit(self, test_size = 0.3, random_state = 42, X = None, y = None, X_test=None, y_test=None, *args, **kwargs):
        self.new_path_config = os.path.join(self.testDir, os.path.basename(self.config_file))
        self._load_config(self.new_path_config)
        start_time = time.time()

        self.X_train, self.X_test, self.y_train, self.y_test = sk.model_selection.train_test_split(self.df.iloc[:, 1:-1], self.df.iloc[:, -1], test_size=test_size, random_state=random_state)

        self.pipeline.fit(self.X_train, self.y_train, *args, **kwargs)

        end_time = time.time()
        self.train_time = end_time - start_time
        return self

    def predict(self):
        y_pred = self.pipeline.predict(self.X_test)
        self._generate_report(y_pred)
        return y_pred

    def _generate_report(self, y_pred):
        report_file = f"{self.testDir}/report_{self.model.__class__.__name__}_{self.timestamp}.html"
        task_type = "classification" if sk.base.is_classifier(self.model) else "regression"

        calculater = CalculateMetrics(self.metrics, task_type)
        metrics_result = calculater.calculate(self.y_test, y_pred)
        draw = Draw(rootdir=self.testDir)

        if "confusion_matrix" in self.pictures and task_type == "classification":
            draw.create_confusion_matrix(self.y_test, y_pred)
        if "roc_curve" in self.pictures and task_type == "classification" and len(set(self.y_test)) == 2:
            draw.create_roc_auc(self.y_test, y_pred)

        self._loadChatModel(self.new_path_config)
        chat = ChatModel(self.base_url, self.lang, self.llm_model, str(self.pipeline) + str(metrics_result))
        report = chat.qa()
        if self.format == "PDF":
            GeneratePdfReport().generatereport(report_file, report)
        elif self.format == "DOCS":
            GenerateDocsReport().generateReport(report_file, report)

