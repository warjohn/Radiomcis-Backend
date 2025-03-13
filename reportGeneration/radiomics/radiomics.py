import logging
import os
import time
import csv
import SimpleITK as sitk
import numpy as np

from joblib import Parallel, delayed
import radiomics
from radiomics import featureextractor
from radiomics.featureextractor import RadiomicsFeatureExtractor

radiomics.logger.setLevel(logging.CRITICAL)

class Radiomcis():

    def __init__(self, csv_file_path : str, output_csv_path : str, filters : list, settings : dict, n_jobs = 1, new_spacing=None):
        self.csv_file_path = csv_file_path
        self.output_csv_path = self.__checks(output_csv_path)
        self.filters = self.__checkFilters(filters)
        self.settings = self.__checkSettings(settings)
        self.variant = 1
        self.n_jobs = n_jobs
        self.new_spacing = new_spacing

    def __checks(self, output_csv_path):
        if not os.path.exists(output_csv_path):
            with open(output_csv_path, 'w'):
                pass
            return output_csv_path
        else:
            return output_csv_path

    def __checkFilters(self, filters : list):
        if len(filters) == 0:
            return filters.append("Original")
        else:
            return filters

    def __checkSettings(self, settings : dict):
        if not settings:
            settings = {}
            settings['binWidth'] = 25
            settings['wavelet'] = 'db2'
            settings['sigma'] = [0.5, 2.5]
            settings['alpha'] = 0.25
            settings['Interpolator'] = sitk.sitkBSpline
            settings['voxelArrayShift'] = 1000
            settings['normalize'] = True
            settings['normalizeScale'] = 100
            settings['label'] = 255
            settings['approximation'] = True
            return settings
        else:
            return settings

    def __initFilters(self, extractor : RadiomicsFeatureExtractor) -> RadiomicsFeatureExtractor:
        for i in self.filters:
            extractor.enableImageTypeByName(str(i))
        return extractor

    def extract_features_from_csv(self):
        start_time = time.time()

        extractor = featureextractor.RadiomicsFeatureExtractor(**self.settings)
        extractor = self.__initFilters(extractor)
        extractor.enableAllFeatures()

        processed_files = set()

        with open(self.output_csv_path, newline='') as result_csvfile:
            reader = csv.DictReader(result_csvfile)
            for row in reader:
                processed_files.add(row['image_path'])

        with open(self.csv_file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            total_images = sum(1 for row in reader)

        with open(self.csv_file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)

            if self.variant == 1:
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.process_image_2D_rgb)(idx, row, extractor, self.output_csv_path, processed_files,
                                                       self.new_spacing) for idx, row in enumerate(reader, 1))

        end_time = time.time()
        print(f"Radiomic processing completed in {end_time - start_time} seconds.")

    def process_image_2D_rgb(self, idx, row, extractor, output_csv_path, processed_files, new_spacing):
        # try:
        image_path = row[0]
        mask_path = row[1]

        name = os.path.dirname(image_path)

        if image_path in processed_files:
            return

        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)

        image = sitk.GetArrayFromImage(image)
        image = image.T
        image = np.expand_dims(image, axis=-1)

        mask = sitk.GetArrayFromImage(mask)
        mask = np.transpose(mask, (2, 1, 0))

        results = {'image_name': name,
                   'image_number': idx}

        image = sitk.GetImageFromArray(image)
        mask = sitk.GetImageFromArray(mask)

        # print("shape image", sitk.GetArrayFromImage(image).shape)
        # print("shape mask", sitk.GetArrayFromImage(mask).shape)

        result = 0
        try:
            result = extractor.execute(image, mask, 1, voxelBased=False)
        except ValueError as e:
            print(f"{e} - mask name - {mask_path} - max number - {sitk.GetArrayFromImage(mask).max()}")

        with open(output_csv_path, 'a', newline='') as result_csvfile:
            if result:
                results.update(result)

                writer = csv.DictWriter(result_csvfile, fieldnames=results.keys())
                if result_csvfile.tell() == 0:
                    writer.writeheader()
                writer.writerow(results)