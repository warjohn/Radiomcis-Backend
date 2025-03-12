import os
import time
import csv
import SimpleITK as sitk
import numpy as np

from joblib import Parallel, delayed
from radiomics import featureextractor


class Radiomcis():

    def __init__(self, csv_file_path, output_csv_path, n_jobs = 1, new_spacing=None):
        self.csv_file_path = csv_file_path
        self.output_csv_path = output_csv_path
        self.variant = 1
        self.n_jobs = n_jobs
        self.new_spacing = new_spacing

    def extract_features_from_csv(self):
        start_time = time.time()

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

        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

        extractor.enableImageTypeByName('Original')
        extractor.enableImageTypeByName('LoG')
        extractor.enableImageTypeByName('Wavelet')
        extractor.enableImageTypeByName('Logarithm')
        extractor.enableImageTypeByName('Square')
        extractor.enableImageTypeByName('Gradient')
        extractor.enableImageTypeByName('Exponential')
        extractor.enableImageTypeByName('LBP2D')
        extractor.enableImageTypeByName('LBP3D')
        extractor.enableImageTypeByName('SquareRoot')
        extractor.enableAllFeatures()
        extractor.enableFeaturesByName(
            firstorder=['Energy', 'TotalEnergy', 'Entropy', 'Minimum', '10Percentile', '90Percentile', 'Maximum',
                        'Mean',
                        'Median', 'InterquartileRange', 'Range', 'MeanAbsoluteDeviation', 'RobustMeanAbsoluteDeviation',
                        'RootMeanSquared', 'StandardDeviation', 'Skewness', 'Kurtosis', 'Variance', 'Uniformity'])
        extractor.enableFeaturesByName(
            shape=['VoxelVolume', 'MeshVolume', 'SurfaceArea', 'SurfaceVolumeRatio', 'Compactness1', 'Compactness2',
                   'Sphericity', 'SphericalDisproportion', 'Maximum3DDiameter', 'Maximum2DDiameterSlice',
                   'Maximum2DDiameterColumn', 'Maximum2DDiameterRow', 'MajorAxisLength', 'MinorAxisLength',
                   'LeastAxisLength', 'Elongation', 'Flatness'])

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
        print(f"Processing completed in {end_time - start_time} seconds.")

    def process_image_2D_rgb(self, idx, row, extractor, output_csv_path, processed_files, new_spacing):
        # try:
        image_path = row[0]
        mask_path = row[1]

        _, name = os.path.dirname(image_path)

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

        print("shape image", sitk.GetArrayFromImage(image).shape)
        print("shape mask", sitk.GetArrayFromImage(mask).shape)

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