import os
import pandas as pd

class Data:

    def __init__(self, input_file_path : str, output_file_path : str):
        self.input_file_path = input_file_path
        self.output_file_path = self.__checks(output_file_path)

    def __checks(self, output_file_path):
        if os.path.exists(output_file_path):
            return output_file_path
        else:
            with open(output_file_path, "w"):
                pass
            return output_file_path

    def __delete_columns(self, df):
        end_idx_df_all = df.columns.get_loc("diagnostics_Mask-original_CenterOfMass")
        return df.drop(columns=df.columns[1:end_idx_df_all + 1])

    def __save_table(self, df):
        df.to_csv(self.output_file_path, index= False)

    def pipeline(self, target_variable) -> pd.DataFrame:
        df = pd.read_csv(self.input_file_path)
        df = self.__delete_columns(df)
        df['outcome'] = target_variable
        self.__save_table(df)
        return df
