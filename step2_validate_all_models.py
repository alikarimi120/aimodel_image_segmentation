import os
import pandas as pd


def read_logs(root_folder):
    data = []
    for network in os.listdir(root_folder):
        network_path = os.path.join(root_folder, network)
        if os.path.isdir(network_path):
            log_path = os.path.join(network_path, 'logs', 'ic_classification_report.csv')
            param_flops_path = os.path.join(network_path, 'logs', 'ic_classification_report2.csv')
            if os.path.isfile(log_path) and os.path.isfile(param_flops_path):
                df_log = pd.read_csv(log_path)
                df_params_flops = pd.read_csv(param_flops_path, header=None)
                

                flops = df_params_flops.iloc[1, 1]
                parameters = df_params_flops.iloc[2, 1]
                
                df_log['network'] = network
                df_log['parameters'] = parameters
                df_log['FLOPs'] = flops
                
                data.append(df_log)
    return pd.concat(data, ignore_index=True)

root_folder = '../aimodel_image_classificationv2'


data = read_logs(root_folder)


print(data)
data.to_csv('comparison_report.csv', index=False)

