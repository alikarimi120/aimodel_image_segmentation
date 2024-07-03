
model_list=[
"yolov8n",
"yolov8s",
"yolov8m",
"yolov8l",
"yolov8x",
"yolov5nu",
"yolov5su",
"yolov5mu",
"yolov5lu",
"yolov5xu",
"yolov5n6u",
"yolov5s6u",
"yolov5m6u",
"yolov5l6u",
"yolov5x6u",
"yolov6-n",
"yolov6-s",
"yolov6-m",
"yolov6-l",
"yolov6-l6",
"yolov9t",
"yolov9s",
"yolov9m",
"yolov9c",
"yolov9e",
"yolov10n",
"yolov10s",
"yolov10m",
"yolov10l",
"yolov10x",
"rtdetr-l",
"rtdetr-x",
"yolov8s-worldv2",
"yolov8m-worldv2",
"yolov8l-worldv2",
"yolov8x-worldv2"
]


f = open("config/default_config.json", "r")
default_config = f.read()
for i in range(0,len(model_list)):
    model_config = default_config.replace("model_name_value",model_list[i]+".pt")
    model_config = model_config.replace("data_path_value", "../../my_data/soccer_ball/data.yaml")
    model_config = model_config.replace("root_output_path_value","../../detection_results_"+model_list[i])
    model_config = model_config.replace("gpu_device_value","1")
    
    f = open("config/"+model_list[i]+".json", "w")
    f.write(model_config)
    f.close()


