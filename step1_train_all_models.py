
import os
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


for i in range(0,len(model_list)):
	os.system("cd train && python3 train_step1.py ../config/"+model_list[i]+".json")
	os.system("cd train && python3 train_step2.py ../config/"+model_list[i]+".json >> output.txt")
	os.system("cd train && python3 train_step3.py ../config/"+model_list[i]+".json")
	os.system("cd train && python3 train_step4.py ../config/"+model_list[i]+".json")


