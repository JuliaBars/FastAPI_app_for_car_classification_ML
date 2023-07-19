# Car models classification

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![Colab](https://img.shields.io/badge/google_colaboratory-F9AB00?style=for-the-badge&logo=google-colab&logoColor=white)

For now it will classify only 75 models, check them on **/available_models**, but later will have more than 3K models.

To classify cars go to **/car_brand_model_classification** and download jpg image, make sure this car model is in list, no matter if there are another objects on the picture, but car should be the main.

After execute you will get the answer in Response Body. 

---

The model is very light and works fast, but the accuracy sometimes is not extremly high. Anyway hope you will like it.
I used mobelinet_v3 with transfer learning on PyTorch.
You can find details in my repository: [link](https://github.com/JuliaBars/cars_model_classification)

---

You can also try Yolo8 functionality on **/img_object_detection_to_img**.
Go to **/yolo_classes** to check all what Yolo can classify.
On **/car_model_segment_and_crop** you can see how Yolo and OpenCV work together, car segmentation and cropping helped a lot to improve accuracy of model.

### Examples
![classification](https://github.com/JuliaBars/JuliaBars/assets/107411145/d3b6e28b-13d1-464c-b5d2-56cec46aaf8e)
<details>
<summary>For more examples click below</summary>
  
![segment and crop](https://github.com/JuliaBars/JuliaBars/assets/107411145/767247d2-de11-47b4-98bb-ba07443d896b)
  
![detection](https://github.com/JuliaBars/JuliaBars/assets/107411145/be2d4d31-2e8d-4ba9-a16d-1200344bd7a4)

![2023-07-16_22-33-30](https://github.com/JuliaBars/FastAPI_app_for_car_classification_ML/assets/107411145/a08026d4-5f85-4efa-896a-bd398395c486)
</details>


---
### Getting Started

You have two options to start the application: using Docker or locally on your machine.

#### Using Docker
Start the application with the following command:
```
docker-compose up
```

#### Locally
To start the application locally, follow these steps:

1. Install the required packages:

```
pip install -r requirements.txt
```
2. Start the application:
```
uvicorn main:app --reload --port 8001
``` 

---
### Tests
This repository contains functional tests for a program to ensure the proper operation of the service.
To get started with the testing process, you first need to set up the necessary environment. 
This can be achieved by either installing the required packages or by running the Docker container.
```
pytest -v --disable-warnings
```
---

# Many thanks

The baseline was cloned from this repository: [Alex-Lekov](https://github.com/Alex-Lekov/yolov8-fastapi/blob/main/README.md)
