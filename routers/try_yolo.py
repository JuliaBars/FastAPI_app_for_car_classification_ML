from fastapi import APIRouter, File, status
from fastapi.responses import StreamingResponse

from loguru import logger
from logger_cfg import config

import bll
from models.sample_model.yolo_labels import yolo_classes


logger.configure(**config)

router = APIRouter(
    prefix="",
    tags=["try_yolo8"],
)


@router.get("/yolo_classes", status_code=status.HTTP_200_OK)
@logger.catch()
def get_yolo_classes():
    '''To check the list of cars' models,avalable for recognition.'''
    return yolo_classes


@router.post("/object_detection_with_yolo")
@logger.catch()
def object_detection_with_yolo(file: bytes = File(...)):
    '''Object Detection from an image plot bbox on image. Using Yolo8.'''
    input_image = bll.get_image_from_bytes(file)
    predict = bll.detect_sample_model(input_image)
    final_image = bll.add_bboxs_on_img(image=input_image, predict=predict)

    return StreamingResponse(
        content=bll.get_bytes_from_image(final_image), media_type="image/jpeg")
