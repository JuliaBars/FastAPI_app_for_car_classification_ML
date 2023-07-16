import sys

from fastapi import APIRouter, File, status
from fastapi.responses import StreamingResponse
from loguru import logger

import bll
from models.sample_model.labels import cars_models
from logger_cfg import config


logger.configure(**config)


router = APIRouter(
    prefix="",
    tags=["classify"],
)


@router.get("/available_models", status_code=status.HTTP_200_OK)
@logger.catch()
def get_all_models():
    '''To check the list of cars' models, avalable for recognition.'''
    return cars_models


@router.post("/car_brand_model_classification")
@logger.catch()
def car_brand_model_classification(file: bytes = File(...)):
    '''Object Detection from an image plot bbox on image. Using Yolo8.'''
    input_image = bll.get_image_from_bytes(file)
    predict = bll.predict_brand_and_model(input_image)

    return StreamingResponse(
        content=bll.get_bytes_from_image(predict), media_type="image/jpeg")


@router.post("/car_model_segment_and_crop")
@logger.catch()
def car_model_segment_and_crop(file: bytes = File(...)):
    '''Object Detection from an image plot bbox on image. Using Yolo8.'''
    input_image = bll.get_image_from_bytes(file)
    predict = bll.extract_segment_image(input_image)

    return StreamingResponse(
        content=bll.get_bytes_from_image(predict), media_type="image/jpeg")
