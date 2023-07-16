from PIL import Image, ImageDraw, ImageFont
import io
import pandas as pd
import numpy as np

import cv2
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator, colors
import torch
import torchvision.transforms as transforms

from models.sample_model.labels import cars_models

# Initialize the models
model_sample_model = YOLO("./models/sample_model/yolov8n.pt")


def get_image_from_bytes(binary_image: bytes) -> Image:
    """ Convert image from bytes to PIL RGB format. """
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image


def get_bytes_from_image(image: Image) -> bytes:
    """ Convert PIL image to Bytes. """
    return_image = io.BytesIO()
    image.save(return_image, format='JPEG', quality=100)
    return_image.seek(0)
    return return_image


def transform_predict_to_df(results: list, labeles_dict: dict) -> pd.DataFrame:
    """
    Transform predict from yolov8 (torch.Tensor) to pandas DataFrame.   
    """
    predict_bbox = pd.DataFrame(
        results[0].to("cpu").numpy().boxes.xyxy, columns=[
            'xmin', 'ymin', 'xmax', 'ymax']
        )
    predict_bbox['confidence'] = results[0].to("cpu").numpy().boxes.conf
    predict_bbox['class'] = (
        results[0].to("cpu").numpy().boxes.cls).astype(int)
    predict_bbox['name'] = predict_bbox["class"].replace(labeles_dict)
    return predict_bbox


def get_model_predict(
        model: YOLO, input_image: Image,
        save: bool = False, image_size: int = 1248,
        conf: float = 0.5, augment: bool = False) -> pd.DataFrame:
    """ Get the predictions of a model on an input image. """
    predictions = model.predict(
                        imgsz=image_size,
                        source=input_image,
                        conf=conf,
                        save=save,
                        augment=augment,
                        flipud=0.0,
                        fliplr=0.0,
                        mosaic=0.0,
                        )
    predictions = transform_predict_to_df(predictions, model.model.names)
    return predictions


# ----------------------- BBOX Func -----------------------

def add_bboxs_on_img(image: Image, predict: pd.DataFrame()) -> Image:
    """ Add a bounding box on the image. """
    annotator = Annotator(np.array(image))
    predict = predict.sort_values(by=['xmin'], ascending=True)
    for i, row in predict.iterrows():
        text = f"{row['name']}: {int(row['confidence']*100)}%"
        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        annotator.box_label(bbox, text, color=colors(row['class'], True))
    return Image.fromarray(annotator.result())


def detect_sample_model(input_image: Image) -> pd.DataFrame:
    """ Predict from sample_model. Base on YoloV8. """
    predict = get_model_predict(
        model=model_sample_model,
        input_image=input_image,
        save=False,
        image_size=640,
        augment=False,
        conf=0.5,
    )
    return predict


#------------------------MODELS CLASSIFICATION------------------------

class YOLOSegmentation:
    """
    Useful class to get bboxes, classes, segmentations, scores in correct
    format to pass them to cv2 image processed functions.
    """
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img):
        height, width, channels = img.shape

        results = self.model.predict(source=img, save=False, save_txt=False)
        result = results[0]
        segmentation_contours_idx = []
        for seg in result.masks.xyn:
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)
            segmentation_contours_idx.append(segment)

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
        return bboxes, class_ids, segmentation_contours_idx, scores


#model for segmentation
model_for_classify = YOLOSegmentation("./models/sample_model/yolov8m-seg.pt")

# for model classification project we will segment only cars, busses, truckes
classes_ids = [
    2,  # Car
    7,  # Truck
    5,  # Bus
]


def extract_segment_image(img, segmentator=model_for_classify) -> Image:
    """ Extract segmented and cropped car, truck, bus and return as PIL"""
    open_cv_image = np.array(img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    bboxes, classes, segmentations, scores = segmentator.detect(open_cv_image)
    for item, object_class in enumerate(classes):
        if object_class not in classes_ids:
            continue
        else:
            break
    points = np.array(segmentations[item])
    mask = np.zeros(open_cv_image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    img_segm = cv2.bitwise_and(open_cv_image, open_cv_image, mask=mask)
    (x, y, x2, y2) = bboxes[item]
    crop_img = img_segm[y:y2, x:x2]
    im_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    PIL_im = Image.fromarray(im_rgb)
    return PIL_im


def load_model():
    """ Load and evaluate saved model. """
    model = torch.load(
        './models/sample_model/model_mob_netv3_79_perc.pth',
        map_location=torch.device('cpu'))
    model.eval()
    return model


def image_to_tensor(cv2_img: np.ndarray) ->  torch.Tensor:
    """ Converting cv2 output to torch tensor. """
    image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    tensor_img = transform(image)
    tensor_img = tensor_img.unsqueeze(0)
    return tensor_img


def draw_predictions_on_image(img: Image, label: str) -> Image:
    img_1 = ImageDraw.Draw(img)
    myFont = ImageFont.truetype('./utils/arial_bolditalicmt.ttf', size=25)
    img_1.text((10, 10), label, font=myFont, fill=(255, 0, 0))
    return img


def predict_brand_and_model(img, segmentator=model_for_classify):
    open_cv_image = np.array(img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    bboxes, classes, segmentations, scores = segmentator.detect(open_cv_image)
    for item, object_class in enumerate(classes):
        if object_class not in classes_ids:
            continue
        else:
            break
    points = np.array(segmentations[item])
    mask = np.zeros(open_cv_image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    img_segm = cv2.bitwise_and(open_cv_image, open_cv_image, mask=mask)
    (x, y, x2, y2) = bboxes[item]
    crop_img = img_segm[y:y2, x:x2]
    model = load_model()
    tensor_img = image_to_tensor(crop_img)
    predict = model(tensor_img)
    answer = predict.argmax(-1)
    name = cars_models.get(answer.item()).split('_')
    name = f'Brand: {name[0]}, model: {name[1]}'
    im_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    # PIL_im = Image.fromarray(im_rgb)
    PIL_im = draw_predictions_on_image(img, name)
    return PIL_im
