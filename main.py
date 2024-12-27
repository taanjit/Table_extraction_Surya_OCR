from typing import Annotated
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse

import os
from PIL import Image
import cv2

from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

from io import BytesIO
from io import StringIO

import tensorflow as tf
import numpy as np


app = FastAPI()

langs = ["en"] # Replace with your languages - optional but recommended
det_processor, det_model = load_det_processor(), load_det_model()
rec_model, rec_processor = load_rec_model(), load_rec_processor()


def table_structure(contents):
    image = Image.open(BytesIO(contents))
    predictions = run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor)
    # print(predictions)
    texts = [(l.text) for p in predictions for l in p.text_lines]
    boxes = [(l.bbox) for p in predictions for l in p.text_lines]
    probabilities = [(l.confidence) for p in predictions for l in p.text_lines]

    
    for text,box,probability in zip(texts,boxes,probabilities):
        print(text,box,probability)

    return boxes,texts,probabilities

def intersection(box_1, box_2):
    return [box_2[0], box_1[1],box_2[2], box_1[3]]

def iou(box_1, box_2):

    x_1 = max(box_1[0], box_2[0])
    y_1 = max(box_1[1], box_2[1])
    x_2 = min(box_1[2], box_2[2])
    y_2 = min(box_1[3], box_2[3])

    inter = abs(max((x_2 - x_1, 0)) * max((y_2 - y_1), 0))
    if inter == 0:
        return 0

    box_1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
    box_2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))

    return inter / float(box_1_area + box_2_area - inter)

def main(contents):
    boxes,texts,probabilities=table_structure(contents)
    # Convert the bytes to a NumPy array
    nparr = np.frombuffer(contents, np.uint8)

    # Decode the NumPy array into an image using cv2.imdecode
    image_boxes = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED) # Or cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, etc.
    image_height=image_boxes.shape[0]
    image_width=image_boxes.shape[1]

    im=image_boxes.copy()
    vert_boxes,horiz_boxes=[],[]

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x_h,x_v=0,int(x1)
        y_h,y_v=int(y1),0
        
        width_h,width_v=image_width,int(x2-x1)
        height_h,height_v=int(y2-y1),image_height

        horiz_boxes.append([x_h,y_h,x_h+width_h,y_h+height_h])
        vert_boxes.append([x_v,y_v,x_v+width_v,y_v+height_v])  


    horiz_out = tf.image.non_max_suppression(
        horiz_boxes,
        probabilities,
        max_output_size = 1000,
        iou_threshold=0.1,
        score_threshold=float('-inf'),
        name=None
    )
    horiz_lines = np.sort(np.array(horiz_out))
    im_nms = image_boxes.copy()  

    vert_out = tf.image.non_max_suppression(
            vert_boxes,
            probabilities,
            max_output_size = 100000,
            iou_threshold=0.1,
            score_threshold=float('-inf'),
            name=None
        )
    vert_lines = np.sort(np.array(vert_out))    

    out_array = [["" for i in range(len(vert_lines))] for j in range(len(horiz_lines))]
    unordered_boxes = []
    for i in vert_lines:
        unordered_boxes.append(vert_boxes[i][0])
    ordered_boxes = np.argsort(unordered_boxes)

    for i in range(len(horiz_lines)):
        for j in range(len(vert_lines)):
            resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]] )

            for b in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[b])
                the_box = [x1,y1,x2,y2]
                if(iou(resultant,the_box)>0.05):
                    out_array[i][j] = texts[b]


    import pandas as pd
    pd.DataFrame(out_array).to_csv(f'table.csv')
    print('Finished')

@app.post("/uploadfile/",tags=["Upload"])
async def create_upload_file(file: UploadFile):
    contents = await file.read()
    main(contents)
    # return FileResponse(filepath="table.csv", filename="table.csv", media_type="text/csv")
    try:
        filename = "table.csv"  # Construct the full filename
        filepath = os.path.join(".", filename)  # Construct the full path

        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="table not found.")

        return FileResponse(filepath, filename=filename, media_type="text/csv")

    except HTTPException as http_exc:
        raise http_exc # Re-raise HTTP exceptions to be handled by FastAPI
    except Exception as e:
        print(f"Error serving file: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while retrieving the table.")





