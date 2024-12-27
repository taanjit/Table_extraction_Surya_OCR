from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
import cv2
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import pandas as pd

langs = ["en"] # Replace with your languages - optional but recommended
det_processor, det_model = load_det_processor(), load_det_model()
rec_model, rec_processor = load_rec_model(), load_rec_processor()


class FileReader:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = []
        self.excel_files = []

    def read_files(self):
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".png"):
                    file_path = os.path.join(root, file)
                    # img = Image.open(file_path)
                    self.image_files.append(file_path)
                elif file.endswith(".xlsx"):
                    file_path = os.path.join(root, file)
                    # df = pd.read_excel(file_path)
                    self.excel_files.append(file_path)
        self.image_files.sort()

    def get_image_files(self):
        return self.image_files

    def get_excel_files(self):
        return self.excel_files
    
def table_structure(file_path):
    directory, filename = os.path.split(file_path)
    print(directory, filename)
    image = Image.open(file_path)

    predictions = run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor)
    # print(predictions)
    texts = [(l.text) for p in predictions for l in p.text_lines]
    boxes = [(l.bbox) for p in predictions for l in p.text_lines]
    probabilities = [(l.confidence) for p in predictions for l in p.text_lines]

    
    # for text,box,probability in zip(texts,boxes,probabilities):
    #     print(text,box,probability)

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





if __name__=="__main__":
    #function calls
    # Example usage:
    reader = FileReader("pages")
    reader.read_files()

    image_files = reader.get_image_files()
    print(image_files)
    # excel_files = reader.get_excel_files()
    # print(excel_files)

    for i,image_file in enumerate(image_files):
        boxes,texts,probabilities=table_structure(image_file)
        image_boxes=cv2.imread(image_file)
        image_height=image_boxes.shape[0]
        image_width=image_boxes.shape[1]
        for bbox,text in zip(boxes,texts):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red rectangle, thickness 2
        cv2.imwrite(f'Detections_image_boxes_{i}.jpg',image_boxes)


        # Reconstructions           

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

            cv2.rectangle(im,(x_h,y_h),(x_h+width_h,y_h+height_h),(0,255,0),1)
            cv2.rectangle(im,(x_v,y_v),(x_v+width_v,y_v+height_v),(255,0,0),1)
        cv2.imwrite(f'Detections_H_V_boxes_{i}.jpg',im)

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

        for val in horiz_lines:
            cv2.rectangle(im_nms, (int(horiz_boxes[val][0]),int(horiz_boxes[val][1])), (int(horiz_boxes[val][2]),int(horiz_boxes[val][3])),(0,0,255),1)
        cv2.imwrite(f'im_nms_H_{i}.jpg',im_nms)

        vert_out = tf.image.non_max_suppression(
            vert_boxes,
            probabilities,
            max_output_size = 100000,
            iou_threshold=0.1,
            score_threshold=float('-inf'),
            name=None
        )
        vert_lines = np.sort(np.array(vert_out))
        for val in vert_lines:
            cv2.rectangle(im_nms, (int(vert_boxes[val][0]),int(vert_boxes[val][1])), (int(vert_boxes[val][2]),int(vert_boxes[val][3])),(255,0,0),1)
        cv2.imwrite(f'im_nms_V_{i}.jpg',im_nms)

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
                    # print(b,box,'\n ************',x1, y1, x2, y2,'\n')
                    the_box = [x1,y1,x2,y2]
                    if(iou(resultant,the_box)>0.05):
                        out_array[i][j] = texts[b]
        print(out_array)
        import pandas as pd
        pd.DataFrame(out_array).to_csv(f'CSV_dataout/sample{i}.csv')
        print('Finished')