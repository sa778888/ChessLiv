import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import argparse

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12  # Your number of classes
    cfg.MODEL.WEIGHTS = "model_final.pth"
    cfg.MODEL.DEVICE = "cpu"  # Use CPU for inference
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Set a threshold for detection confidence
    return cfg

def detect_chess_pieces(predictor, frame):
    outputs = predictor(frame)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    return boxes, classes, scores

def map_to_chessboard(frame, boxes, classes):
    board_state = [[""] * 8 for _ in range(8)]
    height, width = frame.shape[:2]
    for box, cls in zip(boxes, classes):
        x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        row = int(8 * y / height)
        col = int(8 * x / width)
        if 0 <= row < 8 and 0 <= col < 8:
            board_state[row][col] = class_names[cls]
    return board_state

def generate_fen(board_state):
    fen = ""
    for row in board_state:
        empty = 0
        for piece in row:
            if piece == "":
                empty += 1
            else:
                if empty > 0:
                    fen += str(empty)
                    empty = 0
                fen += piece
        if empty > 0:
            fen += str(empty)
        fen += "/"
    fen = fen[:-1] + " w - - 0 1"  # Assuming it's white's turn to move
    return fen

def process_frame(frame, predictor):
    boxes, classes, scores = detect_chess_pieces(predictor, frame)
    board_state = map_to_chessboard(frame, boxes, classes)
    fen = generate_fen(board_state)

    # Draw bounding boxes and FEN on the frame
    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_names[cls]}: {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"FEN: {fen}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame, fen

def process_input(input_path, predictor):
    if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Process image
        frame = cv2.imread(input_path)
        processed_frame, fen = process_frame(frame, predictor)
        cv2.imshow("Chess Board Detection", processed_frame)
        print(f"FEN: {fen}")
        cv2.waitKey(0)
    elif input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        # Process video
        cap = cv2.VideoCapture(input_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame, fen = process_frame(frame, predictor)
            cv2.imshow("Chess Board Detection", processed_frame)
            print(f"Current FEN: {fen}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
    else:
        print("Unsupported file format. Please use an image or video file.")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    

    # Define class names (adjust based on your training)
    class_names = {
        0: "K", 1: "Q", 2: "R", 3: "B", 4: "N", 5: "P",
        6: "k", 7: "q", 8: "r", 9: "b", 10: "n", 11: "p"
    }

    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)
    arg="d2.jpeg"
    process_input(arg, predictor)
