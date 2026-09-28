from http.server import HTTPServer, BaseHTTPRequestHandler
from abc import ABC, abstractmethod
import json
import math
import os
import sys
import cv2
import joblib
from PIL import Image
import torch
from torchvision import transforms
import re
import numpy as np
from color_face_map import color_face

sys.path.insert(0, os.path.abspath('solver'))
from kociemba_solver import KociembaCubeSolver

sys.path.insert(0, os.path.abspath('color_recognition'))
from color_cnn import ColorCNN


# Strategy Pattern: Base predictor interface
class ColorPredictor(ABC):
    """Abstract base class for color prediction strategies"""

    @abstractmethod
    def predict_pixel(self, patch, pixel_x, pixel_y):
        """Predict color at a pixel location"""
        pass

    @abstractmethod
    def get_name(self):
        """Return the model name"""
        pass


class KNNPredictor(ColorPredictor):
    """KNN-based color prediction"""

    def __init__(self, classifier_path, scaler_path, label_mapping_path):
        self.classifier = joblib.load(classifier_path)
        self.scaler = joblib.load(scaler_path)
        self.label_mapping = joblib.load(label_mapping_path)

    def predict_pixel(self, patch):
        """Predict color from image patch"""
        patch_pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        train_transforms = transforms.Compose(
            [transforms.Resize((20, 20)),
             transforms.ToTensor()])
        patch_tensor = train_transforms(patch_pil)
        patch_flat = patch_tensor.view(1, -1).numpy()

        patch_scaled = self.scaler.transform(patch_flat)
        prediction_idx = self.classifier.predict(patch_scaled)[0]
        return self.label_mapping[prediction_idx]

    def get_name(self):
        return "KNN"


class CNNPredictor(ColorPredictor):
    """CNN-based color prediction"""

    def __init__(self, model_path, label_mapping_path):
        torch.serialization.add_safe_globals([ColorCNN])
        self.model = torch.load(model_path, weights_only=False)
        self.model.eval()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.label_mapping = joblib.load(label_mapping_path)

    def predict_pixel(self, patch):
        """Predict color from image patch"""
        patch_pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        train_transforms = transforms.Compose(
            [transforms.Resize((20, 20)),
             transforms.ToTensor()])
        patch_tensor = train_transforms(patch_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(patch_tensor)
            prediction_idx = output.argmax(dim=1).item()

        return self.label_mapping.get(prediction_idx,
                                      f"class_{prediction_idx}")

    def get_name(self):
        return "CNN"


class Solver():

    def __init__(self, solver="kociemba"):
        if solver == "kociemba":
            self.solver = KociembaCubeSolver()

    def solve(self, configuration):
        return self.solver.solve(configuration)


class SimpleJSONHandler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin",
                         "http://127.0.0.1:5500")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_POST(self):
        if self.path == "/predict":
            self.handle_predict()
        elif self.path == "/solve":
            self.handle_solve()
        else:
            self.send_error(404, "Endpoint not found")

    def handle_predict(self):
        """Handle image prediction requests"""
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_error(415, "Expected multipart/form-data")
            return

        boundary = content_type.split("boundary=")[1].encode()
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        parts = body.split(b"--" + boundary)
        for part in parts:
            if b'Content-Disposition' in part and b'name="image"' in part:
                match = re.search(b'\r\n\r\n(.*)\r\n$', part, re.DOTALL)
                if not match:
                    continue
                image_bytes = match.group(1)
                break
        else:
            self.send_error(400, "Missing 'image' field")
            return

        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8),
                             cv2.IMREAD_COLOR)
        if image is None:
            self.send_error(400, "Invalid image data")
            return

        predictor = self.server.predictor
        predictions = predict(image, predictor)

        response_json = json.dumps({
            "predictions": predictions,
            "model": predictor.get_name(),
            "status": "ok"
        })
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin",
                         "http://127.0.0.1:5500")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_json)))
        self.end_headers()
        self.wfile.write(response_json.encode("utf-8"))

    def handle_solve(self):
        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            self.send_error(415, "Expected application/json")
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        body = json.loads(body)

        # Extract cube sides
        frontSide = body["front"]
        topSide = body["top"]
        leftSide = body["left"]
        rightSide = body["right"]
        bottomSide = body["bottom"]
        backSide = body["back"]

        try:
            solving_steps = self.server.solver.solve("".join(
                map(
                    lambda x: color_face[x], topSide + rightSide + frontSide +
                    bottomSide + leftSide + backSide)))
            print(solving_steps)
            response_json = json.dumps({"steps": solving_steps})
            response_code = 200
        except Exception as e:
            response_code = 400
            response_json = json.dumps({"error": "Illegal cube state"})

        self.send_response(response_code)
        self.send_header("Access-Control-Allow-Origin",
                         "http://127.0.0.1:5500")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_json)))
        self.end_headers()
        self.wfile.write(response_json.encode("utf-8"))


def run(server_class=HTTPServer,
        handler_class=SimpleJSONHandler,
        solver=Solver("kociemba"),
        port=8000,
        predictor=None):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    httpd.solver = solver
    httpd.predictor = predictor
    print(
        f"Server running on http://localhost:{port} with {predictor.get_name()} model"
    )
    httpd.serve_forever()


def find_edges(img):
    rows, cols = img.shape[:2]

    for i in range(rows):
        for j in range(cols):
            gray = 0.2989 * img[i, j][2] + 0.5870 * img[
                i, j][1] + 0.1140 * img[i, j][0]
            img[i, j] = [gray, gray, gray]
    img = cv2.GaussianBlur(img, (15, 15), 0)

    edges = cv2.Canny(img, 55, 150)
    cv2.imshow("edg", img)
    return edges


def get_cube_corners(edges):
    rows, cols = edges.shape[:2]
    max_x_point = (-1, 0)
    max_y_point = (0, -1)
    min_x_point = (cols + 1, 0)
    min_y_point = (0, rows + 1)

    for i in range(rows):
        for j in range(cols):
            if (edges[i, j] == 0):
                continue
            else:
                if i <= min_y_point[1]:
                    min_y_point = (j, i)
                if i > max_y_point[1]:
                    max_y_point = (j, i)
                if j < min_x_point[0]:
                    min_x_point = (j, i)
                if j >= max_x_point[0]:
                    max_x_point = (j, i)

    pi = math.pi

    side_1 = math.dist(min_x_point, min_y_point)
    side_2 = math.dist(min_y_point, max_x_point)
    side_3 = math.dist(max_x_point, max_y_point)
    side_4 = math.dist(max_y_point, min_x_point)
    diag_1 = math.dist(min_x_point, max_x_point)
    diag_2 = math.dist(min_y_point, max_y_point)

    alpha_1 = math.acos(
        (side_1**2 + side_2**2 - diag_1**2) / (2 * side_1 * side_2))
    alpha_2 = math.acos(
        (side_2**2 + side_3**2 - diag_2**2) / (2 * side_2 * side_3))
    alpha_3 = math.acos(
        (side_3**2 + side_4**2 - diag_1**2) / (2 * side_3 * side_4))
    alpha_4 = math.acos(
        (side_4**2 + side_1**2 - diag_2**2) / (2 * side_4 * side_1))

    area = (side_1 * side_2 * math.sin(alpha_1) / 2) + (side_3 * side_4 *
                                                        math.sin(alpha_3) / 2)
    diagonals_intersection_angle = math.asin(2 * area / (diag_1 * diag_2))

    angle_treshold = 20 * pi / 180
    if (abs(alpha_1 - pi / 2)
            > angle_treshold) or (abs(alpha_2 - pi / 2) > angle_treshold) or (
                abs(alpha_3 - pi / 2) > angle_treshold) or (
                    abs(alpha_4 - pi / 2)
                    > angle_treshold) or (abs(diagonals_intersection_angle -
                                              pi / 2) > angle_treshold):
        return (min_x_point[0], min_y_point[1]), (
            max_x_point[0], min_y_point[1]), (max_x_point[0],
                                              max_y_point[1]), (min_x_point[0],
                                                                max_y_point[1])
    else:
        vertical_treshold = (max_x_point[0] + min_x_point[0]) / 2
        if min_y_point[0] < vertical_treshold:
            return min_y_point, max_x_point, max_y_point, min_x_point
        else:
            return min_x_point, min_y_point, max_x_point, max_y_point


def contour_image(image):
    image_resized = cv2.resize(image, (540, 540),
                               interpolation=cv2.INTER_LINEAR)
    img = image_resized.copy()
    edges = find_edges(img)
    u_l, u_r, l_r, l_l = get_cube_corners(edges)
    return image_resized, u_l, u_r, l_r, l_l


def side_colors(image, up_left, up_right, low_right, low_left, predictor):
    primary_diag = (low_right[0] - up_left[0], low_right[1] - up_left[1])
    secondary_diag = (up_right[0] - low_left[0], up_right[1] - low_left[1])
    up_side = (up_right[0] - up_left[0], up_right[1] - up_left[1])
    left_side = (low_left[0] - up_left[0], low_left[1] - up_left[1])

    p_1 = (int(up_left[0] + (1 / 6) * primary_diag[0]),
           int(up_left[1] + (1 / 6) * primary_diag[1]))
    p_9 = (int(up_left[0] + (5 / 6) * primary_diag[0]),
           int(up_left[1] + (5 / 6) * primary_diag[1]))
    p_5 = (int(up_left[0] + (1 / 2) * primary_diag[0]),
           int(up_left[1] + (1 / 2) * primary_diag[1]))
    p_7 = (int(low_left[0] + (1 / 6) * secondary_diag[0]),
           int(low_left[1] + (1 / 6) * secondary_diag[1]))
    p_3 = (int(low_left[0] + (5 / 6) * secondary_diag[0]),
           int(low_left[1] + (5 / 6) * secondary_diag[1]))
    p_2 = (int(up_left[0] + (1 / 2) * up_side[0] + (1 / 6) * left_side[0]),
           int(up_left[1] + (1 / 2) * up_side[1] + (1 / 6) * left_side[1]))
    p_8 = (int(up_left[0] + (1 / 2) * up_side[0] + (5 / 6) * left_side[0]),
           int(up_left[1] + (1 / 2) * up_side[1] + (5 / 6) * left_side[1]))
    p_4 = (int(up_left[0] + (1 / 6) * up_side[0] + (1 / 2) * left_side[0]),
           int(up_left[1] + (1 / 6) * up_side[1] + (1 / 2) * left_side[1]))
    p_6 = (int(up_left[0] + (5 / 6) * up_side[0] + (1 / 2) * left_side[0]),
           int(up_left[1] + (5 / 6) * up_side[1] + (1 / 2) * left_side[1]))

    points = [p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8, p_9]

    colors = []
    for point in points:
        colors.append(
            predictor.predict_pixel(
                image[max(0, point[1] - 10):min(image.shape[0], point[1] + 10),
                      max(0, point[0] - 10):min(image.shape[1], point[0] +
                                                10)]))

    return colors


def predict(image, predictor):
    img, u_l, u_r, l_r, l_l = contour_image(image)
    side_squares_colors = side_colors(img, u_l, u_r, l_r, l_l, predictor)
    return side_squares_colors


def main():
    # Choose your model here
    use_cnn = True

    if use_cnn:
        predictor = CNNPredictor(
            "color_classification_models/cnn_color_predictor.pth",
            "color_classification_models/label_mapping_cnn.sav")
    else:
        predictor = KNNPredictor(
            "color_classification_models/knn_color_predictor.sav",
            "color_classification_models/scaler_knn.sav",
            "color_classification_models/label_mapping_knn.sav")

    run(handler_class=SimpleJSONHandler, predictor=predictor)


if __name__ == "__main__":
    main()
