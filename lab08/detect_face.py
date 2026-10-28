import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import warnings
from scipy.ndimage import convolve, gaussian_filter

MODEL_PATH = "version-RFB-320.onnx"

warnings.filterwarnings("ignore", category=UserWarning)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3  

# 3. Use only available providers to prevent the provider fallback warning
desired_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = [p for p in desired_providers if p in ort.get_available_providers()]

# face detection
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]

    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]

    picked = []

    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes]

        x1 = np.maximum(current_box[0], rest_boxes[:, 0])
        y1 = np.maximum(current_box[1], rest_boxes[:, 1])
        x2 = np.minimum(current_box[2], rest_boxes[:, 2])
        y2 = np.minimum(current_box[3], rest_boxes[:, 3])
        inter = np.maximum(0.0, x2-x1) * np.maximum(0.0, y2-y1)

        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area_rest = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (rest_boxes[:, 3] - rest_boxes[:, 1])
        union = area_current + area_rest - inter
        iou = inter / (union + 1e-5)
 
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    resized = image.resize((320,240))
    x = np.asarray(resized).astype(np.float32)
    x = (x - 127.0) / 128.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)

    session = ort.InferenceSession(MODEL_PATH, sess_options=sess_options)
    input_name = session.get_inputs()[0].name
    confidences, boxes = session.run(None, {input_name: x})

    confidences = confidences[0]
    boxes = boxes[0]
    probs = confidences[:, 1]
    mask = probs > conf_threshold
    if np.any(mask):
        box_probs = np.concatenate([boxes[mask], probs[mask].reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs, iou_threshold=0.3, top_k=-1)
    else:
        box_probs = np.zeros((0, 5), dtype=np.float32)
    
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2, score in box_probs:
        x1, x2 = x1 * orig_w, x2 * orig_w
        y1, y2 = y1 * orig_h, y2 * orig_h
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)
        draw.text((x1, max(0, y1 - 15)), f"{score:.2f}", fill=(0, 255, 0))
    image.save(output_path)
    print(f"Detected {len(box_probs)} face(s). Saved to {output_path}")
    return box_probs


# Edge detection
def non_maximum_suppression(magnitude, angle):
    H, W = magnitude.shape
    out = np.zeros((H,W), dtype=np.float32)
    deg = np.rad2deg(angle)
    deg[deg < 0] += 180

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            a = deg[i, j]

            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q, r = magnitude[i, j - 1], magnitude[i, j + 1]
            elif 22.5 <= a < 67.5:
                q, r = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
            elif 67.5 <= a < 112.5:
                q, r = magnitude[i - 1, j], magnitude[i + 1, j]
            else:
                q, r = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
 
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                out[i, j] = magnitude[i, j]
    return out

def hysteresis_thresholding(img, low_thresh, high_thresh):
    STRONG, WEAK = 255, 50

    res = np.zeros(img.shape, dtype=np.uint8)
    res[img >= high_thresh] = STRONG
    res[(img >= low_thresh) & (img < high_thresh)] = WEAK

    H,W = res.shape
    weak_i, weak_j = np.where(res == WEAK)
    for i, j in zip(weak_i, weak_j):
        window = res[max(0, i-1):min(H, i+2), max(0, j-1):min(W,j+2)]
        if np.any(window == STRONG):
            res[i,j] = STRONG
        else:
            res[i,j] = 0
        
    return res

def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    img = np.asarray(Image.open(image_path).convert("L")).astype(np.float32)
    smoothed = gaussian_filter(img, sigma=1.4)

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]], dtype=np.float32)
    Gx = convolve(smoothed, Kx)
    Gy = convolve(smoothed, Ky)
    magnitude = np.hypot(Gx, Gy)
    magnitude = magnitude/ magnitude.max() * 255.0
    angle = np.arctan2(Gy, Gx)

    thin = non_maximum_suppression(magnitude, angle)
    edges = hysteresis_thresholding(thin, low, high)
    Image.fromarray(edges.astype(np.uint8)).save(output_path)
    print(f"Edge map saved to {output_path}")
    return edges

if __name__ == "__main__":
    detect_faces("face.jpg")
    canny_scipy("face.jpg")