import face_alignment
from skimage import img_as_ubyte
from skimage.transform import resize
import numpy as np
from tqdm import tqdm
import imageio
import argparse
from pathlib import Path


def extract_bbox(frame, fa):
    if max(frame.shape[0], frame.shape[1]) > 640:
        scale_factor = max(frame.shape[0], frame.shape[1]) / 640.0
        frame = resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
        frame = img_as_ubyte(frame)
    else:
        scale_factor = 1
    frame = frame[..., :3]
    bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
    if len(bboxes) == 0:
        return []
    return np.array(bboxes)[:, :-1] * scale_factor


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def join(tube_bbox, bbox):
    xA = min(tube_bbox[0], bbox[0])
    yA = min(tube_bbox[1], bbox[1])
    xB = max(tube_bbox[2], bbox[2])
    yB = max(tube_bbox[3], bbox[3])
    return (xA, yA, xB, yB)


def compute_bbox(bbox, frame_shape, increase_area=0.1):
    left, top, right, bot = bbox
    width = right - left
    height = bot - top

    # Computing aspect preserving bbox
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    top, bot, left, right = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])
    h, w = bot - top, right - left

    return w, h, left, top


def compute_bbox_trajectories(trajectories, frame_shape):
    commands = []
    for i, (bbox, tube_bbox) in enumerate(trajectories):
        res = compute_bbox(tube_bbox, frame_shape, increase_area=0.1)

    return res


def detect_face(frame, scale):
    device = 'cuda'  # if args.cpu else 'cuda'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)

    frame_shape = frame.shape
    bboxes = extract_bbox(frame, fa)

    trajectories = []
    bbox = bboxes[0]
    trajectories.append([bbox, bbox])

    w, h, left, top = compute_bbox(bbox, frame_shape, scale)

    return w, h, left, top


def crop(frame, w, h, left, top):

    size = min(h, w)

    center_y = top + h // 2
    center_x = left + w // 2

    # Calculate the new top-left corner for the square crop
    new_top = center_y - size // 2
    new_left = center_x - size // 2

    # Ensure the coordinates are within the image bounds
    new_top = max(0, new_top)
    new_left = max(0, new_left)

    bottom = new_top + size
    right = new_left + size

    # Adjust if the crop exceeds the image dimensions
    if bottom > frame.shape[0]:
        new_top -= (bottom - frame.shape[0])
    if right > frame.shape[1]:
        new_left -= (right - frame.shape[1])

    # Recalculate bottom and right after adjustments
    bottom = new_top + size
    right = new_left + size

    # Ensure final coordinates are within bounds
    new_top = max(0, new_top)
    new_left = max(0, new_left)
    bottom = min(bottom, frame.shape[0])
    right = min(right, frame.shape[1])

    # Crop the square
    cropped_image = frame[new_top:bottom, new_left:right]

    return cropped_image


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='')
    parser.add_argument("--mode", type=str, choices=['img', 'vid'])
    parser.add_argument("--scale", type=float, default=0.25)
    args = parser.parse_args()

    print("==> running")
    if args.mode == 'img':
        save_path = './data/source/' + Path(args.data_path).stem + "_crop.png"
        img = imageio.v3.imread(args.path)
        w, h, left, top = detect_face(img, args.scale)
        img_crop = crop(img, w, h, left, top)
        # print("The size of cropped image is: ", img_crop.shape)

        imageio.imwrite(save_path, img_crop)
        print("Save at: ", save_path)

    elif args.mode == 'vid':
        save_path = './data/driving/' + Path(args.data_path).stem + "_crop.mp4"
        reader = imageio.get_reader(args.data_path, "ffmpeg")
        fps = reader.get_meta_data()['fps']

        frames = []
        for i, frame in tqdm(enumerate(reader)):
            # print(frame.shape)
            if i == 0:
                w, h, left, top = detect_face(frame, args.scale)
            frame_crop = crop(frame, w, h, left, top)
            frames.append(frame_crop)

        imageio.mimsave(save_path, frames, fps=fps)
        print("Save at: ", save_path)
    else:
        raise NotImplementedError