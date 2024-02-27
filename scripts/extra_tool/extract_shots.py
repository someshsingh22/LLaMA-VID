import argparse
import os
from scenedetect import (
    AdaptiveDetector,
    ContentDetector,
    SceneManager,
    detect,
    open_video,
    save_images,
    scene_manager,
)
from scenedetect.platform import get_and_create_path, get_cv2_imwrite_params, tqdm
import cv2
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract shots from a video using the scenedetect library."
    )
    parser.add_argument(
        "--process_idx",
        type=int,
        default=0,
        help="Index of the current process (default: 0)",
    )
    parser.add_argument(
        "--num_process",
        type=int,
        default=16,
        help="Total number of processes (default: 16)",
    )
    return parser.parse_args()

args = parse_arguments()

def get_shots_numpy(image_paths):
    '''
    Given a list of image paths, load all images as numpy array of shape (n, h, w, c)
    '''
    images = [cv2.imread(image_path) for image_path in image_paths]
    images = np.stack(images, axis=0)
    return images

_suffix = os.listdir('videos')[args.process_idx :: args.num_process]
if not os.path.exists(f"video_scenes/npy"):
    os.makedirs(f"video_scenes")

if __name__ == "__main__":
    for suffix in tqdm(_suffix):
        try:
            suffix = "videos/" + suffix
            video = open_video(suffix)
            scene_manager = SceneManager()
            scene_manager.add_detector(AdaptiveDetector())
            scene_manager.detect_scenes(video)
            scene_list = scene_manager.get_scene_list()
            if len(scene_list) == 0:
                scene_manager = SceneManager()
                scene_manager.add_detector(ContentDetector(20))
                scene_manager.detect_scenes(video)
                scene_list = scene_manager.get_scene_list()
                
            video_id = suffix.split('.')[0]
            save_images(
                scene_list=scene_list,
                video=video,
                image_name_template=f"{video_id}/$SCENE_NUMBER",
                output_dir="video_scenes",
                num_images=1,
            )
            nparry = get_shots_numpy(sorted(os.listdir(f"video_scenes/{video_id}")))
            np.save(f"video_scenes/npy/{video_id}.npy", nparry)
        except Exception as e:
            with open(f"error_csv_{args.process_idx}.text", "a") as f:
                f.write(suffix + "\n")
