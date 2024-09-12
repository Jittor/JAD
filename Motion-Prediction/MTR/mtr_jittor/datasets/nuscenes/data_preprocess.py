# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved


import sys, os
import numpy as np
import pickle
import multiprocessing
import glob
from tqdm import tqdm
from nuscenes import NuScenes

from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper

from scipy.spatial.transform import Rotation


def get_type(type):
    if type.startswith("human"):
        return 2
    if type.startswith("vehicle.bicycle"):
        return 3
    if type.startswith("vehicle.motorcycle"):
        return 3
    if type.startswith("vehicle"):
        return 1
    return 0


def get_rotation(quat):
    r = Rotation.from_quat(quat)
    return r.as_euler("xyz", degrees=False)[0]


def get_velocity(helper, instance_token, sample_token):
    annotation = helper.get_sample_annotation(instance_token, sample_token)

    if annotation["prev"] == "":
        return 0

    prev = helper.data.get("sample_annotation", annotation["prev"])

    current_time = 1e-6 * helper.data.get("sample", sample_token)["timestamp"]
    prev_time = 1e-6 * helper.data.get("sample", prev["sample_token"])["timestamp"]
    time_diff = current_time - prev_time
    diff = (
        np.array(annotation["translation"]) - np.array(prev["translation"])
    ) / time_diff
    return diff[0], diff[1]


def get_polyline_dir(polyline):
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(
        np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000
    )
    return polyline_dir


def get_map_features(nusc_map, rangex, rangey):
    map_infos = {
        "lane": [],
        "road_line": [],
        "road_edge": [],
        "stop_sign": [],
        "crosswalk": [],
        "speed_bump": [],
    }
    polylines = []

    point_cnt = 0
    data = nusc_map.discretize_centerlines(1)
    id = 0
    for cur_data in data:
        cur_info = {"id": id}
        id += 1

        # cur_info["speed_limit_mph"] = cur_data.lane.speed_limit_mph
        cur_info["type"] = 1
        # 0: undefined, 1: freeway, 2: surface_street, 3: bike_lane

        global_type = 1
        cur_polyline = []
        num_points = cur_data.shape[0]
        valid = False
        for i in range(num_points):
            if (
                cur_data[i][0] > rangex[0]
                and cur_data[i][0] < rangex[1]
                and cur_data[i][1] > rangey[0]
                and cur_data[i][1] < rangey[1]
            ):
                valid = True
            cur_polyline.append(np.concatenate([cur_data[i], np.array([global_type])]))
        if not valid:
            continue
        cur_polyline = np.stack(
            cur_polyline,
            axis=0,
        )
        cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
        cur_polyline = np.concatenate(
            (cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1
        )

        map_infos["lane"].append(cur_info)

        polylines.append(cur_polyline)
        cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
        point_cnt += len(cur_polyline)

    try:
        polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    except:
        polylines = np.zeros((0, 7), dtype=np.float32)
        print("Empty polylines: ")
    map_infos["all_polylines"] = polylines
    return map_infos


"""
def decode_dynamic_map_states_from_proto(dynamic_map_states):
    dynamic_map_infos = {"lane_id": [], "state": [], "stop_point": []}
    for cur_data in dynamic_map_states:  # (num_timestamp)
        lane_id, state, stop_point = [], [], []
        for cur_signal in cur_data.lane_states:  # (num_observed_signals)
            lane_id.append(cur_signal.lane)
            state.append(signal_state[cur_signal.state])
            stop_point.append(
                [
                    cur_signal.stop_point.x,
                    cur_signal.stop_point.y,
                    cur_signal.stop_point.z,
                ]
            )

        dynamic_map_infos["lane_id"].append(np.array([lane_id]))
        dynamic_map_infos["state"].append(np.array([state]))
        dynamic_map_infos["stop_point"].append(np.array([stop_point]))

    return dynamic_map_infos
"""


def process_nuscenes_data(root, output_path=None, train=True):
    nuscenes = NuScenes("v1.0-mini", dataroot=root)
    from nuscenes.eval.prediction.splits import (
        create_splits_scenes,
        get_prediction_challenge_split,
    )
    from nuscenes.map_expansion.map_api import NuScenesMap

    prediction_pairs = []
    if train:
        prediction_pairs = get_prediction_challenge_split("mini_train", root)
    else:
        prediction_pairs = get_prediction_challenge_split("mini_val", root)

    helper = PredictHelper(nuscenes)

    ret_infos = []
    scene_id = 0
    for prediction_pair in prediction_pairs:
        scene_id += 1
        instance_token, sample_token = prediction_pair.split("_")

        past_annotations = helper.get_past_for_agent(
            instance_token, sample_token, seconds=5, in_agent_frame=False, just_xy=False
        )

        future_annotations = helper.get_future_for_agent(
            instance_token, sample_token, seconds=5, in_agent_frame=False, just_xy=False
        )

        if len(past_annotations) < 10 or len(future_annotations) < 10:
            continue

        scene = nuscenes.get(
            "scene", nuscenes.get("sample", sample_token)["scene_token"]
        )

        info = {}

        info["scenario_id"] = scene_id
        first = past_annotations[-1]["sample_token"]
        last = future_annotations[-1]["sample_token"]
        instances = []
        cur_sample_token = first
        num_samples = 21
        ego_track = np.zeros((num_samples, 10))

        info["timestamps_seconds"] = []
        sample_tokens = {}
        for i in range(num_samples):
            sample = nuscenes.get("sample", cur_sample_token)
            sample_tokens[cur_sample_token] = i
            cam_front_data = nuscenes.get("sample_data", sample["data"]["CAM_FRONT"])
            ego_pose = nuscenes.get("ego_pose", cam_front_data["ego_pose_token"])
            info["timestamps_seconds"].append(sample["timestamp"] * 1e-6)

            ego_track[i, 0:3] = ego_pose["translation"]
            ego_track[i, 3] = 4.084  # Renault Zoe production data
            ego_track[i, 4] = 1.73
            ego_track[i, 5] = 1.562
            ego_track[i, 6] = get_rotation(ego_pose["rotation"])
            if i > 0:
                ego_track[i, 7:9] = (
                    ego_pose["translation"][0:2] - ego_track[i - 1, 0:2]
                ) / (info["timestamps_seconds"][-1] - info["timestamps_seconds"][-2])
            ego_track[i, 9] = 1

            for annotation_token in sample["anns"]:
                annotation = nuscenes.get("sample_annotation", annotation_token)
                if annotation["category_name"].split(".")[0] in [
                    "vehicle",
                    "human",
                ] and not (annotation["instance_token"] in instances):
                    instances.append(annotation["instance_token"])
            num_objects = len(instances) + 1

            cur_sample_token = sample["next"]
            if cur_sample_token == "":
                break
        info["current_time_index"] = 10
        info["sdc_track_index"] = num_objects - 1

        tracks = np.zeros((num_objects, num_samples, 10))  # last one is ego
        object_type = []  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
        for k in range(num_objects - 1):
            inst = instances[k]
            obj = nuscenes.get("instance", inst)

            start_token = nuscenes.get(
                "sample_annotation", obj["first_annotation_token"]
            )["sample_token"]
            start = sample_tokens.get(start_token, 0)
            cur_ann = nuscenes.get("sample_annotation", obj["first_annotation_token"])
            object_type.append(get_type(cur_ann["category_name"]))

            for i in range(start, num_samples):
                tracks[k, i, 0:3] = cur_ann["translation"]
                tracks[k, i, 3] = cur_ann["size"][1]
                tracks[k, i, 4] = cur_ann["size"][0]
                tracks[k, i, 5] = cur_ann["size"][2]
                tracks[k, i, 6] = get_rotation(cur_ann["rotation"])
                tracks[k, i, 7:9] = get_velocity(helper, inst, cur_ann["sample_token"])
                tracks[k, i, 9] = 1
                if cur_ann["next"] == "":
                    break
                cur_ann = nuscenes.get("sample_annotation", cur_ann["next"])

        tracks[-1] = ego_track
        object_type.append(4)

        info["objects_of_interest"] = []  # list, could be empty list

        info["tracks_to_predict"] = {"track_index": []}

        for i in range(num_objects):
            if object_type[i] == 1 and tracks[i, 10, 9] == 1:
                info["tracks_to_predict"]["track_index"].append(i)

        # for training: suggestion of objects to train on, for val/test: need to be predicted

        track_infos = {"object_id": [], "object_type": [], "trajs": []}
        track_infos["object_id"] = np.array(instances + ["0"])
        track_infos["trajs"] = np.array(tracks)
        track_infos["object_type"] = np.array(object_type)
        info["tracks_to_predict"]["object_type"] = [
            track_infos["object_type"][cur_idx]
            for cur_idx in info["tracks_to_predict"]["track_index"]
        ]
        info["tracks_to_predict"]["difficulty"] = [
            1 for i in info["tracks_to_predict"]["track_index"]
        ]

        log = nuscenes.get("log", scene["log_token"])
        map_name = log["location"]
        nusc_map = NuScenesMap(dataroot=root, map_name=map_name)
        minx = np.min(ego_track[:, 0], axis=0) - 50
        maxx = np.min(ego_track[:, 0], axis=0) + 50
        miny = np.min(ego_track[:, 1], axis=0) - 50
        maxy = np.min(ego_track[:, 1], axis=0) + 50
        map_infos = get_map_features(nusc_map, [minx, maxx], [miny, maxy])
        # dynamic_map_infos = decode_dynamic_map_states_from_proto(
        #    scenario.dynamic_map_states
        # )

        save_infos = {
            "track_infos": track_infos,
            "dynamic_map_infos": None,
            "map_infos": map_infos,
        }
        save_infos.update(info)

        output_file = os.path.join(output_path, f"sample_{info['scenario_id']}.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(save_infos, f)

        ret_infos.append(info)
    return ret_infos


def get_infos_from_protos(data_path, output_path=None, train=True):
    from functools import partial

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    all_infos = process_nuscenes_data(data_path, output_path, train)
    return all_infos


def create_infos(raw_data_path, output_path):
    train_infos = get_infos_from_protos(
        data_path=raw_data_path,
        output_path=os.path.join(output_path, "processed_scenes_training"),
        train=True,
    )
    train_filename = os.path.join(output_path, "processed_scenes_training_infos.pkl")
    with open(train_filename, "wb") as f:
        pickle.dump(train_infos, f)
    print(
        "----------------nuScenes info train file is saved to %s----------------"
        % train_filename
    )

    val_infos = get_infos_from_protos(
        data_path=raw_data_path,
        output_path=os.path.join(output_path, "processed_scenes_validation"),
        train=False,
    )
    val_filename = os.path.join(output_path, "processed_scenes_val_infos.pkl")
    with open(val_filename, "wb") as f:
        pickle.dump(val_infos, f)
    print(
        "----------------nuScenes info val file is saved to %s----------------"
        % val_filename
    )


if __name__ == "__main__":
    create_infos(raw_data_path=sys.argv[1], output_path=sys.argv[2])
