# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved


import numpy as np
import os
from mtr_jittor.utils import common_utils

object_type_to_id = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}


def transform_preds_to_waymo_format(pred_dicts, top_k_for_eval=-1, eval_second=8):
    print(f"Total number for evaluation (intput): {len(pred_dicts)}")
    temp_pred_dicts = []
    for k in range(len(pred_dicts)):
        if isinstance(pred_dicts[k], list):
            temp_pred_dicts.extend(pred_dicts[k])
        else:
            temp_pred_dicts.append(pred_dicts[k])
    pred_dicts = temp_pred_dicts
    print(f"Total number for evaluation (after processed): {len(pred_dicts)}")

    scene2preds = {}
    num_max_objs_per_scene = 0
    for k in range(len(pred_dicts)):
        cur_scenario_id = pred_dicts[k]["scenario_id"]
        if cur_scenario_id not in scene2preds:
            scene2preds[cur_scenario_id] = []
        scene2preds[cur_scenario_id].append(pred_dicts[k])
        num_max_objs_per_scene = max(
            num_max_objs_per_scene, len(scene2preds[cur_scenario_id])
        )
    num_scenario = len(scene2preds)
    topK, num_future_frames, _ = pred_dicts[0]["pred_trajs"].shape

    if top_k_for_eval != -1:
        topK = min(top_k_for_eval, topK)

    sampled_interval = 1
    if num_future_frames in [30, 50, 80]:
        sampled_interval = 5
    assert (
        num_future_frames % sampled_interval == 0
    ), f"num_future_frames={num_future_frames}"
    num_frame_to_eval = num_future_frames // sampled_interval

    num_frames_in_total = 21

    batch_pred_trajs = np.zeros(
        (num_scenario, num_max_objs_per_scene, topK, 1, num_frame_to_eval, 2)
    )
    batch_pred_scores = np.zeros((num_scenario, num_max_objs_per_scene, topK))
    gt_trajs = np.zeros((num_scenario, num_max_objs_per_scene, num_frames_in_total, 7))
    gt_is_valid = np.zeros(
        (num_scenario, num_max_objs_per_scene, num_frames_in_total), dtype=int
    )
    pred_gt_idxs = np.zeros((num_scenario, num_max_objs_per_scene, 1))
    pred_gt_idx_valid_mask = np.zeros(
        (num_scenario, num_max_objs_per_scene, 1), dtype=int
    )
    object_type = np.zeros((num_scenario, num_max_objs_per_scene), dtype=object)
    object_id = np.zeros((num_scenario, num_max_objs_per_scene), dtype=int)
    scenario_id = np.zeros((num_scenario), dtype=object)

    object_type_cnt_dict = {}
    for key in object_type_to_id.keys():
        object_type_cnt_dict[key] = 0

    for scene_idx, val in enumerate(scene2preds.items()):
        cur_scenario_id, preds_per_scene = val
        scenario_id[scene_idx] = cur_scenario_id
        for obj_idx, cur_pred in enumerate(preds_per_scene):
            sort_idxs = cur_pred["pred_scores"].argsort()[::-1]
            cur_pred["pred_scores"] = cur_pred["pred_scores"][sort_idxs]
            cur_pred["pred_trajs"] = cur_pred["pred_trajs"][sort_idxs]

            cur_pred["pred_scores"] = (
                cur_pred["pred_scores"] / cur_pred["pred_scores"].sum()
            )

            batch_pred_trajs[scene_idx, obj_idx] = cur_pred["pred_trajs"][
                :topK, np.newaxis, :, :
            ][:, :, :num_frame_to_eval, :]
            batch_pred_scores[scene_idx, obj_idx] = cur_pred["pred_scores"][:topK]
            gt_trajs[scene_idx, obj_idx] = cur_pred["gt_trajs"][
                :num_frames_in_total, [0, 1, 3, 4, 6, 7, 8]
            ]  # (num_timestamps_in_total, 10), [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            gt_is_valid[scene_idx, obj_idx] = cur_pred["gt_trajs"][
                :num_frames_in_total, -1
            ]
            pred_gt_idxs[scene_idx, obj_idx, 0] = obj_idx
            pred_gt_idx_valid_mask[scene_idx, obj_idx, 0] = 1
            object_type[scene_idx, obj_idx] = object_type_to_id[cur_pred["object_type"]]

            object_type_cnt_dict[cur_pred["object_type"]] += 1

    gt_infos = {
        "scenario_id": scenario_id.tolist(),
        "object_type": object_type.tolist(),
        "gt_is_valid": gt_is_valid,
        "gt_trajectory": gt_trajs,
        "pred_gt_indices": pred_gt_idxs,
        "pred_gt_indices_mask": pred_gt_idx_valid_mask,
    }
    return batch_pred_scores, batch_pred_trajs, gt_infos, object_type_cnt_dict


def motion_metrics(
    prediction_trajectory,
    prediction_score,
    ground_truth_trajectory,
    ground_truth_is_valid,
    prediction_ground_truth_indices,
    prediction_ground_truth_indices_mask,
    object_type,
):
    """
    prediction_trajectory: (batch_size, num_pred_groups, top_k, 1, num_pred_steps, 2)
    prediction_score: (batch_size, num_pred_groups, top_k)
    ground_truth_trajectory: (batch_size, num_total_agents, num_gt_steps, 7)
    ground_truth_is_valid: (batch_size, num_total_agents, num_gt_steps)
    prediction_ground_truth_indices: (batch_size, num_pred_groups, 1)
    prediction_ground_truth_indices_mask: (batch_size, num_pred_groups, 1)
    object_type: (batch_size, num_total_agents)
    """
    prediction_ground_truth_indices = prediction_ground_truth_indices.astype(int)
    ret = np.zeros((5))
    batch_size = prediction_trajectory.shape[0]
    num_pred_groups = prediction_trajectory.shape[1]
    num_pred_steps = prediction_trajectory.shape[4]
    num_gt_steps = ground_truth_trajectory.shape[2]
    num_obj_per_group = prediction_trajectory.shape[3]
    top_k = prediction_trajectory.shape[2]
    gt_traj = np.zeros(
        (batch_size, num_pred_groups, num_obj_per_group, num_pred_steps, 7)
    )
    gt_mask = np.zeros((batch_size, num_pred_groups, num_obj_per_group, num_pred_steps))

    for i in range(batch_size):
        gt_traj[i] = ground_truth_trajectory[
            i, prediction_ground_truth_indices[i], -num_pred_steps:
        ]  # (num_pred_groups, 1, num_pred_steps, 7)
        gt_mask[i] = ground_truth_is_valid[
            i, prediction_ground_truth_indices[i], -num_pred_steps:
        ]  # (num_pred_groups, 1, num_pred_steps)
    gt_mask = np.expand_dims(prediction_ground_truth_indices_mask, axis=3) * gt_mask
    gt_traj = np.broadcast_to(
        np.expand_dims(gt_traj, axis=2),
        (batch_size, num_pred_groups, top_k, num_obj_per_group, num_pred_steps, 7),
    )  # (batch_size, num_pred_groups, top_k, 1, num_pred_steps, 7)
    gt_mask = np.broadcast_to(
        np.expand_dims(gt_mask, axis=2),
        (batch_size, num_pred_groups, top_k, num_obj_per_group, num_pred_steps),
    )  # (batch_size, num_pred_groups, top_k, 1, num_pred_steps)
    ret[0] = np.sum(
        np.min(
            np.sum(
                gt_mask
                * np.linalg.norm(
                    gt_traj[:, :, :, :, :, 0:2] - prediction_trajectory, axis=-1
                ),
                axis=(-1, -2),
            ),
            axis=-1,
        )
    ) / (np.sum(gt_mask) / top_k)

    # print(
    #     gt_mask
    #     * np.linalg.norm(gt_traj[:, :, :, :, :, 0:2] - prediction_trajectory, axis=-1)
    # )

    ret[1] = np.sum(
        np.min(
            np.sum(
                gt_mask[:, :, :, :, -1]
                * np.linalg.norm(
                    gt_traj[:, :, :, :, -1, 0:2]
                    - prediction_trajectory[:, :, :, :, -1],
                    axis=-1,
                ),
                axis=-1,
            ),
            axis=-1,
        )
    ) / (np.sum(gt_mask[:, :, :, :, -1]) / top_k)

    obj_trajs = np.zeros_like(prediction_trajectory)
    obj_trajs[:, :, :, :, :, 0:2] = common_utils.rotate_points_along_z(
        points=(
            gt_traj[:, :, :, :, :, 0:2] - prediction_trajectory[:, :, :, :, :]
        ).reshape(-1, 1, 2),
        angle=-gt_traj[:, :, :, :, :, 4].reshape(-1),
    ).reshape(batch_size, num_pred_groups, top_k, num_obj_per_group, num_pred_steps, 2)

    vH = 11
    vL = 1.4
    gamma = (
        np.maximum(0, np.minimum(1, (gt_traj[:, :, :, :, :, 5:7] - vL) / (vH - vL))) / 2
        + 0.5
    )

    lon0 = 3.6
    lat0 = 1.8
    lon = lon0 * gamma[:, :, :, :, :, 0]
    lat = lat0 * gamma[:, :, :, :, :, 1]
    a = np.abs(obj_trajs[:, :, :, :, :, 0]) > lon
    b = np.abs(obj_trajs[:, :, :, :, :, 1]) > lat
    miss = (a + b) > 0  # (batch_size, num_pred_groups, top_k, 1, num_pred_steps)
    miss_timestep = (np.sum(gt_mask * miss, axis=-2) > 0).astype(
        int
    )  # (batch_size, num_pred_groups, top_k, num_pred_steps)
    timestep_mask = (np.sum(gt_mask, axis=-2) > 0).astype(int)

    ret[3] = np.sum(
        np.min(
            miss_timestep,
            axis=-2,
        ),
    ) / (np.sum(timestep_mask) / top_k)

    true_positive = (
        np.expand_dims(prediction_score, axis=3) * (1 - miss_timestep) * timestep_mask
    )  # (batch_size, num_pred_groups, top_k, num_pred_steps)
    ret[4] = np.sum(np.max(true_positive, axis=-2)) / (np.sum(timestep_mask) / top_k)

    return ret


"""
def _CreateTestScenario(batch_size):
    gt_scenario_id = ["test"]
    gt_object_id = [[1, 2]]
    gt_object_type = [[1, 1]]
    gt_is_valid = np.ones([1, 2, 5], dtype=bool)
    gt_trajectory = np.reshape(
        [
            [
                [2, 2, 1, 1, 0.78539816, 20.0, 20.0],
                [4, 4, 1, 1, 0.78539816, 20.0, 20.0],
                [6, 6, 1, 1, 0.78539816, 20.0, 20.0],
                [8, 8, 1, 1, 0.78539816, 20.0, 20.0],
                [10, 10, 1, 1, 0.78539816, 20.0, 20.0],
            ],
            [
                [-1, 0, 1, 1, 3.14159, -10.0, 0.0],
                [-2, 0, 1, 1, 3.14159, -10.0, 0.0],
                [-3, 0, 1, 1, 3.14159, -10.0, 0.0],
                [-4, 0, 1, 1, 3.14159, -10.0, 0.0],
                [-5, 0, 1, 1, 3.14159, -10.0, 0.0],
            ],
        ],
        [1, 2, 5, 7],
    )

    pred_gt_indices = np.reshape([0, 1], (1, 1, 2))
    pred_gt_indices_mask = np.ones((1, 1, 2)) > 0.0

    return {
        "scenario_id": [[gt_scenario_id[0] + "%s" % i] for i in range(batch_size)],
        "object_id": [gt_object_id for _ in range(batch_size)],
        "object_type": [gt_object_type for _ in range(batch_size)],
        "gt_is_valid": [gt_is_valid for _ in range(batch_size)],
        "gt_trajectory": [gt_trajectory for _ in range(batch_size)],
        "pred_gt_indices": [pred_gt_indices for _ in range(batch_size)],
        "pred_gt_indices_mask": [pred_gt_indices_mask for _ in range(batch_size)],
    }


pred_score = np.reshape([0.5, 0.5], (1, 1, 2))
pred_trajectory = np.reshape(
    [
        [[[4, 4], [6, 0], [8, 0], [10, 0]], [[-2, 0], [-3, 0], [0, 4], [0, 5]]],
        [[[14, 0], [16, 0], [18, 0], [20, 0]], [[0, 22], [0, 23], [0, 24], [0, 25]]],
    ],
    (1, 1, 2, 2, 4, 2),
)
gt = _CreateTestScenario(1)
print(
    motion_metrics(
        pred_trajectory,
        pred_score,
        np.concatenate(gt["gt_trajectory"]),
        np.concatenate(gt["gt_is_valid"]),
        np.concatenate(gt["pred_gt_indices"]),
        np.concatenate(gt["pred_gt_indices_mask"]),
        np.concatenate(gt["object_type"]),
    )
)
"""


def nuscenes_evaluation(pred_dicts, top_k=-1, eval_second=8, num_modes_for_eval=6):

    pred_score, pred_trajectory, gt_infos, object_type_cnt_dict = (
        transform_preds_to_waymo_format(
            pred_dicts,
            top_k_for_eval=top_k,
            eval_second=eval_second,
        )
    )

    pred_trajs = pred_trajectory
    gt_trajs = gt_infos["gt_trajectory"]
    gt_is_valid = gt_infos["gt_is_valid"]
    pred_gt_indices = gt_infos["pred_gt_indices"]
    pred_gt_indices_mask = gt_infos["pred_gt_indices_mask"]

    object_type = gt_infos["object_type"]

    metric_results = motion_metrics(
        prediction_trajectory=pred_trajs,  # (batch_size, num_pred_groups, top_k, 1, num_pred_steps, 2)
        prediction_score=pred_score,  # (batch_size, num_pred_groups, top_k)
        ground_truth_trajectory=gt_trajs,  # (batch_size, num_total_agents, num_gt_steps, 7)
        ground_truth_is_valid=gt_is_valid,  # (batch_size, num_total_agents, num_gt_steps)
        prediction_ground_truth_indices=pred_gt_indices,  # (batch_size, num_pred_groups, 1)
        prediction_ground_truth_indices_mask=pred_gt_indices_mask,  # (batch_size, num_pred_groups, 1)
        object_type=object_type,  # (batch_size, num_total_agents)
    )  # ([min_ade, min_fde, miss_rate, overlap_rate, mean_ap])

    result_dict = {}
    avg_results = {}
    for i, m in enumerate(["minADE", "minFDE", "OverlapRate", "MissRate", "mAP"]):
        avg_results.update(
            {
                f"{m} - VEHICLE": [0.0, 0],
                f"{m} - PEDESTRIAN": [0.0, 0],
                f"{m} - CYCLIST": [0.0, 0],
            }
        )
        avg_results[f"{m} - VEHICLE"][0] += float(metric_results[i])
        avg_results[f"{m} - VEHICLE"][1] += 1
        result_dict[f"{m}\t"] = float(metric_results[i])

    for key in avg_results:
        avg_results[key] = avg_results[key][0] / (avg_results[key][1] + 1e-8)

    result_dict["-------------------------------------------------------------"] = 0
    result_dict.update(avg_results)

    final_avg_results = {}
    result_format_list = [
        ["Nuscenes", "mAP", "minADE", "minFDE", "MissRate", "\n"],
        ["VEHICLE", None, None, None, None, "\n"],
        ["PEDESTRIAN", None, None, None, None, "\n"],
        ["CYCLIST", None, None, None, None, "\n"],
        ["Avg", None, None, None, None, "\n"],
    ]
    name_to_row = {"VEHICLE": 1, "PEDESTRIAN": 2, "CYCLIST": 3, "Avg": 4}
    name_to_col = {"mAP": 1, "minADE": 2, "minFDE": 3, "MissRate": 4}

    for cur_metric_name in ["minADE", "minFDE", "MissRate", "mAP"]:
        final_avg_results[cur_metric_name] = 0
        for cur_name in ["VEHICLE", "PEDESTRIAN", "CYCLIST"]:
            final_avg_results[cur_metric_name] += avg_results[
                f"{cur_metric_name} - {cur_name}"
            ]

            result_format_list[name_to_row[cur_name]][name_to_col[cur_metric_name]] = (
                "%.4f," % avg_results[f"{cur_metric_name} - {cur_name}"]
            )

        final_avg_results[cur_metric_name] /= 3
        result_format_list[4][name_to_col[cur_metric_name]] = (
            "%.4f," % final_avg_results[cur_metric_name]
        )

    result_format_str = " ".join(
        [x.rjust(12) for items in result_format_list for x in items]
    )

    result_dict["--------------------------------------------------------------"] = 0
    result_dict.update(final_avg_results)
    result_dict["---------------------------------------------------------------"] = 0
    result_dict.update(object_type_cnt_dict)
    result_dict[
        "-----Note that this evaluation may have marginal differences with the official Waymo evaluation server-----"
    ] = 0

    return result_dict, result_format_str


def main():
    import pickle
    import argparse

    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--pred_infos", type=str, default=None, help="pickle file")
    parser.add_argument("--top_k", type=int, default=-1, help="")
    parser.add_argument("--eval_second", type=int, default=8, help="")
    parser.add_argument("--num_modes_for_eval", type=int, default=6, help="")

    args = parser.parse_args()
    print(args)

    assert args.eval_second in [3, 5, 8]
    pred_infos = pickle.load(open(args.pred_infos, "rb"))

    result_format_str = ""
    print("Start to evaluate the waymo format results...")

    metric_results, result_format_str = nuscenes_evaluation(
        pred_dicts=pred_infos,
        top_k=args.top_k,
        eval_second=args.eval_second,
        num_modes_for_eval=args.num_modes_for_eval,
    )

    print(metric_results)
    metric_result_str = "\n"
    for key in metric_results:
        metric_results[key] = metric_results[key]
        metric_result_str += "%s: %.4f \n" % (key, metric_results[key])
    print(metric_result_str)
    print(result_format_str)


if __name__ == "__main__":
    main()
