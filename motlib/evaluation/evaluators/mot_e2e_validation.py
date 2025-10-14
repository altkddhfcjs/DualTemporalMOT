# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import copy

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”). All Bytedance's Modifications are Copyright (2023) Bytedance Ltd. and/or its affiliates. 

import cv2
from util.utils import slprint, to_device
from . import EVALUATOR_REGISTRY
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os
import time
import pickle
import torch
import logging
from copy import deepcopy
from pathlib import Path
import util.misc as utils
import torch.distributed as dist
from motlib.evaluation.evaluate_metrics.cocoeval import CocoEvaluator
from motlib.evaluation.evaluate_metrics import mot_eval_metrics
from motlib.utils import set_dir, time_synchronized
from motlib.evaluation.utils.result_process import filter_result
from motlib.mot_models.network.dino_mot.tracker.manager import E2ETrackManager
from motlib.tracker.interpolation import GSInterpolation as iptrack

# matching
from motlib.mot_models.structures import Instances, Boxes, pairwise_iou
from scipy.optimize import linear_sum_assignment
from torchvision.ops import nms
from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
import numpy as np

__all__ = ['evaluate_e2e', 'evaluate_e2e_debug']

def matching_pred_gt(pred, gt):
    ious = pairwise_iou(Boxes(pred.boxes), Boxes(gt.boxes))  # .squeeze(1)
    # if len(gt_instances) == 1:
    #     ious = ious.unsqueeze(-1)
    similarity = ious.cpu().numpy()

    sim_iou_denom = ious.sum(0)[np.newaxis, :] + ious.sum(1)[:, np.newaxis] - ious
    sim_iou = np.zeros_like(similarity)
    sim_iou_mask = sim_iou_denom > 0 + np.finfo('float').eps
    sim_iou[torch.where(sim_iou_mask)] = similarity[torch.where(sim_iou_mask)] / sim_iou_denom[
        torch.where(sim_iou_mask)]
    potential_matches_count = sim_iou
    global_alignment_score = potential_matches_count / (
                len(gt) + len(pred) - potential_matches_count)
    score_mat = global_alignment_score * similarity
    match_rows, match_cols = linear_sum_assignment(-score_mat)
    actually_matched_mask = similarity[match_rows, match_cols] >= 0.3
    src_idx = torch.tensor(match_rows[actually_matched_mask])
    tgt_idx = torch.tensor(match_cols[actually_matched_mask])
    return src_idx, tgt_idx

def draw_image(image, track_boxes, track_ids, cmap):
    output_image = image.copy()
    # Generate unique colors for each track ID
    colors = {track_id: (np.array(cmap(track_id)[:3]) * 255).astype(int).tolist() for track_id in track_ids}
    for track_id, box in zip(track_ids, track_boxes):
        x1, y1, x2, y2 = map(int, box)
        box_color = colors[track_id]
        cv2.rectangle(output_image, (x1, y1), (x2, y2), box_color, 5)
        cv2.putText(output_image, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color, 3)
    return output_image


@EVALUATOR_REGISTRY.register()
@torch.no_grad()
def evaluate_e2e_debug(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False,
                 args=None):
    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    savefolder = output_dir / 'results'
    cmap = plt.cm.get_cmap('tab20', 20)
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        logger.info("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    _cnt = 0
    output_state_dict = {'target': [], 'result': []}  # for debug only

    # load gt
    gt_path = "mot_files/dataset/dancetrack/val"
    # gt_image_path = ""
    track_instances = None
    valid_frame_num = 1  # 3189 #len(data_loader.dataset.coco.dataset['videos']) #data_loader.sampler.valid_num
    track_mannager = E2ETrackManager(args, valid_frame_num, deepcopy(data_loader.dataset.coco.dataset['videos']))

    inference_time = 0
    track_time = 0
    n_samples = 0
    warmup_iter = min(100, len(data_loader) // 3)

    false_frame = pickle.load(open("false_record.pkl", "rb"))

    video_information = data_loader.dataset.coco.videoToImgs
    video_seq_info = {video_information[i][0]['video_id']: video_information[i][0]['file_name'].split("/")[4] for i in
                      video_information}
    video_dict = {video_information[i][0]['video_id']: len(video_information[i]) for i in video_information}

    current_video_id = -1
    for cur_iter, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header, logger=logger)):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        video_id = int(targets[0]['video_id'])

        if current_video_id != video_id:
            current_video_id = video_id
            track_mannager.valid_frame_num = video_dict[video_id]
            track_mannager.num = 0

        is_time_record = (cur_iter < len(data_loader) - 1) and cur_iter > warmup_iter
        if is_time_record:
            n_samples += len(targets)
            start = time.time()

        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples, targets, track_instances)
            track_instances = outputs['track_instances']

        if is_time_record:
            infer_end = time_synchronized()
            inference_time += infer_end - start

        pred_ids = outputs['pred'][0].obj_idxes
        results = track_mannager.update(outputs['pred'], targets)

        if is_time_record:
            track_end = time_synchronized()
            track_time += track_end - infer_end

        _cnt += len(results)

        if len(results) > 0:

            if coco_evaluator is not None:
                coco_results = {}
                for img_id, det_res in results:
                    coco_results[img_id] = det_res
                coco_evaluator.update(coco_results)
            if args.save_results:
                for i, (img_id, res) in enumerate(results):
                    tgt = targets[i]
                    assert tgt['image_id'].item() == img_id
                    _image_id = tgt['image_id']
                    _frame_id = tgt['frame_id']
                    _video_id = tgt['video_id']
                    _res_box = res['boxes']
                    _res_score = res['scores']
                    _res_label = res['labels']

                    gt_info = torch.cat((_image_id, _frame_id, _video_id), 0)

                    res_info = torch.cat((_res_box, _res_score.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                    # import ipdb;ipdb.set_trace()

                    output_state_dict['target'].append(gt_info.cpu())
                    output_state_dict['result'].append(res_info.cpu())
        del targets, samples, results

    if args.save_results:
        savepath = savefolder / 'results-{}.pkl'.format(utils.get_rank())
        savepath = str(savepath)
        logger.info("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: {}".format(str(metric_logger)))
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    panoptic_res = None
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()

        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    # dist.barrier()
    res_record = torch.FloatTensor(5).cuda()
    res_record_dti = torch.FloatTensor(5).cuda()
    if utils.get_rank() in [-1, 0]:

        track_results_path = '{}/{}/track_results'.format(args.output_dir, args.tracker_name)
        interpolation_track_results_path = '{}/{}/track_results'.format(args.output_dir, 'IPTrack')
        set_dir(interpolation_track_results_path)
        iptrack(txt_path=track_results_path, save_path=interpolation_track_results_path)

        if not args.just_inference_flag:
            dataset_name = data_loader.dataset.dataset_name
            assert len(dataset_name) == 1
            output_res, _ = mot_eval_metrics(args, dataset_name=dataset_name[0],
                                             eval_config=data_loader.dataset.coco.eval_config)
            res_record = filter_result(stats, output_res, args.tracker_name,
                                       data_loader.dataset.coco.eval_config['dataset_class'], res_record)
            res_record_dti = filter_result(stats, output_res, 'IPTrack',
                                           data_loader.dataset.coco.eval_config['dataset_class'], res_record_dti)
    if utils.is_dist_avail_and_initialized():
        dist.barrier()
        dist.broadcast(res_record, 0)
        dist.broadcast(res_record_dti, 0)
    output = {'Det': res_record[0].item(), 'HOTA': res_record[1].item(), 'MOTA': res_record[2].item(),
              'IdSW': res_record[3].item(), 'IDF1': res_record[4].item(), 'HOTA_dti': res_record_dti[1].item(),
              'MOTA_dti': res_record_dti[2].item(), 'IdSW_dti': res_record_dti[3].item(),
              'IDF1_dti': res_record_dti[4].item()}
    return output

    # cur_img = cv2.imread(os.path.join(gt_path, seq_name, "img1/{0:08d}.jpg".format(track_mannager.num)))
    # prev_img = cv2.imread(os.path.join(gt_path, seq_name, "img1/{0:08d}.jpg".format(track_mannager.num - 1)))
    #
    # prev_track_boxes = previous_track_instances.boxes.cpu().numpy().tolist()
    # prev_track_ids = previous_track_instances.obj_idxes.cpu().numpy().tolist()
    # draw_prev_img = draw_image(prev_img, prev_track_boxes, prev_track_ids, cmap)
    #
    # track_boxes = track_instances.boxes.cpu().numpy().tolist()
    # track_ids = track_instances.obj_idxes.cpu().numpy().tolist()
    # draw_cur_img = draw_image(img, track_boxes, track_ids, cmap)
    #
    # fig, axes = plt.subplots(1, 2, figsize=(32, 16))
    # axes[0].imshow(draw_prev_img[:, :, ::-1])
    # axes[0].set_title("previous")
    # axes[1].imshow(draw_cur_img[:, :, ::-1])
    # axes[1].set_title("current")
    # plt.show()




def check_draw(gt_path, seq_name, track_mannager, det_outputs, track_instances, cmap):
    img = cv2.imread(os.path.join(gt_path, seq_name, "img1/{0:08d}.jpg".format(track_mannager.num)))
    track_boxes = det_outputs.boxes.cpu().numpy().tolist()
    track_ids = torch.arange(0, len(det_outputs)).cpu().numpy().tolist()
    vis_img = draw_image(img, track_boxes, track_ids, cmap)

    track_boxes = track_instances.boxes.cpu().numpy().tolist()
    track_ids = track_instances.obj_idxes.cpu().numpy().tolist()
    vis_img2 = draw_image(img, track_boxes, track_ids, cmap)

    fig, axes = plt.subplots(1, 2, figsize=(32, 16))
    axes[0].imshow(vis_img[:, :, ::-1])
    axes[0].set_title("detected")
    axes[1].imshow(vis_img2[:, :, ::-1])
    axes[1].set_title("track")
    plt.show()


@EVALUATOR_REGISTRY.register()
@torch.no_grad()
def evaluate_e2e(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None):
    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    savefolder = output_dir / 'results'
    cmap = plt.cm.get_cmap('tab20', 20)
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    _cnt = 0
    output_state_dict = {'target': [], 'result': []} # for debug only

    # load gt
    gt_path = "mot_files/dataset/dancetrack/val"
    # gt_image_path = ""
    track_instances = None
    valid_frame_num = 1
    track_mannager = E2ETrackManager(args, valid_frame_num, deepcopy(data_loader.dataset.coco.dataset['videos']))

    inference_time = 0
    track_time = 0
    n_samples = 0
    warmup_iter = min(100, len(data_loader)//3)

    false_frame = pickle.load(open("false_record.pkl", "rb"))

    video_information = data_loader.dataset.coco.videoToImgs
    video_seq_info = {video_information[i][0]['video_id']: video_information[i][0]['file_name'].split("/")[4] for i in video_information}
    video_dict = {video_information[i][0]['video_id']: len(video_information[i]) for i in video_information}
    
    labels_full = None
    
    trajectory = None
    traj_samples = None
    traj_targets = None
    traj_gt = None

    gt_traj = {}

    # false_seq = ["dancetrack0019", "dancetrack0004"] # "dancetrack0081"] #, "dancetrack0047"]

    seq_name = ""
    total_miss_match_scores = []
    detection_not_matching = 0
    reverse_not_matching = 0
    previous_track_instances = None
    current_video_id = -1
    for cur_iter, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header, logger=logger)):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        video_id = int(targets[0]['video_id'])

        if current_video_id != video_id:
            previous_track_instances = None
            current_video_id = video_id
            track_mannager.valid_frame_num = video_dict[video_id]
            track_mannager.num = 0

            ## gt matching ##
            seq_name = video_seq_info[video_id]
            trajectory = {}
            traj_samples = {}
            traj_targets = {}
            traj_gt = {}
            seq_gt_path = os.path.join(gt_path, seq_name, "gt/gt.txt")
            labels_full = {}
            record_traj_ids = []
            for l in open(seq_gt_path):
                t, i, *xywh, mark, label = l.strip().split(',')[:8]
                t, i, mark, label = map(int, (t, i, mark, label))
                if t-1 not in labels_full:
                    labels_full[t-1] = []
                if mark == 0:
                    continue
                if label in [3, 4, 5, 6, 9, 10, 11]:  # Non-person
                    continue
                else:
                    crowd = False
                x, y, w, h = map(float, (xywh))
                labels_full[t-1].append([x, y, x+w, y+h, i, crowd])

            error_frames = false_frame[seq_name]
            seq_error_frame = {}
            for e in error_frames:
                gt_id = e['gt_id']
                pred_id = e['pred_id']
                frames = e['frames']
                for f_id in frames:
                    f_id = int(f_id)
                    if f_id == -1:
                        continue
                    if f_id not in seq_error_frame:
                        seq_error_frame[f_id] = []
                    seq_error_frame[f_id].append({
                        "gt_id": gt_id,
                        "pred_id": pred_id - 1
                    })
            ####################################################3

        # if seq_name not in false_seq:
        #     continue

        is_time_record = (cur_iter < len(data_loader) - 1) and cur_iter > warmup_iter
        if is_time_record:
            n_samples += len(targets)
            start = time.time()

        if track_instances is not None:
            previous_track_instances = copy.deepcopy(track_instances)

        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples, targets, track_instances)
            track_instances = outputs['track_instances']
        
        if is_time_record:
            infer_end = time_synchronized()
            inference_time += infer_end - start

        gt_tensor = torch.tensor(labels_full[track_mannager.num])

        gt_instances = Instances((1, 1))
        gt_instances.boxes = gt_tensor[..., :4]
        gt_instances.track_id = gt_tensor[..., 4].to(torch.int)
        src_idx, tgt_idx = matching_pred_gt(track_instances.to('cpu'), gt_instances.to('cpu'))

        track_instances.gt_ids = torch.zeros(len(track_instances), dtype=torch.int).to(track_instances.boxes.device) - 1
        track_instances.gt_ids[src_idx] = gt_instances.track_id[tgt_idx].to(track_instances.boxes.device)

        track_id = gt_instances.track_id.cpu().numpy().tolist()
        for i in track_id:
            if i not in gt_traj:
                gt_traj[i] = []
        track_instances.matched_indices = torch.zeros(len(track_instances), dtype=torch.int).to(track_instances.boxes.device)# - 1

        if len(trajectory) > 1:
            det_outputs = model(samples, targets, None, detected=True, threshold=0.4)['track_instances']
            # reversed_instances = model(traj_samples[track_mannager.num - 1], traj_targets[track_mannager.num - 1], track_instances, reverse=True)['track_instances']
            nms_indices = nms(det_outputs.boxes, det_outputs.scores, 0.5)
            det_outputs = det_outputs[nms_indices]
            if len(det_outputs) != len(track_instances):
                det_src, det_tgt = matching_pred_gt(det_outputs.to('cpu'), gt_instances.to('cpu'))
                det_outputs.gt_ids = torch.zeros(len(det_outputs), dtype=torch.int).to(det_outputs.boxes.device) - 1
                det_outputs.gt_ids[det_src] = gt_instances.track_id[det_tgt].to(det_outputs.boxes.device)

                # det_pos = det_outputs.query_pos
                # track_pos = track_instances.query_pos
                # appearance_matrix = torch.softmax(det_pos @ track_pos.T, dim=1).cpu().numpy()
                iou_matrix = pairwise_iou(Boxes(det_outputs.boxes), Boxes(track_instances.boxes)).cpu().numpy()
                similarity_matrix = iou_matrix #appearance_matrix * 0.2 + iou_matrix * 0.8
                cost_matrix = -similarity_matrix
                cost_matrix[similarity_matrix < 0.3] = 1e9
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                matched_indices = [
                    (row, col) for row, col in zip(row_indices, col_indices)
                    if similarity_matrix[row, col] >= 0.3
                ]
                matched_indices = torch.tensor(matched_indices)

                matched_check = torch.zeros_like(track_instances.matched_indices) - 1
                matched_check[matched_indices[..., 1]] = 0
                track_instances.matched_indices[matched_check == -1] = -1
                det_outputs.obj_idxes[matched_indices[..., 0]] = track_instances.obj_idxes[matched_indices[..., 1]].clone()
                # matched_indices = torch.tensor(matched_indices)
                # det_gt_ids = det_outputs[matched_indices[..., 0]].gt_ids
                # track_gt_ids = track_instances[matched_indices[..., 1]].gt_ids
                #
                # if torch.sum(det_gt_ids != track_gt_ids) > 0:
                #     print("not matching")

        # if track_mannager.num in seq_error_frame:
        #     print("error frame")

        if torch.sum(track_instances.matched_indices == -1) > 0:
            not_detected_track = track_instances[track_instances.matched_indices == -1]
            # total_miss_match_scores.append(not_detected_track.scores)
            not_detected_track = not_detected_track[not_detected_track.scores < 0.9]
            if len(not_detected_track) > 0:
                not_obj_idxes = not_detected_track.obj_idxes.cpu().numpy().tolist()
                # previous_t1 = previous_track_instances
                # previous_t2 = trajectory[track_mannager.num - 2]
                # previous_t3 = trajectory[track_mannager.num - 3]
                for tmp_idx in not_obj_idxes:
                    # matched_t1 = previous_t1[previous_t1.obj_idxes == tmp_idx]
                    # matched_t2 = previous_t2[previous_t2.obj_idxes == tmp_idx]
                    # matched_t3 = previous_t3[previous_t3.obj_idxes == tmp_idx]

                    # previous_ = Instances.cat([matched_t1, matched_t2) #, matched_t3])

                    matched_prev_track_instances = previous_track_instances[previous_track_instances.obj_idxes == tmp_idx]
                    if len(matched_prev_track_instances) < 1:
                        continue

                    # max_idx = int(torch.max(previous_.scores, dim=0)[1])
                    # matched_prev_track_instances = previous_[max_idx]

                    track_idx = int(torch.where(track_instances.obj_idxes == tmp_idx)[0])
                    # if matched_prev_track_instances.scores > track_instances[track_idx].scores:
                    track_instances.mem_bank[track_idx] = matched_prev_track_instances.mem_bank[0]
                    track_instances.ref_bank[track_idx] = matched_prev_track_instances.ref_bank[0]
                    # track_instances.query_pos[track_instances.obj_idxes == tmp_idx] = matched_prev_track_instances.query_pos
                    # track_instances.output_embedding[track_instances.obj_idxes == tmp_idx] = matched_prev_track_instances.output_embedding
                    # track_instances.ref_pts[track_instances.obj_idxes == tmp_idx] = matched_prev_track_instances.ref_pts
                    track_instances.scores[track_instances.obj_idxes == tmp_idx] = 0.5
                # print("")

        # if previous_track_instances is not None:
        #     for prev_idx in range(len(previous_track_instances)):
        #         prev_instance = previous_track_instances[prev_idx]
        #         prev_obj_idx = int(prev_instance.obj_idxes)
        #         mathced_track_instance = track_instances[track_instances.obj_idxes == prev_obj_idx]
        #         if mathced_track_instance.gt_ids != prev_instance.gt_ids:
        #             print("track false")

        # matched_track_id = gt_instances.track_id[tgt_idx].cpu().numpy().tolist()
        # for det_idx, gt_track_id in enumerate(matched_track_id):
        #     track_obj_idxes = int(track_instances[int(src_idx[det_idx])].obj_idxes)
        #     if len(gt_traj[gt_track_id]) != 0 and gt_traj[gt_track_id][-1] != track_obj_idxes:
        #         print("unmatched")
        #     gt_traj[gt_track_id].append(track_obj_idxes)

        # for inst_idx, i in enumerate(track_instances.obj_idxes):
        #     if i not in record_traj_ids:
        #         if len(trajectory) > 0:
        #             reverse_instances = model(traj_samples[track_mannager.num - 1], traj_targets[track_mannager.num - 1], track_instances, reverse=True)['track_instances']
        #             if len(det_outputs) != len(track_instances):
        #                 detection_not_matching += 1
        #                 # print("")
        #             if len(reverse_instances) != len(track_instances):
        #                 reverse_not_matching += 1
        #                 # print("")
        #         else:
        #             record_traj_ids.append(i)

        # for inst_idx, i in enumerate(track_instances.obj_idxes):
        #     if i not in record_traj_ids:
        #         if len(trajectory) > 0:
        #             matched_check = track_instances.matched_indices[inst_idx]
        #             if matched_check == 0:
        #                 record_traj_ids.append(i)
        #         else:
        #             record_traj_ids.append(i)
        #     else:
        #         # # First frame
        #         # if len(trajectory) > 0:
        #         matched_check = track_instances.matched_indices[inst_idx]
        #         if matched_check == -1:
        #             prev_inst = previous_track_instances[previous_track_instances.obj_idxes == i]
        #
        #             track_instances.query_pos[inst_idx] = prev_inst.query_pos[0]
        #             track_instances.ref_pts[inst_idx] = prev_inst.ref_pts[0]
        #             track_instances.output_embedding[inst_idx] = prev_inst.output_embedding[0]
        #             track_instances.mem_bank[inst_idx] = prev_inst.mem_bank[0]
        #             track_instances.disappear_time[inst_idx] += 1
        #             track_instances.scores[inst_idx] = 0.1
        #             track_instances.ref_bank[inst_idx] = prev_inst.ref_bank[0]

        # matched_track_id = gt_instances.track_id[tgt_idx].cpu().numpy().tolist()
        # for det_idx, gt_track_id in enumerate(matched_track_id):
        #     track_obj_idxes = int(track_instances[int(src_idx[det_idx])].obj_idxes)
        #     if len(gt_traj[gt_track_id]) != 0 and gt_traj[gt_track_id][-1] != track_obj_idxes:
        #         # print("")
        #         previous_track_instances = trajectory[track_mannager.num - 1]
        #         false_instance = previous_track_instances[previous_track_instances.obj_idxes == track_obj_idxes]
        #
        #         img = cv2.imread(os.path.join(gt_path, seq_name, "img1/{0:08d}.jpg".format(track_mannager.num)))
        #         track_boxes = track_instances.boxes.cpu().numpy().tolist()
        #         track_ids = track_instances.obj_idxes.cpu().numpy().tolist()
        #         vis_img = draw_image(img, track_boxes, track_ids, cmap)
        #
        #     gt_traj[gt_track_id].append(track_obj_idxes)

        trajectory[track_mannager.num] = track_instances
        traj_samples[track_mannager.num] = samples
        traj_targets[track_mannager.num] = targets
        traj_gt[track_mannager.num] = gt_instances

        for i in track_instances.obj_idxes:
            if i not in record_traj_ids:
                record_traj_ids.append(i)
                if track_mannager.num != 0:
                    # print("new object")
                    if track_mannager.num - 1 < 0:
                        continue

                    reversed_instances = model(traj_samples[track_mannager.num - 1], traj_targets[track_mannager.num - 1], track_instances, reverse=True)['track_instances']
                    reversed_score = reversed_instances.scores[reversed_instances.obj_idxes == i][0]
                    if reversed_score > 0.8:
                        # maybe..
                        # print(reversed_score)
                        ious = pairwise_iou(Boxes(reversed_instances[reversed_instances.obj_idxes == i].boxes), Boxes(trajectory[track_mannager.num - 1].boxes))
                        if torch.max(ious) > 0.5:
                            # if torch.sum(ious > 0.2) > 0:
                            #     print(torch.max(ious))
                            track_instances = track_instances[track_instances.obj_idxes != i]

        # if 4 in track_instances.obj_idxes:
        #     new_instances = track_instances[track_instances.obj_idxes == 4]
        #     reversed_instances = model(traj_samples[track_mannager.num - 1], traj_targets[track_mannager.num - 1], track_instances)['track_instances']
        #     similarity = torch.softmax(track_instances.query_pos @ track_instances.query_pos.T, dim=0)
        #
        #     print("")

        # if track_mannager.num == 12:
        #     print("")

        # if track_mannager.num == 597:
        #     print("")

        # if track_mannager.num > 0:
        #     match_labels = previous_track_instances.gt_ids.view(-1, 1) == track_instances.gt_ids.view(1, -1)
        #     match_labels[previous_track_instances.gt_ids == -1] = False
        #     match_labels[:, track_instances.gt_ids == -1] = False
        #     similarity = torch.softmax(previous_track_instances.query_pos @ track_instances.query_pos.T, dim=0)
        #     max_score, max_indices = torch.max(similarity, dim=1)
        #     # max_indices[max_score < 0.5]
        #
        #     for idx, ind in enumerate(max_indices):
        #         if previous_track_instances[int(idx)].gt_ids == -1:
        #             continue
        #         if track_instances[int(ind)].gt_ids == -1:
        #             continue
        #         if not match_labels[idx, ind]:
        #             print("Error")

        # if track_mannager.num == 605:
        #     print("")

        # if track_mannager.num - 1 != 0 and track_mannager.num - 1 in seq_error_frame:
        #     error_instances = seq_error_frame[track_mannager.num - 1]
        #     current_ti = track_instances[track_instances.obj_idxes == error_instances[0]['pred_id']]
        #     # previous_ti = previous_track_instances[track_instances.obj_idxes == error_instances[0]['pred_id']]
        #
        #     # previous_mem_query_pos = torch.stack([p[0] for p in previous_track_instances.mem_bank])
        #     # previous_mem_query_pos_t2 = torch.stack([p[1] for p in previous_track_instances.mem_bank])
        #     # previous_mem_query_pos_t3 = torch.stack([p[2] for p in previous_track_instances.mem_bank])
        #
        #     # torch.softmax(current_ti.query_pos @ previous_mem_query_pos.T, dim=1)
        #     # torch.softmax(current_ti.query_pos @ previous_mem_query_pos_t2.T, dim=1)
        #     # torch.softmax(current_ti.query_pos @ previous_mem_query_pos_t3.T, dim=1)
        #
        #     pred_ids_matches = previous_track_instances.obj_idxes.view(-1, 1) == track_instances.obj_idxes.view(1, -1)
        #     # track_instances_t3 = trajectory[track_mannager.num - 10]
        #     # outputs = model(samples, targets, track_instances_t3)
        #     # new_track_instances = outputs['track_instances']
        #     # new_src_idx, new_tgt_idx = matching_pred_gt(new_track_instances.to('cpu'), gt_instances.to('cpu'))
        #
        #     # img = cv2.imread(os.path.join(gt_path, seq_name, "img1/{0:08d}.jpg".format(track_mannager.num)))
        #     print("Error frame")

        pred_ids = outputs['pred'][0].obj_idxes
        results = track_mannager.update(outputs['pred'], targets)

        if is_time_record:
            track_end = time_synchronized()
            track_time += track_end - infer_end

        _cnt += len(results)

        if len(results) > 0:

            if coco_evaluator is not None:
                coco_results = {}
                for img_id, det_res in results:
                    coco_results[img_id] = det_res
                coco_evaluator.update(coco_results)
            if args.save_results:
                for i, (img_id, res) in enumerate(results):
                    tgt = targets[i]
                    assert tgt['image_id'].item() == img_id
                    _image_id = tgt['image_id']
                    _frame_id = tgt['frame_id']
                    _video_id = tgt['video_id']
                    _res_box = res['boxes']
                    _res_score = res['scores']
                    _res_label = res['labels']

                    gt_info = torch.cat((_image_id, _frame_id, _video_id), 0)
                    
                    res_info = torch.cat((_res_box, _res_score.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                    # import ipdb;ipdb.set_trace()

                    output_state_dict['target'].append(gt_info.cpu())
                    output_state_dict['result'].append(res_info.cpu())
        del targets, samples, results

    # resnet_forward_time = model.module.dino.timer_resnet.avg_seconds()
    # encoder_forward_time = model.module.dino.timer_encoder.avg_seconds()
    # decoder_forward_time = model.module.dino.timer_decoder.avg_seconds()
    # tracking_forward_time = model.module.dino.timer_tracking.avg_seconds()
    
    # statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples, resnet_forward_time, encoder_forward_time, decoder_forward_time, tracking_forward_time])
    # if utils.is_dist_avail_and_initialized():
    #     torch.distributed.reduce(statistics, dst=0)
    
    # inference_time = statistics[0].item()
    # track_time = statistics[1].item()
    # n_samples = statistics[2].item()

    # resnet_forward_time = statistics[3].item()
    # encoder_forward_time = statistics[4].item()
    # decoder_forward_time = statistics[5].item()
    # tracking_forward_time = statistics[6].item()

    # a_infer_time = 1000 * inference_time / n_samples
    # a_track_time = 1000 * track_time / n_samples

    # time_info = ", ".join(
    #     [
    #         "Average {} time: {:.2f} ms".format(k, v)
    #         for k, v in zip(
    #             ["forward", "track", "inference"],
    #             [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
    #         )
    #     ]
    # )
    # time_info += ' fps {:.2f}'.format(1000 / (a_infer_time + a_track_time))
    # logger.info(time_info)

    # logger.info('Timer Resnet {:.2f} Encoder {:.2f} Decoder {:.2f} Track update {:.2f} Total {:.2f}'.format(resnet_forward_time, encoder_forward_time, decoder_forward_time, tracking_forward_time, inference_time))

    if args.save_results:
        savepath = savefolder/ 'results-{}.pkl'.format(utils.get_rank())
        savepath = str(savepath)
        logger.info("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: {}".format(str(metric_logger)))
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
            
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    # dist.barrier()
    res_record = torch.FloatTensor(5).cuda()
    res_record_dti = torch.FloatTensor(5).cuda()
    if utils.get_rank() in [-1, 0]:

        track_results_path = '{}/{}/track_results'.format(args.output_dir, args.tracker_name)
        interpolation_track_results_path = '{}/{}/track_results'.format(args.output_dir, 'IPTrack')
        set_dir(interpolation_track_results_path)
        iptrack(txt_path=track_results_path, save_path=interpolation_track_results_path)

        if not args.just_inference_flag:
            dataset_name = data_loader.dataset.dataset_name
            assert len(dataset_name) == 1
            output_res, _ = mot_eval_metrics(args, dataset_name=dataset_name[0], eval_config=data_loader.dataset.coco.eval_config)
            res_record = filter_result(stats, output_res, args.tracker_name, data_loader.dataset.coco.eval_config['dataset_class'], res_record)
            res_record_dti = filter_result(stats, output_res, 'IPTrack', data_loader.dataset.coco.eval_config['dataset_class'], res_record_dti)
    if utils.is_dist_avail_and_initialized():
            dist.barrier()
            dist.broadcast(res_record, 0)
            dist.broadcast(res_record_dti, 0)
    output = {'Det':res_record[0].item(),'HOTA': res_record[1].item(), 'MOTA': res_record[2].item(), 'IdSW': res_record[3].item(), 'IDF1': res_record[4].item(), 'HOTA_dti': res_record_dti[1].item(), 'MOTA_dti': res_record_dti[2].item(), 'IdSW_dti': res_record_dti[3].item(), 'IDF1_dti': res_record_dti[4].item()}
    return output
