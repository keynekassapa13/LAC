# coding=utf-8
import torch
import numpy as np
from scipy.spatial.distance import cdist

DATASET_TO_NUM_CLASSES = {
    'pouring': 5,
    'baseball_pitch': 4,
    'baseball_swing': 3,
    'bench_press': 2,
    'bowl': 3,
    'clean_and_jerk': 6,
    'golf_swing': 3,
    'jumping_jacks': 4,
    'pushup': 2,
    'pullup': 2,
    'situp': 2,
    'squat': 4,
    'tennis_forehand': 3,
    'tennis_serve': 4,
}

from loguru import logger

class Retrieval(object):
    """Calculate Retrieval AP."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.downstream_task = True
        self.K_list = cfg.eval.retrieval_ks
        self.dist_type = cfg.eval.kendall_tau_distance
        self.stride = cfg.eval.kendall_tau_stride

    def evaluate(self, dataset, cur_epoch, summary_writer):
        """Labeled evaluation."""
        self.num_classes = DATASET_TO_NUM_CLASSES[dataset['name']]
        val_embs = dataset['val_dataset']['embs']
        val_labels = dataset['val_dataset']['labels']
        val_APs = []
        for K in self.K_list:
            val_APs.append(self.get_AP(val_embs, val_labels, K, 
                            cur_epoch, summary_writer, '%s_val' % dataset['name'], visualize=True))
        return val_APs[0]
    
    def get_AP(self, embs_list, label_list, K, cur_epoch, summary_writer, split, visualize=False):
        """Get topK in embedding space and calculate average precision."""
        num_seqs = len(embs_list)
        precisions = np.zeros(num_seqs)
        idx = 0
        for i in range(num_seqs):
            query_feats = embs_list[i][::self.stride]
            query_label = label_list[i][::self.stride]

            candidate_feats = []
            candidate_label = []
            for j in range(num_seqs):
                if i != j:
                    candidate_feats.append(embs_list[j][::self.stride])
                    candidate_label.append(label_list[j][::self.stride])
            candidate_feats = np.concatenate(candidate_feats, axis=0)
            candidate_label = np.concatenate(candidate_label, axis=0)
            dists = cdist(query_feats, candidate_feats, self.dist_type)
            topk = np.argsort(dists, axis=1)[:, :K]
            ap = 0
            for t in range(len(query_feats)):
                ap += np.mean(int(query_label[t]) == candidate_label[topk[t]])
            precisions[idx] = ap / len(query_feats)
            idx += 1
        # Remove NaNs.
        precisions = precisions[~np.isnan(precisions)]
        precision = np.mean(precisions)

        logger.info('epoch[{}/{}] {} set AP@{} precision: {:.2%}'.format(
            cur_epoch, self.cfg.trainer.epochs, split, K, precision))

        summary_writer.add_scalar(f'AP/{split} set {K}_align_precision', precision, cur_epoch)
        return precision