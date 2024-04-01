from evaluation.classification import Classification
from evaluation.event_completion import EventCompletion
from evaluation.kendalls_tau import KendallsTau
from evaluation.retrieval import Retrieval
from easydict import EasyDict as edict

"""
References:
[1] https://github.com/google-research/google-research/tree/master/tcc
[2] https://github.com/minghchen/CARL_code
[3] https://github.com/taeinkwon/CASA
"""

TASKNAME = {
    'classification': Classification,
    'event_completion': EventCompletion,
    'kendalls_tau': KendallsTau,
    'retrieval': Retrieval
}

def get_tasks(cfg):
    cfg.eval = edict()

    embedding_tasks = {}

    cfg.eval.classification_fractions = [0.1, 0.5, 1.0]
    cfg.eval.kendall_tau_distance = "sqeuclidean"
    cfg.eval.kendall_tau_stride = 5
    cfg.eval.retrieval_ks = [5, 10, 15]
    cfg.eval.val_interval = 50

    for task_name, task in TASKNAME.items():
        task = TASKNAME[task_name](cfg)
        embedding_tasks[task_name] = task
    return embedding_tasks