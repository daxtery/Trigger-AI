from trigger.train.transformers.user_transformer import UserInstance
from trigger.train.transformers.opening_transformer import OpeningInstance

from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.spatial.distance import cosine
from typing import Dict

def quality_metric(user: User, opening: OpeningInstance):

    u_h = set(user.hardSkills)
    u_s = set(user.softSkills)

    o_h = set(opening.hardSkills)
    o_s = set(opening.softSkills)

    if len(o_h) == 0:
        HS_s = 1
    else:
        HS_s = len(o_h.intersection(u_h))/len(o_h)

    if len(o_s) == 0:
        SS_s = 1

    else:
        SS_s = len(o_s.intersection(u_s))/len(o_s)

    Mq = 0.6*HS_s + 0.4*SS_s

    return Mq

def eval_cluster(cluster) -> Dict[str, float]:

    X = cluster.instances
    labels = []

    for x in X:

        labels.append(cluster.get_cluster(x))

    try:

        Ss = silhouette_score(X, labels)
    except:

        Ss = -1.0
    try:
        CHs = calinski_harabasz_score(X, labels)

    except:
        CHs = 0

    return {'ss': Ss, 'chs': CHs}

def computeScore(userInstance: UserInstance, openingInstance: OpeningInstance) -> float:

    return 1 - cosine(userInstance.embedding, openingInstance.embedding)