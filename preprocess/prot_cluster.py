import pickle

import numpy as np
import json
from scipy.cluster.hierarchy import fcluster, linkage, single


def protein_clustering(protein_list, idx_list):
    print('start protein clustering...')
    protein_sim_mat = np.load('/home/s2136015/Code/DeepTarget_final/data/prot_distance.npy').astype(np.float32)
    sim_mat = protein_sim_mat[idx_list, :]
    sim_mat = sim_mat[:, idx_list]
    print('original protein sim_mat', protein_sim_mat.shape, 'subset sim_mat', sim_mat.shape)
    # np.save('../preprocessing/'+MEASURE+'_protein_sim_mat.npy', sim_mat)
    P_dist = []
    for i in range(sim_mat.shape[0]):
        P_dist += (sim_mat[i, (i + 1):]).tolist()
    P_dist = np.array(P_dist)
    P_link = single(P_dist)
    for thre in [0.3, 0.4, 0.5, 0.6]:
        P_clusters = fcluster(P_link, thre, 'distance')
        len_list = []
        for i in range(1, max(P_clusters) + 1):
            len_list.append(P_clusters.tolist().count(i))
        print('thre', thre, 'total num of proteins', len(protein_list), 'num of clusters', max(P_clusters),
              'max length', max(
                len_list))
        P_cluster_dict = {protein_list[i]: P_clusters[i] for i in range(len(protein_list))}
        with open('_protein_cluster_dict_' + str(thre), 'wb') as f:
            pickle.dump(P_cluster_dict, f, protocol=0)


if __name__ == '__main__':
    """path = '/home/s2136015/Code/DeepTarget_final/data/pdbbind_protein_sim_mat.npy'
    protein_sim_mat = np.load(path).astype(np.float32)"""

    prot_dict_path = '/home/s2136015/Code/DeepTarget_final/data/prot_dict.json'
    with open(prot_dict_path, 'r') as load_f:
        load_dict = json.load(load_f)
    protein_list = load_dict.keys()
    idx_list = range(len(protein_list))
    protein_clustering(protein_list, idx_list)
