## Data warehousing pracitce
## written by Lei Xie

import numpy as np
import heapq

###################### PART 1 #######################################
## Use K-means algorithm to group data set by its most significant values

def pq(data, P, init_centroids, max_iter):
    data_partitions = np.hsplit(data, P)
    codebooks = np.zeros((P, init_centroids.shape[1], init_centroids.shape[2]), dtype=np.float32)
    codes = np.zeros((P, data.shape[0]), dtype=np.uint8)
    for i in range(P):
        codebook, code = kmeans(data_partitions[i], init_centroids[i], max_iter)
        codebooks[i] = np.add(codebooks[i], codebook)
        codes[i] = np.add(codes[i], code)
    codes = np.transpose(codes)

    return codebooks, codes


def kmeans(data_partition, init_centroid, max_iter):
    centroids = init_centroid
    for _ in range(max_iter):
        min_index = min_distance(data_partition, centroids)
        centroids = update_centroids(data_partition, min_index, centroids)
    min_index = min_distance(data_partition, centroids)

    return centroids, min_index


def min_distance(data_partition, centroids):
    l1_distance = np.absolute(data_partition - centroids[:, np.newaxis])
    sum_distance = l1_distance.sum(axis=2)
    min_index = np.argmin(sum_distance, axis=0)

    return min_index


def update_centroids(data_partition, min_index, centroids):
    updated_centroids = np.empty((0,data_partition.shape[1]), dtype=np.float32)
    for k in range(centroids.shape[0]):  
        centroid = data_partition[min_index==k]
        if centroid.shape[0] > 0:
            centroid = np.median(centroid, axis=0)
        else:
            centroid = centroids[k]               
        updated_centroids = np.vstack((updated_centroids, centroid))
    
    return updated_centroids

######################## PART 2 ###########################
## Use reverse multi-index algorithm to improve query proficiency

def query(queries, codebooks, codes, T):
    Q = queries.shape[0]
    P = codebooks.shape[0]
    K = codebooks.shape[1]
    candidates = [set() for _ in range(Q)]   #[{}, {}, {}]
    if P == 2:
        for q in range(Q):
            dist_books = {}  
            q_partitions = np.hsplit(queries[q], P)
            for i in range(P):   ## P = 2
                sum_dis = np.absolute(q_partitions[i] - codebooks[i]).sum(axis=1)
                min_index = np.argsort(sum_dis)
                dist_book = []   ### [{u3: 0.5}, {u4: 0.7}, ...]
                for _, value in enumerate(min_index):
                    dist_book.append((value, sum_dis[value]))
                dist_books[i] = dist_book
            candidates[q] = multi_index_P2(dist_books[0], dist_books[1], codes, T)
    else:
        print('P={P} cannot be handled')

    return candidates



def multi_index_P2(dist_books1, dist_books2, codes, T):
    l1 = len(dist_books1)  # l1 = K = 256
    l2 = len(dist_books2)  # l2 = K = 256
    traversed = [[False]*l2 for _ in range(l1)]  # traversed = [[false, false....], [false,fasle,...], ...[false,fasle,...]]   256x256
    pqueue = []    ### pqueue = [((0,0), [208,41], 30.688416589995853), ...] -->  [((inverted_index), (cluster_index), sum_distance)]
    candidate = set()
    ## dist_books1 = [(208, 15.624244810014847), (), ...]      dist_books2[0] = (41, 15.064171779981004)
    pqueue.append( ((0,0), [dist_books1[0][0], dist_books2[0][0]], dist_books1[0][1] + dist_books2[0][1]) )   
    while len(candidate) < T:
        pqueue = sorted(pqueue, key=lambda x: x[2])   ## sort pqueue via the third value sum_distance, [((inverted_index), (cluster_index), sum_distance)]
        pop_item = pqueue.pop(0)   # pop_item = ((0,0), [208,41], 30.688416589995853)
        cur_min_index = pop_item[0]    # cur_min_index = (0,0)
        i = cur_min_index[0]
        j = cur_min_index[1]
        traversed[cur_min_index[0]][cur_min_index[1]] = True
        if i < l1-1 and (j== 0 or traversed[i+1][j-1]):
            pqueue.append(  ((i+1,j), [dist_books1[i+1][0], dist_books2[j][0]], dist_books1[i+1][1] + dist_books2[j][1]) )
        if j < l2-1 and (i== 0 or traversed[i-1][j+1]):
            pqueue.append(  ((i,j+1), [dist_books1[i][0], dist_books2[j+1][0]], dist_books1[i][1] + dist_books2[j+1][1]) ) 
        cluster_index = pop_item[1]   # cluster_index = [208,41]
        exist = np.all(codes==cluster_index, axis=1) # find if [208,41] exists in codes, exist = [false, false, false, true, false, true, ...]
        candi_index = np.where(exist == True)        # find the index of True candi_index = [3, 5]
        if candi_index[0].shape[0] > 0:              # candi_index[0] = [] skip, candi_index[0] = [3, 5] add item in candidate
            for item in candi_index[0]:
                candidate.add(item)

    return candidate

def multi_index_P4(dist_books1, dist_books2, dist_books3, dist_books2, codes, T):
    pass
