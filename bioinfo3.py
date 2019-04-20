import numpy as np

#######Week1##########
def DP_change(money, coins):
    min_num_coins = {}
    min_num_coins[0] = 0
    for m in range(1, money + 1):
        min_num_coins[m] = float("inf")
        for coin_value in coins:
            if m >= coin_value:
                if min_num_coins[m - coin_value] + 1 < min_num_coins[m]:
                    min_num_coins[m] =  min_num_coins[m - coin_value] + 1
    return min_num_coins[money]

def manhattan_tourist(n,m, down, right):
    s = np.zeros([n+1, m+1])
    for i in range(1,n+1):
        s[i][0] = s[i-1][0] + down[i-1][0]
    for j in range(1,m+1):
        s[0][j] = s[0][j-1] + right[0][j-1]
    for i in range(1, n+1):
        for j in range(1, m+1):
            s[i][j] = max(s[i-1][j] + down[i-1][j], s[i][j-1] + right[i][j-1])
    return int(s[n][m])

def lcs_backtrack(v,w):
    s = np.zeros([len(v)+1, len(w)+1])
    backtrack = np.zeros([len(v)+1, len(w)+1])
    backtrack = [list(i) for i in backtrack]
    for i in range(1,len(v)+1):
        for j in range(1, len(w)+1):
            match = 0
            if v[i-1] == w[j-1]:
                match = 1
            s[i][j] = max(s[i-1][j], 
                          s[i][j-1], 
                          s[i-1][j-1] + match)
            if s[i][j] == s[i-1][j]:
                backtrack[i][j] = "down"
            elif s[i][j] == s[i][j-1]:
                backtrack[i][j] = "right"
            elif s[i][j] == s[i-1][j-1] + match:
                backtrack[i][j] = "diag"
    return backtrack

def output_lcs(backtrack, v, i, j):
    if i == 0 or j == 0:
        return ""
    if backtrack[i][j] == "down":
        return output_lcs(backtrack, v, i-1, j)
    elif backtrack[i][j] == "right":
        return output_lcs(backtrack, v, i, j-1)
    elif backtrack[i][j] == "diag":
        return output_lcs(backtrack, v, i-1, j-1) + v[i-1]

#######Week2########

def global_alignment_backtrack(v,w,score_matrix):
    s = np.zeros([len(v)+1, len(w)+1])
    for i in range(len(v)+1):
        s[i,0] = -5 * i
    for j in range(len(w)+1):
        s[0,j] = -5 * j    
    backtrack = np.zeros([len(v)+1, len(w)+1])
    backtrack = [list(i) for i in backtrack]
    for i in range(1,len(v)+1):
        for j in range(1, len(w)+1):
            score = score_matrix[v[i-1]][w[j-1]]
            s[i][j] = max(s[i-1][j] - 5, 
                          s[i][j-1] - 5, 
                          s[i-1][j-1] + score)
            if s[i][j] == s[i-1][j] - 5:
                backtrack[i][j] = "down"
            elif s[i][j] == s[i][j-1] - 5:
                backtrack[i][j] = "right"
            elif s[i][j] == s[i-1][j-1] + score:
                backtrack[i][j] = "diag"
    return backtrack,s

def output_GA_v(backtrack, v, i, j):
    if i == 0 and j == 0:
        return ""
    elif i == 1 and j == 0:
        return v[i-1]
    elif i == 0 and j == 1:
        return "-"
    if backtrack[i][j] == "down":
        return output_GA_v(backtrack, v, i-1, j) + v[i-1]
    elif backtrack[i][j] == "right":
        return output_GA_v(backtrack, v, i, j-1) + "-"
    elif backtrack[i][j] == "diag":
        return output_GA_v(backtrack, v, i-1, j-1) + v[i-1]
    
def output_GA_w(backtrack, w, i, j):
    if i == 0 and j == 0:
        return ""
    elif i == 1 and j == 0:
        return "-"
    elif i == 0 and j == 1:
        return w[j-1]
    if backtrack[i][j] == "down":
        return output_GA_w(backtrack, w, i-1, j) + "-"
    elif backtrack[i][j] == "right":
        return output_GA_w(backtrack, w, i, j-1) + w[j-1]
    elif backtrack[i][j] == "diag":
        return output_GA_w(backtrack, w, i-1, j-1) + w[j-1]


def local_alignment_backtrack(v,w,score_matrix):
    s = np.zeros([len(v)+1, len(w)+1])
    backtrack = np.zeros([len(v)+1, len(w)+1])
    backtrack = [list(i) for i in backtrack]
    for i in range(1,len(v)+1):
        for j in range(1, len(w)+1):
            score = score_matrix[v[i-1]][w[j-1]]
            s[i][j] = max(0,
                          s[i-1][j] - 5, 
                          s[i][j-1] - 5, 
                          s[i-1][j-1] + score)
            if s[i][j] == s[i-1][j] - 5:
                backtrack[i][j] = "down"
            elif s[i][j] == s[i][j-1] - 5:
                backtrack[i][j] = "right"
            elif s[i][j] == s[i-1][j-1] + score:
                backtrack[i][j] = "diag"
            elif s[i][j] == 0:
                backtrack[i][j] = "free"
    return backtrack,s


def output_LA_v(backtrack, v, i, j):
    if backtrack[i][j] == "free":
        return ""
    elif i == 0 and j == 0:
        return ""
    elif i == 1 and j == 0:
        return v[i-1]
    elif i == 0 and j == 1:
        return "-"
    if backtrack[i][j] == "down":
        return output_LA_v(backtrack, v, i-1, j) + v[i-1]
    elif backtrack[i][j] == "right":
        return output_LA_v(backtrack, v, i, j-1) + "-"
    elif backtrack[i][j] == "diag":
        return output_LA_v(backtrack, v, i-1, j-1) + v[i-1]
    
def output_LA_w(backtrack, w, i, j):
    if backtrack[i][j] == "free":
        return ""
    elif i == 0 and j == 0:
        return ""
    elif i == 1 and j == 0:
        return "-"
    elif i == 0 and j == 1:
        return w[j-1]
    if backtrack[i][j] == "down":
        return output_LA_w(backtrack, w, i-1, j) + "-"
    elif backtrack[i][j] == "right":
        return output_LA_w(backtrack, w, i, j-1) + w[j-1]
    elif backtrack[i][j] == "diag":
        return output_LA_w(backtrack, w, i-1, j-1) + w[j-1]

def editds(v,w):
    s = np.zeros([len(v)+1, len(w)+1])
    for i in range(len(v)+1):
        s[i,0] = i
    for j in range(len(w)+1):
        s[0,j] = j
    for i in range(1,len(v)+1):
        for j in range(1, len(w)+1):
            delt={True:0, False:1}
            s[i][j] = min(s[i-1][j] + 1, 
                          s[i][j-1] + 1, 
                          s[i-1][j-1] + delt[v[i-1] == w[j-1]])
    return s[-1,-1]

####WEEEK 3##########
##AFFINE GAP####
def affine_gap_GA(v,w,score_matrix):
#Initiate
    sigma = 11
    epsilon = 1
    s_lower = np.zeros([len(v)+1, len(w)+1])
    s_middle = np.zeros([len(v)+1, len(w)+1])
    s_upper = np.zeros([len(v)+1, len(w)+1])
    
    b_lower = np.zeros([len(v)+1, len(w)+1])
    b_lower = [list(i) for i in b_lower]
    b_middle = np.zeros([len(v)+1, len(w)+1])
    b_middle = [list(i) for i in b_middle]
    b_upper = np.zeros([len(v)+1, len(w)+1])
    b_upper = [list(i) for i in b_upper]
#Populate   
    s_lower[0,0] = -float("inf")
    s_upper[0,0] = -float("inf")

    for i in range(1,len(v)+1):
        s_lower[i,0] = 0 - sigma - (i-1)*epsilon
        s_upper[i,0] = -float("inf")
        s_middle[i,0] = 0 - sigma - (i-1)*epsilon
    for j in range(1,len(w)+1):
        s_lower[0,j] = -float("inf")
        s_upper[0,j] = 0 - sigma - (j-1)*epsilon
        s_middle[0,j] = 0 - sigma - (j-1)*epsilon

#GO    
    for i in range(1,len(v)+1):
        for j in range(1, len(w)+1):
            s_lower[i,j] = max(s_lower[i-1,j] - epsilon,
                               s_middle[i-1,j] - sigma)
            if s_lower[i,j] == s_lower[i-1,j] - epsilon:
                b_lower[i][j] = "down"
            elif s_lower[i,j] == s_middle[i-1,j] - sigma:
                b_lower[i][j] = "middle"            
            
            s_upper[i,j] = max(s_upper[i,j-1] - epsilon,
                               s_middle[i,j-1] - sigma)
            if s_upper[i,j] == s_upper[i,j-1] - epsilon:
                b_upper[i][j] = "right"
            elif s_upper[i,j] == s_middle[i,j-1] - sigma:
                b_upper[i][j] = "middle"
                
            s_middle[i,j] = max(s_lower[i,j], 
                                s_upper[i,j], 
                                s_middle[i-1,j-1] + score_matrix[v[i-1]][w[j-1]])
            if s_middle[i,j] == s_lower[i,j]:
                b_middle[i][j] = "lower"
            elif s_middle[i,j] == s_upper[i,j]:
                b_middle[i][j] = "upper"
            elif s_middle[i,j] == s_middle[i-1,j-1] + score_matrix[v[i-1]][w[j-1]]:
                b_middle[i][j] = "diag"

    return [[s_lower,s_middle,s_upper],[b_lower, b_middle, b_upper]]

def output_affinegap_GA_v(pos, backtrack_list, v, i, j):
    b_lower, b_middle, b_upper = backtrack_list
    if i == 0 and j == 0:
        return ""
    elif i == 1 and j == 0:
        return v[i-1]
    elif i == 0 and j == 1:
        return "-"
    if pos == "middle":
        if b_middle[i][j] == "lower":
            return output_affinegap_GA_v("lower", backtrack_list, v, i, j)
        elif b_middle[i][j] == "upper":
            return output_affinegap_GA_v("upper", backtrack_list, v, i, j)
        elif b_middle[i][j] == "diag":
            return output_affinegap_GA_v("middle", backtrack_list, v, i-1, j-1) + v[i-1]
    
    elif pos == "lower":
        if b_lower[i][j] == "down":
            return output_affinegap_GA_v("lower", backtrack_list, v, i-1, j) + v[i-1]
        if b_lower[i][j] == "middle":
            return output_affinegap_GA_v("middle", backtrack_list, v, i-1, j) + v[i-1]

    elif pos == "upper":
        if b_upper[i][j] == "right":
            return output_affinegap_GA_v("upper", backtrack_list, v, i, j-1) + "-"
        if b_upper[i][j] == "middle":
            return output_affinegap_GA_v("middle", backtrack_list, v, i, j-1) + "-"

        
        
def output_affinegap_GA_w(pos, backtrack_list, w, i, j):
    b_lower, b_middle, b_upper = backtrack_list
    if i == 0 and j == 0:
        return ""
    elif i == 1 and j == 0:
        return "-"
    elif i == 0 and j == 1:
        return w[j-1]
    if pos == "middle":
        if b_middle[i][j] == "lower":
            return output_affinegap_GA_w("lower", backtrack_list, w, i, j)
        elif b_middle[i][j] == "upper":
            return output_affinegap_GA_w("upper", backtrack_list, w, i, j)
        elif b_middle[i][j] == "diag":
            return output_affinegap_GA_w("middle", backtrack_list, w, i-1, j-1) + w[j-1]
    
    elif pos == "lower":
        if b_lower[i][j] == "down":
            return output_affinegap_GA_w("lower", backtrack_list, w, i-1, j) + "-"
        if b_lower[i][j] == "middle":
            return output_affinegap_GA_w("middle", backtrack_list, w, i-1, j) + "-"

    elif pos == "upper":
        if b_upper[i][j] == "right":
            return output_affinegap_GA_w("upper", backtrack_list, w, i, j-1) + w[j-1]
        if b_upper[i][j] == "middle":
            return output_affinegap_GA_w("middle", backtrack_list, w, i, j-1) + w[j-1]



###LINEAR SPACE ALGORITHIM
def np_index(condition):
    result = np.where(condition)
    return [a for a in result[0]]

def from_source(v,w,score_matrix):
    s0 = np.zeros([len(v)+1, 1])
    middle = len(w) // 2
    for i in range(len(v)+1):
        s0[i] = -5 * i   
    for j in range(1, middle + 1):
        s1 = np.zeros([len(v)+1, 1])
        for i in range(0, len(v)+1):
            score = score_matrix[v[i-1]][w[j-1]]
            if i == 0 :
                s1[i] = s0[i] - 5
            else:
                s1[i] = max(s1[i-1] - 5, 
                              s0[i] - 5, 
                              s0[i-1] + score)
        s0 = s1
    return s0

def to_sink(v,w,score_matrix):
    v = v[::-1]
    w = w[::-1]
    s0 = np.zeros([len(v)+1, 1])
    middle = len(w) - len(w) // 2
    for i in range(len(v)+1):
        s0[i] = -5 * i

    for j in range(1, middle + 1):
        s1 = np.zeros([len(v)+1, 1])
        for i in range(0, len(v)+1):
            score = score_matrix[v[i-1]][w[j-1]]
            if i == 0 :
                s1[i] = s0[i] - 5
            else:
                s1[i] = max(s1[i-1] - 5, 
                              s0[i] - 5, 
                              s0[i-1] + score)
        s0 = s1
    return s0[::-1]

def middle_node(v,w,score_matrix):
    middle = len(w)//2
    middle_column = from_source(v,w,score_matrix) + to_sink(v,w,score_matrix)
    middle_i = np_index(middle_column == middle_column.max())
    middle_nodes = [(i,middle) for i in middle_i]
    return middle_nodes

def from_source_2(v,w,score_matrix):
    s0 = np.zeros([len(v)+1, 1])
    middle = len(w) // 2
    for i in range(len(v)+1):
        s0[i] = -5 * i   
    for j in range(1, middle + 2):
        s1 = np.zeros([len(v)+1, 1])
        for i in range(0, len(v)+1):
            score = score_matrix[v[i-1]][w[j-1]]
            if i == 0 :
                s1[i] = s0[i] - 5
            else:
                s1[i] = max(s1[i-1] - 5, 
                              s0[i] - 5, 
                              s0[i-1] + score)
        if j <= middle:
            s0 = s1
    return s0,s1

def to_sink_2(v,w,score_matrix):
    v = v[::-1]
    w = w[::-1]
    s0 = np.zeros([len(v)+1, 1])
    middle = len(w) - len(w) // 2
    for i in range(len(v)+1):
        s0[i] = -5 * i

    for j in range(1, middle + 1):
        s1 = np.zeros([len(v)+1, 1])
        for i in range(0, len(v)+1):
            score = score_matrix[v[i-1]][w[j-1]]
            if i == 0 :
                s1[i] = s0[i] - 5
            else:
                s1[i] = max(s1[i-1] - 5, 
                              s0[i] - 5, 
                              s0[i-1] + score)
        if j < middle:        
            s0 = s1
    return s0[::-1],s1[::-1]


def middle_edge(v,w,score_matrix):
    source1, source2 = from_source_2(v,w,score_matrix)
    sink1,sink2 = to_sink_2(v,w,score_matrix)
    j_column = source1 + sink2
    j_plus_column = source2 + sink1
    middle = len(w) // 2
    middle_nodes = np_index(j_column == j_column.max())
    
    start = (middle_nodes[-1],middle)
    end = (j_plus_column.argmax(), middle+1)
    return start,end