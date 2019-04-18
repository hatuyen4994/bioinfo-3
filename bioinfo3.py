import numpy as np


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