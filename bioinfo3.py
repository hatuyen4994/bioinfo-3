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