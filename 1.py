import heapq

class node:
    def __init__(self, f, s, l=None, r=None):
        self.f = f
        self.s = s
        self.l = l
        self.r = r
        self.huff = ''

    def __lt__(self, next):
        return self.f < next.f


message = 'abbbbcccccdddeeeffff'
f_dict = {}
nodes = []
for ch in message:
    if ch in f_dict:
        f_dict[ch] += 1
    else:
        f_dict[ch] = 1

for ch, f in f_dict.items():
    heapq.heappush(nodes, node(f, ch))

while len(nodes) > 1:
    left = heapq.heappop(nodes)
    right = heapq.heappop(nodes)

    left.huff = 0
    right.huff = 1

    newNode = node(left.f + right.f, left.s + right.s, l=left, r=right)
    heapq.heappush(nodes, newNode)

def printNodes(n, val=''):
    newVal = val + str(n.huff)
    if n.l:
        printNodes(n.l, val=newVal)
    if n.r:
        printNodes(n.r, val=newVal)
    
    if not n.r and not n.l:
        print(f'{n.s}-{n.f}-{newVal}')

printNodes(nodes[0], '')
    

class Object:
    def __init__(self, w, v):
        self.w = w
        self.v = v
        self.c = v / w
    
    def __lt__(self, other):
        return self.cost < other.cost


def solve(W, V, M, n):
    packs = []

    for i in range(n):
        packs.append(Object(W[i], V[i]))
    packs.sort(reverse=True)

    remain = M
    stop = False
    i = 0
    while not stop:
        if packs[i].w <= remain:
            remain -= packs[i].w
            result += packs[i].v

        if packs[i].w > remain:
            i += 1

        if i == n:
            stop = True 

string = 'BCAADDDCCACACAC'
# Creating tree nodes
class NodeTree(object):
 
    def _init_(self, left=None, right=None):
        self.left = left
        self.right = right
 
    def children(self):
        return (self.left, self.right)
 
    def nodes(self):
        return (self.left, self.right)
 
    def _str_(self):
        return '%s_%s' % (self.left, self.right)
 
# Main function implementing huffman coding
def huffman_code_tree(node, left=True, binString=''):
    if type(node) is str:
        return {node: binString}
    (l, r) = node.children()
    d = dict()
    d.update(huffman_code_tree(l, True, binString + '0'))
    d.update(huffman_code_tree(r, False, binString + '1'))
    return d
 
 
## Calculating frequency
freq = {}
for c in string:
    if c in freq:
        freq[c] += 1
    else:
        freq[c] = 1
 
freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
 
nodes = freq
 
while len(nodes) > 1:
    (key1, c1) = nodes[-1]
    (key2, c2) = nodes[-2]
    nodes = nodes[:-2]
    node = NodeTree(key1, key2)
    nodes.append((node, c1 + c2))
 
    nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
 
 
huffmanCode = huffman_code_tree(nodes[0][0])
 
print(' Char | Huffman code ')
print('----------------------')
for (char, frequency) in freq:
    print((char, huffmanCode[char]))

# 0/1 knapsack

def ks(M, w, v, n):
    k = [[0 for i in range(w + 1)] for j in range(n + 1)]

    for i in range(n + 1):
        for j in range(w + 1):
            if i == 0 or j == 0:
                k[i][j] = 0
            elif w[i - 1] <= j:
                k[i][j] = max(k[i-1][w], v[i-1] + k[i-1][j - w[i - 1]]) 



N = 4

def safe(board, row, col):
    for i in range(col):
        if board[row][i] == 1:
            return False
    
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    for i, j in zip(range(row, N), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    
    return True

def solve(board, col):
    if col >= N:
        return True
    
    for i in range(N):
        if safe(board, i, col):
            board[i][col] = 1

            if solve(board, col + 1):
                return True
            
            board[i][col] = 0
