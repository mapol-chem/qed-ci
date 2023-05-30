import itertools
import re 
from itertools import permutations




def graph(N,n_o):
    '''
        lexical ordering graph with unoccupied arc set to be zero, return vertex weight
    '''
    rows, cols = ((N+1)*(n_o-N+1), 3)
    rows1, cols1 = (N+1, n_o+1)
    graph = [[0 for i in range(cols)] for j in range(rows)]
    graph_big = [[0 for i in range(cols1)] for j in range(rows1)]
    graph_big[N][n_o]=1
    #weight of vertex
    for e in range(N,-1,-1):
        for o in range(n_o-1,-1,-1):
            #print(e,o)
            if e==N and o>=e:
                graph_big[e][o] = graph_big[e][o+1]
            elif e<=N and o<e:
                graph_big[e][o] = 0 
            else:
                graph_big[e][o] = graph_big[e+1][o+1] + graph_big[e][o+1]
            
    
    count = 0
    for e in range(0,N+1):
        for o in range(0,n_o+1):
            #print(e,o,graph_big[e][o])
            if graph_big[e][o] !=0:
                graph[count][0] = e
                graph[count][1] = o
                graph[count][2] = graph_big[e][o]
                count +=1
    
    return graph,graph_big
def arc_weight(graph,graph_big):
    '''
        return diagonal (occupied) arc weight for lexical ordering graph above
        Y=arc weight
    '''
    Y=[]
    for row in range (0,len(graph)):
        #print(graph[i])
        e = graph[row][0]   
        o = graph[row][1]
        B=[]
        if e == N:
            continue
        i = o - e
        c = 0
        if i == 0:
            c=0
            B.extend([e,o,c])
            #print(B)
        else:
            for j in range (1, i+1):
                c +=graph_big[e+1][o+2-j] 
            B.extend([e,o,c])
            #print(B)
        Y.append(B)
    
    return Y
def string_to_binary(string,n_o):
    b=[int(d) for d in str(bin(string))[2:]]
    if len(b)<n_o:
        c = [0]*(n_o-len(b))
        d = c+b
    else:
        d=b
    d.reverse()
    return(d)
def string_to_index(string,N,n_o,Y):
    '''
        return index using arc weight for a string(integer represents binary string)
    '''
    count=0
    index=1
    b=[int(d) for d in str(bin(string))[2:]]
    if len(b)<n_o:
        c = [0]*(n_o-len(b))
        d = c+b
    else:
        d=b
    d.reverse()
    for i in range(n_o):
        if d[i] == 1:
            e = count
            o = i
            for j in range(0,len(Y)):
                if  Y[j][0] == e and Y[j][1] == o: 
                    index +=Y[j][2]
            count +=1        
    return index

def binary_to_index(binary_string,N,n_o,Y):
    count=0
    index=1
    string=0
    for i in range(0,len(binary_string)):
        if binary_string[i] == 1:
            e = count
            o = i
            string += pow(2,i)
            for j in range(0,len(Y)):
                if  Y[j][0] == e and Y[j][1] == o: 
                    index +=Y[j][2]
            count +=1        
    #print(string)        
    return index,string
def index_to_string(index, N, n_o, Y, return_binary=False):
    '''
       return string for an index using graph
    '''
    I=1
    e=N
    o=n_o
    count=0
    record=[]
    while(I<=index and e <= o):
        #print(count,e,o,I,record)
        if (e == o and I<index):
            count2 = 0
            for i in range(len(record)-1,-1,-1):
                if record[i]==0:
                    count2 += 1
                    record.pop(i)
                else:    
                    record[i]=0
                    break
            o=o+count2
            e=e+1
            for j in range(0,len(Y)):
                if  Y[j][0] == e-1 and Y[j][1] == o:
                    b=Y[j][2]
            I=I-b        
            #print(record,o,e,b,I)    
    
        else:
            if e>0:
                for j in range(0,len(Y)):
                    if  Y[j][0] == e-1 and Y[j][1] == o-1:
                        a=Y[j][2]
                        if a<=index-I:
                            e=e-1
                            o=o-1
                            I=I+a
                            record.append(1)
                        else:
                            o=o-1
                            record.append(0)
            else:
                o = o-1
                record.append(0)
        count +=1
        if count == 15000000 or (e==0 and o==0 and I==index): break
    string = 0
    for i in range(len(record)):
        if record[i] == 1:
            string += pow(2,len(record)-i-1)
    #print(string)
    #print(record)

    if return_binary == True:
        return record,string
    else:
        return string

def phase_single_excitation(p,q,string):
    '''
        getting the phase(-1 or 1) for E_pq\ket{I_a} where I_a is an alpha or beta string 
        p=particle,q=hole
    '''
    if p>q:
        mask=(1<<p)-(1<<(q+1))
    else:
        mask=(1<<q)-(1<<(p+1))
    if (bin(mask & string).count('1') %2):
        return -1
    else:
        return 1
def single_replacement_list(index,N,n_o,Y):
    '''
        getting the sign, string address and pq for sign(pq)E_pq\ket{I_a}
        p=particle, q=hole
        this excludes E_pp which should return the original string
    '''
    rows, cols = (N*(n_o-N), 4)
    table = [[0 for i in range(cols)] for j in range(rows)]
    string = index_to_string(index,N,n_o,Y)
    d = string_to_binary(string,n_o)
    print('single replacement list for binary string',d,'string',string,'index',index)
    occ=[]
    vir=[]
    #print(occ,vir)
    for i in range(n_o):
        if (string &(1<<i)):
            occ.append(i)
        else:
            vir.append(i)
    count=0        
    for i in range(N):
        for a in range(n_o-N):
            string1 = (string^(1<<occ[i])) | (1<<vir[a])
            #c=string_to_binary(string1,n_o)
            table[count][0] = string_to_index(string1,N,n_o,Y)
            table[count][1] = phase_single_excitation(vir[a],occ[i],string)
            table[count][2] = vir[a]
            table[count][3] = occ[i]
            count += 1
    return table 

#test
N = 3 
n_o = 8
fixed_length = n_o
perms = [''.join(p) for p in permutations('111' + '0' * (fixed_length - N))]
unique_perms = set(perms)
unique_perms1 = str(unique_perms)
print(unique_perms1)


temp1=re.sub(r'\'',r'',unique_perms1)
#print(temp1)
temp=re.findall(r'(\d\d\d\d\d\d\d\d)',temp1)
temp2=re.sub(r'(\d)(\d)(\d)(\d)(\d)(\d)(\d)(\d)',r'\1,\2,\3,\4,\5,\6,\7,\8',temp[0])
binary_list = []
for i in range(0,len(temp)):
    b = [int(j) for j in list(temp[i])]
    binary_list.append(b)
#print(binary_list)   


graph,graph_big=graph(N,n_o)
Y=arc_weight(graph,graph_big)
print('graph_big')
print(graph_big)
    
print('graph')
print(graph)
    
print('Y')
print(Y)


print('binary','index','string')
for a in range(0,len(binary_list)):
    index1,string1=binary_to_index(binary_list[a],N,n_o,Y)
    print(binary_list[a],index1,string1)   
print('binary(inverse)','index','string')
for index2 in range(1,len(binary_list)+1):
    binary,string=index_to_string(index2,N,n_o,Y,return_binary=True)
    print(binary,index2,string)

#lets build a single replacement list for a string

for index2 in range(1,len(binary_list)+1):
    table2 = single_replacement_list(index2,N,n_o,Y)
    print('address','sign','particle','hole')
    for i in range(len(table2)):
        print table2[i]

