import numpy as np
from numpy import linalg as LA
import os
import time
from scipy import sparse
from sklearn.preprocessing import normalize

def set_style(name, height, bold=False):
    style = xlwt.XFStyle()

    font = xlwt.Font()
    font.name = name  # 'Times New Roman'
    font.bold = bold
    font.color_index = 4
    font.height = height

    style.font = font

    return style

def update(V, V_exp, V_exp_bf, V_exprec):
    [nne,id] = np.shape(V_exprec)
    [k,d] = np.shape(V_exp)
    for i in range(nne):
        addg = np.zeros((1,d))
        for j in range (id):
            if((V_exprec[i,j]=='-1')|(V_exprec[i,j]=='')):
                V[i, :] = V[i, :] + (addg/j)
                break
            else:
                tmp = (V_exp[int(V_exprec[i,j]),:])-(V_exp_bf[int(V_exprec[i,j]),:])
                addg = addg+tmp

    return V

def expand(n_word,d,V,V_rec):
    V_exp = np.zeros((n_word,d))
    [nnc,indc] = np.shape(V_rec)
    for i in range (nnc):
        adde = np.zeros((1,d))
        for j in range (indc):
            if((V_rec[i,j]=='-1')|(V_rec[i,j]=='')):
                if j == 0 :
                    V_exp[i, :] = V_exp[i, :] + adde
                else:
                    V_exp[i, :] = V_exp[i, :] + (adde/j)
                break
            else:
                #print(V_rec[i,j])
                adde = adde + V[int(V_rec[i,j]),:]
    return V_exp

def word_analogy(W,vocab_pt,compga,compvb,pinga,pinvb,lc,lp,datasetname):
  #  k = eval_analogy_not_val.measure(W,vocab_pt,compga,compvb,pinga,pinvb,lc,lp,datasetname)
    k = 0
    return k

def word_similar(W,vocab_pt,compga,compvb,pinga,pinvb,lc,lp,datasetname):
  #  k1, k2 = eval_similar_val.measure_similar(W, vocab_pt,datasetname)
  #  k3, k4 = eval_similar_not_val.measure_similar(W,vocab_pt,compga,compvb,pinga,pinvb,lc,lp,datasetname)
    k1,k2,k3,k4 = 0
    return k1, k2, k3, k4

def updaterow(V, V_exp, V_rec,i):
    [nne,id] = np.shape(V_rec)
    [k,d] = np.shape(V_exp)

    for j in range(id):
        if((V_rec[i,j]=='-1')|(V_rec[i,j]=='')):
            break
        else:
            V[int(V_rec[i,j]), :] = V_exp[i, :]
    return V

def expandrow(V, V_exp, V_rec, V_exprec, i):
    [nne, id] = np.shape(V_rec)
    [nne2,id2] = np.shape(V_exprec)
    [k, d] = np.shape(V_exp)
    [n,d] = np.shape(V)
    for j in range(id):
        if ((V_rec[i, j] == '-1') | (V_rec[i, j] == '')):
            break
        else:
            recV_value = int(V_rec[i,j])
            for j2 in range(id2):
                if ((V_exprec[recV_value, j2] == '-1') | (V_exprec[recV_value, j2] == '')):
                    break
                else:
                    V_exp[int(V_exprec[recV_value, j2]),:] = V[recV_value,:]

    return V_exp

if __name__ == '__main__':

    #datasetname = 'NLPIR_V4'
    #n_word = 30772
    #n_comp = 429
    #n_pin = 1055
    #n_char = 3347
   
    datasetname='yidu_new'
    n_word = 7573
    n_comp = 360
    n_pin = 777
    n_char = 1613
   
    Datadir = '/yan/patch/temp/csr/SPMIadd_pro_/dataset_new/' + datasetname + '/'
    R_file = Datadir + 'rel_btw_comp_and_pin.txt'
    R = np.zeros([n_comp,n_pin])
    R_pt = open(R_file,'r',encoding='UTF-8')
    ind = 0
    while True:
        tmp = R_pt.readline()
        if not tmp:
            break
        #print(tmp)
        tmp = tmp.strip().split()
        for i in range(n_pin):
            R[ind,i] = int(tmp[i])
        ind = ind+1
    R = normalize(R, axis=1, norm='l2')
    print('FInish R matrix')

    M_file = Datadir + 'M_matrix.txt'
    vocab_pt = Datadir  + '/vocab.txt'
    compvb = Datadir + '/compvb.txt'
    pinvb = Datadir + '/pinyinvb.txt'
    charvb = Datadir+ '/charvb.txt'
    V2_rec = np.loadtxt(Datadir + '/comprecm.txt', dtype='str', delimiter=' ', skiprows=0)
    V2_exprec = np.loadtxt(Datadir + '/compexprecm.txt', dtype='str', delimiter=' ', skiprows=0)
    V3_rec = np.loadtxt(Datadir + '/pinyinrecm.txt', dtype='str', delimiter=' ', skiprows=0)
    V3_exprec = np.loadtxt(Datadir + '/pinyinexprecm.txt', dtype='str', delimiter=' ', skiprows=0)
    V4_rec = np.loadtxt(Datadir+ '/charrecm.txt', dtype='str', delimiter=' ', skiprows=0)
    V4_exprec = np.loadtxt(Datadir + '/charexprecm.txt', dtype='str', delimiter=' ', skiprows=0)

    ResultDir = '/yan/patch/temp/csr/SPMIadd_pro_/resultforGS/' + datasetname + "/"
    curTime = time.strftime('%m_%d%Hh%Mm%Ss', time.localtime(time.time()))
    ResDir = os.path.join(ResultDir, "SPMI_add_GridSearch_" + curTime)
    os.mkdir(ResDir)
    #output = open(os.path.join(ResDir, "Log"), 'w')

    max_iter = 15
    alpha_comp_range = [0.4]
    alpha_pin_range = [0.6]
    k_ns_range = [5]
    d_range = [300]
    lam_sparse_range = [0.1]
    randthrd = 1  #random initialiation count
    index_arg = 0
    for s3 in range(len(k_ns_range)):
        k_ns = k_ns_range[s3]
        M = np.zeros((n_word, n_word))
        M_pt = open(M_file, 'r')
        while True:
           tmp = M_pt.readline()
           if not tmp:
               break
           tmp = tmp.strip().split()
           M[int(tmp[0]), int(tmp[1])] = max(float(tmp[2]) - np.log(k_ns), 0)

        for s1 in range(len(alpha_comp_range)):
            alpha_comp = alpha_comp_range[s1]
            for s2 in range(len(alpha_pin_range)):
                alpha_pin = alpha_pin_range[s2]
                for s4 in range(len(d_range)):
                    d = d_range[s4]
                    for s5 in range(len(lam_sparse_range)):
                        lam_sparse = lam_sparse_range[s5]
                        for randiter in range(randthrd):
                            W = np.random.randn(n_word, d).astype(np.float32)  # C
                            V1 = np.random.randn(n_word, d).astype(np.float32)  # A

                            V2 = np.random.randn(n_comp, d).astype(np.float32)
                            V2_exp = expand(n_word, d, V2, V2_rec)
                            V3 = np.random.randn(n_pin, d).astype(np.float32)
                            V3_exp = expand(n_word, d, V3, V3_rec)
                            V4 = np.random.randn(n_char, d).astype(np.float32)
                            V4_exp = expand(n_word, d, V4, V4_rec)

                            curArgument = [float(alpha_comp), float(alpha_pin), int(k_ns), int(d), float(lam_sparse),int(randiter)]
                            string_Arg = "[" + ", ".join([str(s) for s in curArgument]) + "]"
                            curString_Arg = '_'.join([str(s) for s in curArgument])
                            print("index %d: %s " % (index_arg, string_Arg))

                            for iter in range(max_iter):
                                print(iter)
                                V = V1 + alpha_comp * V2_exp + alpha_pin * V3_exp
                                W = (LA.solve(np.dot(V.T, V) + lam_sparse * np.eye(d, dtype=W.dtype), np.dot(V.T, M.T))).T
                                V1 = (LA.solve(np.dot(W.T, W) + lam_sparse * np.eye(d, dtype=W.dtype),
                                               np.dot(W.T, M))).T - alpha_comp * V2_exp - alpha_pin * V3_exp
                                if alpha_comp != 0:
                                    V2_exp_bf = V2_exp
                                    V2_exp = ((LA.solve(np.dot(W.T, W) + lam_sparse * np.eye(d, dtype=W.dtype),
                                                        np.dot(W.T, M))).T - V1 - alpha_pin * V3_exp) / alpha_comp
                                    V2 = update(V2, V2_exp, V2_exp_bf, V2_exprec)
                                    V2 = (LA.solve(np.dot(V3.T,V3),np.dot(V3.T,R.T))).T
                                    V2_exp = expand(n_word, d, V2, V2_rec)
                                if alpha_pin != 0:
                                    V3_exp_bf = V3_exp
                                    V3_exp = ((LA.solve(np.dot(W.T, W) + lam_sparse * np.eye(d, dtype=W.dtype),
                                                        np.dot(W.T, M))).T - V1 - alpha_comp * V2_exp) / alpha_pin
                                    V3 = update(V3, V3_exp, V3_exp_bf, V3_exprec)
                                    V3 = (LA.solve(np.dot(V2.T,V2),np.dot(V2.T,R))).T
                                    V3_exp = expand(n_word, d, V3, V3_rec)

                                V4_exp_bf = V4_exp
                                V4_exp = (LA.solve(np.dot(W.T, W) + lam_sparse * np.eye(d, dtype=W.dtype),
                                               np.dot(W.T, M))).T - V1 - alpha_comp * V2_exp - alpha_pin * V3_exp 
                                V4 = update(V4, V4_exp, V4_exp_bf, V4_exprec)
                                V4_exp = expand(n_word, d, V4, V4_rec)
                                np.savez(os.path.join(ResDir,'comp'+str(alpha_comp)+'_pin'+str(alpha_pin)+'_iter' + str(iter) +'_Embeddings_randthrd.npz'), W=W, V1=V1, V2=V2, V3=V3, V4=V4)
                        
