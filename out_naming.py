from function_types import FunctionType
from intervention_types import IvType
from out import Out
from vsn import Vsn
import os
from pathlib import Path

def unique_name(C_n, D_n, node_n, fun_type : FunctionType, iv_type: IvType, vsn: Vsn, iv_per_node):
    name = str(C_n)+'_' + str(D_n)+'_' +str(node_n)+'_' \
              +str(fun_type)+'_' +str(iv_type)+ '_gain.' + str(vsn.mdl_gain) + \
              '_rff.'+ str(vsn.rff) +  '_pair.' +str(vsn.regression_per_pair) + \
              '_NumIv.' + str(iv_per_node[0])  + str(iv_per_node[1])
    return (name)

def folders_test_tree_search(file_nm):
    dag_folder = "logs/dags/" + file_nm + "/"  # "/C"+ str(C_n) + "N" + str(node_n) + "/" #str(run_id)+"/"

    if not Path("logs").exists():
        os.mkdir("logs")
    if not Path("logs/dags").exists():
        os.mkdir("logs/dags")
    if not Path(dag_folder).exists():
        os.mkdir(dag_folder)
    if not Path("logs/tree").exists():
        os.mkdir("logs/tree")
    out = Out("tree/" + "/eval_" + file_nm + '.txt', vb=False, tofile=False)
    outres = Out("tree/" + "/res_" + file_nm + '.txt')
    return out, outres, dag_folder