from tigramite.independence_tests.parcorr_mult import ParCorrMult


import lingam
import numpy as np

def fit_jpcmci(instance, cond_ind_test=ParCorrMult(significance='analytic')):
    # Create a J-PCMCI+ object, passing the dataframe and (conditional)
    # independence test objects, as well as the observed temporal and spatial context nodes
    # and the indices of the dummies.
    from tigramite.jpcmciplus import JPCMCIplus
    JPCMCIplus = JPCMCIplus(dataframe=instance.dataframe_jpcmci,
                            cond_ind_test=cond_ind_test,
                            node_classification=instance.node_classification_jpcmci,
                            verbosity=0)

    # Define the analysis parameters.
    tau_max = instance.tau_max-1#diff naming conventions
    pc_alpha = 0.01

    # Run J-PCMCI+
    results = JPCMCIplus.run_jpcmciplus(tau_min=0,
                                        tau_max=tau_max,
                                        pc_alpha=pc_alpha)
    graph = results["graph"]
    sig_links = (graph != "")*(graph != "<--")

    return sig_links



def fit_varlingams(instance):
    model = lingam.VARLiNGAM()
    adj_C = []
    data_C = None
    for c in instance.data_C:
        from sklearn import preprocessing
        data_c = instance.data_C[c][:, range(instance.N)] #drop unobserved columns for cnode, snode
        data_c = preprocessing.normalize(data_c) # otherwise linalg error
        model.fit(data_c)
        adj_c = model.adjacency_matrices_[0]
        adj_C.append(adj_c)
        data_C = data_c if data_C is None else np.vstack((data_C, data_c))
    model.fit(data_C)
    adj_c = model.adjacency_matrices_[0]
    return adj_C, adj_c