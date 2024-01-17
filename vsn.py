class Vsn:
    def __init__(self, clus, ilp, rff, rff_num_features=200):
        assert not (clus and ilp)
        self.CLUS = clus
        self.ILP = ilp
        self.RFF = rff
        self.RFF_n = rff_num_features
