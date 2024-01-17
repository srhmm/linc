from pulp import *
from scipy.stats import wasserstein_distance

from utils import *
from regression import *
from utils_pi import pi_group_map
import time


"""Gaussian Process Regression Info (per context/pairs)"""

class ContextRegression:
    def __init__(self,
                 gp_c, kr_c, gp_ij):
        self.GP_c = gp_c
        self.KR_c = kr_c
        self.GP_ij = gp_ij


"""Search for a partition Pi of contexts into groups (given data and a fixed target Y)"""

class PI:
    def __init__(self,
                 Xc, yc, rst,
                 info_regression: ContextRegression = None,
                 skip_regression=False,
                 skip_regression_pairs=True,
                 normalize=False,
                 min_max=True,
                 rff=False,
                 pi_search=False):
        self.Xc = np.nan_to_num(Xc) #Xc.mean()
        self.yc = np.nan_to_num(yc)
        self.X_n = Xc.shape[2]
        self.C_n = len(self.Xc)
        self.contexts = range(self.C_n)
        self.dims = range(self.X_n)
        self.normalize = normalize
        self.min_max= min_max
        self.regression_per_pair = not skip_regression_pairs
        self.rff=rff
        self.pi_search=pi_search

        self.pi_cache = {}

        assert Xc.shape[0] == yc.shape[0]
        assert Xc.shape[1] == yc.shape[1]

        # Pooled data
        Pi_one = [[i for i in self.contexts]]
        self.Xall, self.yall = data_groupby_pi(Xc, yc, Pi_one)

        # Linear regression per context
        if False:#(self.X_n< 20):
            self.coef_c = group_linearRegression(self.Xc, self.yc, scale=True)
            self.coef_c_noscaling = group_linearRegression(self.Xc, self.yc, scale=False)
        else:
            self.coef_c= np.array([[0 for _ in range(self.X_n)] for _ in range(self.C_n)])

        assert self.coef_c.shape[0] == self.C_n and self.coef_c.shape[1] == self.X_n

        # GP regression per context
        if skip_regression:
            if info_regression is not None:
                self.GP_c, self.KR_c = info_regression.GP_c, info_regression.KR_c
                self.info_regression = info_regression
                if skip_regression_pairs:
                    self.GP_ij = None
                else:
                    self.GP_ij = info_regression.GP_ij
        else:
            #st = time.time()
            self.__regress_nonlinear_context()
            #print("* GP regression time: ", round(time.time() - st, 2))
            if skip_regression_pairs:
                self.GP_ij = None
            else:
                #print ("GP Regresssion on each context pair")
                self.__regress_nonlinear_pairs()
            self.info_regression = ContextRegression(self.GP_c, self.KR_c, self.GP_ij)

    def pi_lookup(self, partition):

        hash_key = f'pi_{str(partition)}'

        if self.pi_cache.__contains__(hash_key):
            assert self.pi_cache.__contains__(hash_key)
            GP_k = self.pi_cache[hash_key]
            return GP_k
        else:
            #st = time.perf_counter()
            GP_k = self.regress_nonlinear_groups(partition)
            self.pi_cache[hash_key] = GP_k
            #print("GP fit per group time: ", round(time.perf_counter() - st, 5), "sec")
            return GP_k

    def cmp_distances(self, from_G_ij=False):
        if from_G_ij:
            assert self.GP_ij is not None
            self.__predict_jointly()
        else:
            assert self.GP_c is not None
            self.__predict_pairwise()
        self.__distances_mdl()

    #def cmp_scores_group(self, partition):
    #    self.__regress_nonlinear_groups(partition)

    def __regress_nonlinear_context(self, kr=False):
        """ GP and kernel regression in each context

        :param kr: whether to do kernel regression
        :return: sets self.GP_c
        """
        self.GP_c = group_gaussianProcessRegression(self.Xc, self.yc, rand_fourier_features=self.rff, pi_search=self.pi_search)
        self.KR_c = None
        if kr:
            self.KR_c = group_kernelRidgeRegression(self.Xc, self.yc, show_plt=False)

    def __regress_nonlinear_pairs(self):
        """ GP regression over each pair of contexts

        :return: sets self.GP_ij
        """
        self.GP_ij = [None for _ in itertools.combinations(self.contexts, 2)]
        comb_i : int = 0
        for c_i, c_j in itertools.combinations(self.contexts, 2):
            Xij, yij = data_groupby_pi(self.Xc, self.yc, [[c_i, c_j]])
            self.GP_ij[comb_i] = group_gaussianProcessRegression(Xij, yij, rand_fourier_features=self.rff,pi_search=self.pi_search)[0]
            comb_i = comb_i + 1


    def regress_nonlinear_groups(self, partition):
        """GP regression in each of the groups (dep on a partition)

        :param partition: partition
        :return: GP_k for each group k in partition
        """
        X_grouped, y_grouped = data_groupby_pi(self.Xc, self.yc, partition)
        GP_k = group_gaussianProcessRegression(X_grouped, y_grouped, rand_fourier_features=self.rff,pi_search=self.pi_search)

        return GP_k

    def distances_one_context(self):
        return self.GP_c[0].mdl_score_ytrain()

    def __distances_mdl_standard(self):
        """ MDL distances for pairs of contexts.
        Distance d(ci, cj) is MDL score gain: max(1,2 in ci, cj) L(y1 | GP(c1)) - L(y1 | GP(c2))

        :return:
        """
        self.pair_mdl_dists = [0 for _ in itertools.combinations(self.contexts, 2)]
        self.pair_mdl_gains = [0 for _ in itertools.combinations(self.contexts, 2)]
        comb_i : int = 0
        combs = itertools.combinations(self.contexts, 2)
        for c_i, c_j in combs:
            # 1. Distances for ILP: analogously to Wasserstein distances, look at true GP for c_i vs. c_j's GP,
            # and compare MDL scores of these models for c_i
            # negative distances, min is choosing the worst one/max the better one
            mdl_j, _, _, _ = self.GP_c[c_j].mdl_score_ytrain()
            mdl_j_test, _, _, _ = self.GP_c[c_i].mdl_score_ytest(self.Xc[c_j], self.yc[c_j])

            mdl_i, _, _, _ = self.GP_c[c_i].mdl_score_ytrain()
            mdl_i_test, _, _, _ = self.GP_c[c_j].mdl_score_ytest(self.Xc[c_i], self.yc[c_i])

            self.pair_mdl_dists[comb_i] = max(mdl_j - mdl_j_test, mdl_i - mdl_i_test)#min(mdl_j_test - mdl_j,
                                          #    mdl_i_test - mdl_i)

            # 2. Gain of using a group model instead of two context models
            # To avoid computing all pairwise-pooled GPs: choose one of the two as the group model
            # A. max: use the best of both (cheap encoding wanted), as with wasserstein barycenter
            # B. min: use the worst of both, min-max problem
            Xpi, ypi = data_groupby_pi(self.Xc, self.yc, Pi=[[c_i, c_j]])
            Xij, yij = Xpi[0], ypi[0]
            mdl_i_joint, _, _, _ = self.GP_c[c_i].mdl_score_ytest(Xij, yij)
            mdl_j_joint, _, _, _ = self.GP_c[c_j].mdl_score_ytest(Xij, yij)

            self.pair_mdl_gains[comb_i] = max(mdl_i + mdl_j - mdl_i_joint,
                                              mdl_i + mdl_j - mdl_j_joint)
            if self.min_max:#doesnt seem to do much?
                self.pair_mdl_gains[comb_i] = min(mdl_i + mdl_j - mdl_i_joint,
                                                  mdl_i + mdl_j - mdl_j_joint)

            comb_i = comb_i + 1

    def __distances_mdl(self):
        """ MDL distances for pairs of contexts.
        Distance d(ci, cj) is MDL score gain: max(1,2 in ci, cj) L(y1 | GP(c1)) - L(y1 | GP(c2))

        :return:
        """
        self.pair_mdl_dists = [0 for _ in itertools.combinations(self.contexts, 2)]
        self.pair_mdl_gains = [0 for _ in itertools.combinations(self.contexts, 2)]

        # For printing
        self.pair_model_gains = [(0,0,0) for _ in itertools.combinations(self.contexts, 2)]
        self.pair_data_gains = [(0,0,0) for _ in itertools.combinations(self.contexts, 2)]
        comb_i : int = 0
        combs = itertools.combinations(self.contexts, 2)
        for c_i, c_j in combs:
            # 1. Distances for ILP: analogously to Wasserstein distances, look at true GP for c_i vs. c_j's GP,
            # and compare MDL scores of these models for c_i
            # negative distances, min is choosing the worst one/max the better one

            mdl_j, mdldata_j, mdlmodel_j, _ = self.GP_c[c_j].mdl_score_ytrain()
            mdl_i, mdldata_i, mdlmodel_i, _ = self.GP_c[c_i].mdl_score_ytrain()

            # 2. Gain of using a group model instead of two context models
            # To avoid computing all pairwise-pooled GPs: choose one of the two as the group model
            # A. max: use the best of both (cheap encoding wanted), as with wasserstein barycenter
            # B. min: use the worst of both, min-max problem

            Xpi, ypi = data_groupby_pi(self.Xc, self.yc, Pi=[[c_i, c_j]])
            Xij, yij = Xpi[0], ypi[0]


            if True:
                mdl_i_joint, mdldata_i_joint, mdlmodel_i_joint, _ = self.GP_c[c_i].mdl_score_ytest(Xij, yij)
                mdl_j_joint,  mdldata_j_joint,  mdlmodel_j_joint, _ = self.GP_c[c_j].mdl_score_ytest(Xij, yij)
                self.pair_mdl_gains[comb_i] = min(mdl_i + mdl_j - mdl_i_joint,
                                                      mdl_i + mdl_j - mdl_j_joint)
                self.pair_model_gains[comb_i] = (min(mdlmodel_i + mdlmodel_j - mdlmodel_i_joint,
                                                         mdlmodel_i + mdlmodel_j - mdlmodel_j_joint),
                                                     mdlmodel_i + mdlmodel_j, mdlmodel_i_joint)
                self.pair_data_gains[comb_i] = (min(mdldata_i + mdldata_j - mdldata_i_joint,
                                                        mdldata_i + mdldata_j - mdldata_j_joint), mdldata_i + mdldata_j,
                                                    mdldata_i_joint)
            self.pair_mdl_dists[comb_i] = self.pair_mdl_gains[comb_i]
            comb_i = comb_i + 1


    def __predict_pairwise(self):
        """ Wasserstein Distances on pairs of contexts. The GP fi of ONE context is applied to the other
        and we compare the resulting residual distributions,
        max { D(resid(y1, f1); resid (y2, f1)), D(resid(y1, f2); resid (y2, f2)) }.

        :return: sets self.pair_wass_GP
        """
        GPmeans = [0 for _ in self.contexts]
        self.GPresids_self = [0 for _ in itertools.combinations(self.contexts, 2)]
        self.GPresids_cross = [0 for _ in itertools.combinations(self.contexts, 2)]

        for c_i in self.contexts:
            GP_c_i = self.GP_c[c_i]
            GPmeans[c_i] = GP_c_i.predict(self.Xc[c_i])

        combs = itertools.combinations(self.contexts, 2)
        comb_n = math.comb(self.C_n, 2)
        self.pair_wass_GP = [0 for _ in range(comb_n)]
        self.pair_wass_sq_GP = [0 for _ in range(comb_n)]
        self.pair_mdl_itoj = [(0, 0) for _ in range(comb_n)]
        self.pair_mdl_jtoi = [(0, 0) for _ in range(comb_n)]
        self.pair_mdl_diff = [0 for _ in range(comb_n)]
        self.pair_mdl_joint = [0 for _ in range(comb_n)]
        self.pair_combs = [pair for pair in itertools.combinations(self.contexts, 2)]

        comb_i: int = 0
        for pair in combs:
            c_i, c_j = pair[0], pair[1]
            # back: j to i vs. i to i
            GPmean_back_i = GPmeans[c_i]
            GPmean_back_j = self.GP_c[c_j].predict(self.Xc[c_i])
            GPresid_back_i = (GPmean_back_i - self.yc[c_i])
            GPresid_back_j = (GPmean_back_j - self.yc[c_i])

            # forward: i to j vs. j to j
            GPmean_forward_i = self.GP_c[c_i].predict(self.Xc[c_j])
            GPmean_forward_j = GPmeans[c_j]
            GPresid_forward_i = (GPmean_forward_i - self.yc[c_j])
            GPresid_forward_j = (GPmean_forward_j - self.yc[c_j])

            self.GPresids_self[comb_i] = max(sum(GPresid_forward_j ** 2), sum(GPresid_back_i ** 2))
            self.GPresids_cross[comb_i] = max(sum(GPresid_forward_i ** 2), sum(GPresid_back_j ** 2))

            self.pair_wass_GP[comb_i] = max(wasserstein_distance(GPresid_forward_i, GPresid_forward_j),
                                            wasserstein_distance(GPresid_back_j, GPresid_back_i))
            self.pair_wass_sq_GP[comb_i] = max(wasserstein_distance(GPresid_forward_i ** 2, GPresid_forward_j ** 2),
                                               wasserstein_distance(GPresid_back_j ** 2, GPresid_back_i ** 2))
            comb_i = comb_i + 1




    def __predict_jointly(self):
        """ Wasserstein Distances on pairs of contexts. The GP f12 over both context is applied to each
        and we compare the resulting residual distributions,
        D(resid(y1, f12); resid (y2, f12))

        :return: sets self.pair_wass_GP
        """
        GPmeans = [0 for _ in self.contexts]
        self.GPresids_joint = [(0, 0) for _ in itertools.combinations(self.contexts, 2)]

        combs = itertools.combinations(self.contexts, 2)
        comb_n = math.comb(self.C_n, 2)
        self.pair_wass_jointGP = [0 for _ in range(comb_n)]
        self.pair_wass_sq_jointGP = [0 for _ in range(comb_n)]

        self.pair_combs = [pair for pair in itertools.combinations(self.contexts, 2)]

        for c_i in self.contexts:
            GP_c_i = self.GP_c[c_i]
            GPmeans[c_i] = GP_c_i.predict(self.Xc[c_i])

        comb_i: int = 0
        for c_i, c_j in combs:
            GP_ij = self.GP_ij[comb_i]

            # Context c_i
            GPmean_i = GPmeans[c_i]
            GPmean_ij = GP_ij.predict(self.Xc[c_i])
            GPresid_i = (GPmean_i - self.yc[c_i])
            GPresid_ij = (GPmean_ij - self.yc[c_i])

            # Context c_j
            GPmean_j = GPmeans[c_j]
            GPmean_ji = GP_ij.predict(self.Xc[c_j])
            GPresid_j = (GPmean_j - self.yc[c_j])
            GPresid_ji = (GPmean_ji - self.yc[c_j])

            self.GPresids_joint[comb_i] = (max(sum(GPresid_i ** 2), sum(GPresid_ij ** 2)),
                                           max(sum(GPresid_j ** 2), sum(GPresid_ji ** 2)))

            self.pair_wass_jointGP[comb_i] = max(wasserstein_distance(GPresid_i, GPresid_ij),
                                                 wasserstein_distance(GPresid_j, GPresid_ji))
            self.pair_wass_sq_jointGP[comb_i] = max(wasserstein_distance(GPresid_i ** 2, GPresid_ij ** 2),
                                                    wasserstein_distance(GPresid_j ** 2, GPresid_ji ** 2))
            comb_i = comb_i + 1

    def cmp_distances_linear(self, pi_test, emp=False, linc=False):
        """ Euclidean Distances and MDL score of linear parameters

        :param pi_test: partition
        :param emp: empirical score on euclidean distance (as in Vario)
        :param linc: full MDL score (linear-linc) or only the model score (as in Vario)
        :return: score
        """
        map = pi_group_map(pi_test, self.C_n)

        coef_pi = [[sum([self.coef_c[:, x_i][c_i] for c_i in pi_test[pi_k]]) / len(pi_test[pi_k])
                    for pi_k in range(len(pi_test))]
                   for x_i in self.dims]
        #[[c(x1, pi1),...],
        #[c(x2, pi1),...]]

        coef_error = [[(self.coef_c[:, x_i][c_i] - coef_pi[x_i][map[c_i]]) ** 2
                      for c_i in self.contexts]
                      for x_i in self.dims]
        coef_onegroup = [[sum(self.coef_c[:, x_i]) / self.C_n]
                      for x_i in self.dims]
        coef_error_onegroup = [[(self.coef_c[:, x_i][c_i] - coef_onegroup[x_i][0]) ** 2 for c_i in self.contexts]
                               for x_i in self.dims]

        # Vario empirical ----------
        if emp:
            sse =  [logg(sum(coef_error[x_i]))
                    for x_i in self.dims]
            sse_one = [logg(sum(coef_error_onegroup[x_i]))
                    for x_i in self.dims]
            vario_score = sum([(sse_one[x_i] - sse[x_i]) / max(len(pi_test) - 1, 1)
                               for x_i in self.dims])
        else:
            sig = 1
            mdl_coef = sum([(self.C_n / 2) * logg(2 * np.pi * (sig ** 2)) + 1 / (2 * (sig ** 2)) * sum(coef_error[x_i])
                            for x_i in self.dims]) # coefficient errors are per dimension
            mdl_alpha = (logg(self.C_n) / 2) * len(pi_test)
            mdl_pi = self.C_n * logg(self.C_n) + logg(self.C_n) # TODO constant model cost aspects?

            # MDL linear ----------
            if linc:
                num_samples = sum([len(self.yc[c_i]) for c_i in self.contexts])
                #residuals_c = [(sum([self.coef_c[c_i, x_i]*self.Xc[c_i][:,x_i] for x_i in self.dims]) - self.yc[c_i]) ** 2
                #               for c_i in self.contexts]
                sse_data_c = [sum((sum([self.coef_c[c_i, x_i]*self.Xc[c_i][:,x_i] for x_i in self.dims])
                                   - self.yc[c_i]) ** 2) for c_i in self.contexts]
                sse_data = sum(sse_data_c)
                mdl_data = (num_samples / 2) * logg(2 * np.pi * (sig ** 2)) + 1 / (2 * (sig ** 2)) * (logg(sse_data))
                vario_score = mdl_coef + mdl_alpha + mdl_pi + mdl_data
            # MDL model clustering ----------
            else:
                vario_score = mdl_coef + mdl_alpha + mdl_pi
        return vario_score
