from ..ca_environment.active_ca import ActiveCAEnv
from ..ca_environment.acive_ca_proba import ProbaActiveCAEnv
from .findscope_core import FindScopeBase
from ..utils import get_kappa, get_con_subset, get_scope
from .findscope_obj import cb_split_proba, cb_split_half_vars

available_split_funcs = [cb_split_proba, cb_split_half_vars]

class CBFindScope(FindScopeBase):
    """
    This is the new version for the FindScope function that uses a constraint-based approach
    """

    def __init__(self, ca_env: ActiveCAEnv = None, split_func=None, time_limit=0.2):
        """
        Initialize the FindScope2 class.

        :param ca_env: The constraint acquisition environment.
        :param time_limit: The time limit for findscope query generation.
        :param split_func: The function used to split the variables in findscope.
        """

        if split_func is None:
            split_func = cb_split_proba if isinstance(ca_env, ProbaActiveCAEnv) else cb_split_half_vars

        super().__init__(ca_env, time_limit, split_func=split_func)
        self._kappaB = []

    @property
    def split_func(self):
        """
        Get the split function to be used in findscope.

        :return: The split function.
        """
        return self._split_func

    @split_func.setter
    def split_func(self, split_func):
        """
        Set the split function to be used in findscope.

        :param split_func: The split function.
        :raises AssertionError: If the split function is not available.
        """
        assert split_func in available_split_funcs, "Split function given for FindScope is not available"
        self._split_func = split_func

    def run(self, Y, kappa=None):
        """
        Run the FindScope2 algorithm.

        :param Y: A set of variables.
        :return: The scope of the partial example.
        :raises Exception: If the partial example is not a negative example.
        """
        assert self.ca is not None

        kappaB = kappa if kappa is not None else get_kappa(self.ca.instance.bias, Y)
        self._kappaB = kappaB
        if len(self._kappaB) == 0:
            raise Exception(f"The partial example e_Y, on the subset of variables Y given in FindScope, "
                            f"must be a negative example")
        scope = self._find_scope(Y)

        return scope

    def _find_scope(self, Y):
        """
        Find the scope of the partial example.

        :param Y: A set of variables.
        :return: The scope of the partial example.
        :raises Exception: If kappaB is not part of the bias.
        """

        while True:

            Y1, Y2 = self.split_func(Y=Y, kappaB=self._kappaB, P_c=self.ca.bias_proba)

            if len(Y1) == 0:
                return get_scope(self._kappaB)
            
            kappaBY1 = get_con_subset(self._kappaB, list(Y1))

            self.ca.metrics.increase_findscope_queries()
            if self.ca.ask_membership_query(Y1):
                self.ca.remove_from_bias(kappaBY1)
                self._kappaB = list(set(self._kappaB) - set(kappaBY1))

            else:
                Y = Y1
                self._kappaB = kappaBY1

