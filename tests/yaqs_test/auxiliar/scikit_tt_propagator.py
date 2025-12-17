#%%
import matplotlib.pyplot as plt
import numpy as np


from pathlib import Path



from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel, NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable

from mqt.yaqs.noise_char.optimization import trapezoidal
import copy


import scikit_tt.tensor_train as tt
from scikit_tt.tensor_train import TT
import scikit_tt.solvers.ode as ode

import multiprocessing
import os


from mqt.yaqs.simulator import available_cpus

from mqt.yaqs.core.libraries.gate_library import X, Y, Z, Create, Destroy, Id

# from auxiliar.write import *
from mqt.yaqs.core.libraries.gate_library import GateLibrary, Zero

import sys

def noise_model_to_operator_list(noise_model: NoiseModel) -> list[Observable]:
    """Converts a noise model to a list of observables.

    Args:
        noise_model (NoiseModel): The noise model to convert.

    Returns:
        list[Observable]: A list of observables corresponding to the noise processes in the noise model.
    """
    noise_list: list[Observable] = []

    for proc in noise_model.processes:
        gate = getattr(GateLibrary, proc["name"])
        noise_list.extend(Observable(gate(), site) for site in proc["sites"])
    return noise_list


def process_k(k, n_new_obs, scikit_new_obs_list, scikit_hamiltonian, scikit_jump_operator_list, scikit_jump_parameter_list, n_t, dt, scikit_tt_solver, init_state):


    exp_vals = np.zeros([n_new_obs,n_t], dtype=complex)

    scikit_initial_state = copy.deepcopy(init_state)

    for j in range(n_new_obs):
        exp_vals[j,0] = scikit_initial_state.transpose(conjugate=True)@scikit_new_obs_list[j]@scikit_initial_state

    for i in range(n_t - 1):
        scikit_initial_state = ode.tjm(scikit_hamiltonian, scikit_jump_operator_list, scikit_jump_parameter_list, scikit_initial_state, dt, 1, solver=scikit_tt_solver)[-1]

        for j in range(n_new_obs):
            exp_vals[j,i+1] = scikit_initial_state.transpose(conjugate=True)@scikit_new_obs_list[j]@scikit_initial_state

    return exp_vals







# import numpy as np
# from scikit_tt.tensor_train import TT
# import scikit_tt.tensor_train as tt

# # parameters

# T = 5
# dt = 0.1
# L = 10
# J = 1
# g = 0.5
# gamma_rel = 0.1
# gamma_deph = 0.1
# rank = 5

# # operators and observables

# X = np.array([[0, 1], [1, 0]])
# Y = np.array([[0, -1j], [1j, 0]])
# Z = np.array([[1, 0], [0, -1]])
# I = np.eye(2)
# L_1 = np.array([[0, 1], [0, 0]])
# L_2 = np.array([[1, 0], [0, -1]])

# # Hamiltonian

# cores = [None] * L
# cores[0] = tt.build_core([[-g * X, - J * Z, I]])
# for i in range(1, L - 1):
#     cores[i] = tt.build_core([[I, 0, 0], [Z, 0, 0], [-g * X, - J * Z, I]])
# cores[-1] = tt.build_core([I, Z, -g*X])
# hamiltonian = TT(cores)

# # ------------------------------------------------
# # - CONSTRUCT MPO FOR DENSITY MATRIX FORMULATION -
# # ------------------------------------------------

# # Hamiltonian part (commutator)

# cores_1 = [None] * L
# cores_2 = [None] * L
# for i in range(L):
#     r_l = hamiltonian.ranks[i]
#     r_r = hamiltonian.ranks[i+1]
#     cores_1[i] = np.einsum('ijkl,mn->ijmknl', hamiltonian.cores[i], I).reshape([r_l, 4, 4,r_r])
#     cores_2[i] = np.einsum('mn,ijkl->imjnkl', I, hamiltonian.cores[i].transpose([0,2,1,3])).reshape([r_l, 4, 4,r_r])
# hamiltonian_part = -1j*(TT(cores_1) - TT(cores_2))

# # non-Hamiltonian part

# cores = [None] * L
# S = gamma_rel * np.kron(L_1,np.conj(L_1)) + gamma_deph * np.kron(L_2,np.conj(L_2))
# S = S - 0.5 * (gamma_rel * np.kron(np.conj(L_1).T@L_1,np.eye(2)) + gamma_deph * np.kron(np.eye(2),(np.conj(L_2).T@L_2).T))
# I = np.eye(4)
# cores[0] = tt.build_core([[S, I]])
# for i in range(1, L-1): 
#     cores[i] = tt.build_core([[I, 0], [S, I]])
# cores[-1] = tt.build_core([I, S])
# non_hamiltonian_part = TT(cores)

# mpo_dmf = hamiltonian_part + non_hamiltonian_part

# print(mpo_dmf)























class Propagator:
    r"""High-level propagator that runs an MPS-based Lindblad simulation.

    The class wraps simulator inputs, performs
    consistency checks between noise models and the Hamiltonian, augments the
    observable set with Lindblad-derived A_kn operators (sensitivities of
    expectation values w.r.t. jump rates), runs the underlying simulator, and
    post-processes simulator outputs into convenient arrays for analysis.

    Attributes:
    obs_list : list[Observable]
        (Set after set_observable_list) Deep copy of user-provided observables.
    n_obs : int
        (Set after set_observable_list) Number of observables.
    times : array-like
        Time grid used by the most recent run (copied from sim_params.times).
    obs_traj : list[Observable]
        Observables returned by the simulator corresponding to the original
        user-requested observables (populated by run).
    obs_array : numpy.ndarray
        Array of observable trajectories with shape (n_obs, n_timesteps).
    d_on_d_gk : numpy.ndarray
        Object-array of Observable entries (shape [n_jump, n_obs]) corresponding
        to A_kn-like operators (or zero placeholders) computed by the simulator
        and integrated in time.
    d_on_d_gk_array : numpy.ndarray
        Numeric array of integrated A_kn trajectories (shape [n_jump, n_obs]).
    Other internal fields may be set during execution (e.g., temporary lists and
    simulator-specific containers).
    Public methods
    set_observable_list(obs_list: list[Observable]) -> None
        Store and validate a deep copy of obs_list. Validates that every site index
        referenced by the observables is within the range [0, sites-1]. Sets
        n_obs and flips set_observables to True. Raises ValueError for empty lists
        or out-of-range site indices.
    run(noise_model: CompactNoiseModel) -> None
        Execute the propagation. Requires that set_observables has been called.
        Validates that the provided compact noise_model matches the one used to
        construct this propagator (same process names and site assignments).
        Constructs A_kn-like observables for each matching jump/operator pair,
        appends them to the observable list, builds a new AnalogSimParams instance
        for the simulator, and invokes the underlying simulator with the expanded
        noise model. Post-processes results by trapezoidally integrating A_kn
        trajectories, arranging them into object and numeric arrays (d_on_d_gk and
        d_on_d_gk_array) and extracting obs_traj and obs_array for the original
        observables.
        - During initialization: if any site index in compact_noise_model.expanded_noise_model
          exceeds the number of sites in the Hamiltonian.
        - set_observable_list: if obs_list is empty or contains observables that
          reference out-of-range site indices.
        - run: if observables have not been set (set_observables is False) or if the
          provided noise_model does not match the initialized compact_noise_model
          in process names or site indices.
    - All constructor inputs are deep-copied to avoid accidental external mutation.
    - The class expects external types (AnalogSimParams, MPO, MPS, CompactNoiseModel,
      Observable) to expose particular attributes (for example, `times`, `length`,
      `expanded_noise_model`, `compact_processes`, `gate`, `sites`, and `results`).
    - The A_kn operators constructed in run follow the Lindblad derivative form:
      L_k^\dagger O L_k - 0.5 {L_k^\dagger L_k, O}, computed only for observables
      that act on the same site(s) as the corresponding jump operator.
    - The user-facing numeric arrays (obs_array and d_on_d_gk_array) are convenient
      summaries for optimization or analysis tasks (e.g., gradient-based fitting of
      jump rates).
    """

    def __init__(
        self,
        *,
        sim_params: AnalogSimParams,
        hamiltonian: MPO,
        compact_noise_model: CompactNoiseModel,
        init_state: MPS,
        tjm: bool = True
    ) -> None:
        """Initialize a Propagation object for simulating open quantum system dynamics.

        This constructor deep-copies the provided inputs and derives internal
        structures needed for propagation of an MPS under a Hamiltonian with
        a compact noise model.
        Parameters.
        ----------
        sim_params : AnalogSimParams
            Simulation parameters container. A deep copy is stored as
            self.sim_params. It is expected to provide a sequence/array
            `times` used to determine the number of time steps.
        hamiltonian : MPO
            Matrix product operator representing the Hamiltonian. A deep copy
            is stored as self.hamiltonian. The MPO must expose a `length`
            attribute indicating the number of sites.
        compact_noise_model : CompactNoiseModel
            Compact representation of the noise model. A deep copy is stored
            as self.compact_noise_model. Its `expanded_noise_model` attribute
            is deep-copied to self.expanded_noise_model and converted into a
            list of jump operators.
        init_state : MPS
            Initial many-body quantum state as a matrix product state.
            A deep copy is stored as self.init_state.
        Attributes set
        --------------
        sim_params : AnalogSimParams
            Deep copy of the provided simulation parameters.
        hamiltonian : MPO
            Deep copy of the provided Hamiltonian MPO.
        compact_noise_model : CompactNoiseModel
            Deep copy of the provided compact noise model.
        init_state : MPS
            Deep copy of the provided initial state.
        expanded_noise_model
            Deep copy of compact_noise_model.expanded_noise_model.
        noise_list : list[Observable]
            List of noise (jump) operators produced by converting the expanded
            noise model via noise_model_to_operator_list.
        n_jump : int
            Number of jump operators (len(self.noise_list)).
        n_t : int
            Number of time steps (len(self.sim_params.times)).
        sites : int
            Number of sites in the chain (self.hamiltonian.length).
        set_observables : bool
            Flag indicating whether observables have been set (initialized to False).

        Raises:
        ------
        ValueError: If any site index referenced in expanded_noise_model.processes is
            greater than or equal to the number of sites in the Hamiltonian,
            a ValueError is raised with the message
            "Noise site index exceeds number of sites in the Hamiltonian."

        Notes:
        -----
        - All provided inputs are deep-copied to avoid accidental external mutation.
        - This method performs basic consistency checking between the noise
          model and the Hamiltonian site count.
        """
        self.sim_params: AnalogSimParams = copy.deepcopy(sim_params)
        self.hamiltonian: MPO = copy.deepcopy(hamiltonian)
        self.compact_noise_model: CompactNoiseModel = copy.deepcopy(compact_noise_model)
        self.init_state: MPS = copy.deepcopy(init_state)

        self.expanded_noise_model = copy.deepcopy(self.compact_noise_model.expanded_noise_model)

        self.noise_list: list[Observable] = noise_model_to_operator_list(self.expanded_noise_model)

        self.n_jump: int = len(self.noise_list)  # number of jump operators

        self.n_t: int = len(self.sim_params.times)  # number of time steps

        self.sites: int = self.hamiltonian.length  # number of sites in the chain

        self.set_observables: bool = False

        self.tjm = tjm

        if max(proc["sites"][0] for proc in self.expanded_noise_model.processes) >= self.sites:
            msg = "Noise site index exceeds number of sites in the Hamiltonian."
            raise ValueError(msg)

    def scikit_tt_noise_list(self, noise_model: NoiseModel):
        
        jump_operator_list = [[] for _ in range(self.sites)]
        jump_parameter_list = [[] for _ in range(self.sites)]

        for proc in noise_model.processes:
            site = proc["sites"][0]
            if site < self.sites:
                jump_operator_list[site].append(getattr(GateLibrary, proc["name"])().matrix)
                jump_parameter_list[site].append(proc["strength"])

        return jump_operator_list, jump_parameter_list
    

    def scikit_tt_obs_list(self, obs_list: list[Observable]):
        scikit_obs_list = []

        for obs in obs_list:
            site = obs.sites
            mat = obs.gate.matrix
            if site < self.sites:
                scikit_tt_obs= tt.eye(dims=[2]*self.sites)
                scikit_tt_obs.cores[site]=np.zeros([1,2,2,1], dtype=complex)
                scikit_tt_obs.cores[site][0,:,:,0]=mat
                scikit_obs_list.append(scikit_tt_obs)

        return scikit_obs_list
    
    def scikit_tt_init_state(self, init_state: MPS):
        indices = [int(np.argmax(np.isclose(a, 1+0j))) for a in init_state.tensors]
        
        initial_state = tt.unit([2] * self.sites, indices)
        for i in range(self.sim_params.max_bond_dim - 1):
            initial_state += tt.unit([2] * self.sites, indices)
        initial_state = initial_state.ortho()
        initial_state = (1 / initial_state.norm()) * initial_state

        return initial_state
    
    def extract_J_g(self, mpo: MPO) -> tuple[float, float]:
        """Extract J and g from an Ising MPO instance."""
        # Grab the first tensor (left boundary)
        left = mpo.tensors[0]  # shape (2, 2, 1, 3) after transpose (sigma, sigma', left, right)
        left = np.transpose(left, (2, 3, 0, 1)) if left.shape[2] != 1 else left  # just in case


        # Extract relevant operator blocks
        # Depending on shape, you may need to index [0,1] vs [:,1]
        # Here we assume the original (1, 3, 2, 2) structure before transpose:
        if mpo.tensors[0].shape[-1] == 3:  # sanity check
            I_block, J_block, g_block = [np.squeeze(a) for a in np.transpose(mpo.tensors[0], (2, 3, 0, 1))[0]]
        else:
            # If shapes differ, we handle it more generically
            left_untransposed = np.transpose(mpo.tensors[0], (2, 3, 0, 1))
            I_block, J_block, g_block = left_untransposed[0]

        # Estimate J, g by projection
        # J_block ≈ -J * Z
        J = -np.real(J_block[0, 0])
        g = -np.real(g_block[0, 1])

        return J, g
    

    def scikit_tt_hamiltonian(self, hamiltonian: MPO):

        j, g = self.extract_J_g(hamiltonian)



        cores = [None] * self.sites
        cores[0] = tt.build_core([[-g * X().matrix, - j * Z().matrix, Id().matrix]])
        for i in range(1, self.sites - 1):
            cores[i] = tt.build_core([[Id().matrix, 0, 0], [Z().matrix, 0, 0], [-g * X().matrix, - j * Z().matrix, Id().matrix]])
        cores[-1] = tt.build_core([Id().matrix, Z().matrix, -g*X().matrix])

        hamiltonian = TT(cores)# jump operators and parameters


        return hamiltonian

    def set_observable_list(self, obs_list: list[Observable]) -> None:
        """Set the list of observables to be used for propagation.

        This method stores a deep copy of the provided observable list on the instance,
        validates that all referenced site indices lie within the allowed range of the
        Hamiltonian, and updates bookkeeping attributes.

        Args:
            obs_list (list[Observable]): Sequence of Observable objects. Each Observable
                must expose a `sites` attribute that is either an int (single site) or
                a list of ints (multiple sites).
        Side effects:
            - self.obs_list is set to a deep copy of obs_list.
            - self.n_obs is set to the number of observables (len(self.obs_list)).
            - self.set_observables is set to True.

        Raises:
            ValueError: If any site index in the observables is greater than or equal
                to self.sites (i.e., outside the range of available sites).
            ValueError: If obs_list is empty (which makes site-index validation via max()
                impossible) or if observables do not provide valid site information.
        """
        self.obs_list = copy.deepcopy(obs_list)

        all_obs_sites = [
            site for obs in obs_list for site in (obs.sites if isinstance(obs.sites, list) else [obs.sites])
        ]

        if max(all_obs_sites) >= self.sites:
            msg = "Observable site index exceeds number of sites in the Hamiltonian."
            raise ValueError(msg)

        self.n_obs = len(self.obs_list)  # number of measurement operators

        self.set_observables = True

    def run(self, noise_model: CompactNoiseModel) -> None:
        """Run the propagation routine with augmented Lindblad-derived operators.

        Parameters
        ----------
        noise_model : CompactNoiseModel
            The compact representation of the noise model to use for propagation.
            The method verifies that the list of compact processes and their sites
            in `noise_model` match the model used to initialize this propagator
            (self.compact_noise_model). The expanded form of this model is passed
            to the underlying simulator.

        Side effects / State changes
        ----------------------------
        On successful completion, several attributes of self are set or updated:
        - self.obs_traj : list[Observable]
            The list of original observables (with their computed time trajectories)
            extracted from the simulator results.
        - self.d_on_d_gk : numpy.ndarray of shape (n_jump, n_obs) with Observable entries
            A matrix of the A_kn-like operators (or zero placeholders) corresponding
            to each jump operator / observable pair; entries are Observable objects
            whose .results have been integrated (trapezoidally) over time.
        - self.d_on_d_gk_array : numpy.ndarray
            2D array of numeric trajectories corresponding to d_on_d_gk (shape
            [n_jump, n_obs, n_timesteps]).
        - self.obs_array : numpy.ndarray
            2D array of numeric trajectories for the original observables
            (shape [n_obs, n_timesteps]).
        - self.times : array-like
            Time grid used by the simulation (copied from self.sim_params.times).

        Raises:
            ValueError: If the observable list has not been initialized (self.set_observables is False).
            ValueError: If any process name or site in the provided noise_model does not match
              the corresponding entry in self.compact_noise_model.

        Notes:
        -----
        - The purpose of the added A_kn observables is to provide sensitivity-like
          quantities (derivatives of observable expectations with respect to
          jump rates) that are computed by the same underlying simulator and then
          post-processed into arrays suitable for analysis or parameter updates.
        """
        if not self.set_observables:
            msg = "Observable list not set. Please use the set_observable_list method to set the observables."
            raise ValueError(msg)

        for i, proc in enumerate(noise_model.compact_processes):
            for j, site in enumerate(proc["sites"]):
                if (
                    proc["name"] != self.compact_noise_model.compact_processes[i]["name"]
                    or site != self.compact_noise_model.compact_processes[i]["sites"][j]
                ):
                    msg = "Noise model processes or sites do not match the initialized noise model."
                    raise ValueError(msg)




        ##### Scikitt-tt part #####

        self.scikit_obs_list = self.scikit_tt_obs_list(self.obs_list)


        self.scikit_hamiltonian = self.scikit_tt_hamiltonian(self.hamiltonian)
        self.scikit_jump_operator_list, self.scikit_jump_parameter_list = self.scikit_tt_noise_list(noise_model.expanded_noise_model)


        self.scikit_tt_solver: dict = {"solver": 'tdvp'+str(self.sim_params.order), "method": 'krylov', "dimension": 5}

        self.scikit_initial_state = self.scikit_tt_init_state(self.init_state)
        


        if self.tjm:

            arg_list = [ (k, self.n_obs, self.scikit_obs_list, self.scikit_hamiltonian, self.scikit_jump_operator_list, self.scikit_jump_parameter_list, self.n_t, self.sim_params.dt, self.scikit_tt_solver, self.scikit_initial_state) for k in range(self.sim_params.num_traj) ]

            with multiprocessing.Pool(processes=available_cpus()-1) as pool:
                results = pool.starmap(process_k, arg_list)


        exp_vals = np.sum(results, axis=0)/self.sim_params.num_traj


        self.obs_array = np.real(exp_vals)

        self.times = self.sim_params.times











class PropagatorWithGradients(Propagator):
    r"""High-level propagator that runs an MPS-based Lindblad simulation.

    The class wraps simulator inputs, performs
    consistency checks between noise models and the Hamiltonian, augments the
    observable set with Lindblad-derived A_kn operators (sensitivities of
    expectation values w.r.t. jump rates), runs the underlying simulator, and
    post-processes simulator outputs into convenient arrays for analysis.

    Attributes:
    obs_list : list[Observable]
        (Set after set_observable_list) Deep copy of user-provided observables.
    n_obs : int
        (Set after set_observable_list) Number of observables.
    times : array-like
        Time grid used by the most recent run (copied from sim_params.times).
    obs_traj : list[Observable]
        Observables returned by the simulator corresponding to the original
        user-requested observables (populated by run).
    obs_array : numpy.ndarray
        Array of observable trajectories with shape (n_obs, n_timesteps).
    d_on_d_gk : numpy.ndarray
        Object-array of Observable entries (shape [n_jump, n_obs]) corresponding
        to A_kn-like operators (or zero placeholders) computed by the simulator
        and integrated in time.
    d_on_d_gk_array : numpy.ndarray
        Numeric array of integrated A_kn trajectories (shape [n_jump, n_obs]).
    Other internal fields may be set during execution (e.g., temporary lists and
    simulator-specific containers).
    Public methods
    set_observable_list(obs_list: list[Observable]) -> None
        Store and validate a deep copy of obs_list. Validates that every site index
        referenced by the observables is within the range [0, sites-1]. Sets
        n_obs and flips set_observables to True. Raises ValueError for empty lists
        or out-of-range site indices.
    run(noise_model: CompactNoiseModel) -> None
        Execute the propagation. Requires that set_observables has been called.
        Validates that the provided compact noise_model matches the one used to
        construct this propagator (same process names and site assignments).
        Constructs A_kn-like observables for each matching jump/operator pair,
        appends them to the observable list, builds a new AnalogSimParams instance
        for the simulator, and invokes the underlying simulator with the expanded
        noise model. Post-processes results by trapezoidally integrating A_kn
        trajectories, arranging them into object and numeric arrays (d_on_d_gk and
        d_on_d_gk_array) and extracting obs_traj and obs_array for the original
        observables.
        - During initialization: if any site index in compact_noise_model.expanded_noise_model
          exceeds the number of sites in the Hamiltonian.
        - set_observable_list: if obs_list is empty or contains observables that
          reference out-of-range site indices.
        - run: if observables have not been set (set_observables is False) or if the
          provided noise_model does not match the initialized compact_noise_model
          in process names or site indices.
    - All constructor inputs are deep-copied to avoid accidental external mutation.
    - The class expects external types (AnalogSimParams, MPO, MPS, CompactNoiseModel,
      Observable) to expose particular attributes (for example, `times`, `length`,
      `expanded_noise_model`, `compact_processes`, `gate`, `sites`, and `results`).
    - The A_kn operators constructed in run follow the Lindblad derivative form:
      L_k^\dagger O L_k - 0.5 {L_k^\dagger L_k, O}, computed only for observables
      that act on the same site(s) as the corresponding jump operator.
    - The user-facing numeric arrays (obs_array and d_on_d_gk_array) are convenient
      summaries for optimization or analysis tasks (e.g., gradient-based fitting of
      jump rates).
    """
    def run(self, noise_model: CompactNoiseModel) -> None:
        """Run the propagation routine with augmented Lindblad-derived operators.

        Parameters
        ----------
        noise_model : CompactNoiseModel
            The compact representation of the noise model to use for propagation.
            The method verifies that the list of compact processes and their sites
            in `noise_model` match the model used to initialize this propagator
            (self.compact_noise_model). The expanded form of this model is passed
            to the underlying simulator.

        Side effects / State changes
        ----------------------------
        On successful completion, several attributes of self are set or updated:
        - self.obs_traj : list[Observable]
            The list of original observables (with their computed time trajectories)
            extracted from the simulator results.
        - self.d_on_d_gk : numpy.ndarray of shape (n_jump, n_obs) with Observable entries
            A matrix of the A_kn-like operators (or zero placeholders) corresponding
            to each jump operator / observable pair; entries are Observable objects
            whose .results have been integrated (trapezoidally) over time.
        - self.d_on_d_gk_array : numpy.ndarray
            2D array of numeric trajectories corresponding to d_on_d_gk (shape
            [n_jump, n_obs, n_timesteps]).
        - self.obs_array : numpy.ndarray
            2D array of numeric trajectories for the original observables
            (shape [n_obs, n_timesteps]).
        - self.times : array-like
            Time grid used by the simulation (copied from self.sim_params.times).

        Raises:
            ValueError: If the observable list has not been initialized (self.set_observables is False).
            ValueError: If any process name or site in the provided noise_model does not match
              the corresponding entry in self.compact_noise_model.

        Notes:
        -----
        - The purpose of the added A_kn observables is to provide sensitivity-like
          quantities (derivatives of observable expectations with respect to
          jump rates) that are computed by the same underlying simulator and then
          post-processed into arrays suitable for analysis or parameter updates.
        """
        if not self.set_observables:
            msg = "Observable list not set. Please use the set_observable_list method to set the observables."
            raise ValueError(msg)

        for i, proc in enumerate(noise_model.compact_processes):
            for j, site in enumerate(proc["sites"]):
                if (
                    proc["name"] != self.compact_noise_model.compact_processes[i]["name"]
                    or site != self.compact_noise_model.compact_processes[i]["sites"][j]
                ):
                    msg = "Noise model processes or sites do not match the initialized noise model."
                    raise ValueError(msg)

        a_kn_site_list: list[Observable] = []

        for lk in self.noise_list:
            a_kn_site_list.extend(
                Observable(
                    lk.gate.dag() * on.gate * lk.gate
                    - 0.5 * on.gate * lk.gate.dag() * lk.gate
                    - 0.5 * lk.gate.dag() * lk.gate * on.gate,
                    lk.sites,
                )
                for on in self.obs_list
                if lk.sites == on.sites
            )

        new_obs_list = self.obs_list + a_kn_site_list


        ##### Scikitt-tt part #####

        scikit_new_obs_list = self.scikit_tt_obs_list(new_obs_list)

        n_new_obs = len(new_obs_list)

        scikit_hamiltonian = self.scikit_tt_hamiltonian(self.hamiltonian)
        scikit_jump_operator_list, scikit_jump_parameter_list = self.scikit_tt_noise_list(noise_model.expanded_noise_model)


        scikit_tt_solver: dict = {"solver": 'tdvp'+str(self.sim_params.order), "method": 'krylov', "dimension": 5}

        scikit_initial_state = self.scikit_tt_init_state(self.init_state)


        self.scikit_new_obs_list = scikit_new_obs_list
        self.scikit_hamiltonian = scikit_hamiltonian
        self.scikit_jump_operator_list = scikit_jump_operator_list
        self.scikit_jump_parameter_list = scikit_jump_parameter_list
        self.scikit_tt_solver = scikit_tt_solver
        self.scikit_initial_state = scikit_initial_state
        

        arg_list = [ (k, n_new_obs, scikit_new_obs_list, scikit_hamiltonian, scikit_jump_operator_list, scikit_jump_parameter_list, self.n_t, self.sim_params.dt, scikit_tt_solver, scikit_initial_state) for k in range(self.sim_params.num_traj) ]

        with multiprocessing.Pool(processes=available_cpus()-1) as pool:
            results = pool.starmap(process_k, arg_list)


        exp_vals = np.sum(results, axis=0)/self.sim_params.num_traj



        ##### Scikitt-tt part #####




        # Separate original and new expectation values from result_lindblad.
        self.obs_array = np.real(exp_vals[:self.n_obs])

        d_on_d_gk_list = np.real(exp_vals[self.n_obs:]) # these correspond to the A_kn operators

        self.d_on_d_gk_array = np.zeros((self.n_jump, self.n_obs, self.n_t), dtype=float)

        count = 0
        for i, lk in enumerate(self.noise_list):
            for j, on in enumerate(self.obs_list):
                if lk.sites == on.sites:
                    self.d_on_d_gk_array[i, j] = trapezoidal(d_on_d_gk_list[count], self.sim_params.times)
                    count += 1
                else:
                    self.d_on_d_gk_array[i, j] = np.zeros(self.n_t, dtype=float)

        self.times = self.sim_params.times

        self.obs_traj = copy.deepcopy(self.obs_list)
        
        for i in range(self.n_obs):
            self.obs_traj[i].results = self.obs_array[i]




###########################
###########################
###########################
###########################
###########################
###########################
###########################
###########################
###########################



if __name__ == "__main__":
    #%%
    work_dir=f"test/characterizer/"

    work_dir_path = Path(work_dir)

    work_dir_path.mkdir(parents=True, exist_ok=True)



    ## Defining Hamiltonian and observable list
    L=3

    J=1
    g=0.5


    H_0 = MPO()
    H_0.init_ising(L, J, g)


    # Define the initial state
    init_state = MPS(L, state='zeros')


    # obs_list = [Observable(X(), site) for site in range(L)]  + [Observable(Y(), site) for site in range(L)] + [Observable(Z(), site) for site in range(L)]
    obs_list = [Observable(Y(), site) for site in range(L)]



    #%%
    ## Defining simulation parameters

    T=3

    dt=0.1

    N=1

    max_bond_dim=8

    threshold=1e-6

    order=2

    sim_params = AnalogSimParams(observables=obs_list, elapsed_time=T, dt=dt, num_traj=N, max_bond_dim=max_bond_dim, threshold=threshold, order=order, sample_timesteps=True)





    #%%


    #%%
    ## Defining reference noise model and reference trajectory
    gamma_rel = 0.1

    gamma_deph = 0.1
    # ref_noise_model =  CompactNoiseModel([{"name": "lowering", "sites": [i for i in range(L)], "strength": gamma_rel}] + [{"name": "pauli_z", "sites": [i for i in range(L)], "strength": gamma_deph}])
    ref_noise_model =  CompactNoiseModel( [{"name": "pauli_z", "sites": [i for i in range(L)], "strength": gamma_deph} ])

    # ref_noise_model =  CompactNoiseModel([{"name": noise_operator, "sites": [i], "strength": gamma_rel} for i in range(L)] )


    ## Writing reference gammas to file
    np.savetxt(work_dir + "gammas.txt", ref_noise_model.strength_list, header="##", fmt="%.6f")


    propagator = PropagatorWithGradients(
        sim_params=sim_params,
        hamiltonian=H_0,
        compact_noise_model=ref_noise_model,
        init_state=init_state
    )
    # %%
    # scikit_init_state=propagator.scikit_tt_init_state(init_state)
    # print("scikit_init_state shape:", scikit_init_state.col_dims, scikit_init_state.row_dims)
    # %%
