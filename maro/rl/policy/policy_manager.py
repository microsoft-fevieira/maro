# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from multiprocessing import Pipe, Process
from os import getcwd
from typing import Callable, Dict

from maro.communication import Proxy, SessionMessage, SessionType
from maro.rl.experience import ExperienceSet
from maro.rl.policy import AbsCorePolicy
from maro.rl.utils import MsgKey, MsgTag
from maro.utils import Logger

from .trainer import trainer_process


class AbsPolicyManager(ABC):
    """Manage all policies.

    The actual policy instances may reside here or be distributed on a set of processes or remote nodes.

    Args:
        policy_dict (Dict[str, AbsCorePolicy]): A list of policies managed by the manager.
        update_trigger (Dict[str, int]): A dictionary of (policy_name, trigger), where "trigger" indicates the
            required number of new experiences to trigger a call to ``learn`` for each policy. Defaults to None,
            all triggers will be set to 1.
        warmup (Dict[str, int]): A dictionary of (policy_name, warmup_size), where "warmup_size" indicates the
            minimum number of experiences in the experience memory required to trigger a call to ``learn`` for
            each policy. Defaults to None, in which case all warm-up sizes will be set to 1.
        post_update (Callable): Custom function to process whatever information is collected by each
            trainer (local or remote) at the end of ``update`` calls. The function signature should be (trackers,
            ) -> None, where tracker is a list of environment wrappers' ``tracker`` members. Defaults to
            None.
    """
    def __init__(
        self,
        policy_dict: Dict[str, AbsCorePolicy],
        update_trigger: Dict[str, int] = None,
        warmup: Dict[str, int] = None,
        post_update: Callable = None
    ):
        for policy in policy_dict.values():
            if not isinstance(policy, AbsCorePolicy):
                raise ValueError("Only 'AbsCorePolicy' instances can be managed by a policy manager.")

        super().__init__()
        self.policy_dict = policy_dict
        if not update_trigger:
            self.update_trigger = {name: 1 for name in self.policy_dict}
        else:
            self.update_trigger = update_trigger
        if not warmup:
            self.warmup = {name: 1 for name in self.policy_dict}
        else:
            self.warmup = warmup

        self._post_update = post_update

        self._update_history = [set(policy_dict.keys())]
        self.tracker = {}

    @property
    def version(self):
        return len(self._update_history) - 1

    @abstractmethod
    def update(self, exp_by_policy: Dict[str, ExperienceSet]):
        """Logic for handling incoming experiences is implemented here."""
        raise NotImplementedError

    def get_state(self, cur_version: int = None):
        if cur_version is None:
            cur_version = self.version - 1
        updated = set()
        for version in range(cur_version + 1, len(self._update_history)):
            updated |= self._update_history[version]
        return {name: self.policy_dict[name].get_state() for name in updated}


class LocalPolicyManager(AbsPolicyManager):
    """Policy manager that contains the actual policy instances.

    Args:
        policy_dict (Dict[str, AbsCorePolicy]): Policies managed by the manager.
        update_trigger (Dict[str, int]): A dictionary of (policy_name, trigger), where "trigger" indicates the
            required number of new experiences to trigger a call to ``learn`` for each policy. Defaults to None,
            all triggers will be set to 1.
        warmup (Dict[str, int]): A dictionary of (policy_name, warmup_size), where "warmup_size" indicates the
            minimum number of experiences in the experience memory required to trigger a call to ``learn`` for
            each policy. Defaults to None, in which case all warm-up sizes will be set to 1.
        post_update (Callable): Custom function to process whatever information is collected by each
            trainer (local or remote) at the end of ``update`` calls. The function signature should be (trackers,
            ) -> None, where tracker is a list of environment wrappers' ``tracker`` members. Defaults to
            None.
        log_dir (str): Directory to store logs in. A ``Logger`` with tag "POLICY_MANAGER" will be created at init
            time and this directory will be used to save the log files generated by it. Defaults to the current
            working directory.
    """
    def __init__(
        self,
        policy_dict: Dict[str, AbsCorePolicy],
        update_trigger: Dict[str, int] = None,
        warmup: Dict[str, int] = None,
        post_update: Callable = None,
        log_dir: str = getcwd()
    ):
        super().__init__(policy_dict, update_trigger=update_trigger, warmup=warmup, post_update=post_update)
        self._new_exp_counter = defaultdict(int)
        self._logger = Logger("LOCAL_POLICY_MANAGER", dump_folder=log_dir)

    def update(self, exp_by_policy: Dict[str, ExperienceSet]):
        """Store experiences and update policies if possible.

        The incoming experiences are expected to be grouped by policy ID and will be stored in the corresponding
        policy's experience manager. Policies whose update conditions have been met will then be updated.
        """
        t0 = time.time()
        updated = set()
        for policy_name, exp in exp_by_policy.items():
            policy = self.policy_dict[policy_name]
            policy.experience_store.put(exp)
            self._new_exp_counter[policy_name] += exp.size
            if (
                self._new_exp_counter[policy_name] >= self.update_trigger[policy_name] and
                policy.experience_store.size >= self.warmup[policy_name]
            ):
                policy.learn()
                updated.add(policy_name)
                self._new_exp_counter[policy_name] = 0

        if updated:
            self._update_history.append(updated)
            self._logger.info(f"Updated policies {updated}")

        if self._post_update:
            self._post_update([policy.tracker for policy in self.policy_dict.values()])

        self._logger.debug(f"policy update time: {time.time() - t0}")


class MultiProcessPolicyManager(AbsPolicyManager):
    """Policy manager that spawns a set of trainer processes for parallel training.

    Args:
        policy_dict (Dict[str, AbsCorePolicy]): Policies managed by the manager.
        num_trainers (int): Number of trainer processes to be forked.
        create_policy_func_dict (dict): A dictionary mapping policy names to functions that create them. The policy
            creation function should have exactly one parameter which is the policy name and return an ``AbsPolicy``
            instance.
        update_trigger (Dict[str, int]): A dictionary of (policy_name, trigger), where "trigger" indicates the
            required number of new experiences to trigger a call to ``learn`` for each policy. Defaults to None,
            all triggers will be set to 1.
        warmup (Dict[str, int]): A dictionary of (policy_name, warmup_size), where "warmup_size" indicates the
            minimum number of experiences in the experience memory required to trigger a call to ``learn`` for
            each policy. Defaults to None, in which case all warm-up sizes will be set to 1.
        post_update (Callable): Custom function to process whatever information is collected by each
            trainer (local or remote) at the end of ``update`` calls. The function signature should be (trackers,)
            -> None, where tracker is a list of environment wrappers' ``tracker`` members. Defaults to None.
        log_dir (str): Directory to store logs in. A ``Logger`` with tag "POLICY_MANAGER" will be created at init
            time and this directory will be used to save the log files generated by it. Defaults to the current
            working directory.
    """
    def __init__(
        self,
        policy_dict: Dict[str, AbsCorePolicy],
        num_trainers: int,
        create_policy_func_dict: Dict[str, Callable],
        update_trigger: Dict[str, int] = None,
        warmup: Dict[str, int] = None,
        post_update: Callable = None,
        log_dir: str = getcwd(),
    ):
        super().__init__(policy_dict, update_trigger=update_trigger, warmup=warmup, post_update=post_update)
        self._policy2trainer = {}
        self._trainer2policies = defaultdict(list)
        self._exp_cache = defaultdict(ExperienceSet)
        self._num_experiences_by_policy = defaultdict(int)

        for i, name in enumerate(self.policy_dict):
            trainer_id = i % num_trainers
            self._policy2trainer[name] = f"TRAINER.{trainer_id}"
            self._trainer2policies[f"TRAINER.{trainer_id}"].append(name)

        self._logger = Logger("MULTIPROCESS_POLICY_MANAGER", dump_folder=log_dir)

        self._trainer_processes = []
        self._manager_end = {}
        for trainer_id, policy_names in self._trainer2policies.items():
            manager_end, trainer_end = Pipe()
            self._manager_end[trainer_id] = manager_end
            trainer = Process(
                target=trainer_process,
                args=(
                    trainer_id,
                    trainer_end,
                    {name: create_policy_func_dict[name] for name in policy_names},
                    {name: self.policy_dict[name].get_state() for name in self._trainer2policies[trainer_id]}
                ),
                kwargs={"log_dir": log_dir}
            )
            self._trainer_processes.append(trainer)
            trainer.start()

    def update(self, exp_by_policy: Dict[str, ExperienceSet]):
        exp_to_send, updated = {}, set()
        for policy_name, exp in exp_by_policy.items():
            self._num_experiences_by_policy[policy_name] += exp.size
            self._exp_cache[policy_name].extend(exp)
            if (
                self._exp_cache[policy_name].size >= self.update_trigger[policy_name] and
                self._num_experiences_by_policy[policy_name] >= self.warmup[policy_name]
            ):
                exp_to_send[policy_name] = self._exp_cache.pop(policy_name)
                updated.add(policy_name)

        for trainer_id, conn in self._manager_end.items():
            conn.send({
                "type": "train",
                "experiences": {name: exp_to_send[name] for name in self._trainer2policies[trainer_id]}
            })

        trackers = []
        for conn in self._manager_end.values():
            result = conn.recv()
            trackers.append(result["tracker"])
            for policy_name, policy_state in result["policy"].items():
                self.policy_dict[policy_name].set_state(policy_state)

        if updated:
            self._update_history.append(updated)
            self._logger.info(f"Updated policies {updated}")

        if self._post_update:
            self._post_update(trackers)

    def exit(self):
        """Tell the trainer processes to exit."""
        for conn in self._manager_end.values():
            conn.send({"type": "quit"})


class MultiNodePolicyManager(AbsPolicyManager):
    """Policy manager that communicates with a set of remote nodes for parallel training.

    Args:
        policy_dict (Dict[str, AbsCorePolicy]): Policies managed by the manager.
        group (str): Group name for the training cluster, which includes all trainers and a training manager that
            manages them.
        num_trainers (int): Number of trainers. The trainers will be identified by "TRAINER.i", where
            0 <= i < num_trainers.
        update_trigger (Dict[str, int]): A dictionary of (policy_name, trigger), where "trigger" indicates the
            required number of new experiences to trigger a call to ``learn`` for each policy. Defaults to None,
            all triggers will be set to 1.
        warmup (Dict[str, int]): A dictionary of (policy_name, warmup_size), where "warmup_size" indicates the
            minimum number of experiences in the experience memory required to trigger a call to ``learn`` for
            each policy. Defaults to None, in which case all warm-up sizes will be set to 1.
        post_update (Callable): Custom function to process whatever information is collected by each
            trainer (local or remote) at the end of ``update`` calls. The function signature should be (trackers,)
            -> None, where tracker is a list of environment wrappers' ``tracker`` members. Defaults to None.
        log_dir (str): Directory to store logs in. A ``Logger`` with tag "POLICY_MANAGER" will be created at init
            time and this directory will be used to save the log files generated by it. Defaults to the current
            working directory.
        proxy_kwargs: Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to the empty dictionary.
    """
    def __init__(
        self,
        policy_dict: Dict[str, AbsCorePolicy],
        group: str,
        num_trainers: int,
        update_trigger: Dict[str, int] = None,
        warmup: Dict[str, int] = None,
        post_update: Callable = None,
        log_dir: str = getcwd(),
        proxy_kwargs: dict = {}
    ):
        super().__init__(policy_dict, update_trigger=update_trigger, warmup=warmup, post_update=post_update)
        peers = {"trainer": num_trainers}
        self._proxy = Proxy(group, "policy_manager", peers, component_name="POLICY_MANAGER", **proxy_kwargs)

        self._policy2trainer = {}
        self._trainer2policies = defaultdict(list)
        self._exp_cache = defaultdict(ExperienceSet)
        self._num_experiences_by_policy = defaultdict(int)

        self._logger = Logger("MULTINODE_POLICY_MANAGER", dump_folder=log_dir)

        for i, name in enumerate(self.policy_dict):
            trainer_id = i % num_trainers
            self._policy2trainer[name] = f"TRAINER.{trainer_id}"
            self._trainer2policies[f"TRAINER.{trainer_id}"].append(name)

        self._logger.info("Initializing policy states on trainers...")
        for trainer_name, policy_names in self._trainer2policies.items():
            self._proxy.send(
                SessionMessage(
                    MsgTag.INIT_POLICY_STATE, self._proxy.name, trainer_name,
                    body={MsgKey.POLICY_STATE: {name: self.policy_dict[name].get_state() for name in policy_names}}
                )
            )

    def update(self, exp_by_policy: Dict[str, ExperienceSet]):
        exp_to_send, updated = {}, set()
        for policy_name, exp in exp_by_policy.items():
            self._num_experiences_by_policy[policy_name] += exp.size
            self._exp_cache[policy_name].extend(exp)
            if (
                self._exp_cache[policy_name].size >= self.update_trigger[policy_name] and
                self._num_experiences_by_policy[policy_name] >= self.warmup[policy_name]
            ):
                exp_to_send[policy_name] = self._exp_cache.pop(policy_name)
                updated.add(policy_name)

        msg_body_by_dest = defaultdict(dict)
        for policy_name, exp in exp_to_send.items():
            trainer_id = self._policy2trainer[policy_name]
            if MsgKey.EXPERIENCES not in msg_body_by_dest[trainer_id]:
                msg_body_by_dest[trainer_id][MsgKey.EXPERIENCES] = {}
            msg_body_by_dest[trainer_id][MsgKey.EXPERIENCES][policy_name] = exp

        trackers = []
        for reply in self._proxy.scatter(MsgTag.LEARN, SessionType.TASK, list(msg_body_by_dest.items())):
            trackers.append(reply.body[MsgKey.TRACKER])
            for policy_name, policy_state in reply.body[MsgKey.POLICY_STATE].items():
                self.policy_dict[policy_name].set_state(policy_state)

        if updated:
            self._update_history.append(updated)
            self._logger.info(f"Updated policies {updated}")

        if self._post_update:
            self._post_update(trackers)

    def exit(self):
        """Tell the remote trainers to exit."""
        self._proxy.ibroadcast("trainer", MsgTag.EXIT, SessionType.NOTIFICATION)
        self._proxy.close()
        self._logger.info("Exiting...")
