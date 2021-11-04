# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import numpy as np
import torch

from maro.rl.model import SimpleMultiHeadModel
from maro.utils.exception.rl_toolkit_exception import UnrecognizedTask


from .abs_agent import AbsAgent
from .dqn import DQNConfig


class DQN_ADF(AbsAgent):
    """DQN with action-dependent features
    
    Args:
        model (SimpleMultiHeadModel): Q-value model that takes a state-action pair and returns a number.
        config: Configuration for DQN_ADF algorithm.
    """
    def __init__(self, model: SimpleMultiHeadModel, config: DQNConfig):
        if (config.advantage_type is not None and
                (model.task_names is None or set(model.task_names) != {"state_value", "advantage"})):
            raise UnrecognizedTask(
                f"Expected model task names 'state_value' and 'advantage' since dueling DQN is used, "
                f"got {model.task_names}"
            )
        super().__init__(model, config)
        self._training_counter = 0
        self._target_model = model.copy() if model.trainable else None
        print(f'################ Update Frequency: {self.config.target_update_freq} ########################')
    
    def choose_action(self, state) -> Union[int, np.ndarray]:
        q_estimates = self._apply_model(state, is_eval=True, training=False).cpu().numpy()
            # Obtain Q(s,a) estimates for each action characterized by the state
        num_actions = np.shape(q_estimates)[0]
        greedy_action = q_estimates.argmax()
            # Determines the greedy action by taking argmax of the Q estimates
        # No exploration
        if self.config.epsilon == .0:
            return state[greedy_action][0]
        # Otherwise sample uniformly from the set of actions based on the epsilon parameter
        return state[greedy_action][0] if np.random.random() > self.config.epsilon \
            else state[np.random.choice(num_actions)][0]

    def _compute_td_errors(self, states, actions, rewards, next_states, discount_exponent):
        # Compute the one step TD errors for a given (s,a,r,s') tuple

        current_q_values_for_all_actions = self._batched_apply_model(states, is_eval=True, training=True)
        # Get the Q values for all actions in the current batch states list

        state_offsets = np.cumsum([0] + [len(state) for state in states[:-1]])
        current_q_values = current_q_values_for_all_actions[state_offsets+actions]
        # Evaluate it at the selected actions.  State_Offsets is used in order to deal w/ stacking

        next_q_values = self._get_next_q_values(next_states)  # (N,)
        # Get the next Q values for the next_states

        target_q_values = ((self.config.reward_discount**discount_exponent) * next_q_values + rewards).detach()
        
        return self.config.loss_func(current_q_values, target_q_values)

    def learn(self, states, actions: np.ndarray, rewards: np.ndarray, next_states):
        # Compute the TD errors and take a step on the loss
        old_ticks = torch.from_numpy(rewards[:, 0].astype(np.float32)).to(self.device)
        next_ticks = torch.from_numpy(rewards[:, 1].astype(np.float32)).to(self.device)

        
        rewards = torch.from_numpy(rewards[:, 2].astype(np.float32)).to(self.device)

        discount_exponent = next_ticks - old_ticks

        loss = self._compute_td_errors(states, actions, rewards, next_states, discount_exponent)
        self.model.step(loss.mean())
        self._training_counter += 1
        if self._training_counter % self.config.target_update_freq == 0: # Soft updates of the target model
            self._target_model.soft_update(self.model, self.config.tau)
        return loss.detach().numpy()

    def set_exploration_params(self, epsilon):
        if type(epsilon) is dict:
            self.config.epsilon = epsilon["epsilon"]
        else:
            self.config.epsilon = epsilon

    def _get_q_values(self, state, is_eval: bool = True, training: bool = True):
        output = self.model(state, training=training) if is_eval else self._target_model(state, training=False)
        if self.config.advantage_type is None:
            return output
        else:    
            state_values = output["state_value"]
            advantages = output["advantage"]
            # Use mean or max correction to address the identifiability issue
            corrections = advantages.mean(1) if self.config.advantage_type == "mean" else advantages.max(1)[0]
            return state_values + advantages - corrections.unsqueeze(1)
        
    def _get_next_q_values(self, next_states):
        next_q_values_for_all_actions = self._batched_apply_model(next_states, is_eval=False, training=False)

        # Applies function f to subsets of q_values corresponding to states
        def qmap(q_values, f):
            ret = []
            start_idx = 0
            for state in next_states:
                end_idx = start_idx + len(state)
                ret.append(f(q_values[start_idx: end_idx]))
                start_idx = end_idx
            return ret

        if self.config.double:
            next_q_all_eval = self._batched_apply_model(next_states, is_eval=True, training=False)
            actions = qmap(next_q_all_eval, torch.argmax)
            # Argmax indexes into each state, add state offsets to index into the full array
            state_offsets = np.cumsum([0] + [len(state) for state in next_states[:-1]])
            return next_q_values_for_all_actions[state_offsets + actions]
        else:
            max_q_vals = qmap(next_q_values_for_all_actions, torch.max)
            return torch.Tensor(max_q_vals).to(self.device)
            
    def _apply_model(self, state, is_eval: bool = True, training = False):
        num_actions = len(state)
        batched_state = np.stack([state_features for _, state_features in state])
        tensor_state = torch.from_numpy(batched_state.astype(np.float32)).to(self.device)
        q_estimates = self._get_q_values(tensor_state, is_eval, training=training)
        return q_estimates.squeeze(1)

    def _batched_apply_model(self, states, is_eval: bool = True, training = False):
        batched_state = np.stack([state_features for state in states for _, state_features in state])
        tensor_state = torch.from_numpy(batched_state.astype(np.float32)).to(self.device)
        q_estimates = self._get_q_values(tensor_state, is_eval, training=training)
        return q_estimates.squeeze(1)