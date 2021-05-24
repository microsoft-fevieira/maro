# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import numpy as np
import torch

from maro.rl.model import SimpleMultiHeadModel

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
    
    def choose_action(self, state) -> Union[int, np.ndarray]:
        q_estimates = self._apply_model(state, is_eval=True, training=False).cpu().numpy()
        num_actions = np.shape(q_estimates)[0]
        greedy_action = q_estimates.argmax()
        # No exploration
        if self.config.epsilon == .0:
            return state[greedy_action][0]
        return state[greedy_action][0] if np.random.random() > self.config.epsilon \
            else state[np.random.choice(num_actions)][0]

    def _compute_td_errors(self, states, actions, rewards, next_states):
        current_q_values_for_all_actions = self._apply_model(states, is_eval=True, training=True)
        current_q_values = current_q_values_for_all_actions[actions]
        next_q_values = self._get_next_q_values(next_states)  # (N,)
        target_q_values = (rewards + self.config.reward_discount * next_q_values).detach()
        return self.config.loss_func(current_q_values, target_q_values)

    def learn(self, states, actions: np.ndarray, rewards: np.ndarray, next_states):
        #states = torch.from_numpy(states).to(self.device)
        #actions = torch.from_numpy(actions).to(self.device)
        #rewards = torch.from_numpy(rewards).to(self.device)
        #next_states = torch.from_numpy(next_states).to(self.device)
        
        #No support for minibatch yet
        assert len(states) == 1, "DQN_ADF does not support minibatching. Set batch_size to 1."
        
        loss = self._compute_td_errors(states[0], actions[0], rewards[0], next_states[0])
        self.model.step(loss.mean())
        self._training_counter += 1
        if self._training_counter % self.config.target_update_freq == 0:
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
        next_q_values_for_all_actions = self._apply_model(next_states, is_eval=False, training=False)
        if self.config.double:
            next_q_all_eval = self._apply_model(next_states, is_eval=True, training=False)
            actions = next_q_all_eval.argmax()
            return next_q_values_for_all_actions[actions]
        else:
            return next_q_values_for_all_actions.max()
            
    def _apply_model(self, state, is_eval: bool = True, training = False):
        num_actions = len(state)
        batched_state = np.stack([state_features for _, state_features in state])
        tensor_state = torch.from_numpy(batched_state.astype(np.float32)).to(self.device)
        q_estimates = self._get_q_values(tensor_state, is_eval, training=training)
        return q_estimates.squeeze(1)
