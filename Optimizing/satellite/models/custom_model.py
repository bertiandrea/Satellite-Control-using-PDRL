# custom_model.py

import isaacgym #BugFix
import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, hidden_size=256, 
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        #print("Policy: observation_space", observation_space)
        #print("Policy: action_space", action_space)
        #print("Policy: num_observations", self.num_observations)
        self.net = nn.Sequential(nn.Linear(self.num_observations, hidden_size),
                                 nn.ELU(), #Also Tanh() or ReLU()
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ELU(), #Also Tanh() or ReLU()
                                 nn.Linear(hidden_size, hidden_size // 2),
                                 nn.ELU(), #Also Tanh() or ReLU()
                                 )
        self.mean_layer = nn.Linear(hidden_size // 2, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=device))
    
    def act(self, inputs, role):
        inputs["states"] = inputs["states"][:, :self.num_observations]

        return GaussianMixin.act(self, inputs, "policy")

    def compute(self, inputs, role):
        inputs = inputs["states"][:, :self.num_observations]

        return self.mean_layer(self.net(inputs)), self.log_std_parameter, {}
        
class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, hidden_size=256): #observation_space init like state_space
        Model.__init__(self, observation_space, action_space, device) #observation_space init like state_space
        DeterministicMixin.__init__(self, clip_actions)
        #print("Value: observation_space", observation_space)
        #print("Value: action_space", action_space)
        #print("Value: num_observations", self.num_observations)
        self.net = nn.Sequential(nn.Linear(self.num_observations, hidden_size), #num_observations init like num_states
                                 nn.ELU(), #Also Tanh() or ReLU()
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ELU(), #Also Tanh() or ReLU()
                                 nn.Linear(hidden_size, hidden_size // 2),
                                 nn.ELU()  #Also Tanh() or ReLU()
                                 )
        self.value_layer = nn.Linear(hidden_size // 2, 1)

    def act(self, inputs, role):
        inputs["states"] = inputs["states"][:, :self.num_observations]

        return DeterministicMixin.act(self, inputs, "value")

    def compute(self, inputs, role):
        inputs = inputs["states"][:, :self.num_observations]

        return self.value_layer(self.net(inputs)), {}


class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, hidden_size=512,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, hidden_size),
                                 nn.ELU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ELU(),
                                 nn.Linear(hidden_size, hidden_size//2),
                                 nn.ELU())

        self.mean_layer = nn.Linear(hidden_size//2, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(hidden_size//2, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            self._shared_output = self.net(inputs["states"])
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}