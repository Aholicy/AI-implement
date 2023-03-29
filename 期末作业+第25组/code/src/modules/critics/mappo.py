import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm


class MAPPOCritic(nn.Module):
    def __init__(self, scheme, args):
        super(MAPPOCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"
        self.hidden_states = None
        self.critic_hidden_dim = args.critic_hidden_dim

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.fc1.weight.new(1, self.args.critic_hidden_dim).zero_()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def forward(self, batch, t=None):
        inputs = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)
        return v

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observation
        inputs.append(batch["obs"][:, ts])

        # actions (masked out by agent)
        #actions = batch["actions_onehot"][:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        #agent_mask = (1 - th.eye(self.n_agents, device=batch.device))
        #agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        #inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))

        # last actions
        #if t == 0:
        #    inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        #elif isinstance(t, int):
        #    inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        #else:
        #    last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
        #    last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        #    inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1) # [64,1,3,173]
        return inputs.squeeze(1) # [64,3,173]

    def _get_input_shape(self, scheme):
        # MAPPO中的V函数输入为：全局状态 + 自身局部观察 + 智能体的id (OneHot形式)
        # state
        input_shape = scheme["state"]["vshape"] # 85
        # observation
        input_shape += scheme["obs"]["vshape"] # 85
        # last actions
        #input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents # 33
        # agent id
        input_shape += self.n_agents # 3
        return input_shape