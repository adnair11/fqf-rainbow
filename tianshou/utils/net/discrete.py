from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tianshou.data import Batch, to_torch
from tianshou.utils.net.common import MLP, BaseActor, Net, TActionShape, get_output_dim


class Actor(BaseActor):
    """Simple actor network for discrete action spaces.

    :param preprocess_net: a self-defined preprocess_net. Typically, an instance of
        :class:`~tianshou.utils.net.common.Net`.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param softmax_output: whether to apply a softmax layer over the last
        layer's output.
    :param preprocess_net_output_dim: the output dimension of
        `preprocess_net`. Only used when `preprocess_net` does not have the attribute `output_dim`.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        preprocess_net: nn.Module | Net,
        action_shape: TActionShape,
        hidden_sizes: Sequence[int] = (),
        softmax_output: bool = True,
        preprocess_net_output_dim: int | None = None,
        device: str | int | torch.device = "cpu",
    ) -> None:
        super().__init__()
        # TODO: reduce duplication with continuous.py. Probably introducing
        #   base classes is a good idea.
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = int(np.prod(action_shape))
        input_dim = get_output_dim(preprocess_net, preprocess_net_output_dim)
        self.last = MLP(
            input_dim,
            self.output_dim,
            hidden_sizes,
            device=self.device,
        )
        self.softmax_output = softmax_output

    def get_preprocess_net(self) -> nn.Module:
        return self.preprocess

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""Mapping: s_B -> action_values_BA, hidden_state_BH | None.

        Returns a tensor representing the values of each action, i.e, of shape
        `(n_actions, )`, and
        a hidden state (which may be None). If `self.softmax_output` is True, they are the
        probabilities for taking each action. Otherwise, they will be action values.
        The hidden state is only
        not None if a recurrent net is used as part of the learning algorithm.
        """
        x, hidden_BH = self.preprocess(obs, state)
        x = self.last(x)
        if self.softmax_output:
            x = F.softmax(x, dim=-1)
        # If we computed softmax, output is probabilities, otherwise it's the non-normalized action values
        output_BA = x
        return output_BA, hidden_BH


class Critic(nn.Module):
    """Simple critic network for discrete action spaces.

    :param preprocess_net: a self-defined preprocess_net. Typically, an instance of
        :class:`~tianshou.utils.net.common.Net`.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param last_size: the output dimension of Critic network. Default to 1.
    :param preprocess_net_output_dim: the output dimension of
        `preprocess_net`. Only used when `preprocess_net` does not have the attribute `output_dim`.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`..
    """

    def __init__(
        self,
        preprocess_net: nn.Module | Net,
        hidden_sizes: Sequence[int] = (),
        last_size: int = 1,
        preprocess_net_output_dim: int | None = None,
        device: str | int | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = last_size
        input_dim = get_output_dim(preprocess_net, preprocess_net_output_dim)
        self.last = MLP(input_dim, last_size, hidden_sizes, device=self.device)

    # TODO: make a proper interface!
    def forward(self, obs: np.ndarray | torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Mapping: s_B -> V(s)_B."""
        # TODO: don't use this mechanism for passing state
        logits, _ = self.preprocess(obs, state=kwargs.get("state", None))
        return self.last(logits)


class CosineEmbeddingNetwork(nn.Module):
    """Cosine embedding network for IQN. Convert a scalar in [0, 1] to a list of n-dim vectors.

    :param num_cosines: the number of cosines used for the embedding.
    :param embedding_dim: the dimension of the embedding/output.

    .. note::

        From https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master
        /fqf_iqn_qrdqn/network.py .
    """

    def __init__(self, num_cosines: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(num_cosines, embedding_dim), nn.ReLU())
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def forward(self, taus: torch.Tensor) -> torch.Tensor:
        batch_size = taus.shape[0]
        N = taus.shape[1]
        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * torch.arange(
            start=1,
            end=self.num_cosines + 1,
            dtype=taus.dtype,
            device=taus.device,
        ).view(1, 1, self.num_cosines)
        # Calculate cos(i * \pi * \tau).
        cosines = torch.cos(taus.view(batch_size, N, 1) * i_pi).view(
            batch_size * N,
            self.num_cosines,
        )
        # Calculate embeddings of taus.
        return self.net(cosines).view(batch_size, N, self.embedding_dim)


class ImplicitQuantileNetwork(Critic):
    """Implicit Quantile Network.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param num_cosines: the number of cosines to use for cosine embedding.
        Default to 64.
    :param preprocess_net_output_dim: the output dimension of
        preprocess_net.

    .. note::

        Although this class inherits Critic, it is actually a quantile Q-Network
        with output shape (batch_size, action_dim, sample_size).

        The second item of the first return value is tau vector.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: TActionShape,
        hidden_sizes: Sequence[int] = (),
        num_cosines: int = 64,
        preprocess_net_output_dim: int | None = None,
        device: str | int | torch.device = "cpu",
    ) -> None:
        last_size = int(np.prod(action_shape))
        super().__init__(preprocess_net, hidden_sizes, last_size, preprocess_net_output_dim, device)
        self.input_dim = get_output_dim(preprocess_net, preprocess_net_output_dim)
        self.embed_model = CosineEmbeddingNetwork(num_cosines, self.input_dim).to(
            device,
        )

    def forward(  # type: ignore
        self,
        obs: np.ndarray | torch.Tensor,
        sample_size: int,
        **kwargs: Any,
    ) -> tuple[Any, torch.Tensor]:
        r"""Mapping: s -> Q(s, \*)."""
        logits, hidden = self.preprocess(obs, state=kwargs.get("state", None))
        # Sample fractions.
        batch_size = logits.size(0)
        taus = torch.rand(batch_size, sample_size, dtype=logits.dtype, device=logits.device)
        embedding = (logits.unsqueeze(1) * self.embed_model(taus)).view(
            batch_size * sample_size,
            -1,
        )
        out = self.last(embedding).view(batch_size, sample_size, -1).transpose(1, 2)
        return (out, taus), hidden


class FractionProposalNetwork(nn.Module):
    """Fraction proposal network for FQF.

    :param num_fractions: the number of factions to propose.
    :param embedding_dim: the dimension of the embedding/input.

    .. note::

        Adapted from https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master
        /fqf_iqn_qrdqn/network.py .
    """

    def __init__(self, num_fractions: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(embedding_dim, num_fractions)
        torch.nn.init.xavier_uniform_(self.net.weight, gain=0.01)
        torch.nn.init.constant_(self.net.bias, 0)
        self.num_fractions = num_fractions
        self.embedding_dim = embedding_dim

    def forward(
        self,
        obs_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Calculate (log of) probabilities q_i in the paper.
        dist = torch.distributions.Categorical(logits=self.net(obs_embeddings))
        taus_1_N = torch.cumsum(dist.probs, dim=1)
        # Calculate \tau_i (i=0,...,N).
        taus = F.pad(taus_1_N, (1, 0))
        # Calculate \hat \tau_i (i=0,...,N-1).
        tau_hats = (taus[:, :-1] + taus[:, 1:]).detach() / 2.0
        # Calculate entropies of value distributions.
        entropies = dist.entropy()
        return taus, tau_hats, entropies


class FullQuantileFunction(ImplicitQuantileNetwork):
    """Full(y parameterized) Quantile Function.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param num_cosines: the number of cosines to use for cosine embedding.
        Default to 64.
    :param preprocess_net_output_dim: the output dimension of
        preprocess_net.

    .. note::

        The first return value is a tuple of (quantiles, fractions, quantiles_tau),
        where fractions is a Batch(taus, tau_hats, entropies).
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: TActionShape,
        hidden_sizes: Sequence[int] = (),
        num_cosines: int = 64,
        preprocess_net_output_dim: int | None = None,
        device: str | int | torch.device = "cpu",
    ) -> None:
        super().__init__(
            preprocess_net,
            action_shape,
            hidden_sizes,
            num_cosines,
            preprocess_net_output_dim,
            device,
        )

    def _compute_quantiles(self, obs: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        batch_size, sample_size = taus.shape
        embedding = (obs.unsqueeze(1) * self.embed_model(taus)).view(batch_size * sample_size, -1)
        return self.last(embedding).view(batch_size, sample_size, -1).transpose(1, 2)

    def forward(  # type: ignore
        self,
        obs: np.ndarray | torch.Tensor,
        propose_model: FractionProposalNetwork,
        fractions: Batch | None = None,
        **kwargs: Any,
    ) -> tuple[Any, torch.Tensor]:
        r"""Mapping: s -> Q(s, \*)."""
        logits, hidden = self.preprocess(obs, state=kwargs.get("state", None))
        # Propose fractions
        if fractions is None:
            taus, tau_hats, entropies = propose_model(logits.detach())
            fractions = Batch(taus=taus, tau_hats=tau_hats, entropies=entropies)
        else:
            taus, tau_hats = fractions.taus, fractions.tau_hats
        quantiles = self._compute_quantiles(logits, tau_hats)
        # Calculate quantiles_tau for computing fraction grad
        quantiles_tau = None
        if self.training:
            with torch.no_grad():
                quantiles_tau = self._compute_quantiles(logits, taus[:, 1:-1])
        return (quantiles, fractions, quantiles_tau), hidden





class FullQuantileFunctionRainbow(ImplicitQuantileNetwork):
    """Full(y parameterized) Quantile Function with Noisy Networks and Dueling option.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param num_cosines: the number of cosines to use for cosine embedding.
        Default to 64.
    :param preprocess_net_output_dim: the output dimension of
        preprocess_net.
    :param noisy_std: standard deviation for NoisyLinear layers. Default to 0.5.
    :param is_noisy: whether to use noisy layers. Default to True.

    .. note::

        The first return value is a tuple of (quantiles, fractions, quantiles_tau),
        where fractions is a Batch(taus, tau_hats, entropies).
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: TActionShape,
        hidden_sizes: Sequence[int] = (),
        num_cosines: int = 64,
        preprocess_net_output_dim: int | None = None,
        device: str | int | torch.device = "cpu",
        noisy_std: float = 0.5,
        is_noisy: bool = True,
        is_dueling : bool = True
    ) -> None:
        super().__init__(
            preprocess_net,
            action_shape,
            hidden_sizes,
            num_cosines,
            preprocess_net_output_dim,
            device,
        )

        if preprocess_net_output_dim is None:
            raise ValueError("preprocess_net_output_dim must be specified and not None.")
        
        # print(f"preprocess_net_output_dim: {preprocess_net_output_dim}")
        # print(f"hidden_sizes: {hidden_sizes}")

        self.action_shape = action_shape
        self.noisy_std = noisy_std
        self.is_noisy = is_noisy
        self.is_dueling = is_dueling

        print(action_shape,noisy_std)
        print(preprocess_net_output_dim)

        def linear(x: int, y: int) -> nn.Module:
            if self.is_noisy:
                return NoisyLinear(x, y, self.noisy_std)
            return nn.Linear(x, y)

        # Define the advantage network

        self.advantage_net = nn.Sequential(
            linear(preprocess_net_output_dim, 512),
            nn.ReLU(inplace=True),
            linear(512, self.action_shape)
        )

        # print("Advantage net", self.advantage_net)
        

         # Define the value network for dueling architecture
        if self.is_dueling:
            self.value_net = nn.Sequential(
                linear(preprocess_net_output_dim, 512),
                nn.ReLU(inplace=True),
                linear(512, 1) # Output dimension is 1 for the value function
            )
            print("Dueling is True")


        # print("The value net", self.value_net)

        # if self.is_noisy:
        #     self.last = nn.Sequential(
        #     NoisyLinear(3136, 512),
        #     nn.ReLU(inplace=True),
        #     NoisyLinear(512, action_shape)
        #                             )

        # print(self.last)

        # self.embed_model = nn.Linear(num_cosines, preprocess_net_output_dim)

    def _compute_quantiles(self, obs: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        batch_size, sample_size = taus.shape
        embedding = (obs.unsqueeze(1) * self.embed_model(taus)).view(batch_size * sample_size, -1)

        # Compute advantages
        advantage = self.advantage_net(embedding).view(batch_size, sample_size, -1).transpose(1, 2)

        if self.is_dueling:
            # Compute value
            value = self.value_net(embedding).view(batch_size, sample_size, 1).transpose(1, 2)
            # Combine value and advantage to compute quantiles
            quantiles = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            quantiles = advantage
        
        return quantiles

        # return self.last(embedding).view(batch_size, sample_size, -1).transpose(1, 2)




    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        propose_model: FractionProposalNetwork,
        fractions: Batch | None = None,
        **kwargs: Any,
    ) -> tuple[Any, torch.Tensor]:
        r"""Mapping: s -> Q(s, \*)."""
        logits, hidden = self.preprocess(obs, state=kwargs.get("state", None))
        # Propose fractions
        if fractions is None:
            taus, tau_hats, entropies = propose_model(logits.detach())
            fractions = Batch(taus=taus, tau_hats=tau_hats, entropies=entropies)
        else:
            taus, tau_hats = fractions.taus, fractions.tau_hats
        quantiles = self._compute_quantiles(logits, tau_hats)
        # Calculate quantiles_tau for computing fraction grad
        quantiles_tau = None
        if self.training:
            with torch.no_grad():
                quantiles_tau = self._compute_quantiles(logits, taus[:, 1:-1])
        return (quantiles, fractions, quantiles_tau), hidden

class NoisyLinear(nn.Module):
    """Implementation of Noisy Networks. arXiv:1706.10295.

    :param in_features: the number of input features.
    :param out_features: the number of output features.
    :param noisy_std: initial standard deviation of noisy linear layers.

    .. note::

        Adapted from https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master
        /fqf_iqn_qrdqn/network.py .
    """

class NoisyLinear(nn.Module):
    """Implementation of Noisy Networks. arXiv:1706.10295.

    :param in_features: the number of input features.
    :param out_features: the number of output features.
    :param noisy_std: initial standard deviation of noisy linear layers.

    .. note::

        Adapted from https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master
        /fqf_iqn_qrdqn/network.py .
    """

    def __init__(self, in_features: int, out_features: int, noisy_std: float = 0.5) -> None:
        super().__init__()

        # Learnable parameters.
        self.mu_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.sigma_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.FloatTensor(out_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features))

        # Factorized noise parameters.
        self.register_buffer("eps_p", torch.FloatTensor(in_features))
        self.register_buffer("eps_q", torch.FloatTensor(out_features))

        self.in_features = in_features
        self.out_features = out_features
        self.sigma = noisy_std

        self.reset()
        self.sample()

    def reset(self) -> None:
        bound = 1 / np.sqrt(self.in_features)
        self.mu_W.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_W.data.fill_(self.sigma / np.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma / np.sqrt(self.in_features))

    def f(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.randn(x.size(0), device=x.device)
        return x.sign().mul_(x.abs().sqrt_())

    # TODO: rename or change functionality? Usually sample is not an inplace operation...
    def sample(self) -> None:
        self.eps_p.copy_(self.f(self.eps_p))
        self.eps_q.copy_(self.f(self.eps_q))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.mu_W + self.sigma_W * (self.eps_q.ger(self.eps_p))
            bias = self.mu_bias + self.sigma_bias * self.eps_q.clone()
        else:
            weight = self.mu_W
            bias = self.mu_bias

        return F.linear(x, weight, bias)


class IntrinsicCuriosityModule(nn.Module):
    """Implementation of Intrinsic Curiosity Module. arXiv:1705.05363.

    :param feature_net: a self-defined feature_net which output a
        flattened hidden state.
    :param feature_dim: input dimension of the feature net.
    :param action_dim: dimension of the action space.
    :param hidden_sizes: hidden layer sizes for forward and inverse models.
    :param device: device for the module.
    """

    def __init__(
        self,
        feature_net: nn.Module,
        feature_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (),
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.feature_net = feature_net
        self.forward_model = MLP(
            feature_dim + action_dim,
            output_dim=feature_dim,
            hidden_sizes=hidden_sizes,
            device=device,
        )
        self.inverse_model = MLP(
            feature_dim * 2,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            device=device,
        )
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.device = device

    def forward(
        self,
        s1: np.ndarray | torch.Tensor,
        act: np.ndarray | torch.Tensor,
        s2: np.ndarray | torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Mapping: s1, act, s2 -> mse_loss, act_hat."""
        s1 = to_torch(s1, dtype=torch.float32, device=self.device)
        s2 = to_torch(s2, dtype=torch.float32, device=self.device)
        phi1, phi2 = self.feature_net(s1), self.feature_net(s2)
        act = to_torch(act, dtype=torch.long, device=self.device)
        phi2_hat = self.forward_model(
            torch.cat([phi1, F.one_hot(act, num_classes=self.action_dim)], dim=1),
        )
        mse_loss = 0.5 * F.mse_loss(phi2_hat, phi2, reduction="none").sum(1)
        act_hat = self.inverse_model(torch.cat([phi1, phi2], dim=1))
        return mse_loss, act_hat
