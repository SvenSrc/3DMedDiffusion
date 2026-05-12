import torch


class LossAwareTimestepSampler:

    def __init__(self, num_timesteps, history_size=10):
        self.num_timesteps = num_timesteps
        self.history_size = history_size
        self.loss_history = torch.ones(num_timesteps, history_size)
        self.history_idx = 0
        self._weights = None

    def update_losses(self, timesteps, losses):
        timesteps = timesteps.cpu()
        losses = losses.detach().cpu()
        for t, loss in zip(timesteps, losses):
            self.loss_history[t, self.history_idx] = loss.item()
        self.history_idx = (self.history_idx + 1) % self.history_size
        self._weights = None   # invalidate cache; next get_weights() recomputes

    def get_weights(self):
        if self._weights is None:
            w = torch.sqrt(torch.mean(self.loss_history ** 2, dim=1))
            self._weights = w / w.sum()
        return self._weights

    def sample(self, batch_size, device):
        weights = self.get_weights()
        indices = torch.multinomial(weights, batch_size, replacement=True)
        importance = 1.0 / (self.num_timesteps * weights[indices])
        importance = importance / importance.mean()
        return indices.to(device), importance.to(device)
