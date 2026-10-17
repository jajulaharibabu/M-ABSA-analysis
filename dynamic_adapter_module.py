import torch
import torch.nn as nn


class DynamicAdapter(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_adapters=4,
        bottleneck_size=64
    ):
        super().__init__()

        self.num_adapters = num_adapters

        self.adapters = nn.ModuleList()

        for _ in range(num_adapters):
            adapter = nn.Sequential(
                nn.Linear(
                    hidden_size,
                    bottleneck_size
                ),
                nn.ReLU(),
                nn.Linear(
                    bottleneck_size,
                    hidden_size
                )
            )

            self.adapters.append(adapter)

        self.gating_network = nn.Linear(
            hidden_size,
            num_adapters
        )

    def forward(self, hidden_states):
        pooled = hidden_states[:, 0]

        gate_logits = self.gating_network(
            pooled
        )

        gate_weights = torch.softmax(
            gate_logits,
            dim=-1
        )

        adapter_outputs = []

        for adapter in self.adapters:
            output = adapter(
                hidden_states
            )

            adapter_outputs.append(output)

        stacked_outputs = torch.stack(
            adapter_outputs,
            dim=1
        )

        weights = gate_weights.view(
            -1,
            self.num_adapters,
            1,
            1
        )

        fused = (
            stacked_outputs * weights
        ).sum(dim=1)

        return hidden_states + fused