# mypy: disallow-untyped-defs
import torch


class FeedforwardNN(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(FeedforwardNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
