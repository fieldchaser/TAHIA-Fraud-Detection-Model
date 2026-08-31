import torch
import torch.nn as nn


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

class MLP(nn.Module):
    def __init__(self, in_dim, nb_classes):
        super(MLP, self).__init__()
        self.act = nn.ELU(1.5)
        self.dropout = torch.nn.Dropout(p=0.2, inplace=False)
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc3 = nn.Linear(128, nb_classes)

    def forward(self, input):
        out = self.dropout(self.act(self.fc1(input.float())))
        # out = self.dropout(self.act(self.fc2(out)))
        out = self.fc3(out)
        return out