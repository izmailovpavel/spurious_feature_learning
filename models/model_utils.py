import torch

def _replace_fc(model, output_dim):
    d = model.fc.in_features
    model.fc = torch.nn.Linear(d, output_dim)
    return model


class MultiTaskHead(torch.nn.Module):
    def __init__(self, n_features, n_classes_list):
        super(MultiTaskHead, self).__init__()
        self.fc_list = [
            torch.nn.Linear(n_features, n_classes).cuda()
            for n_classes in n_classes_list
        ]

    def forward(self, x):
        outputs = []
        for head in self.fc_list:
            out = head(x)
            outputs.append(out)
        return outputs