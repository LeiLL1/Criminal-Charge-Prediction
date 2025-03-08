import torch
class PGD:
    def __init__(self, model,  eps=0.2, alpha=0.00009, restore_iters=3):
        self.model = model
        self.emb_name = 'embeddings'
        self.eps = eps
        self.alpha = alpha
        self.restore_iters = restore_iters
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=True):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def restore(self):
        for _ in range(self.restore_iters):
            for name, param in self.model.named_parameters():
                if param.requires_grad and self.emb_name in name:
                    assert name in self.emb_backup
                    param.data = self.emb_backup[name]

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.eps:
            r = self.eps * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]
