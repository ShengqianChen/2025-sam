import torch

class SAM(torch.optim.Optimizer):
    
    # 初始化函数，接受终端传进来的命令行参数：基础优化器sgd，rho值（SAM关键参数），是否使用ASAM策略
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        # 检查rho是否为非负数
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # 初始化基础优化器SGD
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    # 第一步：执行梯度的更新（也就是计算sharpness-aware的梯度）
    @torch.no_grad()  
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            # 计算步长比例，rho / 梯度范数（加一个小常数防止除0）
            scale = group["rho"] / (grad_norm + 1e-12)

            # 对每个参数进行操作
            for p in group["params"]:
                if p.grad is None: continue
                # 保存当前参数数据，用于第二步恢复
                self.state[p]["old_p"] = p.data.clone()
                # 根据自适应的策略计算e(w)，并将其添加到参数上，爬升到局部最大值
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  

        if zero_grad: self.zero_grad()

    # 第二步：恢复参数并进行基础优化器的更新
    @torch.no_grad()  
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                # 恢复参数为初始值 w
                p.data = self.state[p]["old_p"]  
        # 调用基础优化器在原本的初始值位置根据之前跳过去的局部最大值的梯度方向更新参数
        self.base_optimizer.step() 

        if zero_grad: self.zero_grad()

    # 统一调用第一步和第二步的更新
    @torch.no_grad()  
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2 
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
