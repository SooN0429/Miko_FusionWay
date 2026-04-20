import torch
from torch import nn
from copy import deepcopy, copy
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import optimize


IGNORE_CFG = {"ignore": None}
IN_CFG = {"in": None}


def get_resnet_perm(
    block=2,
    num_blocks=[3, 3, 3],
    shortcut_name="shortcut",
    fc_name="linear",
    pool_name="avgpool",
    res_start_layer=1,
    ignore_running_val=True,
):
    """
    Temporary perm graph for ResNet-like models.
    You can replace this function with transfernet base-network perm design later.
    """
    running_cfg = IGNORE_CFG if ignore_running_val else {}
    perm = {}
    cur_perm = 0
    perm[cur_perm] = [
        [0, "conv1.weight"],
        [0, "bn1.weight"],
        [0, "bn1.bias"],
        [0, "bn1.running_mean", running_cfg],
        [0, "bn1.running_var", running_cfg],
    ]

    res_perm = cur_perm
    cur_perm += 1
    for l, block_num in enumerate(num_blocks):
        layer_id = l + 1
        for b in range(block_num):
            for c in range(1, block + 1):
                if l >= res_start_layer and b == 0 and c == 1:
                    perm[res_perm].extend([
                        [1, f"layer{layer_id}.{b}.conv{c}.weight", IN_CFG],
                        [1, f"layer{layer_id}.{b}.{shortcut_name}.0.weight", IN_CFG],
                    ])
                elif c == 1:
                    perm[res_perm].extend([
                        [1, f"layer{layer_id}.{b}.conv{c}.weight", IN_CFG],
                    ])
                else:
                    perm[cur_perm].extend([
                        [1, f"layer{layer_id}.{b}.conv{c}.weight", IN_CFG],
                    ])
                    cur_perm += 1

                t = [
                    [0, f"layer{layer_id}.{b}.conv{c}.weight"],
                    [0, f"layer{layer_id}.{b}.bn{c}.weight"],
                    [0, f"layer{layer_id}.{b}.bn{c}.bias"],
                    [0, f"layer{layer_id}.{b}.bn{c}.running_mean", running_cfg],
                    [0, f"layer{layer_id}.{b}.bn{c}.running_var", running_cfg],
                ]
                if c < block:
                    perm[cur_perm] = t
                else:
                    if b == 0 and l >= res_start_layer:
                        res_perm = cur_perm
                        cur_perm += 1
                        perm[res_perm] = [
                            [0, f"layer{layer_id}.{b}.{shortcut_name}.0.weight"],
                            [0, f"layer{layer_id}.{b}.{shortcut_name}.1.weight"],
                            [0, f"layer{layer_id}.{b}.{shortcut_name}.1.bias"],
                            [
                                0,
                                f"layer{layer_id}.{b}.{shortcut_name}.1.running_mean",
                                running_cfg,
                            ],
                            [
                                0,
                                f"layer{layer_id}.{b}.{shortcut_name}.1.running_var",
                                running_cfg,
                            ],
                        ]
                    perm[res_perm].extend(t)

    perm[res_perm].extend([
        [1, f"{fc_name}.weight", {"in": pool_name}],
    ])
    return perm


def refine_perm(perm):
    weight_cfg = {}
    old_perm = perm
    perm = {}
    in_perm = {}
    for p, o in old_perm.items():
        if isinstance(o, list):
            o = {"weights": o}
        for wl in o["weights"]:
            if len(wl) < 3:
                wl.append({})
            axis, w, cfg = wl
            if w not in weight_cfg:
                weight_cfg[w] = []
            weight_cfg[w].append([axis, p, cfg])
            if "in" in cfg:
                if w not in in_perm:
                    in_perm[w] = []
                in_perm[w].append(p)
        perm[p] = o

    for p, o in perm.items():
        if "in" not in o:
            o["in"] = []
        for _, w, cfg in o["weights"]:
            if "in" not in cfg and w in in_perm:
                o["in"].extend(in_perm[w])
        o["in"] = list(set(o["in"]))
    return perm, weight_cfg


def remove_col(x, idx, temp=None):
    if temp is None:
        return torch.cat([x[:, :idx], x[:, idx + 1 :]], dim=-1)
    r, c = x.shape
    temp = temp[:r, :c]
    _, l = x[:, idx + 1 :].shape
    temp[:, :l] = x[:, idx + 1 :]
    x[:, idx : idx + l] = temp[:, :l]
    return x[:, : c - 1]


class CovarianceMetric:
    def __init__(self):
        self.std = None
        self.mean = None
        self.outer = None
        self.numel = 0

    def update(self, *feats):
        batch_size, feature_dim = feats[0].shape[0], feats[0].shape[1]
        self.numel += batch_size
        feats = [torch.transpose(f, 0, 1).reshape(feature_dim, -1) for f in feats]
        feats = torch.cat(feats, dim=0)
        feats = torch.nan_to_num(feats, 0, 0, 0)

        std = feats.std(dim=1)
        mean = feats.mean(dim=1)
        outer = (feats @ feats.T) / feats.shape[1]

        if self.mean is None:
            self.mean = torch.zeros_like(mean)
        if self.outer is None:
            self.outer = torch.zeros_like(outer)
        if self.std is None:
            self.std = torch.zeros_like(std)

        self.mean += mean * batch_size
        self.outer += outer * batch_size
        self.std += std * batch_size

    def finalize(self, eps=1e-7):
        self.outer /= self.numel
        self.mean /= self.numel
        cov = self.outer - torch.outer(self.mean, self.mean)
        std = torch.diagonal(cov).sqrt()
        cov = cov / (torch.clamp(torch.outer(std, std), min=eps))
        return cov


class WeightFusion:
    def __init__(
        self,
        reduce=0.5,
        a=0.5,
        b=1,
        iter=100,
        use_cos=False,
        no_fusion=False,
        fix_sims=False,
        fix_rate=0.5,
        use_permute=False,
        verbose=True,
    ):
        self.reduce = reduce
        self.a = a
        self.b = b
        self.iter = iter
        self.fix_sims = fix_sims
        self.sims_dict = None
        self.fix_rate = fix_rate
        self.verbose = verbose
        self.compute_correlation = self._cossim if use_cos else self._correlation

        self.fusion_weight = self.default_fusion_weight
        if use_permute:
            self.fusion_weight = self.permute_weight
        if no_fusion:
            self.fusion_weight = self.no_fusion_weight

    def transform(
        self,
        nets,
        perm,
        act_loader=None,
        in_weight_space=False,
        return_state_dict=False,
        random_state=0,
    ):
        self.fusion_model: nn.Module = deepcopy(nets[0])
        self.perm, self.weight_cfg = refine_perm(deepcopy(perm))
        self.perm_names = list(perm.keys())
        self.perm_mats = {}
        self.params = [{k: v for k, v in net.state_dict().items()} for net in nets]

        if act_loader is not None:
            self.act_transform(nets, act_loader)
        in_weight_space = in_weight_space or act_loader is None
        if in_weight_space:
            self.iter_transform(random_state=random_state)

        merged_state_dict = self.get_merged_state_dict()
        self.network_adapt_by_state_dict(merged_state_dict, self.fusion_model)
        self.fusion_model.load_state_dict(merged_state_dict)
        if return_state_dict:
            return self.get_merged_state_dict(no_avg=True)
        return self.fusion_model

    def gen_act_sims(self, nets, act_loader: DataLoader):
        sims_dict = {k: CovarianceMetric() for k in self.perm}
        device = list(nets[0].parameters())[0].device
        hooks = []
        feats = [{} for _ in range(len(nets))]

        def add_hooks(net: nn.Module, idx):
            modules = {k: v for k, v in net.named_modules()}
            for k, o in self.perm.items():
                if "act_modules" not in o:
                    continue
                for m_idx, act_module in enumerate(o["act_modules"]):
                    act_name = act_module["name"]
                    act_dim = act_module.get("dim", 1)
                    hook_type = act_module.get("hook", "pre")

                    if k not in feats[idx]:
                        feats[idx][k] = {}

                    def prehook_gen(perm_name, mindex, dim):
                        def prehook(_, x):
                            v = x[0].detach()
                            if dim != 1:
                                v = torch.moveaxis(v, dim, 1)
                            feats[idx][perm_name][mindex] = v
                            return None

                        return prehook

                    def posthook_gen(perm_name, mindex, dim):
                        def posthook(_, __, x):
                            v = x.detach()
                            if dim != 1:
                                v = torch.moveaxis(v, dim, 1)
                            feats[idx][perm_name][mindex] = v
                            return None

                        return posthook

                    module = modules[act_name]
                    if hook_type == "pre":
                        hooks.append(module.register_forward_pre_hook(prehook_gen(k, m_idx, act_dim)))
                    else:
                        hooks.append(module.register_forward_hook(posthook_gen(k, m_idx, act_dim)))

        for i, net in enumerate(nets):
            net.eval().cuda()
            add_hooks(net, i)
        with torch.no_grad():
            for img, _ in tqdm(act_loader, desc="Computing activation"):
                img = img.cuda()
                for net in nets:
                    net(img)
                for k, s in sims_dict.items():
                    if k not in feats[0]:
                        continue
                    fs = [torch.stack([v.float() for v in f[k].values()], dim=-1) for f in feats]
                    s.update(*fs)

        for h in hooks:
            h.remove()
        for net in nets:
            net.to(device)

        return {k: v.finalize() for k, v in sims_dict.items() if v.numel > 0}

    def act_transform(self, nets, act_loader: DataLoader):
        sims_dict = self.gen_act_sims(nets, act_loader)
        if self.fix_sims:
            self.sims_dict = sims_dict
        for k, sims in tqdm(sims_dict.items(), desc="Computing permutation", total=len(sims_dict)):
            merge, unmerge, _ = self.fusion_weight(
                sims, r=self.reduce, a=self.a, b=self.b, get_merge_value=True
            )
            merge = merge * len(self.params)
            self.perm_mats[k] = (merge, unmerge)

    def iter_transform(self, random_state=0, tol=5):
        no_progress_count = 0
        generator = torch.Generator()
        generator.manual_seed(random_state)
        perm_state = {p: 0 for p in self.perm_names}
        total_old = 0
        self.best_perm_mats = None

        for iter_i in range(self.iter):
            for p_ix in torch.randperm(len(self.perm_names), generator=generator):
                p = self.perm_names[p_ix]
                wvs = self.get_weight_vectors(p)
                sims = self.compute_correlation(wvs)
                if self.fix_sims and self.sims_dict is not None and p in self.sims_dict:
                    sims = sims * self.fix_rate + self.sims_dict[p] * (1 - self.fix_rate)

                merge, unmerge, merge_value = self.fusion_weight(
                    sims, r=self.reduce, a=self.a, b=self.b, get_merge_value=True
                )
                merge = merge * len(self.params)
                perm_state[p] = merge_value
                self.perm_mats[p] = (merge, unmerge)

            total_new = sum(v for v in perm_state.values()) / len(perm_state)
            if self.verbose:
                print("iter", iter_i, "no_progress", no_progress_count, "score", total_new)

            if total_old >= total_new and self.best_perm_mats is not None:
                no_progress_count += 1
                if no_progress_count >= tol:
                    break
                self.perm_mats = copy(self.best_perm_mats)
            else:
                no_progress_count = 0
                self.best_perm_mats = copy(self.perm_mats)
                total_old = total_new

        if self.best_perm_mats is not None:
            self.perm_mats = self.best_perm_mats
        self.best_perm_mats = None

    def perm_(self, axis, ws, perm_mat):
        ws_perm = []
        perm_mat = perm_mat.chunk(len(self.params), dim=0)
        for i, w in enumerate(ws):
            w = torch.transpose(w, axis, 0)
            shape = list(w.shape)
            merge_mat = perm_mat[i].T
            raw_dim = shape[0]
            shape[0] = merge_mat.shape[0]
            w = torch.matmul(merge_mat, w.reshape(raw_dim, -1)).reshape(*shape)
            ws_perm.append(torch.transpose(w, 0, axis))
        return ws_perm

    def get_weight_vectors(self, p):
        o = self.perm[p]
        weight_vectors = [[] for _ in range(len(self.params))]
        for axis, w, cfg in o["weights"]:
            if "ignore" in cfg:
                continue
            ws = [param[w] for param in self.params]
            for a_, p_, cfg_ in self.weight_cfg[w]:
                if axis == a_:
                    continue
                if p_ in self.perm_mats:
                    select = 0 if "in" in cfg_ else 1
                    perm_mat = self.perm_mats[p_][select]
                    ws = self.perm_(a_, ws, perm_mat)
            for i, w_ in enumerate(ws):
                n = w_.shape[axis]
                weight_vectors[i].append(torch.transpose(w_, axis, 0).reshape(n, -1))
        return [torch.concat(wv, dim=1) for wv in weight_vectors]

    def _correlation(self, feats):
        feats = torch.cat(feats, dim=0)
        feats = torch.nan_to_num(feats, 0, 0, 0)
        std = feats.std(dim=1)
        mean = feats.mean(dim=1)
        outer = (feats @ feats.T) / feats.shape[1]
        cov = outer - torch.outer(mean, mean)
        std = torch.diagonal(cov).sqrt()
        return cov / (torch.clamp(torch.outer(std, std), min=1e-7))

    def _cossim(self, feats):
        feats = torch.cat(feats, dim=0)
        feats_norm = torch.norm(feats, dim=1, keepdim=True)
        feats = feats / feats_norm.clamp_min(1e-8)
        return feats @ feats.T

    def permute_weight(self, sims, r=0.5, get_merge_value=False, **kwargs):
        correlation = sims
        o = correlation.shape[0]
        n = len(self.params)
        om = o // n
        device = correlation.device

        mats = [torch.eye(om, device=device)]
        merge_value = []
        for i in range(1, n):
            row_ind, col_ind = optimize.linear_sum_assignment(
                correlation[:om, om * i : om * (i + 1)].cpu().numpy(),
                maximize=True,
            )
            mats.append(torch.eye(om, device=device)[torch.tensor(col_ind).long().to(device)].T)
            merge_value.append(
                correlation[:om, om * i : om * (i + 1)]
                .cpu()
                .numpy()[row_ind, col_ind]
                .mean()
                .item()
            )

        unmerge = torch.cat(mats, dim=0)
        merge = torch.cat(mats, dim=0)
        merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)
        if get_merge_value:
            return merge, unmerge, sum(merge_value) / max(len(merge_value), 1e-7)
        return merge, unmerge

    def default_fusion_weight(self, sims, r=0.5, a=0, b=1, get_merge_value=False):
        sims = torch.clone(sims)
        o = sims.shape[0]
        remainder = int(o * (1 - r) + 1e-4)
        permutation_matrix = torch.eye(o, o)

        torch.diagonal(sims)[:] = -torch.inf
        num_models = len(self.params)
        om = o // num_models
        original_model = torch.zeros(o, device=sims.device).long()
        for i in range(num_models):
            original_model[i * om : (i + 1) * om] = i

        to_remove = permutation_matrix.shape[1] - remainder
        budget = torch.zeros(num_models, device=sims.device).long() + int((to_remove // num_models) * b + 1e-4)
        merge_value = []

        while permutation_matrix.shape[1] > remainder:
            best_idx = sims.reshape(-1).argmax().item()
            row_idx = best_idx % sims.shape[1]
            col_idx = best_idx // sims.shape[1]
            merge_value.append(sims[row_idx, col_idx].item())
            if col_idx < row_idx:
                row_idx, col_idx = col_idx, row_idx

            row_origin = original_model[row_idx]
            col_origin = original_model[col_idx]
            permutation_matrix[:, row_idx] += permutation_matrix[:, col_idx]
            permutation_matrix = remove_col(permutation_matrix, col_idx)
            sims[:, row_idx] = torch.minimum(sims[:, row_idx], sims[:, col_idx])

            if a <= 0:
                sims[row_origin * om : (row_origin + 1) * om, row_idx] = -torch.inf
                sims[col_origin * om : (col_origin + 1) * om, row_idx] = -torch.inf
            else:
                sims[:, row_idx] *= a
            sims = remove_col(sims, col_idx)

            sims[row_idx, :] = torch.minimum(sims[row_idx, :], sims[col_idx, :])
            if a <= 0:
                sims[row_idx, row_origin * om : (row_origin + 1) * om] = -torch.inf
                sims[row_idx, col_origin * om : (col_origin + 1) * om] = -torch.inf
            else:
                sims[row_idx, :] *= a
            sims = remove_col(sims.T, col_idx).T

            row_origin, col_origin = original_model[row_idx], original_model[col_idx]
            original_model = remove_col(original_model[None, :], col_idx)[0]
            if row_origin == col_origin:
                origin = original_model[row_idx].item()
                budget[origin] -= 1
                if budget[origin] <= 0:
                    selector = original_model == origin
                    sims[selector[:, None] & selector[None, :]] = -torch.inf

        unmerge = permutation_matrix
        merge = permutation_matrix / (permutation_matrix.sum(dim=0, keepdim=True) + 1e-5)
        merge = merge.to(sims.device)
        unmerge = unmerge.to(sims.device)
        if get_merge_value:
            return merge, unmerge, sum(merge_value) / max(len(merge_value), 1e-7)
        return merge, unmerge

    def no_fusion_weight(self, sims, r=0.5, a=0, b=1, get_merge_value=False, **kwargs):
        o = sims.shape[0]
        om = o // len(self.params)
        permutation_matrix = torch.concat([torch.eye(om) for _ in range(len(self.params))], dim=0)
        unmerge = permutation_matrix
        merge = permutation_matrix / (permutation_matrix.sum(dim=0, keepdim=True) + 1e-5)
        merge = merge.to(sims.device)
        unmerge = unmerge.to(sims.device)
        if get_merge_value:
            return merge, unmerge, 1.0
        return merge, unmerge

    @staticmethod
    @torch.no_grad()
    def network_adapt_by_state_dict(state_dict, net: nn.Module):
        modules = {k: v for k, v in net.named_modules()}
        for wk, v in state_dict.items():
            wk_path = wk.split(".")
            module_name = ".".join(wk_path[:-1])
            param_name = wk_path[-1]
            m = modules[module_name]
            w: nn.Parameter = getattr(m, param_name)

            new_w = w if w.shape == v.shape else torch.empty(v.shape).to(w)
            w.set_(new_w)
            if param_name == "weight":
                if isinstance(m, nn.Linear):
                    m.out_features, m.in_features = new_w.shape[0], new_w.shape[1]
                elif isinstance(m, nn.Conv2d):
                    m.out_channels, m.in_channels = new_w.shape[0], new_w.shape[1]
                elif isinstance(m, nn.BatchNorm2d):
                    m.num_features = new_w.shape[0]

    def get_merged_state_dict(self, no_avg=False):
        merged_dict = {}
        rest_keys = set(self.params[0].keys())
        for wk, v in self.weight_cfg.items():
            rest_keys.remove(wk)
            ws = [param[wk] for param in self.params]
            for a, p, cfg in v:
                select = 1 if "in" in cfg else 0
                perm_mat = self.perm_mats[p][select]
                ws = self.perm_(a, ws, perm_mat)
            merged_dict[wk] = ws if no_avg else sum(w for w in ws) / len(ws)

        for wk in rest_keys:
            ws = [param[wk] for param in self.params]
            merged_dict[wk] = sum(w for w in ws) / len(ws)
        return merged_dict


# ===== 4-step helper API =====
def select_perm_graph_for_resnet20():
    return get_resnet_perm()


def build_fusion_engine(**fusion_kwargs):
    return WeightFusion(**fusion_kwargs)


def compute_alignment_mats(
    fusion_engine: WeightFusion,
    nets,
    perm,
    act_loader=None,
    in_weight_space=False,
    random_state=0,
):
    fusion_engine.transform(
        nets,
        perm,
        act_loader=act_loader,
        in_weight_space=in_weight_space,
        return_state_dict=False,
        random_state=random_state,
    )
    return fusion_engine.perm_mats


def apply_alignment_and_merge(
    fusion_engine: WeightFusion,
    nets,
    perm,
    act_loader=None,
    in_weight_space=False,
    random_state=0,
):
    return fusion_engine.transform(
        nets,
        perm,
        act_loader=act_loader,
        in_weight_space=in_weight_space,
        return_state_dict=False,
        random_state=random_state,
    )
