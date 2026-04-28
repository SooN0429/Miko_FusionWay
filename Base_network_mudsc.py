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
    # 這裡的寫法是為了forward時讓參數需求與base network相同而寫的
    @staticmethod
    def _unpack_act_batch(batch):
        """
        Support common batch formats:
        - (x, y)
        - (x, y, test_flag)
        - {"x": x, "test_flag": bool}
        """
        test_flag = None
        if isinstance(batch, dict):
            x = batch.get("x", batch.get("img", batch.get("image")))
            test_flag = batch.get("test_flag", None)
            if x is None:
                raise ValueError("Dict batch must include one of keys: x/img/image")
            return x, test_flag

        if isinstance(batch, (tuple, list)):
            if len(batch) == 0:
                raise ValueError("Empty batch is not supported")
            x = batch[0]
            if len(batch) >= 3:
                test_flag = batch[2]
            return x, test_flag

        return batch, None

    @staticmethod
    def _normalize_test_flag(test_flag):
        if test_flag is None:
            return None
        if isinstance(test_flag, torch.Tensor):
            if test_flag.numel() == 1:
                return bool(test_flag.item())
            # If a batch of same flag values is passed, use first one.
            return bool(test_flag.reshape(-1)[0].item())
        return bool(test_flag)

    @staticmethod
    def _forward_with_optional_flag(net, x, test_flag):
        test_flag = WeightFusion._normalize_test_flag(test_flag)
        if test_flag is None:
            try:
                return net(x)
            except TypeError:
                # For models like resnet18_multi.forward(x, test_flag)
                return net(x, True)
        try:
            return net(x, test_flag)
        except TypeError:
            return net(x)
    # 到這裡
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
            for batch in tqdm(act_loader, desc="Computing activation"):
                img, test_flag = self._unpack_act_batch(batch)
                img = img.cuda()
                for net in nets:
                    self._forward_with_optional_flag(net, img, test_flag)
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


def get_resnet18_multi_base_perm_by_extracted_layer(extracted_layer):
    """
    Build perm graph for resnet18_multi backbone only (without conv_M2 and later heads).

    Supported extracted_layer:
    - "6_point": up to layer3.0
    - "7_point": up to layer3.1
    - "8_point": up to layer4.0
    """
    # resnet18_multi backbone uses BasicBlock x [2, 2, 2, 2] and downsample naming.
    perm = get_resnet_perm(
        block=2,
        num_blocks=[2, 2, 2, 2],
        shortcut_name="downsample",
        fc_name="linear",
        res_start_layer=1,
    )

    if extracted_layer == "6_point":
        max_layer, max_block = 3, 0
    elif extracted_layer == "7_point":
        max_layer, max_block = 3, 1
    elif extracted_layer == "8_point":
        max_layer, max_block = 4, 0
    else:
        raise ValueError(
            f"Unsupported extracted_layer: {extracted_layer}. "
            "Expected one of ['6_point', '7_point', '8_point']."
        )

    def _is_backbone_weight_in_scope(weight_name):
        # Remove classifier head linkage from generic ResNet perm template.
        if weight_name == "linear.weight" or weight_name.startswith("linear."):
            return False
        if weight_name.startswith("conv1.") or weight_name.startswith("bn1."):
            return True
        if not weight_name.startswith("layer"):
            return False

        parts = weight_name.split(".")
        # Example: layer3.0.conv1.weight / layer4.0.downsample.0.weight
        try:
            layer_id = int(parts[0].replace("layer", ""))
            block_id = int(parts[1])
        except (IndexError, ValueError):
            return False

        if layer_id < max_layer:
            return True
        if layer_id == max_layer and block_id <= max_block:
            return True
        return False

    filtered_perm = {}
    for _, weights in perm.items():
        kept = []
        for wl in weights:
            axis, weight_name = wl[0], wl[1]
            cfg = wl[2] if len(wl) > 2 else {}
            if _is_backbone_weight_in_scope(weight_name):
                kept.append([axis, weight_name, cfg] if len(wl) > 2 else [axis, weight_name])
        if kept:
            filtered_perm[len(filtered_perm)] = kept
    return filtered_perm


def get_convm2_perm(prefix="convm2_layer.0", ignore_running_val=True, include_final_residual=False):
    """
    為 resnet18_multi（backbone_multi.py）中的 conv_M2 建立專屬 perm graph。
    設計原則：
    - 預設採 branch-local 對齊（must-not-mix segments），避免不同 branch 任意混排。
    - 只在明確的分支交會點做耦合：
    * branch1_2_2 + branch2_2_2（第一次相加）
    * branch1_3_2 + branch2_3_2（第二次相加）
    - 不把 cat(128/128/64) 與 residual(320) 強制放進同一個可自由混排的節點。
    若需要額外對齊 residual 路徑，可設 include_final_residual=True。
    重要說明（目前版本範圍）：
    - 本函式會建立「最終輸出座標節點」，把 cat 三個分支輸出與 x_residual 輸出綁到同一約束。
    - Add 層本身沒有參數，實作上是對齊 Add 兩側來源權重的通道座標。
    對應 conv_M2.forward：
    - cat 輸出：x = torch.cat([x_branch1, x_branch2, x_branch3], dim=1)，通道為 128 + 128 + 64。
    - x_residual：x_residual = self.res(x_in)，其中 x_in 是進入三個 branch 前的輸入特徵。
    - 最終輸出：x + x_residual。
    因此，目前 perm 設計會約束：
    - 分支內部與中間兩次相加的一致性；
    - 最終 cat + x_residual 的座標一致性；
    - 並將最終座標串接到 linear_test.conv 的輸入軸。
    """
    running_cfg = IGNORE_CFG if ignore_running_val else {}
    perm = {}
    cur_perm = 0

    # P0：共享輸入座標 x -> (res, branch1_1, branch2_1, branch3_1)
    perm[cur_perm] = [
        [1, f"{prefix}.res.weight", IN_CFG],
        [1, f"{prefix}.branch1_1.conv.weight", IN_CFG],
        [1, f"{prefix}.branch2_1.conv.weight", IN_CFG],
        [1, f"{prefix}.branch3_1.conv.weight", IN_CFG],
    ]
    cur_perm += 1

    # P1：branch1 前段 (96)：branch1_1 -> branch1_2_1(depthwise) -> branch1_2_2(pointwise in)
    perm[cur_perm] = [
        [0, f"{prefix}.branch1_1.conv.weight"],
        [0, f"{prefix}.branch1_1.bn.weight"],
        [0, f"{prefix}.branch1_1.bn.bias"],
        [0, f"{prefix}.branch1_1.bn.running_mean", running_cfg],
        [0, f"{prefix}.branch1_1.bn.running_var", running_cfg],
        [0, f"{prefix}.branch1_2_1.depthwise.weight"],
        [1, f"{prefix}.branch1_2_2.pointwise.weight", IN_CFG],
    ]
    cur_perm += 1

    # P2：branch2 前段 (96)：branch2_1 -> branch2_2_1(depthwise) -> branch2_2_2(pointwise in)
    perm[cur_perm] = [
        [0, f"{prefix}.branch2_1.conv.weight"],
        [0, f"{prefix}.branch2_1.bn.weight"],
        [0, f"{prefix}.branch2_1.bn.bias"],
        [0, f"{prefix}.branch2_1.bn.running_mean", running_cfg],
        [0, f"{prefix}.branch2_1.bn.running_var", running_cfg],
        [0, f"{prefix}.branch2_2_1.depthwise.weight"],
        [1, f"{prefix}.branch2_2_2.pointwise.weight", IN_CFG],
    ]
    cur_perm += 1

    # P3：第一次相加座標 (128)：branch1_2_2 + branch2_2_2，並餵給後續兩條 stage-3 分支
    # perm只是提供相加時的用於座標對齊的依據，實際上進行相加的過程仍於模型內部進行。
    perm[cur_perm] = [
        [0, f"{prefix}.branch1_2_2.pointwise.weight"],
        [0, f"{prefix}.branch1_2_2.bn.weight"],
        [0, f"{prefix}.branch1_2_2.bn.bias"],
        [0, f"{prefix}.branch1_2_2.bn.running_mean", running_cfg],
        [0, f"{prefix}.branch1_2_2.bn.running_var", running_cfg],
        [0, f"{prefix}.branch2_2_2.pointwise.weight"],
        [0, f"{prefix}.branch2_2_2.bn.weight"],
        [0, f"{prefix}.branch2_2_2.bn.bias"],
        [0, f"{prefix}.branch2_2_2.bn.running_mean", running_cfg],
        [0, f"{prefix}.branch2_2_2.bn.running_var", running_cfg],
        [0, f"{prefix}.branch1_3_1.depthwise.weight", IN_CFG],
        [0, f"{prefix}.branch2_3_1.depthwise.weight", IN_CFG],
    ]
    cur_perm += 1

    # P4：第二次相加座標 (128)：branch1_3_2 + branch2_3_2 的分支末端輸出
    perm[cur_perm] = [
        [1, f"{prefix}.branch1_3_2.pointwise.weight", IN_CFG],
        [0, f"{prefix}.branch1_3_2.pointwise.weight"],
        [0, f"{prefix}.branch1_3_2.bn.weight"],
        [0, f"{prefix}.branch1_3_2.bn.bias"],
        [0, f"{prefix}.branch1_3_2.bn.running_mean", running_cfg],
        [0, f"{prefix}.branch1_3_2.bn.running_var", running_cfg],
        [1, f"{prefix}.branch2_3_2.pointwise.weight", IN_CFG],
        [0, f"{prefix}.branch2_3_2.pointwise.weight"],
        [0, f"{prefix}.branch2_3_2.bn.weight"],
        [0, f"{prefix}.branch2_3_2.bn.bias"],
        [0, f"{prefix}.branch2_3_2.bn.running_mean", running_cfg],
        [0, f"{prefix}.branch2_3_2.bn.running_var", running_cfg],
    ]
    cur_perm += 1

    # P5：branch3 主座標節點 (64)：僅保留 n=64 的權重軸。
    # ConvTranspose2d 於 groups>1 時，axis 1 對應 out_c/groups（本模型為 16），
    # 不能與 n=64 的項目放在同一節點，否則 get_weight_vectors 會在 concat 時維度衝突。
    perm[cur_perm] = [
        [0, f"{prefix}.branch3_1.conv.weight"],
        [0, f"{prefix}.branch3_1.bn.weight"],
        [0, f"{prefix}.branch3_1.bn.bias"],
        [0, f"{prefix}.branch3_1.bn.running_mean", running_cfg],
        [0, f"{prefix}.branch3_1.bn.running_var", running_cfg],
        [0, f"{prefix}.branch3_2.deconv.weight", IN_CFG],
        [0, f"{prefix}.branch3_2.bn.weight"],
        [0, f"{prefix}.branch3_2.bn.bias"],
        [0, f"{prefix}.branch3_2.bn.running_mean", running_cfg],
        [0, f"{prefix}.branch3_2.bn.running_var", running_cfg],
        [0, f"{prefix}.branch3_3.deconv.weight", IN_CFG],
        [0, f"{prefix}.branch3_3.bn.weight"],
        [0, f"{prefix}.branch3_3.bn.bias"],
        [0, f"{prefix}.branch3_3.bn.running_mean", running_cfg],
        [0, f"{prefix}.branch3_3.bn.running_var", running_cfg],
    ]

    # P5b：grouped deconv 的 group-內座標節點 (16)。
    # 專門放 ConvTranspose2d 的 axis=1（out_c/groups）以保持節點內 n 一致。
    cur_perm += 1
    perm[cur_perm] = [
        [1, f"{prefix}.branch3_2.deconv.weight"],
        [1, f"{prefix}.branch3_3.deconv.weight"],
    ]
    # 原說法：
            # P6：最終輸出座標，對應 (cat(x_branch1, x_branch2, x_branch3) + x_residual)
            # 將最終相加兩側的來源分支(x_branch1, x_branch2, x_branch3、x_residual)綁到同一座標系，並把該座標串接到 linear_test 的輸入。
            #cat+x_residual的部分，採與add支點相同概念 (由於進入支點的輸出已經是經過跨模型對齊後的結果了，並非是單模型訓練下的狀態，因此需要
            #在相加時進行輸出的通道對齊) 進行實作的話，就是將cat用到的輸入:x_branch1, x_branch2, x_branch3的輸出以及x_residual的輸出，
            #兩者的座標放入perm中約束。
            #"在單模型訓練時，Add 兩側會被共同訓練「自然協調」；
            #在跨模型融合時，進入支點的是「各自模型對齊後」的表示，不一定天然同座標，因此要在支點前施加通道對應約束。"
    # P6：最終輸出座標錨點，對應 conv_M2 最終 320 通道輸出 -> linear_test 輸入。
    # 注意：這裡僅保留「同一 320 座標系」的權重，避免把 branch1/2(128) 與
    # grouped deconv 分支權重軸(out/groups=16)混入同一節點，導致向量拼接維度衝突。
    cur_perm += 1
    perm[cur_perm] = [
        # 最終相加中的 residual 分支輸出（320）
        [0, f"{prefix}.res.weight"],
        # 下游 head（linear_test）輸入（320）
        [1, "linear_test.conv.weight", IN_CFG],
    ]

    return perm


def get_resnet18_multi_merged_perm_by_extracted_layer(
    extracted_layer,
    convm2_prefix="convm2_layer.0",
    ignore_running_val=True,
):
    """
    合併 resnet18_multi 的 base backbone perm 與 conv_M2 perm，回傳可直接給 WeightFusion.transform 的完整 perm。

    合併邏輯：
    1) 先建立 base perm（依 extracted_layer 裁切）。
    2) 再建立 conv_M2 perm。
    3) 重新編號兩者節點，避免 key 衝突。
    4) 補一個橋接節點：把 base 最後輸出座標與 conv_M2 的輸入座標綁在同一節點。
       - base 末端（ResNet18 backbone）使用最後 stage 的 block 末端 bn2 輸出作為錨點。
       - conv_M2 輸入側對應：res/branch1_1/branch2_1/branch3_1 的輸入軸（axis=1, IN_CFG）。
    """
    base_perm = get_resnet18_multi_base_perm_by_extracted_layer(extracted_layer)
    convm2_perm = get_convm2_perm(
        prefix=convm2_prefix,
        ignore_running_val=ignore_running_val,
    )

    merged_perm = {}
    # 1) 先放 base，重新連續編號
    for _, weights in base_perm.items():
        merged_perm[len(merged_perm)] = weights

    # 2) 補 base -> conv_M2 的橋接節點
    if extracted_layer == "6_point":
        # backbone 到 layer3.0 結尾
        base_anchor = "layer3.0.bn2"
    elif extracted_layer == "7_point":
        # backbone 到 layer3.1 結尾
        base_anchor = "layer3.1.bn2"
    elif extracted_layer == "8_point":
        # backbone 到 layer4.0 結尾
        base_anchor = "layer4.0.bn2"
    else:
        raise ValueError(
            f"Unsupported extracted_layer: {extracted_layer}. "
            "Expected one of ['6_point', '7_point', '8_point']."
        )

    merged_perm[len(merged_perm)] = [
        [0, f"{base_anchor}.weight"],
        [0, f"{base_anchor}.bias"],
        [0, f"{base_anchor}.running_mean", IGNORE_CFG if ignore_running_val else {}],
        [0, f"{base_anchor}.running_var", IGNORE_CFG if ignore_running_val else {}],
        [1, f"{convm2_prefix}.res.weight", IN_CFG],
        [1, f"{convm2_prefix}.branch1_1.conv.weight", IN_CFG],
        [1, f"{convm2_prefix}.branch2_1.conv.weight", IN_CFG],
        [1, f"{convm2_prefix}.branch3_1.conv.weight", IN_CFG],
    ]

    # 3) 再放 conv_M2，重新連續編號
    for _, weights in convm2_perm.items():
        merged_perm[len(merged_perm)] = weights

    return merged_perm


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
