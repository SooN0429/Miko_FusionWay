import argparse
import importlib
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import Base_network_mudsc as weight_fusion


DEFAULT_PREFIXES = [
    "base_network.",
    "module.base_network.",
    "model.base_network.",
    "module.",
    "model.",
    "net.",
]

'''
執行範例：
python Base_network_mudsc_concept_merging.py \
  --checkpoints "/media/user0309/ADATA HV620S/lab/trained_model_cpt/models/source/source_badnets_clean.pth" "/media/user0309/ADATA HV620S/lab/trained_model_cpt/models/target/target_clean_refool.pth" \
  --in-weight-space \
  --use-permute \
  --fix-sims \
  --act-data-type image \
  --act-image-root "/media/user0309/ADATA HV620S/lab/poisoned_Cifar-10_v1/train_target" \
  --act-attack-types clean badnets refool \
  --recompute-bn-pre \
  --recompute-bn \
  --bn-reset
'''

def parse_args():
    parser = argparse.ArgumentParser("Merge resnet18_multi base-network checkpoints")
    '''
    輸入/輸出參數：
    '''
    parser.add_argument("--checkpoints", nargs="+", required=True, help="輸入要融合的 checkpoint 路徑（至少 2 個）。")
    parser.add_argument("--output", default="/media/user0309/ADATA HV620S/lab/UsedMudscFusion_cpt/merged_checkpoint.pth", help="融合後 checkpoint 的輸出路徑。")
    parser.add_argument("--save-full", action="store_true", help="儲存完整內容（metadata + state_dict）；預設只儲存純 state_dict。")
    '''
    Backbone 載入設定
    '''
    parser.add_argument("--backbone-module-dir", default="/home/user0309/2024_SooN/tang_Vincent_transfer/O2M", help="`backbone_multi.py` 所在的資料夾路徑。")
    parser.add_argument("--backbone-module-name", default="backbone_multi", help="backbone 的 Python 模組名稱。")
    parser.add_argument("--backbone-class-name", default="resnet18_multi", help="backbone 類別名稱。")
    parser.add_argument("--extracted-layer", default="7_point", choices=["6_point", "7_point", "8_point"])
    '''
    排列對齊（permutation）範圍
    '''
    parser.add_argument(
        "--perm-scope",
        default="base_plus_convm2",
        choices=["base_only", "base_plus_convm2"],
        help="對齊規則覆蓋範圍（非融合策略）：base_only=僅 backbone，base_plus_convm2=backbone + convm2。",
    )
    parser.add_argument("--convm2-prefix", default="convm2_layer.0")
    parser.add_argument("--ignore-running-val", action="store_true")
    '''
    融合/最佳化超參數（傳給 weight fusion）
    '''
    parser.add_argument("--in-weight-space", action="store_true")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--reduce", type=float, default=0.5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--use-cos", action="store_true")
    parser.add_argument("--use-permute", action="store_true")
    parser.add_argument("--no-fusion", action="store_true")
    parser.add_argument("--fix-sims", action="store_true")
    parser.add_argument("--fix-rate", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    '''
    Activation 輔助資料（用來幫助對齊）
    '''
    parser.add_argument("--act-data-type", default="image", choices=["none", "image"], help="對齊輔助資料型態：none/image。")
    parser.add_argument("--act-image-root", default="/media/user0309/ADATA HV620S/lab/poisoned_Cifar-10_v1/train_target/", help="影像資料根目錄（attack_type/digit 兩層結構）。")
    parser.add_argument("--act-attack-types", nargs="+", default=None, help="平衡抽樣的 attack type 名稱清單；未指定時自動從根目錄子資料夾推斷。")
    parser.add_argument("--act-per-digit-k", type=int, default=30, help="每個 attack type 的每個 digit 最多抽取樣本數。<=0 表示不限制。")
    parser.add_argument("--act-seed", type=int, default=0, help="平衡抽樣隨機種子。")

    parser.add_argument("--act-batch-size", type=int, default=32, help="act_loader 的 batch size。")
    parser.add_argument("--act-num-workers", type=int, default=4, help="act_loader 的 num_workers。")
    parser.add_argument("--act-test-flag", type=int, default=1, choices=[0, 1], help="傳給 resnet18_multi 的 test_flag（一般影像建議為 1）。")
    
    parser.add_argument("--act-resize-h", type=int, default=224, help="act_loader 影像 resize 高度（預設對齊原流程為 224）。")
    parser.add_argument("--act-resize-w", type=int, default=224, help="act_loader 影像 resize 寬度（預設對齊原流程為 224）。")
    parser.add_argument("--act-norm-mean", nargs=3, type=float, default=[0.485, 0.456, 0.406], help="act_loader Normalize mean（預設 ImageNet）。")
    parser.add_argument("--act-norm-std", nargs=3, type=float, default=[0.229, 0.224, 0.225], help="act_loader Normalize std（預設 ImageNet）。")
    '''
    BatchNorm 統計重估
    '''
    parser.add_argument("--recompute-bn-pre", action="store_true", help="融合前對每個 base model 先做 BN 重估。")
    parser.add_argument("--recompute-bn", action="store_true", help="融合後重估 BatchNorm 統計（建議用於推論/評估前）。")
    parser.add_argument("--bn-reset", action="store_true", help="重估 BN 前先重置 running stats（running_mean/var）。")
    parser.add_argument("--bn-max-batches", type=int, default=0, help="BN 重估最多使用幾個 batch，<=0 表示用完整 loader。")
    return parser.parse_args()


def load_raw_state_dict(path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    if not isinstance(payload, dict):
        raise ValueError(f"不支援的 checkpoint 格式：{path}")
    return payload


def extract_base_state_dict(raw_state_dict, target_keys, prefixes):
    filtered = {k: v for k, v in raw_state_dict.items() if k in target_keys}
    if filtered:
        return filtered
    for prefix in prefixes:
        candidate = {}
        for k, v in raw_state_dict.items():
            if k.startswith(prefix):
                stripped = k[len(prefix):]
                if stripped in target_keys:
                    candidate[stripped] = v
        if candidate:
            return candidate
    return {}


def build_backbone(module_dir, module_name, class_name, extracted_layer):
    if module_dir:
        sys.path.insert(0, module_dir)
    module = importlib.import_module(module_name)
    # backbone_multi.py uses a module-level extracted_layer variable.
    setattr(module, "extracted_layer", extracted_layer)
    cls = getattr(module, class_name)
    return cls()


def load_backbone_from_checkpoint(ckpt_path, args):
    model = build_backbone(
        module_dir=args.backbone_module_dir,
        module_name=args.backbone_module_name,
        class_name=args.backbone_class_name,
        extracted_layer=args.extracted_layer,
    )
    model_state = model.state_dict()
    raw_state = load_raw_state_dict(ckpt_path)
    loaded = extract_base_state_dict(raw_state, set(model_state.keys()), DEFAULT_PREFIXES)
    if not loaded:
        raise ValueError(f"在 checkpoint 中找不到可相容的 base-network 權重鍵：{ckpt_path}")
    missing, unexpected = model.load_state_dict(loaded, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}) from {ckpt_path}: {missing[:5]}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}) from {ckpt_path}: {unexpected[:5]}")
    return model


def get_perm(args):
    scope = args.perm_scope

    if scope == "base_only":
        return weight_fusion.get_resnet18_multi_base_perm_by_extracted_layer(args.extracted_layer)
    return weight_fusion.get_resnet18_multi_merged_perm_by_extracted_layer(
        extracted_layer=args.extracted_layer,
        convm2_prefix=args.convm2_prefix,
        ignore_running_val=args.ignore_running_val,
    )


class WithTestFlagDataset(Dataset):
    def __init__(self, base_ds, test_flag=True):
        self.base_ds = base_ds
        self.test_flag = bool(test_flag)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        return x, y, self.test_flag


def unpack_batch_for_forward(batch):
    if isinstance(batch, dict):
        x = batch.get("x", batch.get("img", batch.get("image")))
        test_flag = batch.get("test_flag", None)
        if x is None:
            raise ValueError("Dict 型別的 batch 必須包含以下其中一個鍵：x/img/image")
        return x, test_flag

    if isinstance(batch, (tuple, list)):
        if len(batch) == 0:
            raise ValueError("不支援空的 batch。")
        x = batch[0]
        test_flag = batch[2] if len(batch) >= 3 else None
        return x, test_flag

    return batch, None


def normalize_test_flag(test_flag):
    if test_flag is None:
        return None
    if isinstance(test_flag, torch.Tensor):
        if test_flag.numel() == 1:
            return bool(test_flag.item())
        return bool(test_flag.reshape(-1)[0].item())
    return bool(test_flag)


def forward_with_optional_flag(net, x, test_flag):
    test_flag = normalize_test_flag(test_flag)
    if test_flag is None:
        try:
            return net(x)
        except TypeError:
            return net(x, True)
    try:
        return net(x, test_flag)
    except TypeError:
        return net(x)


def recompute_bn_stats(model, loader, device, reset=False, max_batches=0):
    has_bn = False
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            has_bn = True
            if reset:
                m.momentum = None
                m.reset_running_stats()
    if not has_bn:
        print("[INFO] No BatchNorm2d found, skip BN recompute.")
        return model

    model.train()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches > 0 and step >= max_batches:
                break
            x, test_flag = unpack_batch_for_forward(batch)
            x = x.to(device)
            _ = forward_with_optional_flag(model, x, test_flag)
    model.eval()
    return model


def infer_attack_types(root):
    return sorted(
        [
            d
            for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ]
    )


def build_act_loader(args):
    if args.act_data_type == "none":
        return None

    if args.act_data_type == "image":
        if not args.act_image_root:
            raise ValueError("當 --act-data-type=image 時，必須提供 --act-image-root。")
        if args.backbone_module_dir:
            feature_dir = os.path.abspath(
                os.path.join(
                    args.backbone_module_dir,
                    "..",
                    "feature_extract-poisoned",
                )
            )
            if feature_dir not in sys.path and os.path.isdir(feature_dir):
                sys.path.insert(0, feature_dir)
        try:
            from attack_test_dataset import AttackTypeBalancedTestDataset
        except ImportError as e:
            raise ImportError(
                "Failed to import AttackTypeBalancedTestDataset. "
                "Please ensure feature_extract-poisoned is importable."
            ) from e

        tfm = transforms.Compose(
            [
                transforms.Resize((args.act_resize_h, args.act_resize_w)),
                transforms.ToTensor(),
                transforms.Normalize(mean=args.act_norm_mean, std=args.act_norm_std),
            ]
        )
        attack_types = args.act_attack_types
        if not attack_types:
            attack_types = infer_attack_types(args.act_image_root)
        if not attack_types:
            raise ValueError(
                f"在指定路徑下找不到任何 attack type 子資料夾：{args.act_image_root}"
            )
        base_ds = AttackTypeBalancedTestDataset(
            root=args.act_image_root,
            attack_types=attack_types,
            per_digit_k=args.act_per_digit_k,
            transform=tfm,
            seed=args.act_seed,
        )
        ds = WithTestFlagDataset(base_ds, test_flag=(args.act_test_flag == 1))
        print(
            f"[INFO] act_loader balanced sampling: attack_types={attack_types}, "
            f"per_digit_k={args.act_per_digit_k}, seed={args.act_seed}, total={len(ds)}"
        )
        return DataLoader(ds, batch_size=args.act_batch_size, shuffle=False, num_workers=args.act_num_workers)

    return None


def main():
    args = parse_args()
    if len(args.checkpoints) < 2:
        raise ValueError("請至少提供 2 個 checkpoint 進行融合。")

    nets = [load_backbone_from_checkpoint(path, args).to(args.device).eval() for path in args.checkpoints]
    perm = get_perm(args)
    act_loader = build_act_loader(args)

    if args.recompute_bn_pre:
        if act_loader is None:
            raise ValueError("融合前 BN 重估需要可用的 image loader，請設定 --act-data-type image 與 --act-image-root。")
        for idx, net in enumerate(nets):
            print(f"[INFO] Recomputing BN before merge for base model #{idx}")
            recompute_bn_stats(
                model=net,
                loader=act_loader,
                device=args.device,
                reset=args.bn_reset,
                max_batches=args.bn_max_batches,
            )

    fusion_engine = weight_fusion.build_fusion_engine(
        reduce=args.reduce,
        a=args.a,
        b=args.b,
        iter=args.iters,
        use_cos=args.use_cos,
        no_fusion=args.no_fusion,
        fix_sims=args.fix_sims,
        fix_rate=args.fix_rate,
        use_permute=args.use_permute,
    )

    merged_model = weight_fusion.apply_alignment_and_merge(
        fusion_engine=fusion_engine,
        nets=nets,
        perm=perm,
        act_loader=act_loader,
        in_weight_space=args.in_weight_space,
        random_state=args.random_state,
    )

    if args.recompute_bn:
        if act_loader is None:
            raise ValueError("BN 重估需要可用的 image loader，請設定 --act-data-type image 與 --act-image-root。")
        recompute_bn_stats(
            model=merged_model,
            loader=act_loader,
            device=args.device,
            reset=args.bn_reset,
            max_batches=args.bn_max_batches,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged_state_dict = merged_model.state_dict()
    if args.save_full:
        payload = {
            "state_dict": merged_state_dict,
            "source_checkpoints": args.checkpoints,
            "extracted_layer": args.extracted_layer,
            "perm_scope": args.perm_scope,
            "fusion_params": {
                "reduce": args.reduce,
                "a": args.a,
                "b": args.b,
                "iter": args.iters,
                "use_cos": args.use_cos,
                "no_fusion": args.no_fusion,
                "fix_sims": args.fix_sims,
                "fix_rate": args.fix_rate,
                "use_permute": args.use_permute,
                "in_weight_space": args.in_weight_space,
                "random_state": args.random_state,
            },
            "bn_recompute": {
                "pre_enabled": args.recompute_bn_pre,
                "enabled": args.recompute_bn,
                "reset": args.bn_reset,
                "max_batches": args.bn_max_batches,
            },
        }
        torch.save(payload, out_path)
    else:
        torch.save(merged_state_dict, out_path)

    print(f"Merged checkpoint saved to: {out_path}")
    print(f"Num source checkpoints: {len(args.checkpoints)}")
    print(f"Num merged params: {len(merged_state_dict)}")


if __name__ == "__main__":
    main()
