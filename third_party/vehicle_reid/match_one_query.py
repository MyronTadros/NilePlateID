#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torchvision import transforms

from load_model import load_model_from_opts
from dataset import ImageDataset


def fliplr(img: torch.Tensor) -> torch.Tensor:
    """Flip horizontal (N x C x H x W)."""
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().to(img.device)
    return img.index_select(3, inv_idx)


@torch.no_grad()
def extract_features(model, dataloader, device, batchsize: int):
    """Compute L2-normalized features for an entire dataloader (with flip-aug)."""
    img_count = 0

    dummy = next(iter(dataloader))[0].to(device)
    output = model(dummy)
    feature_dim = output.shape[1]

    labels = []
    features = torch.empty((len(dataloader.dataset), feature_dim), dtype=torch.float32)

    for idx, (X, y) in enumerate(dataloader):
        n = X.size(0)
        img_count += n

        ff = torch.zeros((n, feature_dim), dtype=torch.float32, device=device)
        labels.extend(y.tolist())

        for i in range(2):
            if i == 1:
                X = fliplr(X)
            input_X = Variable(X.to(device))
            outputs = model(input_X)
            ff += outputs

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        start = idx * batchsize
        end = min((idx + 1) * batchsize, len(dataloader.dataset))
        features[start:end, :] = ff.cpu()

    return features, np.array(labels)


@torch.no_grad()
def extract_feature(model, img: torch.Tensor, device) -> torch.Tensor:
    """Compute one L2-normalized feature vector (with flip-aug)."""
    if len(img.shape) == 3:
        img = torch.unsqueeze(img, 0)
    img = img.to(device)

    feature = model(img).reshape(-1)
    img_flip = fliplr(img)
    flipped_feature = model(img_flip).reshape(-1)

    feature = feature + flipped_feature
    fnorm = torch.norm(feature, p=2)
    return (feature / fnorm).cpu()


def get_scores(query_feature: torch.Tensor, gallery_features: torch.Tensor) -> np.ndarray:
    """Cosine similarity via dot product because features are L2-normalized."""
    query = query_feature.view(-1, 1)          # (D,1)
    score = torch.mm(gallery_features, query)  # (N,1)
    return score.squeeze(1).numpy()            # (N,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Dataset root (images paths are relative to this)")
    ap.add_argument("--query_csv_path", required=True, help="CSV with query images (must contain columns: path,id)")
    ap.add_argument("--gallery_csv_path", required=True, help="CSV with gallery images (must contain columns: path,id)")
    ap.add_argument("--model_opts", required=True, help="opts.yaml used in training (e.g. model/<name>/opts.yaml)")
    ap.add_argument("--checkpoint", required=True, help="checkpoint (e.g. model/<name>/net_14.pth)")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--batchsize", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--query_index", type=int, help="Row index in query_csv (0-based, matches DataFrame row order)")
    group.add_argument("--query_path", type=str, help="Exact value of the 'path' column in query_csv")

    ap.add_argument("--topk", type=int, default=10, help="Print top-k matches")
    ap.add_argument("--respect_cam", action="store_true",
                    help="If CSVs have a 'cam' column, exclude gallery images from the same cam as the query (like repo viz script)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h = w = args.input_size
    data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    query_df = pd.read_csv(args.query_csv_path)
    gallery_df = pd.read_csv(args.gallery_csv_path)

    # Determine which query row to use
    if args.query_index is not None:
        q_idx = args.query_index
        if q_idx < 0 or q_idx >= len(query_df):
            raise SystemExit(f"--query_index out of range: {q_idx} (0..{len(query_df)-1})")
    else:
        matches = query_df.index[query_df["path"] == args.query_path].tolist()
        if not matches:
            raise SystemExit(f"--query_path not found in query_csv: {args.query_path}")
        q_idx = matches[0]

    # Build datasets
    classes = list(pd.concat([query_df["id"], gallery_df["id"]]).unique())
    query_ds = ImageDataset(args.data_dir, query_df, "id", classes, transform=data_transforms)
    gallery_ds = ImageDataset(args.data_dir, gallery_df, "id", classes, transform=data_transforms)

    gallery_loader = torch.utils.data.DataLoader(
        gallery_ds, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers
    )

    # Load model
    model = load_model_from_opts(args.model_opts, args.checkpoint, remove_classifier=True)
    model.eval()
    model.to(device)

    # Compute gallery features once
    gallery_features, _ = extract_features(model, gallery_loader, device, args.batchsize)
    # Convert to torch for fast mm
    gallery_features_t = torch.tensor(gallery_features.numpy() if hasattr(gallery_features, "numpy") else gallery_features)

    # Compute query feature
    q_img_tensor, _ = query_ds[q_idx]
    q_feature = extract_feature(model, q_img_tensor, device)

    # Optional: camera filtering
    use_cam = args.respect_cam and ("cam" in query_df.columns) and ("cam" in gallery_df.columns)
    if use_cam:
        curr_cam = query_df.loc[q_idx, "cam"]
        ok = (gallery_df["cam"].values != curr_cam)
        gallery_orig_idx = np.where(ok)[0]
        gal_features = gallery_features_t[ok]
    else:
        gallery_orig_idx = np.arange(len(gallery_df))
        gal_features = gallery_features_t

    # Score + rank
    scores = get_scores(q_feature, gal_features)
    order = np.argsort(scores)[::-1]
    topk = order[: args.topk]

    # Print results
    q_row = query_df.loc[q_idx]
    print("QUERY")
    print(f"  query_index: {q_idx}")
    print(f"  query_path : {q_row['path']}")
    print(f"  query_id   : {q_row['id']}")
    if "cam" in query_df.columns:
        print(f"  query_cam  : {q_row['cam']}")

    print("\nTOP MATCHES (gallery)")
    for rank, j in enumerate(topk, start=1):
        g_i = gallery_orig_idx[j]
        g_row = gallery_df.loc[g_i]
        print(
            f"  rank {rank:>2}: score={scores[j]:.6f}  gallery_id={g_row['id']}  gallery_path={g_row['path']}"
            + (f"  gallery_cam={g_row['cam']}" if "cam" in gallery_df.columns else "")
        )

    best_j = topk[0]
    best_g_i = gallery_orig_idx[best_j]
    best_row = gallery_df.loc[best_g_i]
    print("\nBEST MATCH SUMMARY")
    print(f"  best_gallery_id   : {best_row['id']}")
    print(f"  best_gallery_path : {best_row['path']}")
    print(f"  best_score        : {scores[best_j]:.6f}")


if __name__ == "__main__":
    main()
