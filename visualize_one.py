import argparse
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg", force=True)


def _load_npy_dict(file_path: str):
    loaded = np.load(file_path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        return loaded.item()
    if isinstance(loaded, np.lib.npyio.NpzFile):
        return {k: loaded[k] for k in loaded.files}
    raise ValueError(f"Unsupported npy/npz structure: {file_path}")


def _get_view_axes(view: str):
    v = (view or "front").lower()
    if v == "front":
        return 0, 2, 1, 1.0, 1.0, -1.0, "Front X-Z"
    if v == "back":
        return 0, 2, 1, -1.0, 1.0, 1.0, "Back X-Z"
    if v == "left":
        return 1, 2, 0, 1.0, 1.0, -1.0, "Left Y-Z"
    if v == "right":
        return 1, 2, 0, -1.0, 1.0, 1.0, "Right Y-Z"
    if v == "side":
        return 1, 2, 0, 1.0, 1.0, -1.0, "Side Y-Z"
    if v == "top":
        return 0, 1, 2, 1.0, 1.0, -1.0, "Top X-Y"
    if v == "bottom":
        return 0, 1, 2, 1.0, -1.0, 1.0, "Bottom X-Y"
    raise ValueError("Unsupported view: {0}. Use one of: front, back, left, right, top, bottom".format(view))


def _project_uvd(xyz: np.ndarray, view: str):
    ax_u, ax_v, ax_d, flip_u, flip_v, depth_sign, view_title = _get_view_axes(view)
    u = xyz[:, ax_u] * float(flip_u)
    v = xyz[:, ax_v] * float(flip_v)
    d = xyz[:, ax_d] * float(depth_sign)
    return u, v, d, view_title


def _zbuffer_visible_indices(
    xyz: np.ndarray,
    view: str,
    image_size: int = 900,
):
    if len(xyz) == 0:
        return np.zeros((0,), dtype=np.int64)

    u, v, d, _ = _project_uvd(xyz, view=view)

    u_min = float(np.min(u))
    u_max = float(np.max(u))
    v_min = float(np.min(v))
    v_max = float(np.max(v))
    du = max(u_max - u_min, 1e-8)
    dv = max(v_max - v_min, 1e-8)

    px = np.clip(((u - u_min) / du * (image_size - 1)).astype(np.int64), 0, image_size - 1)
    py = np.clip(((v - v_min) / dv * (image_size - 1)).astype(np.int64), 0, image_size - 1)
    pix = py * image_size + px

    # z-buffer equivalent for point clouds: keep nearest (minimum depth) per pixel.
    order = np.lexsort((d, pix))
    pix_sorted = pix[order]
    first = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        first[1:] = pix_sorted[1:] != pix_sorted[:-1]
    return order[first]


def save_rgb_visualization(
    xyz,
    rgb,
    save_path,
    max_pts=6000,
    point_size=3.0,
    view="front",
):
    orig_xyz = np.asarray(xyz, dtype=np.float32)
    _, _, _, view_title = _project_uvd(orig_xyz, view=view)

    if len(xyz) > max_pts:
        idx = np.random.choice(len(xyz), max_pts, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]

    vis_idx = _zbuffer_visible_indices(np.asarray(xyz, dtype=np.float32), view=view, image_size=900)
    xyz = xyz[vis_idx]
    rgb = rgb[vis_idx]
    u, v, _, _ = _project_uvd(np.asarray(xyz, dtype=np.float32), view=view)

    fig = plt.figure(figsize=(6, 6))

    ax1 = fig.add_subplot(111)
    ax1.scatter(u, v, c=rgb, s=point_size)
    ax1.set_title(f"Input RGB ({view_title})")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close()


def visualize_from_file(
    data_npy_path: str,
    output_path: str,
    max_points: int = 6000,
    point_size: float = 3.0,
    views: tuple = ("front", "back", "left", "right", "top", "bottom"),
):
    data = _load_npy_dict(data_npy_path)

    xyz = np.asarray(data["xyz"], dtype=np.float32)
    rgb = np.asarray(data["rgb"], dtype=np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    stem, ext = os.path.splitext(output_path)
    
    for view in views:
        out_path = f"{stem}_{view}{ext}" if len(views) > 1 else output_path
        save_rgb_visualization(
            xyz,
            rgb,
            out_path,
            max_pts=max_points,
            point_size=point_size,
            view=view,
        )


def main():
    parser = argparse.ArgumentParser(description="Visualize original scene point cloud.")
    parser.add_argument("--data-npy", type=str, required=True, help="Path to one *_aug.npy scene file")
    parser.add_argument("--output", type=str, required=True, help="Output image path")
    parser.add_argument("--max-points", type=int, default=300000, help="Max points to draw")
    parser.add_argument("--point-size", type=float, default=3.0, help="Scatter point size")
    parser.add_argument(
        "--views",
        type=str,
        default="front,back,left,right,top,bottom",
        help="Comma-separated: front,back,left,right,top,bottom (or use '6')",
    )
    args = parser.parse_args()

    if args.views.strip().lower() in {"6", "six", "6view", "6views", "all"}:
        views = ("front", "back", "left", "right", "top", "bottom")
    else:
        views = tuple(v.strip().lower() for v in args.views.split(",") if v.strip())
        
    visualize_from_file(
        args.data_npy,
        args.output,
        max_points=args.max_points,
        point_size=args.point_size,
        views=views,
    )
    print(f"Saved visualization to: {args.output}")


if __name__ == "__main__":
    main()