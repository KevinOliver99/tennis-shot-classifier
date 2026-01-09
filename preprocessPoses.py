import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import json

POSES_DIR = r"Data\PosesRaw"
OUT_DIR = r"Data\Poses"

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

def process_trc_file(trc_path, out_path):
    with open(trc_path, 'r') as f:
        lines = f.readlines()

    frame_line_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Frame#"):
            frame_line_idx = i
            break

    if frame_line_idx is None or frame_line_idx + 2 >= len(lines):
        raise ValueError("Could not find Frame# line or marker names line in the .trc file.")

    marker_line = lines[frame_line_idx + 1].strip().split('\t')[2:]
    n_markers = int(lines[2].split('\t')[3])

    marker_names = []
    for entry in lines[frame_line_idx].strip().split('\t')[2:]:
        if entry and not entry.startswith(('X', 'Y', 'Z')):
            marker_names.append(entry)
    if not marker_names:
        marker_names = [lines[frame_line_idx].strip().split('\t')[i] for i in range(2, 2 + 3 * n_markers, 3)]

    try:
        hip_marker = 'Hip'
        neck_marker = 'Neck'
        hip_idx = marker_names.index(hip_marker)
        neck_idx = marker_names.index(neck_marker)
    except ValueError:
        raise ValueError(f"Could not find 'Hip' or 'Neck' in marker names: {marker_names}")

    out_lines = []
    out_lines.append("Frame\tTime\t" + "\t".join(marker_line) + "\n")

    data_start = frame_line_idx + 2

    for line in lines[data_start:]:
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        if len(parts) < 2 + n_markers * 3:
            continue
        frame = parts[0]
        time = parts[1]
        coords = np.array([float(x) for x in parts[2:2 + n_markers * 3] if x.strip() != ''])

        coords = coords.reshape((n_markers, 3))

        hip_xy = coords[hip_idx, :2].copy()
        coords[:, :2] -= hip_xy

        neck_xy = coords[neck_idx, :2]
        neck_dist = np.linalg.norm(neck_xy)

        scale = 50.0 / neck_dist if neck_dist != 0 else 1.0
        coords[:, :2] *= scale

        coords_flat = coords.flatten()
        out_line = f"{frame}\t{time}\t" + "\t".join(f"{v:.6f}" for v in coords_flat) + "\n"
        out_lines.append(out_line)

    with open(out_path, 'w') as f:
        f.writelines(out_lines)

def mirror_pose_file(in_path, out_path, n_markers):
    with open(in_path, 'r') as f:
        lines = f.readlines()
    out_lines = [lines[0]]
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) < 2 + n_markers * 3:
            continue
        frame = parts[0]
        time = parts[1]
        coords = np.array([float(x) for x in parts[2:2 + n_markers * 3]])
        coords = coords.reshape((n_markers, 3))
        coords[:, 0] *= -1
        coords_flat = coords.flatten()
        out_line = f"{frame}\t{time}\t" + "\t".join(f"{v:.6f}" for v in coords_flat) + "\n"
        out_lines.append(out_line)
    with open(out_path, 'w') as f:
        f.writelines(out_lines)

def main():
    for subdir in os.listdir(POSES_DIR):
        subdir_path = os.path.join(POSES_DIR, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for fname in os.listdir(subdir_path):
            if fname.endswith("px_person00.trc"):
                trc_path = os.path.join(subdir_path, fname)
                base = os.path.basename(fname)
                identifier = base.split("_")[0].replace(".trc", "")
                out_fname = f"{identifier}.txt"
                out_path = os.path.join(OUT_DIR, out_fname)
                try:
                    process_trc_file(trc_path, out_path)
                    print(f"Processed {trc_path} -> {out_path}")
                except Exception as e:
                    print(f"Skipping {trc_path} due to error: {e}")
                    if os.path.exists(out_path):
                        os.remove(out_path)
                    continue

                try:
                    with open(out_path, 'r') as f:
                        header = f.readline()
                    header_cols = [col for col in header.strip().split('\t') if col]
                    n_coords = len(header_cols) - 2
                    n_markers = n_coords // 3
                    mirrored_fname = f"{identifier}_mirrored.txt"
                    mirrored_path = os.path.join(OUT_DIR, mirrored_fname)
                    mirror_pose_file(out_path, mirrored_path, n_markers)
                    print(f"Mirrored {out_path} -> {mirrored_path}")
                except Exception as e:
                    print(f"Skipping mirroring for {out_path} due to error: {e}")
                    if os.path.exists(mirrored_path):
                        os.remove(mirrored_path)

    annotations_path = os.path.join("Data", "shot_annotations.json")
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    pose_files = [f for f in os.listdir(OUT_DIR) if f.endswith(".txt") and not f.endswith("_mirrored.txt")]
    pose_keys = set(os.path.splitext(f)[0].strip() for f in pose_files)

    removed_annots = []
    for k in list(annotations.keys()):
        base = k.replace(".mp4", "").strip()
        if base not in pose_keys:
            removed_annots.append(k)
            del annotations[k]

    with open(annotations_path, "w") as f:
        json.dump(annotations, f, indent=4)

    annotation_keys = set(k.replace(".mp4", "").strip() for k in annotations.keys())
    removed_pose_files = []
    for f in os.listdir(OUT_DIR):
        if f.endswith(".txt"):
            base = f.replace("_mirrored.txt", "").replace(".txt", "").strip()
            if base not in annotation_keys:
                file_path = os.path.join(OUT_DIR, f)
                os.remove(file_path)
                removed_pose_files.append(f)

    print(f"Removed {len(removed_annots)} annotation(s) without pose file: {removed_annots}")
    print(f"Removed {len(removed_pose_files)} pose file(s) without annotation: {removed_pose_files}")

if __name__ == "__main__":
    main()