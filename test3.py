import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
import fiftyone.types as fot

# Step 1: 画像フォルダをデータセット化
dataset = fo.Dataset.from_dir(
    dataset_type=fot.ImageDirectory,
    dataset_dir="images",  # 先ほど出力したフォルダ
    name="rosbag_images_dataset"
)

# Step 2: CLIPによるグローバル埋め込み生成
fob.compute_similarity(
    dataset,
    model="clip-vit-base32-torch",
    brain_key="clip_embeddings",
    progress=True,
)

# === 新規: YOLOモデルのバックボーン埋め込みを直接計算 ===
# YOLO検出を行わず、直接パッチ領域の埋め込みを抽出する
# patches_field には ground_truth や predictions 等の検出領域フィールドを指定
# 事前に ground_truth がある前提

# YOLOモデルをロード
yolo_model = foz.load_zoo_model(
    "yolov5s-coco-torch",
    pretrained=True,
)

# パッチ埋め込みを直接計算
fob.compute_similarity(
    dataset,
    brain_key="yolo_patch_embeddings",       # 一意のキー
    model=yolo_model,                         # YOLOバックボーンを使う
    progress=True,
)

# Step 3: 全体埋め込みの可視化
fob.compute_visualization(
    dataset,
    brain_key="clip_embeddings_viz",
    method="umap",
)

# Step 4: YOLOパッチ埋め込みの可視化
fob.compute_visualization(
    dataset,
    brain_key="yolo_patch_embeddings_viz",
    method="umap",
)

# Step 5: アプリで確認
session = fo.launch_app(dataset)
session.wait()
