import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
import fiftyone.types as fot

# Step 1: サンプルデータセットのダウンロード（CIFAR-10）
# dataset = foz.load_zoo_dataset(
#     "cifar10",
#     split="test",
#     max_samples=50,
#     shuffle=True,
#     dataset_name="my_cifar10_sample"
# )

dataset = fo.Dataset.from_dir(
    dataset_type=fot.ImageDirectory,
    dataset_dir="images",   # ← 先ほど画像を出力したフォルダ名
    name="rosbag_images_dataset"
)

# Step 2: CLIPによるEmbedding生成（VLM）
fob.compute_similarity(
    dataset,
    model="clip-vit-base32-torch",  # CLIP埋め込み
    brain_key="clip_embeddings",
    progress=True
)


fob.compute_visualization(
    dataset,
    brain_key="clip_embeddings_viz",
    method="umap",           # or "umap"
    # embeddings_field="embeddings",  # 明示しなくても大抵OK
    label_field=None,        # 分類ラベル等があれば指定可
)

# Step 5: FiftyOneアプリで確認
session = fo.launch_app(dataset)
session.wait()
