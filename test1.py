# sample_fiftyone.py
import fiftyone as fo
import fiftyone.zoo as foz

# CIFAR-10の小さなサブセットをロード
dataset = foz.load_zoo_dataset("cifar10", split="test", max_samples=50)

# アプリを起動
session = fo.launch_app(dataset)
session.wait()
