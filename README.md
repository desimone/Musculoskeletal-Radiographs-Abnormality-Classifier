# Musculoskeletal Radiographs Abnormality Classifier

## Experiments

| Network                | Accuracy (encounter) | Precision (encounter) | Recall (encounter) | F1 (encounter) | Kappa (encounter) |
| ---------------------- | -------------------- | --------------------- | ------------------ | -------------- | ----------------- |
| DenseNet169 (baseline) | .83 (.84)            | .82 (.82)             | .87 (.90)          | .84 (.86)      | .65 (.65)         |
| MobileNet              | .81 (.83)            | .80 (.82)             | .85 (.89)          | .82 (.85)      | .62 (.62)         |
| NASNetMobile           | .82 (.83)            | .78 (.80)             | .89 (.92)          | .83 (.86)      | .63 (.63)         |

Also, ResNet50 in pytorch which achieved equivalent results.

## The [Mura](https://arxiv.org/abs/1712.06957) Dataset

```latex
@misc{1712.06957,
Author = {Pranav Rajpurkar and Jeremy Irvin and Aarti Bagul and Daisy Ding and Tony Duan and Hershel Mehta and Brandon Yang and Kaylie Zhu and Dillon Laird and Robyn L. Ball and Curtis Langlotz and Katie Shpanskaya and Matthew P. Lungren and Andrew Ng},
Title = {MURA Dataset: Towards Radiologist-Level Abnormality Detection in Musculoskeletal Radiographs},
Year = {2017},
Eprint = {arXiv:1712.06957},}
```

|     Study | Normal    | Abnormal  |      Total |
| --------: | :-------- | :-------- | ---------: |
|     Elbow | 1,203     | 768       |      1,971 |
|    Finger | 1,389     | 753       |      2,142 |
|   Forearm | 677       | 380       |      1,057 |
|      Hand | 1,613     | 602       |      2,215 |
|   Humerus | 411       | 367       |        778 |
|  Shoulder | 1,479     | 1,594     |      3,073 |
|     Wrist | 2,295     | 1,451     |      3,746 |
| **Total** | **9,067** | **5,915** | **14,982** |

- Each study contains 1-N views (images)
- 40,895 multi-view radiographic images

### Their results (DenseNet169)

|                  | Radiologists (95% CI)    | Model (95% CI)           |
| ---------------: | :----------------------- | :----------------------- |
|            Elbow | **0.858** (0.707, 0.959) | 0.848 (0.691, 0.955)     |
|           Finger | 0.781 (0.638, 0.871)     | **0.792** (0.588, 0.933) |
|          Forearm | **0.899** (0.804, 0.960) | 0.814 (0.633, 0.942)     |
|             Hand | 0.854 (0.676, 0.958)     | **0.858** (0.658, 0.978) |
|          Humerus | **0.895** (0.774, 0.976) | 0.862 (0.709, 0.968)     |
|         Shoulder | **0.925** (0.811, 0.989) | 0.857 (0.667, 0.974)     |
|            Wrist | 0.958 (0.908, 0.988)     | **0.968** (0.889, 1.000) |
| **Aggregate F1** | **0.884** (0.843, 0.918) | 0.859 (0.804, 0.905)     |
