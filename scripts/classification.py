# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import cv2 as cv
import pandas as pd
import numpy as np


def get_components(image):
    """Get the individual connected components of an image

    Returns:
        tuple (labeled_image, coordinates, components)
    """

    processed = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    processed = cv.threshold(processed, 0, 1, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    _, img, stats, _ = cv.connectedComponentsWithStats(processed)
    coords = pd.DataFrame(
        stats[
            :,
            [
                cv.CC_STAT_LEFT,
                cv.CC_STAT_TOP,
                cv.CC_STAT_WIDTH,
                cv.CC_STAT_HEIGHT,
            ],
        ],
        columns=["x", "y", "w", "h"],
    )
    coords["x2"] = coords.x + coords.w
    coords["y2"] = coords.y + coords.h

    return (
        img,
        coords,
        [processed[r.y : r.y2, r.x : r.x2] for _, r in coords.iterrows()],
    )


# %%
get_components(cv.imread("twinkle2.jpg"))


# %%
def generate_components(write: bool = False):
    from pathlib import Path

    Path("components/").mkdir(exist_ok=True)

    filtered = []

    i = 0
    for img in Path("scores/").iterdir():
        components = get_components(cv.imread(str(img.resolve())))
        for component in components[2]:
            if component.shape[0] * component.shape[1] > 40_000:
                continue  # component too large

            filtered.append(component)

            if not write:
                continue

            cv.imwrite(str(Path("components") / f"{i}.jpg"), component * 255)
            i += 1
            if i % 50 == 0:
                print(i)

    return filtered


# %%
ref = cv.imread("categorized/156.jpg")  # "F" as in a chord
ref = cv.cvtColor(ref, cv.COLOR_BGR2GRAY)
ref = cv.threshold(ref, 0, 1, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
ref = ref.astype("float32")  # switch to float for better resizing, etc.

components = generate_components()
similarities = np.empty(len(components))
for i, component in enumerate(components):
    dimensions = (
        max(
            ref.shape[1], component.shape[1]
        ),  # maximum dimensions to avoid compression and data loss
        max(ref.shape[0], component.shape[0]),
    )
    resized = cv.resize(
        component.astype("float32"),
        dimensions,
        interpolation=cv.INTER_CUBIC,
    )
    resized_ref = cv.resize(
        ref,
        dimensions,
        interpolation=cv.INTER_CUBIC,
    )
    similarity = cv.matchTemplate(resized, resized_ref, cv.TM_CCOEFF_NORMED).item()

    similarities[i] = similarity

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.plot(similarities)
plt.title("Similarities from 'F'")


# %%
def compare(source, target, method: int = cv.TM_CCOEFF_NORMED):
    aspect_ratios = (
        source.shape[1] / target.shape[1],
        source.shape[0] / target.shape[0],
    )
    coef = (
        min(aspect_ratios) / max(aspect_ratios)
    ) ** 2  # penalize large differences in aspect ratio

    source, target = resize(source, target)

    return cv.matchTemplate(source, target, method=method).item() * coef


def resize(source, target):
    dimensions = (
        max(
            source.shape[1], target.shape[1]
        ),  # maximum dimensions to avoid compression and data loss
        max(source.shape[0], target.shape[0]),
    )
    source = cv.resize(
        source.astype("float32"),
        dimensions,
        interpolation=cv.INTER_CUBIC,
    )
    target = cv.resize(
        target.astype("float32"),
        dimensions,
        interpolation=cv.INTER_CUBIC,
    )

    return source, target


# %%
def compare_batch(sources, target, method: int = cv.TM_CCOEFF_NORMED):
    similarities = np.empty(len(sources), dtype="float32")
    for i, source in enumerate(sources):
        similarities[i] = compare(source, target, method=method)

    return similarities


# %%
# compare different thresholds
if False:
    for thres in np.linspace(0.75, 0.95, 21):
        for i, idx in enumerate(np.arange(len(components))[similarities > thres]):
            if i % 36 == 0:
                fig, axs = plt.subplots(6, 6, figsize=(12, 12))
            fig.suptitle(f"{thres=:.2f}")
            j = i % 36
            axs[j // 6, j % 6].imshow(components[idx], cmap="gray_r")
            fig.tight_layout()

# %%
methods = [
    "TM_CCOEFF_NORMED",
    "TM_CCOEFF",
    "TM_CCORR_NORMED",
    "TM_CCORR",
    "TM_SQDIFF_NORMED",
    "TM_SQDIFF",
]

components = generate_components()
for method in methods:
    print(method)
    method_val: int = getattr(cv, method)
    similarities = compare_batch(components, ref, method=method_val)
    plt.figure(figsize=(12, 5))
    plt.plot(similarities)
    plt.title(f"Similarities from 'F' with {method=}")

    continue
    print(len(np.arange(len(components))[similarities > 0.80]))
    for i, idx in enumerate(np.arange(len(components))[similarities > 0.80]):
        if i % 36 == 0:
            fig, axs = plt.subplots(6, 6, figsize=(12, 12))
        fig.suptitle(f"Filtered 'F's for {method=}")
        j = i % 36
        axs[j // 6, j % 6].imshow(components[idx], cmap="gray_r")
        fig.tight_layout()

# %%
from pathlib import Path


def generate_similarities(components, categories: list):
    similarities = np.stack(
        [compare_batch(components, category) for category in categories]
    )

    return similarities


def classify_components(components, categories: list):
    similarities = generate_similarities(components, categories)
    return similarities.argmax(axis=0)


# %%
def read_category(path: Path):
    image = cv.imread(str(path))
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.threshold(image, 0, 1, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    return image


categories = [read_category(path.resolve()) for path in Path("categorized/").iterdir()]
components = generate_components()
similarities = generate_similarities(components, categories)
results = similarities.argmax(axis=0)

# %%
Path("classes.txt").write_text(str(list(results)))


# %%
def compare_component_category(component_idx, category_idx):
    component, category = resize(components[component_idx], categories[category_idx])
    print(component.shape, category.shape)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(components[component_idx], cmap="gray_r")
    axs[1].imshow(categories[category_idx], cmap="gray_r")
    similarity = compare(components[component_idx], categories[category_idx])
    print(f"Similarity: {similarity}")


# %%
indices = []

count = 0
for i in range(len(results)):
    if results[i] == 3:
        count += 1
        if count == 5:
            indices.append(i - 4)
            count = 0
    else:
        count = 0

print(indices)

compare_component_category(680, 3)
components[678]

# %%
similarities[:, 126].T

# %%
similarities[:, 40].T.argmax()

# %%
categories = [path.resolve() for path in Path("categorized/").iterdir()]
categories[39]
