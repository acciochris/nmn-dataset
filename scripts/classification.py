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

components = generate_components()
similarities = np.empty(len(components))
for i, component in enumerate(components):
    resized = cv.resize(
        component, (ref.shape[1], ref.shape[0]), interpolation=cv.INTER_NEAREST
    )
    similarity = cv.matchTemplate(resized, ref, cv.TM_CCOEFF_NORMED).item()

    similarities[i] = similarity

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.plot(similarities)
plt.title("Similarities from 'F'")

# %%
for thres in np.linspace(0.75, 0.95, 21):
    for i, idx in enumerate(np.arange(len(components))[similarities > thres]):
        if i % 36 == 0:
            fig, axs = plt.subplots(6, 6, figsize=(12, 12))
        fig.suptitle(f"{thres=:.2f}")
        j = i % 36
        axs[j // 6, j % 6].imshow(components[idx], cmap="gray_r")
        fig.tight_layout()

# %%
plt.imshow(components[156], cmap="gray_r")

# %%
ref

# %%
components[156]
