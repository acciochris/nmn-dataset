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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

# %%
image = cv.imread("twinkle.jpg")

# plt.figure(figsize=(6, 9))
# plt.imshow(image)

# %%
grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
bandw = cv.threshold(grayscale, 0, 1, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
# plt.figure(figsize=(10, 15))
# plt.imshow(bandw, cmap="gray")


# %%
num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(bandw)
components = stats[
    :,
    [
        cv.CC_STAT_LEFT,
        cv.CC_STAT_TOP,
        cv.CC_STAT_WIDTH,
        cv.CC_STAT_HEIGHT,
    ],
]
components = pd.DataFrame(components, columns=["x", "y", "w", "h"])

# %%
fig, ax = plt.subplots(figsize=(16, 24))
ax.imshow(bandw, cmap="gray_r")
for i, row in components.iterrows():
    patch = patches.Rectangle(
        (row.x, row.y), row.w, row.h, linewidth=1, edgecolor="r", facecolor="none"
    )
    ax.add_patch(patch)
fig.show()

# %%
components["x2"] = components.x + components.w
components["y2"] = components.y + components.h


# %%
def get_components(image):
    """Get the individual connected components of an image

    Returns:
        a three dimensional numpy array with dimension (component, y, x)
    """

    processed = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    processed = cv.threshold(processed, 0, 1, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    _, _, stats, _ = cv.connectedComponentsWithStats(processed)
    components = pd.DataFrame(
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
    components["x2"] = components.x + components.w
    components["y2"] = components.y + components.h

    return [processed[r.y : r.y2, r.x : r.x2] for _, r in components.iterrows()]


# %%
get_components(cv.imread("twinkle2.jpg"))[:5]

# %%
