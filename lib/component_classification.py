from pathlib import Path

import cv2 as cv
import pandas as pd
import numpy as np


def get_components(image):
    """Get the individual connected components of an image

    Returns:
        tuple (coordinates, components)
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
        coords,
        [processed[r.y : r.y2, r.x : r.x2] for _, r in coords.iterrows()],
    )


def generate_components(write: bool = False):
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


def compare_batch(sources, target, method: int = cv.TM_CCOEFF_NORMED):
    similarities = np.empty(len(sources), dtype="float32")
    for i, source in enumerate(sources):
        similarities[i] = compare(source, target, method=method)

    return similarities


def generate_similarities(components, categories: list):
    similarities = np.stack(
        [compare_batch(components, category) for category in categories]
    )

    return similarities


def classify_components(components, categories: list):
    similarities = generate_similarities(components, categories)
    return similarities.argmax(axis=0)


def read_category(path: Path):
    image = cv.imread(str(path))
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.threshold(image, 0, 1, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    return image


def generate_categories():
    paths = list(Path("categorized/").iterdir())
    return paths, [read_category(path) for path in paths]
