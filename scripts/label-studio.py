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
# setup path for the library
import sys

sys.path.insert(0, "../lib")

# %%
from db import *
from sqlalchemy import create_engine

engine = create_engine("sqlite:///nmn.db", echo=False)
Base.metadata.create_all(engine)

# %%
from pathlib import Path

data_dir = Path("nmn-ls")
data_dir.mkdir(exist_ok=True)
# page_dir = data_dir / "pages"
# page_dir.mkdir(exist_ok=True)

# %%
import json

import cv2 as cv
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from utils import numpy_to_bytes, bytes_to_numpy

Session = sessionmaker(bind=engine)
data = []

with Session() as session:
    for p in session.scalars(select(Page).order_by(Page.id)):
        page = bytes_to_numpy(p.content)
        cv.imwrite(str(data_dir / f"{p.id}.png"), page)

# %%
with Session() as session:
    for p in session.scalars(select(Page).order_by(Page.id)):
        metadata = {
            "data": {
                "image": "/data/local-files?d="
                + str((data_dir / f"{p.id}.png").relative_to(data_dir.parent))
            },
            "predictions": [{"model_version": "0.0.1", "result": []}],
        }

        results = metadata["predictions"][0]["result"]

        for c in p.components:
            results.append(
                {
                    "id": str(c.id),
                    "type": "rectanglelabels",
                    "to_name": "image",
                    "from_name": "label",
                    "original_width": p.w,
                    "original_height": p.h,
                    "image_rotation": 0,
                    "value": {
                        "x": c.x / p.w * 100,
                        "y": c.y / p.h * 100,
                        "width": c.w / p.w * 100,
                        "height": c.h / p.h * 100,
                        "rotation": 0,
                        "rectanglelabels": [c.category.name],
                    },
                }
            )

        with open(data_dir / f"{p.id}.json", "w") as f:
            json.dump(metadata, f)
