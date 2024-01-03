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
from sqlalchemy import create_engine, String, ForeignKey
from sqlalchemy.orm import (
    DeclarativeBase,
    MappedAsDataclass,
    Mapped,
    mapped_column,
    relationship,
)


class Base(MappedAsDataclass, DeclarativeBase):
    pass


class ComponentCategory(Base):
    __tablename__ = "component_category"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    name: Mapped[str] = mapped_column(String(20))

    variants: Mapped[list["ComponentCategoryVariant"]] = relationship(
        back_populates="category", default_factory=list
    )
    components: Mapped[list["Component"]] = relationship(
        back_populates="category", default_factory=list
    )


class ComponentCategoryVariant(Base):
    __tablename__ = "component_category_variant"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    category_id: Mapped[int] = mapped_column(
        ForeignKey("component_category.id"), init=False
    )
    content: Mapped[bytes | None] = mapped_column(default=None)

    category: Mapped["ComponentCategory"] = relationship(
        back_populates="variants", default=None
    )


class Component(Base):
    __tablename__ = "component"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    page_id: Mapped[int] = mapped_column(ForeignKey("page.id"), init=False)
    category_id: Mapped[int] = mapped_column(
        ForeignKey("component_category.id"), init=False
    )
    x: Mapped[int]
    y: Mapped[int]
    w: Mapped[int]
    h: Mapped[int]
    content: Mapped[bytes | None] = mapped_column(default=None)

    category: Mapped["ComponentCategory"] = relationship(
        back_populates="components", default=None
    )
    page: Mapped["Page"] = relationship(back_populates="components", default=None)


class Page(Base):
    __tablename__ = "page"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    w: Mapped[int]
    h: Mapped[int]
    name: Mapped[str | None] = mapped_column(default=None)
    content: Mapped[bytes | None] = mapped_column(default=None)

    components: Mapped[list["Component"]] = relationship(
        back_populates="page", default_factory=list
    )


engine = create_engine("sqlite:///nmn.db", echo=False)
Base.metadata.create_all(engine)

# %%
import sys

sys.path.insert(0, "../lib")

from component import (
    generate_categories,
    generate_components,
    classify_components,
)

paths, contents = generate_categories()
# components = generate_components()
# results = classify_components(components, categories)

# %%
# numpy array to bytes and back

import numpy as np
from io import BytesIO


def numpy_to_bytes(arr: np.ndarray) -> bytes:
    with BytesIO() as f:
        np.save(f, arr)
        return f.getvalue()


def bytes_to_numpy(b: bytes) -> np.ndarray:
    with BytesIO(b) as f:
        return np.load(f)


# %%
# insert categories and its variants

from sqlalchemy.orm import sessionmaker

Session = sessionmaker(bind=engine)

segments = [path.stem.split(".") for path in paths]
category_names = {s[0] for s in segments}
categories = {name: ComponentCategory(name) for name in category_names}
variants = []

for s, c in zip(segments, contents):
    variant = ComponentCategoryVariant(content=numpy_to_bytes(c))
    categories[s[0]].variants.append(variant)
    variants.append(variant)

# %%
with Session() as session:
    session.expire_on_commit = False
    session.add_all(categories.values())
    session.commit()

# %%
from component import get_components, classify_components

from pathlib import Path

import cv2 as cv

pages = []
targets = [bytes_to_numpy(v.content) for v in variants]

for path in Path("scores/").iterdir():
    print(path)
    img = cv.imread(str(path))
    coords, components = get_components(img)

    results = classify_components(components, targets)

    with Session(expire_on_commit=False) as session:
        page = Page(
            w=img.shape[1], h=img.shape[0], name=path.name
        )  # numpy_to_bytes(img)
        for (i, row), content in zip(coords.iterrows(), components):
            if row.w * row.h > 40000:
                continue  # too big
            component = Component(
                x=int(row.x), y=int(row.y), w=int(row.w), h=int(row.h)
            )  # numpy_to_bytes(content)
            component.category = categories[variants[results[i]].category.name]
            component.page = page
            session.add(component)

        session.add(page)
        session.commit()

# %%
# test db lib
from sqlalchemy import create_engine

engine = create_engine("sqlite:///nmn.db", echo=False)

# %%
import sys

sys.path.insert(0, "../lib")

from db import *

create_db(engine, "./categorized/", "./scores", workers=4, store_content=True)

# %%
