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
    content: Mapped[bytes]

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
    content: Mapped[bytes]
    x: Mapped[int]
    y: Mapped[int]
    w: Mapped[int]
    h: Mapped[int]

    category: Mapped["ComponentCategory"] = relationship(
        back_populates="components", default=None
    )
    page: Mapped["Page"] = relationship(back_populates="components", default=None)


class Page(Base):
    __tablename__ = "page"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    content: Mapped[bytes]

    components: Mapped[list["Component"]] = relationship(
        back_populates="page", default_factory=list
    )


engine = create_engine("sqlite:///nmn.db", echo=True)
Base.metadata.create_all(engine)

# %%
import sys

sys.path.insert(0, "../lib")

from component_classification import (
    generate_categories,
    generate_components,
    classify_components,
)

paths, contents = generate_categories()
# components = generate_components()
# results = classify_components(components, categories)

# %%
# insert categories and its variants

from sqlalchemy.orm import sessionmaker

Session = sessionmaker(bind=engine)

segments = [path.stem.split(".") for path in paths]
category_names = {s[0] for s in segments}
categories = {name: ComponentCategory(name) for name in category_names}

for s, c in zip(segments, contents):
    categories[s[0]].variants.append(ComponentCategoryVariant(content=c.tobytes()))

# %%
with Session() as session:
    session.add_all(categories.values())
    session.commit()

# %%
from component_classification import get_components, compare_batch

from pathlib import Path

import cv2 as cv
import numpy as np
from sqlalchemy import select

pages = []

for path in Path("scores/").iterdir():
    img = cv.imread(str(path))
    coords, components = get_components(img)

    with Session() as session:
        variants = session.execute(select(ComponentCategoryVariant))
        targets = [np.frombuffer(v.content, dtype=np.uint8).resize() for v in variants]

    for (_, row), content in zip(coords.iterrows(), components):
        component = Component(content.tobytes(), row.x, row.y, row.w, row.h)
