from os import PathLike
from functools import partial
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2 as cv
from sqlalchemy import String, ForeignKey
from sqlalchemy.orm import (
    DeclarativeBase,
    MappedAsDataclass,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

from component import (
    generate_categories,
    get_components,
    classify_components,
)
from utils import numpy_to_bytes, bytes_to_numpy


class Base(MappedAsDataclass, DeclarativeBase):
    pass


class ComponentCategory(Base):
    __tablename__ = "component_category"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    name: Mapped[str] = mapped_column(String(20))

    variants: Mapped[list["ComponentCategoryVariant"]] = relationship(
        back_populates="category", default_factory=list, repr=False
    )
    components: Mapped[list["Component"]] = relationship(
        back_populates="category", default_factory=list, repr=False
    )


class ComponentCategoryVariant(Base):
    __tablename__ = "component_category_variant"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    category_id: Mapped[int] = mapped_column(
        ForeignKey("component_category.id"), init=False
    )
    content: Mapped[bytes | None] = mapped_column(default=None, repr=False)

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
    content: Mapped[bytes | None] = mapped_column(default=None, repr=False)

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
    content: Mapped[bytes | None] = mapped_column(default=None, repr=False)

    components: Mapped[list["Component"]] = relationship(
        back_populates="page", default_factory=list, repr=False
    )


def create_tables(engine):
    Base.metadata.create_all(engine)


def _classify(targets, img_info):
    path, img = img_info
    print(path)
    coords, components = get_components(img)

    results = classify_components(components, targets)
    return path, img, results, coords, components


def _images(pages_path: str | PathLike):
    for path in Path(pages_path).iterdir():
        img = cv.imread(str(path))
        yield path, img


def create_db(
    engine,
    category_path: str | PathLike,
    pages_path: str | PathLike,
    store_content: bool = False,
    workers: int = 1,
):
    create_tables(engine)
    Session = sessionmaker(bind=engine)

    # add categories and variants
    paths, contents = generate_categories(category_path)
    segments = [path.stem.split(".") for path in paths]
    category_names = {s[0] for s in segments}
    categories = {name: ComponentCategory(name) for name in category_names}
    variants = []

    for s, c in zip(segments, contents):
        variant = ComponentCategoryVariant(content=numpy_to_bytes(c))
        categories[s[0]].variants.append(variant)
        variants.append(variant)

    with Session() as session:
        session.expire_on_commit = False
        session.add_all(categories.values())
        session.commit()

    # add pages and components
    targets = [bytes_to_numpy(v.content) for v in variants]

    _classify_func = partial(_classify, targets)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_classify_func, img_info)
            for img_info in _images(pages_path)
        ]
        for future in as_completed(futures):
            path, img, results, coords, components = future.result()

            with Session(expire_on_commit=False) as session:
                page = Page(
                    w=img.shape[1],
                    h=img.shape[0],
                    name=path.name,
                    content=numpy_to_bytes(img) if store_content else None,
                )

                for (i, row), content in zip(coords.iterrows(), components):
                    if row.w * row.h > 40000:
                        continue  # too big

                    component = Component(
                        x=int(row.x),
                        y=int(row.y),
                        w=int(row.w),
                        h=int(row.h),
                        content=numpy_to_bytes(content) if store_content else None,
                    )
                    component.category = categories[variants[results[i]].category.name]
                    component.page = page
                    session.add(component)

                session.add(page)
                session.commit()
