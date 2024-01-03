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
    w: Mapped[int]
    h: Mapped[int]

    components: Mapped[list["Component"]] = relationship(
        back_populates="page", default_factory=list
    )
