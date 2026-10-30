"""YYYYMM helpers for the monthly planning calendar."""

from __future__ import annotations


def validate_yyyymm(value: int) -> int:
    """Return ``value`` as an int, or raise if it is not a real YYYYMM.

    ``201924`` is rejected: the month part must be 01..12. The original
    single-iteration default used that sentinel and only worked by accident
    because later filters compared it as an integer.
    """
    number = int(value)
    year, month = divmod(number, 100)
    if year < 1900 or year > 2200 or not 1 <= month <= 12:
        raise ValueError(
            f"Invalid YYYYMM value {value}: month must be 01..12 "
            "(for example 201912, not 201924)."
        )
    return number


def advance_yyyymm(yyyymm: int, months: int = 1) -> int:
    """Move a YYYYMM value forward by ``months`` (0 is allowed)."""
    if months < 0:
        raise ValueError("months must be >= 0")
    year, month = divmod(int(yyyymm), 100)
    if not 1 <= month <= 12:
        raise ValueError(
            f"Invalid YYYYMM value {yyyymm}: month must be 01..12 "
            "(for example 201912, not 201924)."
        )
    month += int(months)
    year += (month - 1) // 12
    month = (month - 1) % 12 + 1
    return year * 100 + month
