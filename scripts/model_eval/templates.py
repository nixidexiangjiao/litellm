"""Emit starter workbooks so the input format never has to be guessed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .pricing import OPTIONAL_COLUMNS, REQUIRED_COLUMNS

if TYPE_CHECKING:
    from openpyxl import Workbook

_EXAMPLE_BODY = {
    "model": "recorded-model-name",
    "stream": True,
    "messages": [{"role": "user", "content": "hello"}],
}


def write_workload_template(path: Path) -> None:
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.worksheets[0]
    sheet.title = "requests"
    sheet.append(["user_id", "timestamp", "body"])
    sheet.append(["u-1001", "2026-07-30 09:15:00", json.dumps(_EXAMPLE_BODY, ensure_ascii=False)])
    _finish(workbook, sheet_widths=(16, 22, 120), path=path)


def write_pricing_template(path: Path) -> None:
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.worksheets[0]
    sheet.title = "pricing"
    sheet.append([*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS])
    sheet.append(["vendor-a-deepseek-v3", 0.27, 1.10, "vendor A / deepseek-v3", 0.07, 0.0, "USD"])
    sheet.append(["vendor-b-deepseek-v3", 2.0, 8.0, "vendor B / deepseek-v3", 0.4, 0.0, "CNY"])
    _finish(workbook, sheet_widths=(24, 14, 14, 26, 20, 20, 10), path=path)


def _finish(workbook: "Workbook", sheet_widths: tuple[int, ...], path: Path) -> None:
    from openpyxl.styles import Font

    sheet = workbook.worksheets[0]
    for index, width in enumerate(sheet_widths, start=1):
        sheet.column_dimensions[sheet.cell(row=1, column=index).column_letter].width = width
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    workbook.save(path)
