"""Export to code: one finished export as a standalone Python script that someone can run outside the app. It reads
the original data the way the app read it, applies the export's decisions and rebuilds the same model-ready data,
then checks it against what the app read and exported.

- `gather.info_for` reads what a script is written from (the export's frozen snapshot, settings and manifest, the
  data's source, the derived fields it uses, and summaries of the app's data and export for the checks).
- `render.render` writes the script: the decisions as code, the driver (`script.py`) and the app functions they run,
  copied from the app by `bundle`.
"""

from .gather import CodeUnavailable, info_for
from .render import ExportInfo, file_name, render

__all__ = ["CodeUnavailable", "ExportInfo", "file_name", "info_for", "render"]
