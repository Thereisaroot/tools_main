"""ShookLink application package."""

__version__ = "0.1.0"

# Git substitutes the full commit only when exporting a release archive.
_archive_commit = "$Format:%H$"
BUILD_REVISION = _archive_commit if not _archive_commit.startswith("$") else "checkout"
