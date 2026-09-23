"""Allow ``python -m shapeqc`` as an alias for ``qc-report``."""

from shapeqc.cli import report

if __name__ == "__main__":
    report()
