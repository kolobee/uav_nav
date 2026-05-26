"""Батч-рендер PlantUML .puml → .png через plantuml.jar."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JAR = ROOT.parent / "plantuml.jar"
SRC_DIR = ROOT / "diagrams" / "plantuml"
OUT_DIR = ROOT / "diagrams"


def main() -> int:
    if not JAR.exists():
        print(f"ERROR: PlantUML jar not found at {JAR}", file=sys.stderr)
        return 1
    puml_files = sorted(SRC_DIR.glob("*.puml"))
    if not puml_files:
        print(f"No .puml files in {SRC_DIR}", file=sys.stderr)
        return 1
    print(f"Rendering {len(puml_files)} PlantUML files...")
    cmd = [
        "java", "-Djava.awt.headless=true", "-jar", str(JAR),
        "-tpng", "-charset", "UTF-8",
        "-o", str(OUT_DIR.resolve()),
        *[str(p) for p in puml_files],
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr, file=sys.stderr)
        return res.returncode
    pngs = sorted(OUT_DIR.glob("*.png"))
    print(f"OK -> {len(pngs)} PNG files in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
