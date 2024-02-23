import os
from pathlib import Path

BasePath = Path(__file__).resolve().parent.parent
# BasePath = os.path.dirname(os.path.abspath(__file__))
# LogPath = os.path.join(BasePath, "logs")
LogPath = BasePath

print(f"__file__ = {__file__}")
print(f"os.path.basename(__file__) = {os.path.basename(__file__)}")
print(
    f"os.path.basename(__file__).split('.')[0] = {os.path.basename(__file__).split('.')[0]}"
)
print(
    f"os.path.splitext(os.path.basename(__file__))[0] = {os.path.splitext(os.path.basename(__file__))[0]}"
)
print(f"Path(__file__) = {Path(__file__)}")
print(f"Path(__file__).resolve()) = {Path(__file__).resolve()}")
print(f"Path(__file__).resolve().parent) = {Path(__file__).resolve().parent}")
print(
    f"Path(__file__).resolve().parent.parent) = {Path(__file__).resolve().parent.parent}"
)
print(f"BasePath: {BasePath}")
