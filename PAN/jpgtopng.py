from pathlib import Path
from PIL import Image   # pip install pillow

folder = Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\FYP_1\PAN\results_single\pan_baseline\test_latest")      # ← change me
out_dir = folder / "png"                      # converts go here (optional)
out_dir.mkdir(exist_ok=True)

for jpg_file in folder.glob("*.jp*g"):        # matches .jpg and .jpeg
    with Image.open(jpg_file) as im:
        png_path = out_dir / (jpg_file.stem + ".png")
        im.save(png_path, "PNG")              # lossless PNG
        print(f"✓ {jpg_file.name}  →  {png_path.name}")
