from __future__ import annotations  #dont want to deal with type hints

import math  #sin/cos do most of the work
import random
from datetime import datetime
from pathlib import Path

try:
    import numpy as np  #arrays make these per-pixel patterns much less painful
    from PIL import Image, ImageDraw, ImageFilter  #pillow handles resizing, drawing and pngs
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("Install dependencies!")
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "data" / "images" / "procedural"  #same folder the gan scans later

#hand-picked ramps so random shapes still have a coherent color scheme.
PALETTES = [
    [(12, 22, 52), (31, 77, 112), (61, 152, 156), (191, 218, 166), (250, 224, 155)],
    [(30, 18, 50), (81, 45, 95), (160, 74, 105), (239, 142, 119), (255, 220, 167)],
    [(17, 40, 37), (42, 94, 68), (104, 145, 86), (192, 185, 113), (241, 225, 179)],
    [(21, 29, 45), (52, 66, 94), (100, 123, 144), (193, 202, 190), (243, 230, 201)],
    [(34, 22, 16), (106, 46, 32), (179, 87, 46), (237, 157, 79), (255, 224, 148)],
    [(11, 35, 61), (22, 93, 123), (25, 165, 163), (149, 219, 194), (238, 245, 205)],
    [(36, 25, 58), (78, 55, 112), (128, 98, 170), (198, 156, 198), (248, 216, 219)],
]

STYLE_NAMES = {
    1: "Topographic contours",
    2: "Orbital rings",
    3: "Flowing waves",
    4: "Geometric layers",
    5: "Organic cells",
    6: "Mixed collection",
}

#build a soft noise map by mixing several scales, a bit like layered terrain
def _smooth_field(width: int, height: int, rng: np.random.Generator, octaves: int = 5) -> np.ndarray:
    #start with an empty full-size field; each octave (loop through) contributes another scale of texture
    accum = np.zeros((height, width), dtype=np.float32)
    weight_sum = 0.0
    for octave in range(octaves):
        #each pass starts from a different-sized random grid (coarse first, finer later).
        grid_w = max(2, 3 * (2 ** octave) + 1)
        grid_h = max(2, int(grid_w * height / width))
        small = rng.random((grid_h, grid_w), dtype=np.float32)
        #bicubic resizing smooths the tiny grid into a continuous-ish field.
        #the grid passes through 8-bit pil here, which is fine for an art texture.
        layer = np.asarray(
            Image.fromarray(np.uint8(small * 255)).resize((width, height), Image.Resampling.BICUBIC),
            dtype=np.float32,
        ) / 255.0
        #smaller details get less weight; otherwise the whole image looks too noisy.
        #each finer pass matters less than the broad shapes from earlier passes
        weight = 1.0 / (1.75 ** octave)
        accum += layer * weight
        weight_sum += weight
    accum /= max(weight_sum, 1e-9)  #avoid divide-by-zero if settings are changed later
    lo, hi = float(accum.min()), float(accum.max())  #field's current range
    #put every field back into 0..1 so the palette math is consistent.
    return (accum - lo) / max(hi - lo, 1e-6)

#turn one number per pixel into a color picked from the current palette
def _colorize(values: np.ndarray, palette: list[tuple[int, int, int]], bands: float = 0.0) -> np.ndarray:
    #a field can drift a little beyond the palette range after adding waves/noise
    values = np.clip(values, 0.0, 1.0).astype(np.float32)
    if bands > 0:
        #quantizing makes contour-like color bands instead of a smooth gradient
        values = np.floor(values * bands) / max(bands - 1.0, 1.0)
        values = np.clip(values, 0.0, 1.0)
    #these are the points in the gradient where one palette color becomes the next
    #place each palette color evenly along the normalized field
    stops = np.linspace(0.0, 1.0, num=len(palette), dtype=np.float32)
    out = np.empty((*values.shape, 3), dtype=np.float32)  #rgb output, still floating point for blending
    for channel in range(3):
        #interpolate red, green and blue separately using the same scalar field.
        out[..., channel] = np.interp(values, stops, [color[channel] for color in palette])
    return out

#make x and y coordinate maps so the pattern formulas can work in space
def _mesh(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #coordinates run from -1 to 1, which makes formulas behave the same at each resolution.
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)  #the usual x/y coordinate grid
    r = np.sqrt(x * x + y * y)  #distance from image center
    theta = np.arctan2(y, x)  #angle around the center, in radians
    return x, y, r, theta

#make contour-style bands using a noisy field as a pretend elevation map
def _topographic(size: int, rng: np.random.Generator, palette: list[tuple[int, int, int]]) -> np.ndarray:
    #keep the base field low-frequency so contour bands survive the 64px resize.
    #smooth hills become the 'elevation map'; little ripples stop every sample looking alike.
    field = _smooth_field(size, size, rng, 4)
    x, y, _, _ = _mesh(size)
    field = np.clip(field + 0.025 * np.sin(x * 8 + y * 5), 0.0, 1.0)
    image = _colorize(field, palette, bands=0)
    #narrow sin-wave minima turn into thin contour lines, like a simplified topo map.
    contour = np.abs(np.sin((field * 15.0 + 0.1) * math.pi))
    line_strength = np.clip((0.24 - contour) / 0.24, 0.0, 1.0) * 0.50
    dark = np.asarray(palette[0], dtype=np.float32)
    #blend the dark palette color only where a contour line should appear.
    return image * (1.0 - line_strength[..., None]) + dark * line_strength[..., None]

#draw a center-weighted ring pattern with a bit of asymmetry
def _orbital(size: int, rng: np.random.Generator, palette: list[tuple[int, int, int]]) -> np.ndarray:
    x, y, radius, theta = _mesh(size)
    #radians + a random phase shift change the placement of the ring pattern each time
    phase = rng.uniform(-math.pi, math.pi)
    twist = 0.05 * np.sin(theta * rng.integers(3, 8) + phase)
    #the sine makes rings; the exponential gives the center some extra light/weight
    rings = 0.5 + 0.5 * np.sin((radius + twist) * rng.uniform(18, 28) + phase)
    field = 0.58 * rings + 0.42 * np.exp(-radius * rng.uniform(1.5, 2.7))
    field += 0.11 * (_smooth_field(size, size, rng, 4) - 0.5)
    image = _colorize(np.clip(field, 0, 1), palette)
    #a small luminous off-center orb gives the compositions an anchor
    cx, cy = rng.uniform(-0.22, 0.22, size=2)
    orb = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / rng.uniform(0.012, 0.04))
    light = np.asarray(palette[-1], dtype=np.float32)
    alpha = np.clip(orb * 0.64, 0, 0.64)[..., None]
    return image * (1 - alpha) + light * alpha

def _waves(size: int, rng: np.random.Generator, palette: list[tuple[int, int, int]]) -> np.ndarray:
    x, y, _, _ = _mesh(size)
    #combine two waves with different directions/frequencies so they interfere a little
    #random phases keep every wave image from having identical alignment
    phase = rng.uniform(-math.pi, math.pi, size=4)
    f1 = np.sin(y * rng.uniform(9, 15) + 1.8 * np.sin(x * rng.uniform(3, 7) + phase[0]) + phase[1])
    f2 = np.sin(x * rng.uniform(7, 13) - y * rng.uniform(2, 6) + phase[2])
    field = 0.5 + 0.33 * f1 + 0.17 * f2
    field += (_smooth_field(size, size, rng, 4) - 0.5) * 0.10
    return _colorize(np.clip(field, 0, 1), palette)

#draw overlapping vector shapes at full resolution before adding grain
def _geometric(size: int, rng: np.random.Generator, palette: list[tuple[int, int, int]]) -> np.ndarray:
    #draw translucent vector shapes at full source resolution before we shrink them later
    base = Image.new("RGB", (size, size), palette[0])
    #rgba drawing makes it possible to layer partially transparent shapes
    draw = ImageDraw.Draw(base, "RGBA")
    colors = palette[1:]
    cx, cy = size * rng.uniform(0.35, 0.65), size * rng.uniform(0.35, 0.65)
    maxr = size * rng.uniform(0.37, 0.52)
    count = int(rng.integers(8, 17))
    #stack circles/arcs/triangles around a slightly off-center point
    for i in range(count):
        frac = (i + 1) / count
        r = maxr * frac
        color = colors[i % len(colors)]
        alpha = int(rng.integers(95, 220))
        box = (cx - r, cy - r, cx + r, cy + r)
        if i % 3 == 0:
            draw.ellipse(box, fill=(*color, alpha))
        elif i % 3 == 1:
            draw.arc(box, start=int(rng.integers(0, 180)), end=int(rng.integers(200, 360)), fill=(*color, alpha), width=max(2, size // 90))
        else:
            offset = r * 0.55
            draw.polygon([(cx, cy-r), (cx+r, cy+offset), (cx-r*0.8, cy+r*0.6)], fill=(*color, alpha))
    for _ in range(int(rng.integers(3, 8))):
        x0, y0 = rng.uniform(0, size, size=2)
        radius = rng.uniform(size * 0.015, size * 0.06)
        c = colors[int(rng.integers(0, len(colors)))]
        draw.ellipse((x0-radius, y0-radius, x0+radius, y0+radius), fill=(*c, int(rng.integers(120, 230))))
    return np.asarray(base, dtype=np.float32)

#warp a smooth field into cell-like patches with darker borders
def _organic(size: int, rng: np.random.Generator, palette: list[tuple[int, int, int]]) -> np.ndarray:
    #start with a coarse field and bend it using two additional fields
    field = _smooth_field(size, size, rng, 6)
    x, y, _, _ = _mesh(size)
    warp_x = (_smooth_field(size, size, rng, 4) - 0.5) * 0.18
    warp_y = (_smooth_field(size, size, rng, 4) - 0.5) * 0.18
    field = np.clip(field + warp_x * np.sin(y * 8) + warp_y * np.cos(x * 7), 0, 1)
    base = _colorize(field, palette, bands=6)
    cell_border = np.abs(np.sin(field * 18 * math.pi))
    border_strength = np.clip((0.15 - cell_border) / 0.15, 0, 1) * 0.25
    dark = np.asarray(palette[0], dtype=np.float32)
    return base * (1 - border_strength[..., None]) + dark * border_strength[..., None]

#choose the pattern recipe for this one image
def _generate_base(style: int, size: int, rng: np.random.Generator) -> np.ndarray:
    #pick a palette once per picture; style 6 is handled by the caller as a mix of families
    palette = random.choice(PALETTES)
    if style == 1:
        return _topographic(size, rng, palette)
    if style == 2:
        return _orbital(size, rng, palette)
    if style == 3:
        return _waves(size, rng, palette)
    if style == 4:
        return _geometric(size, rng, palette)
    if style == 5:
        return _organic(size, rng, palette)
    raise ValueError(f"Unknown pattern style: {style}")

#add broad and fine noise separately so some texture survives downsampling
def add_grain(rgb: np.ndarray, rng: np.random.Generator, amount: float) -> np.ndarray:
    """Add subtle coarse grain that remains visible after reducing to 64x64."""
    if amount <= 0:
        return np.clip(rgb, 0, 255).astype(np.uint8)
    size = rgb.shape[0]

    #a resize spreads this low-res noise over the large source image
    #this noise starts at training scale, so resizing does not erase all of it
    coarse = rng.normal(0.0, 7.0 * amount, size=(64, 64)).astype(np.float32)
    coarse_image = Image.fromarray(np.clip(coarse + 128, 0, 255).astype(np.uint8), mode="L")
    coarse = (np.asarray(coarse_image.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32) - 128.0)
    #fine source-resolution grain adds surface texture; the resize filters most of it out
    #the finer noise mostly changes local surface texture in the source png
    fine = rng.normal(0.0, 3.0 * amount, size=(size, size, 1)).astype(np.float32)
    #a smaller amount of independent chroma noise prevents overly synthetic color blocks
    chroma = rng.normal(0.0, 1.2 * amount, size=rgb.shape).astype(np.float32)
    out = rgb.astype(np.float32) + coarse[..., None] + fine + chroma
    return np.clip(out, 0, 255).astype(np.uint8)

#make the requested images and return their paths for anyone calling this as a function
def generate_dataset(
    count: int,
    resolution: int = 512,
    style: int = 6,
    noise: float = 0.45,
    seed: int | None = None,
    output_dir: Path | str = DEFAULT_OUTPUT,
    preview_every: int = 0,
) -> list[Path]:
    """Create PNGs in output_dir and return written paths."""
    #validate settings here too, since the headless runner calls this without the menu
    if count < 1:
        raise ValueError("count must be at least 1")
    if resolution not in (256, 512, 1024):
        raise ValueError("resolution must be 256, 512, or 1024")
    if style not in range(1, 6) and style != 6:
        raise ValueError("style must be between 1 and 6")
    if not 0.0 <= noise <= 1.0:
        raise ValueError("noise must be in the range 0.0 to 1.0")
    #numpy and python's random module are used for different bits of the generator
    #giving both the same seed makes a run repeatable when a seed is supplied
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    out_folder = Path(output_dir)
    out_folder.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    #a timestamp prevents a new run from overwriting the previous set of images
    #give this batch a timestamp so running the tool twice will not overwrite images
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    styles = list(range(1, 6))
    for index in range(count):
        #style 6 means mix the five generators, rather than a sixth separate formula
        active_style = py_rng.choice(styles) if style == 6 else style
        #the palette helpers use module-level random.choice, so save/restore its state
        global_state = random.getstate()
        random.seed(py_rng.randrange(0, 2**32))
        base = _generate_base(active_style, resolution, rng)
        random.setstate(global_state)
        image_array = add_grain(base, rng, noise)  #grain is deliberately added after the shapes
        image = Image.fromarray(image_array, mode="RGB")
        filename = f"proc_{active_style}_{run_tag}_{index + 1:05d}_{resolution}px.png"
        destination = out_folder / filename
        #png keeps the samples lossless
        #modest compression saves some cpu time
        image.save(destination, format="PNG", compress_level=3)
        written.append(destination)
        if preview_every > 0 and (index + 1) % preview_every == 0:
            print(f"Generated {index + 1}/{count} images...")
    return written

#keep asking until the user gives a valid integer
def ask_int(prompt: str, default: int, minimum: int, maximum: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
            if minimum <= value <= maximum:
                return value
        except ValueError:
            pass
        print(f"Enter a whole number from {minimum} to {maximum}.")

#same idea as ask_int, but decimal values are allowed here
def ask_float(prompt: str, default: float, minimum: float, maximum: float) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
            if minimum <= value <= maximum:
                return value
        except ValueError:
            pass
        print(f"Enter a number from {minimum} to {maximum}.")

#ask for generation settings when this file is run by itself
def create_interactive() -> None:
    print("\nProcedural Pattern Dataset Generator")
    print("Creates high-resolution PNG patterns with subtle grain.")
    print("The GAN will rescale them to 64x64 when training.")
    for number, name in STYLE_NAMES.items():
        print(f"{number}. {name}")
    style = ask_int("Pattern style", 1, 1, 6)
    count = ask_int("Number of images to create", 300, 1, 50000)
    resolution = ask_int("Source image resolution (256, 512, or 1024)", 512, 256, 1024)
    while resolution not in (256, 512, 1024):
        print("Choose exactly 256, 512, or 1024.")
        resolution = ask_int("Source image resolution", 512, 256, 1024)
    noise = ask_float("Grain strength (0.0 none, 1.0 strongest; subtle default)", 0.45, 0.0, 1.0)
    seed = ask_int("Random seed (-1 means choose automatically)", -1, -1, 2_147_483_647)
    use_default = input(f"Save to {DEFAULT_OUTPUT}? [Y/n]: ").strip().lower()
    if use_default in ("", "y", "yes"):
        output_dir = DEFAULT_OUTPUT
    else:
        output_dir = Path(input("Output directory: ").strip()).expanduser()
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
    #-1 means don't force a repeatable sequence; any other seed is recorded by behavior
    actual_seed = None if seed == -1 else seed
    print(f"Creating {count} image(s) at {resolution}x{resolution}; style: {STYLE_NAMES[style]}; grain: {noise:.2f}.")
    print(f"Output: {output_dir}")
    files = generate_dataset(count, resolution, style, noise, actual_seed, output_dir, preview_every=50)
    print(f"Done. Wrote {len(files)} PNG files to {output_dir}")
    print("They will be included in training automatically when stored under data/images/.")

#show a small numbered menu for making images without opening the training app
def main() -> None:
    while True:
        print("\n=======================================")
        print(" Procedural Pattern Dataset Generator")
        print("=======================================")
        print("1. Create high-resolution training images")
        print("2. Show pattern styles")
        print("3. Quit")
        choice = input("Select a number: ").strip()
        if choice == "1":
            create_interactive()
        elif choice == "2":
            for number, name in STYLE_NAMES.items():
                print(f"{number}. {name}")
            print("Style 6 selects a randomized mix of styles 1-5 for a broader dataset.")
        elif choice == "3":
            return
        else:
            print("Choose 1, 2, or 3.")

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")