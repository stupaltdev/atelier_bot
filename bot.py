import asyncio
import io
import os
import tempfile
from pathlib import Path
import random
import cv2
import discord
import numpy as np
from PIL import Image, ImageSequence

ASSETS_DIR = Path(__file__).parent / "sticker_assets"
HEALING_DIR = Path(__file__).parent / "healing-assets"
STICKER_EXTS = {".png", ".jpg", ".jpeg"}
MARGIN_RATIO = 0.02  # gap between sticker and image edge, as fraction of image width
DETECT_MAX_SIDE = 2048  # downscale big images before face detection, for speed
FACE_MODEL = HEALING_DIR / "face_detection_yunet_2023mar.onnx"
FACE_MIN_SCORE = 0.7  # YuNet confidence needed to count as a face
VIDEO_MAX_SECONDS = 10  # longer videos are cut off, to keep the GIF small
VIDEO_MAX_FPS = 15
VIDEO_MAX_WIDTH = 480

TINT_COLORS = {
    "atelier pink": "#e43dac",
    "red": "#ff3b30",
    "orange": "#ff9500",
    "yellow": "#ffd60a",
    "green": "#34c759",
    "blue": "#0a84ff",
    "purple": "#af52de",
}

CORNERS = ["top left", "top right", "bottom left", "bottom right"]
OUTPUT_FORMATS = ["gif", "jpeg"]
EXTENSIONS = {"gif": "gif", "jpeg": "jpg"}


class NoFaceFound(Exception):
    pass


def load_stickers() -> dict[str, Path]:
    """Every image in the sticker_assets folder is a sticker, named by its filename."""
    return {
        p.stem: p
        for p in sorted(ASSETS_DIR.iterdir())
        if p.suffix.lower() in STICKER_EXTS
    }


def load_eyes() -> tuple[Image.Image, list[complex]]:
    """The healing eyes overlay, plus the centres of its left and right eye (as x + yj)."""
    eyes = Image.open(HEALING_DIR / "eyes.png").convert("RGBA")
    solid = eyes.getchannel("A").point(lambda v: 255 if v > 128 else 0)
    half = eyes.width // 2
    centres = []
    for x0, x1 in ((0, half), (half, eyes.width)):
        left, top, right, bottom = solid.crop((x0, 0, x1, eyes.height)).getbbox()
        centres.append(complex(x0 + (left + right) / 2, (top + bottom) / 2))
    return eyes, centres


STICKERS = load_stickers()
EYES, EYE_ANCHORS = load_eyes()


def quantize(frame: Image.Image, keep: Image.Image | None = None) -> Image.Image:
    """Reduce a frame to a 256-colour GIF palette.

    The overlay in keep is usually a tiny part of the frame, so a plain quantize spends
    the palette on the photo and loses the overlay's colours. Reserve some for it.
    """
    if keep is None:
        return frame.quantize(colors=256, method=Image.MEDIANCUT)

    rgba = np.asarray(keep.convert("RGBA"))
    opaque = rgba[rgba[..., 3] > 128][:, :3]
    if not len(opaque):
        return frame.quantize(colors=256, method=Image.MEDIANCUT)
    overlay_pal = Image.fromarray(opaque.reshape(1, -1, 3)).quantize(colors=32).getpalette()
    reserved = len(overlay_pal) // 3
    frame_pal = frame.quantize(colors=256 - reserved, method=Image.MEDIANCUT).getpalette()

    pal = Image.new("P", (1, 1))
    pal.putpalette((overlay_pal + frame_pal)[: 256 * 3])
    return frame.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG)


def render(src: Image.Image, fmt: str, edit, keep: Image.Image | None = None) -> io.BytesIO:
    """Run edit() on each RGBA frame and save the result as fmt.

    Animated inputs keep all frames when the output is a GIF; JPEG gets the first frame.
    keep is an overlay whose colours must survive GIF quantization.
    """
    animated = fmt == "gif" and getattr(src, "n_frames", 1) > 1
    frames, durations = [], []
    for frame in ImageSequence.Iterator(src):
        frames.append(edit(frame.convert("RGBA")).convert("RGB"))
        durations.append(frame.info.get("duration", 100))
        if not animated:
            break

    out = io.BytesIO()
    if fmt == "jpeg":
        frames[0].save(out, "JPEG", quality=95)
    else:
        frames = [quantize(f, keep) for f in frames]
        animation = (
            dict(
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=src.info.get("loop", 0),
                disposal=2,
            )
            if animated
            else {}
        )
        frames[0].save(out, "GIF", **animation)
    out.seek(0)
    return out


def apply_sticker(
    base_bytes: bytes, sticker_path: Path, corner: str, size_pct: float, fmt: str = "gif"
) -> io.BytesIO:
    src = Image.open(io.BytesIO(base_bytes))
    sticker = Image.open(sticker_path).convert("RGBA")

    new_w = max(1, round(src.width * size_pct / 100))
    new_h = max(1, round(sticker.height * new_w / sticker.width))
    sticker = sticker.resize((new_w, new_h), Image.LANCZOS)

    margin = round(src.width * MARGIN_RATIO)
    x = margin if "left" in corner else src.width - new_w - margin
    y = margin if "top" in corner else src.height - new_h - margin
    # Don't let a huge sticker hang off the image.
    x, y = max(0, x), max(0, y)

    def stamp(frame: Image.Image) -> Image.Image:
        frame.alpha_composite(sticker, (x, y))
        return frame

    return render(src, fmt, stamp, keep=sticker)


def decode_video(video_bytes: bytes) -> tuple[list[Image.Image], float]:
    """Frames (RGBA, downscaled and frame-rate limited) and the fps of a video."""
    with tempfile.NamedTemporaryFile(suffix=".video") as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        try:
            if not cap.isOpened():
                raise ValueError("couldn't read that video")
            source_fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_MAX_FPS
            fps = min(source_fps, VIDEO_MAX_FPS)
            step = source_fps / fps
            max_frames = round(VIDEO_MAX_SECONDS * fps)

            frames, index, next_keep = [], 0, 0.0
            while len(frames) < max_frames:
                ok, bgr = cap.read()
                if not ok:
                    break
                index += 1
                if index - 1 < next_keep:
                    continue  # drop frames to hit the target frame rate
                next_keep += step

                frame = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if frame.width > VIDEO_MAX_WIDTH:
                    h = round(frame.height * VIDEO_MAX_WIDTH / frame.width)
                    frame = frame.resize((VIDEO_MAX_WIDTH, h), Image.LANCZOS)
                frames.append(frame.convert("RGBA"))
        finally:
            cap.release()

    if not frames:
        raise ValueError("that video has no frames")
    return frames, fps


def frames_to_gif(frames: list[Image.Image], fps: float, keep: Image.Image | None = None) -> io.BytesIO:
    out = io.BytesIO()
    pal = [quantize(f.convert("RGB"), keep) for f in frames]
    pal[0].save(
        out, "GIF", save_all=True, append_images=pal[1:],
        duration=round(1000 / fps), loop=0, disposal=2,
    )
    out.seek(0)
    return out


def video_to_gif(
    video_bytes: bytes, sticker_path: Path, corner: str, size_pct: float
) -> io.BytesIO:
    """Convert a video to a looping GIF with the sticker stamped on every frame."""
    frames, fps = decode_video(video_bytes)
    first = frames[0]
    sticker = Image.open(sticker_path).convert("RGBA")
    new_w = max(1, round(first.width * size_pct / 100))
    new_h = max(1, round(sticker.height * new_w / sticker.width))
    sticker = sticker.resize((new_w, new_h), Image.LANCZOS)
    margin = round(first.width * MARGIN_RATIO)
    x = margin if "left" in corner else first.width - new_w - margin
    y = margin if "top" in corner else first.height - new_h - margin
    for frame in frames:
        frame.alpha_composite(sticker, (max(0, x), max(0, y)))
    return frames_to_gif(frames, fps, keep=sticker)


def parse_color(text: str) -> tuple[int, int, int]:
    """A preset name or a hex code (#ff8800, ff8800 or #f80) as an RGB tuple."""
    text = text.strip().lower()
    text = TINT_COLORS.get(text, text).lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    try:
        if len(text) != 6:
            raise ValueError
        return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        raise ValueError(
            f"unknown color. Use one of {', '.join(TINT_COLORS)} or a hex code like #ff8800"
        ) from None


def tint_frame(frame: Image.Image, color: tuple[int, int, int], opacity: float) -> Image.Image:
    layer = Image.new("RGBA", frame.size, (*color, round(255 * opacity)))
    return Image.alpha_composite(frame, layer)


def apply_tint(
    data: bytes, color: tuple[int, int, int], opacity: float, fmt: str, is_video: bool
) -> io.BytesIO:
    if is_video:
        frames, fps = decode_video(data)
        return frames_to_gif([tint_frame(f, color, opacity) for f in frames], fps)
    src = Image.open(io.BytesIO(data))
    return render(src, fmt, lambda f: tint_frame(f, color, opacity))


def find_eyes(frame: Image.Image) -> list[tuple[complex, complex]]:
    """(left eye, right eye) centres, as x + yj, for every face found in the frame."""
    scale = min(1.0, DETECT_MAX_SIDE / max(frame.size))
    size = (max(1, round(frame.width * scale)), max(1, round(frame.height * scale)))
    bgr = cv2.cvtColor(np.asarray(frame.convert("RGB").resize(size)), cv2.COLOR_RGB2BGR)

    # A fresh detector per frame: they hold per-size state, and commands run in parallel threads.
    detector = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", size, FACE_MIN_SCORE)
    _, faces = detector.detect(bgr)
    # Each row is the face box, then five landmarks; the first two are the eye
    # on the image's left and the eye on the image's right.
    return [
        (complex(f[4], f[5]) / scale, complex(f[6], f[7]) / scale)
        for f in (faces if faces is not None else [])
    ]


def place_eyes(frame: Image.Image, pairs: list[tuple[complex, complex]]) -> Image.Image:
    src_left, src_right = EYE_ANCHORS
    for left, right in pairs:
        # Scale + rotate + shift the overlay so its eye centres land on the detected ones.
        # Pillow wants the inverse mapping (output pixel -> overlay pixel).
        z = (src_right - src_left) / (right - left)
        shift = src_left - z * left
        coeffs = (z.real, -z.imag, shift.real, z.imag, z.real, shift.imag)
        # Premultiplied alpha so the transparent edges don't blur in as dark fringes.
        layer = EYES.convert("RGBa").transform(frame.size, Image.AFFINE, coeffs, Image.BICUBIC)
        frame.alpha_composite(layer.convert("RGBA"))
    return frame


def apply_healing(base_bytes: bytes, fmt: str = "gif") -> io.BytesIO:
    src = Image.open(io.BytesIO(base_bytes))
    faces = []

    def heal(frame: Image.Image) -> Image.Image:
        nonlocal faces
        # In animations, keep the last frame's faces if detection drops out for a frame.
        faces = find_eyes(frame) or faces
        return place_eyes(frame, faces)

    result = render(src, fmt, heal, keep=EYES)
    if not faces:
        raise NoFaceFound
    return result


bot = discord.Bot()


async def sticker_autocomplete(ctx: discord.AutocompleteContext):
    typed = ctx.value.lower()
    return [name for name in STICKERS if typed in name.lower()][:25]


@bot.slash_command(name="sticker", description="Put a sticker in a corner of your image")
async def sticker(
    ctx: discord.ApplicationContext,
    image: discord.Option(discord.Attachment, "The image to stamp"),
    sticker: discord.Option(
        str, "Which sticker to use", autocomplete=sticker_autocomplete
    ),
    corner: discord.Option(
        str, "Which corner (default: top left)", choices=CORNERS, default="top left"
    ),
    size: discord.Option(
        int,
        "Sticker width as % of image width (default: 50)",
        min_value=1,
        max_value=100,
        default=50,
    ),
    output_format: discord.Option(
        str, "Output format (default: gif)", name="format", choices=OUTPUT_FORMATS, default="gif"
    ),
):
    if sticker not in STICKERS:
        await ctx.respond(
            f"Unknown sticker. Choose one of: {', '.join(STICKERS)}", ephemeral=True
        )
        return
    if not (image.content_type or "").startswith("image/"):
        await ctx.respond("That attachment isn't an image.", ephemeral=True)
        return

    await ctx.defer()
    try:
        data = await image.read()
        result = apply_sticker(data, STICKERS[sticker], corner, size, output_format)
    except Exception as e:
        await ctx.followup.send(f"Couldn't process that image: {e}")
        return

    await ctx.followup.send(file=discord.File(result, filename=f"stickered.{EXTENSIONS[output_format]}"))

@bot.slash_command(name="stickervideo", description="Put a sticker in a corner of your video, as a GIF")
async def sticker_video(
    ctx: discord.ApplicationContext,
    video: discord.Option(discord.Attachment, "The video to stamp"),
    sticker: discord.Option(
        str, "Which sticker to use", autocomplete=sticker_autocomplete
    ),
    corner: discord.Option(
        str, "Which corner (default: top left)", choices=CORNERS, default="top left"
    ),
    size: discord.Option(
        int,
        "Sticker width as % of video width (default: 50)",
        min_value=1,
        max_value=100,
        default=50,
    ),
):
    if sticker not in STICKERS:
        await ctx.respond(
            f"Unknown sticker. Choose one of: {', '.join(STICKERS)}", ephemeral=True
        )
        return
    if not (video.content_type or "").startswith("video/"):
        await ctx.respond("That attachment isn't a video.", ephemeral=True)
        return

    await ctx.defer()
    try:
        data = await video.read()
        result = await asyncio.to_thread(video_to_gif, data, STICKERS[sticker], corner, size)
    except Exception as e:
        await ctx.followup.send(f"Couldn't process that video: {e}")
        return

    await ctx.followup.send(file=discord.File(result, filename="stickered.gif"))


async def color_autocomplete(ctx: discord.AutocompleteContext):
    typed = ctx.value.lower()
    return [name for name in TINT_COLORS if typed in name][:25]


@bot.slash_command(name="tint", description="Cover an image, GIF or video with a layer of color")
async def tint(
    ctx: discord.ApplicationContext,
    image: discord.Option(discord.Attachment, "The image, GIF or video to tint"),
    color: discord.Option(
        str,
        "Preset or hex code like #ff8800 (default: atelier pink)",
        autocomplete=color_autocomplete,
        default="atelier pink",
    ),
    opacity: discord.Option(
        int, "How strong the color is, in % (default: 50)", min_value=1, max_value=100, default=50
    ),
    output_format: discord.Option(
        str, "Output format (default: gif)", name="format", choices=OUTPUT_FORMATS, default="gif"
    ),
):
    content_type = image.content_type or ""
    is_video = content_type.startswith("video/")
    if not (is_video or content_type.startswith("image/")):
        await ctx.respond("That attachment isn't an image or video.", ephemeral=True)
        return
    try:
        rgb = parse_color(color)
    except ValueError as e:
        await ctx.respond(f"Couldn't use that color: {e}", ephemeral=True)
        return

    await ctx.defer()
    try:
        data = await image.read()
        result = await asyncio.to_thread(apply_tint, data, rgb, opacity / 100, output_format, is_video)
    except Exception as e:
        await ctx.followup.send(f"Couldn't process that file: {e}")
        return

    ext = "gif" if is_video else EXTENSIONS[output_format]
    await ctx.followup.send(file=discord.File(result, filename=f"tinted.{ext}"))


@bot.slash_command(name='howgay', description='Let the Atelier bot estimate how much of a homosexual you are')
async def howgay(
    ctx: discord.ApplicationContext,
    user: discord.Option(discord.Member, "Who to rate (default: you)", required=False),
):
    gayp = random.randint(1,100)
    name = user.mention if user else "You"
    verb = "are" if user is None else "is"
    await ctx.respond(f'{name} {verb} {gayp}% gay.')
    if gayp == 100:
        await ctx.send("FAGGOT!")


@bot.slash_command(name="ship", description="Rate the compatibility of two people")
async def ship(
    ctx: discord.ApplicationContext,
    first: discord.Option(discord.Member, "The first person"),
    second: discord.Option(discord.Member, "The second person"),
):
    if first == second:
        await ctx.respond("You can't ship someone with themselves.", ephemeral=True)
        return
    # Seed from both IDs (sorted, so the order doesn't matter): the same pair always gets the same score.
    score = random.Random(min(first.id, second.id) * max(first.id, second.id)).randint(1, 100)
    await ctx.respond(f"💘 {first.mention} x {second.mention}: {score}% compatible")


@bot.slash_command(name="healing", description="Put healing eyes on the faces in your image")
async def healing(
    ctx: discord.ApplicationContext,
    image: discord.Option(discord.Attachment, "The image to heal"),
    output_format: discord.Option(
        str, "Output format (default: gif)", name="format", choices=OUTPUT_FORMATS, default="gif"
    ),
):
    if not (image.content_type or "").startswith("image/"):
        await ctx.respond("That attachment isn't an image.", ephemeral=True)
        return

    await ctx.defer()
    try:
        data = await image.read()
        # Face detection is slow on big images and GIFs; keep it off the event loop.
        result = await asyncio.to_thread(apply_healing, data, output_format)
    except NoFaceFound:
        await ctx.followup.send("Couldn't find a face in that image.")
        return
    except Exception as e:
        await ctx.followup.send(f"Couldn't process that image: {e}")
        return

    await ctx.followup.send(file=discord.File(result, filename=f"healed.{EXTENSIONS[output_format]}"))


@bot.slash_command(name="ping", description="Pong")
async def ping(ctx: discord.ApplicationContext):
    await ctx.respond(f"Pong!")


@bot.event
async def on_ready():
    await bot.change_presence(activity=discord.CustomActivity("i'm gay"))
    print(f"Logged in as {bot.user} with {len(STICKERS)} stickers: {list(STICKERS)}")


if __name__ == "__main__":
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise SystemExit("Set the DISCORD_TOKEN environment variable first.")
    bot.run(token)
