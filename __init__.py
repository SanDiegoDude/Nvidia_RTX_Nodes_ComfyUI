import math

import torch
import nvvfx
from enum import Enum
from typing import TypedDict
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


class ResizeMethod(str, Enum):
    SCALE_BY = "scale by multiplier"
    TARGET_DIMENSIONS = "target dimensions"
    MEGAPIXEL = "megapixel target"
    SHORT_SIDE = "short side resolution"


UPSCALE_QUALITY_MAP = {
    ("Upscale", "LOW"): nvvfx.effects.QualityLevel.LOW,
    ("Upscale", "MEDIUM"): nvvfx.effects.QualityLevel.MEDIUM,
    ("Upscale", "HIGH"): nvvfx.effects.QualityLevel.HIGH,
    ("Upscale", "ULTRA"): nvvfx.effects.QualityLevel.ULTRA,
    ("High-Bitrate Upscale", "LOW"): nvvfx.effects.QualityLevel.HIGHBITRATE_LOW,
    ("High-Bitrate Upscale", "MEDIUM"): nvvfx.effects.QualityLevel.HIGHBITRATE_MEDIUM,
    ("High-Bitrate Upscale", "HIGH"): nvvfx.effects.QualityLevel.HIGHBITRATE_HIGH,
    ("High-Bitrate Upscale", "ULTRA"): nvvfx.effects.QualityLevel.HIGHBITRATE_ULTRA,
}

ENHANCE_QUALITY_MAP = {
    ("Denoise", "LOW"): nvvfx.effects.QualityLevel.DENOISE_LOW,
    ("Denoise", "MEDIUM"): nvvfx.effects.QualityLevel.DENOISE_MEDIUM,
    ("Denoise", "HIGH"): nvvfx.effects.QualityLevel.DENOISE_HIGH,
    ("Denoise", "ULTRA"): nvvfx.effects.QualityLevel.DENOISE_ULTRA,
    ("Deblur", "LOW"): nvvfx.effects.QualityLevel.DEBLUR_LOW,
    ("Deblur", "MEDIUM"): nvvfx.effects.QualityLevel.DEBLUR_MEDIUM,
    ("Deblur", "HIGH"): nvvfx.effects.QualityLevel.DEBLUR_HIGH,
    ("Deblur", "ULTRA"): nvvfx.effects.QualityLevel.DEBLUR_ULTRA,
}


class RTXVideoSuperResolution(io.ComfyNode):
    class ResizeTypedDict(TypedDict):
        resize_type: ResizeMethod
        scale: float
        width: int
        height: int
        megapixels: float
        short_side: int

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RTXVideoSuperResolution",
            display_name="RTX Video Super Resolution",
            category="image/upscaling",
            search_aliases=["rtx", "nvidia", "upscale", "super resolution", "vsr"],
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input(
                    "mode",
                    options=["Upscale", "High-Bitrate Upscale"],
                    default="Upscale",
                    tooltip="Upscale: AI upscaling for compressed/general sources. "
                            "High-Bitrate Upscale: optimized for already-clean sources "
                            "(ProRes, high-quality H.265) — skips artifact suppression to preserve existing detail.",
                ),
                io.DynamicCombo.Input(
                    "resize_type",
                    tooltip="Choose how to determine the output resolution.",
                    options=[
                        io.DynamicCombo.Option(ResizeMethod.SCALE_BY, [
                            io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.01, tooltip="Scale factor (e.g., 2.0 doubles the size)."),
                        ]),
                        io.DynamicCombo.Option(ResizeMethod.TARGET_DIMENSIONS, [
                            io.Int.Input("width", default=1920, min=64, max=8192, step=8, tooltip="Target width in pixels."),
                            io.Int.Input("height", default=1080, min=64, max=8192, step=8, tooltip="Target height in pixels."),
                        ]),
                        io.DynamicCombo.Option(ResizeMethod.MEGAPIXEL, [
                            io.Float.Input("megapixels", default=2.0, min=0.1, max=16.0, step=0.1, tooltip="Target total megapixels (e.g., 2.0 = ~1920x1080). Aspect ratio is preserved."),
                        ]),
                        io.DynamicCombo.Option(ResizeMethod.SHORT_SIDE, [
                            io.Int.Input("short_side", default=1080, min=64, max=8192, step=8, tooltip="Target pixel length for the shortest side. Aspect ratio is preserved."),
                        ]),
                    ],
                ),
                io.Combo.Input(
                    "intensity",
                    options=["LOW", "MEDIUM", "HIGH", "ULTRA"],
                    default="ULTRA",
                    tooltip="Processing intensity. Higher values produce better results but are slower.",
                ),
            ],
            outputs=[
                io.Image.Output("upscaled_images"),
            ],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, mode: str, resize_type: ResizeTypedDict, intensity: str) -> io.NodeOutput:
        _, h, w, _ = images.shape

        selected_method = resize_type["resize_type"]
        if selected_method == ResizeMethod.SCALE_BY:
            scale = resize_type["scale"]
            output_width = int(w * scale)
            output_height = int(h * scale)
        elif selected_method == ResizeMethod.TARGET_DIMENSIONS:
            output_width = resize_type["width"]
            output_height = resize_type["height"]
        elif selected_method == ResizeMethod.MEGAPIXEL:
            target_pixels = resize_type["megapixels"] * 1_000_000
            current_pixels = w * h
            scale = math.sqrt(target_pixels / current_pixels)
            output_width = int(w * scale)
            output_height = int(h * scale)
        elif selected_method == ResizeMethod.SHORT_SIDE:
            target = resize_type["short_side"]
            if h <= w:
                scale = target / h
            else:
                scale = target / w
            output_width = int(w * scale)
            output_height = int(h * scale)
        else:
            raise ValueError(f"Unsupported resize method: {selected_method}")

        output_width = max(8, round(output_width / 8) * 8)
        output_height = max(8, round(output_height / 8) * 8)

        selected_quality = UPSCALE_QUALITY_MAP.get(
            (mode, intensity), nvvfx.effects.QualityLevel.HIGH
        )

        return _run_vsr(images, output_width, output_height, selected_quality)


class RTXVideoEnhance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RTXVideoEnhance",
            display_name="RTX Video Denoise / Deblur",
            category="image/upscaling",
            search_aliases=["rtx", "nvidia", "denoise", "deblur", "enhance", "sharpen", "cleanup"],
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input(
                    "mode",
                    options=["Denoise", "Deblur"],
                    default="Denoise",
                    tooltip="Denoise: removes noise and compression artifacts (macro-blocking, mosquito noise) while preserving detail. "
                            "Deblur: sharpens soft or motion-blurred footage. "
                            "Both process at input resolution — output dimensions match the input.",
                ),
                io.Combo.Input(
                    "intensity",
                    options=["LOW", "MEDIUM", "HIGH", "ULTRA"],
                    default="MEDIUM",
                    tooltip="Processing strength. LOW preserves the most texture, ULTRA applies maximum correction.",
                ),
            ],
            outputs=[
                io.Image.Output("enhanced_images"),
            ],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, mode: str, intensity: str) -> io.NodeOutput:
        _, h, w, _ = images.shape

        selected_quality = ENHANCE_QUALITY_MAP.get(
            (mode, intensity), nvvfx.effects.QualityLevel.DENOISE_MEDIUM
        )

        return _run_vsr(images, w, h, selected_quality)


def _run_vsr(
    images: torch.Tensor,
    output_width: int,
    output_height: int,
    quality: nvvfx.effects.QualityLevel,
) -> io.NodeOutput:
    MAX_PIXELS = 1024 * 1024 * 16
    out_pixels = output_width * output_height
    batch_size = max(1, MAX_PIXELS // out_pixels)

    result_batches = []

    with nvvfx.VideoSuperRes(quality) as sr:
        sr.output_width = output_width
        sr.output_height = output_height
        sr.load()

        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i + batch_size]
            batch_cuda = batch.cuda().permute(0, 3, 1, 2).contiguous()

            batch_outputs = []
            for j in range(batch_cuda.shape[0]):
                input_frame = batch_cuda[j]
                dlpack_out = sr.run(input_frame).image
                output = torch.from_dlpack(dlpack_out).clone()
                batch_outputs.append(output)

            batch_out_tensor = torch.stack(batch_outputs, dim=0)
            batch_out_tensor = batch_out_tensor.permute(0, 2, 3, 1).cpu()
            result_batches.append(batch_out_tensor)

    final_images = torch.cat(result_batches, dim=0)
    return io.NodeOutput(final_images)


class NVVFXVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            RTXVideoSuperResolution,
            RTXVideoEnhance,
        ]


async def comfy_entrypoint() -> NVVFXVideoExtension:
    return NVVFXVideoExtension()
