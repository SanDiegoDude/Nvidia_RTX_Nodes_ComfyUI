import math

import torch
import nvvfx
from enum import Enum
from typing import TypedDict
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


class UpscaleType(str, Enum):
    SCALE_BY = "scale by multiplier"
    TARGET_DIMENSIONS = "target dimensions"
    MEGAPIXEL = "megapixel target"
    SHORT_SIDE = "short side resolution"


class RTXVideoSuperResolution(io.ComfyNode):
    class UpscaleTypedDict(TypedDict):
        resize_type: UpscaleType
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
                io.DynamicCombo.Input(
                    "resize_type",
                    tooltip="Choose how to determine the output resolution.",
                    options=[
                        io.DynamicCombo.Option(UpscaleType.SCALE_BY, [
                            io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.01, tooltip="Scale factor (e.g., 2.0 doubles the size)."),
                        ]),
                        io.DynamicCombo.Option(UpscaleType.TARGET_DIMENSIONS, [
                            io.Int.Input("width", default=1920, min=64, max=8192, step=8, tooltip="Target width in pixels."),
                            io.Int.Input("height", default=1080, min=64, max=8192, step=8, tooltip="Target height in pixels."),
                        ]),
                        io.DynamicCombo.Option(UpscaleType.MEGAPIXEL, [
                            io.Float.Input("megapixels", default=2.0, min=0.1, max=16.0, step=0.1, tooltip="Target total megapixels (e.g., 2.0 = ~1920x1080). Aspect ratio is preserved."),
                        ]),
                        io.DynamicCombo.Option(UpscaleType.SHORT_SIDE, [
                            io.Int.Input("short_side", default=1080, min=64, max=8192, step=8, tooltip="Target pixel length for the shortest side. Aspect ratio is preserved."),
                        ]),
                    ],
                ),
                io.Combo.Input("quality", options=["LOW", "MEDIUM", "HIGH", "ULTRA"], default="ULTRA"),
            ],
            outputs=[
                io.Image.Output("upscaled_images"),
            ],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, resize_type: UpscaleTypedDict, quality: str) -> io.NodeOutput:
        _, h, w, _ = images.shape

        selected_type = resize_type["resize_type"]
        if selected_type == UpscaleType.SCALE_BY:
            scale = resize_type["scale"]
            output_width = int(w * scale)
            output_height = int(h * scale)
        elif selected_type == UpscaleType.TARGET_DIMENSIONS:
            output_width = resize_type["width"]
            output_height = resize_type["height"]
        elif selected_type == UpscaleType.MEGAPIXEL:
            target_pixels = resize_type["megapixels"] * 1_000_000
            current_pixels = w * h
            scale = math.sqrt(target_pixels / current_pixels)
            output_width = int(w * scale)
            output_height = int(h * scale)
        elif selected_type == UpscaleType.SHORT_SIDE:
            target = resize_type["short_side"]
            if h <= w:
                scale = target / h
            else:
                scale = target / w
            output_width = int(w * scale)
            output_height = int(h * scale)
        else:
            raise ValueError(f"Unsupported resize type: {selected_type}")

        output_width = max(8, round(output_width / 8) * 8)
        output_height = max(8, round(output_height / 8) * 8)

        MAX_PIXELS = 1024 * 1024 * 16

        out_pixels = output_width * output_height
        batch_size = max(1, MAX_PIXELS // out_pixels)

        quality_mapping = {
            "LOW": nvvfx.effects.QualityLevel.LOW,
            "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM,
            "HIGH": nvvfx.effects.QualityLevel.HIGH,
            "ULTRA": nvvfx.effects.QualityLevel.ULTRA,
        }
        selected_quality = quality_mapping.get(quality, nvvfx.effects.QualityLevel.HIGH)

        upscaled_batches = []

        with nvvfx.VideoSuperRes(selected_quality) as sr:
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
                upscaled_batches.append(batch_out_tensor)

        final_images = torch.cat(upscaled_batches, dim=0)

        return io.NodeOutput(final_images)


class NVVFXVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            RTXVideoSuperResolution,
        ]


async def comfy_entrypoint() -> NVVFXVideoExtension:
    return NVVFXVideoExtension()
