"""
onnx_export.py — Export CLIP model to ONNX format for faster inference.

Usage:
    python onnx_export.py [--output-dir ./models]
"""

import os
import argparse
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


def export_clip_to_onnx(
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    output_dir: str = "./models",
):
    """Export CLIP visual and text encoders to ONNX."""
    import open_clip

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading CLIP model: {model_name} ({pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained,
    )
    model.eval()

    # --- Export Visual Encoder ---
    visual_path = os.path.join(output_dir, "clip_visual.onnx")
    print(f"Exporting visual encoder to {visual_path}...")

    dummy_image = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model.visual,
        dummy_image,
        visual_path,
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        opset_version=14,
    )
    print(f"  ✓ Visual encoder saved ({os.path.getsize(visual_path) / 1e6:.1f} MB)")

    # --- Export Text Encoder ---
    text_path = os.path.join(output_dir, "clip_text.onnx")
    print(f"Exporting text encoder to {text_path}...")

    tokenizer = open_clip.get_tokenizer(model_name)
    dummy_text = tokenizer(["a photo of a cat"])

    # Text encoder wrapping
    class TextEncoder(torch.nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.model = clip_model

        def forward(self, text):
            return self.model.encode_text(text)

    text_encoder = TextEncoder(model)
    text_encoder.eval()

    try:
        torch.onnx.export(
            text_encoder,
            dummy_text,
            text_path,
            input_names=["text"],
            output_names=["embedding"],
            dynamic_axes={"text": {0: "batch_size"}, "embedding": {0: "batch_size"}},
            opset_version=14,
        )
        print(f"  ✓ Text encoder saved ({os.path.getsize(text_path) / 1e6:.1f} MB)")
    except Exception as e:
        print(f"  ⚠ Text encoder export failed (common with some CLIP variants): {e}")
        print("  Visual encoder is still available for image inference.")

    print(f"\nONNX export complete! Models saved to: {output_dir}/")
    return {"visual": visual_path, "text": text_path if os.path.exists(text_path) else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CLIP to ONNX")
    parser.add_argument("--output-dir", default="./models", help="Output directory")
    parser.add_argument("--model", default="ViT-B-32", help="CLIP model name")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k", help="Pretrained weights")
    args = parser.parse_args()

    export_clip_to_onnx(
        model_name=args.model,
        pretrained=args.pretrained,
        output_dir=args.output_dir,
    )
