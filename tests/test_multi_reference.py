"""Standalone regression tests for Anima/Cosmos ordered reference images."""
import importlib.util
import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from comfy_api.latest import _io

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("cosmos_reference_test", os.path.join(PACK, "__init__.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class FakeInner:
    def __init__(self):
        self.process_calls = []

    def process_latent_in(self, samples):
        self.process_calls.append(samples)
        return samples * 2

    def extra_conds(self, **kwargs):
        return {"original": True}


class FakePatcher:
    def __init__(self, model=None, model_options=None, object_patches=None):
        self.model = model or FakeInner()
        self.model_options = dict(model_options or {})
        self.object_patches = dict(object_patches or {})

    def clone(self):
        return FakePatcher(self.model, self.model_options, self.object_patches)

    def add_object_patch(self, name, value):
        self.object_patches[name] = value

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


def make_latent(value, batch=1, h=2, w=2):
    return {"samples": torch.full((batch, 16, 1, h, w), float(value))}


def make_image(value, h=32, w=48, batch=1, channels=3):
    return torch.full((batch, h, w, channels), float(value) / 10)


def conditioning(marker):
    return [[torch.zeros(1, 1), {"marker": marker, "keep": "yes", "reference_latents": ["old"]}]]


class FakeVAE:
    def __init__(self, events=None):
        self.calls = []
        self.events = events if events is not None else []

    def encode(self, pixels):
        value = float(pixels[0, 0, 0, 0])
        self.calls.append((value, tuple(pixels.shape)))
        self.events.append(("vae", value))
        return torch.full((pixels.shape[0], 16, 1, pixels.shape[1] // 8, pixels.shape[2] // 8), value)


class MultiReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upscale_calls = []
        cls.original_upscale = mod.comfy.utils.common_upscale

        def common_upscale(samples, width, height, method, crop):
            cls.upscale_calls.append((tuple(samples.shape), width, height, method, crop))
            return torch.nn.functional.interpolate(samples[:, :3].float(), (height, width), mode="nearest")

        mod.comfy.utils.common_upscale = common_upscale

    @classmethod
    def tearDownClass(cls):
        mod.comfy.utils.common_upscale = cls.original_upscale

    def _encode(self, count, images=None, megapixels=0.01):
        events = []
        vae = FakeVAE(events)
        positive = conditioning("pos")
        negative = conditioning("neg")
        originals = (positive[0][1].copy(), negative[0][1].copy())
        if images is None:
            images = {f"image_{i - 1}": make_image(i) for i in range(1, count + 1)}
        result = mod.CosmosReferenceImagesEncode.execute(
            positive, negative, vae, megapixels=megapixels, images=images
        )
        return result, vae, events, originals, positive, negative

    def test_one_to_four_references_are_encoded_once_and_replace_metadata(self):
        for count in range(1, 5):
            with self.subTest(count=count):
                result, vae, events, originals, positive, negative = self._encode(count)
                pos, neg, latent = result.args
                refs = pos[0][1]["reference_latents"]
                self.assertEqual(len(refs), count)
                self.assertIs(refs, neg[0][1]["reference_latents"])
                self.assertEqual([round(float(ref[0, 0, 0, 0, 0]), 1) for ref in refs], [i / 10 for i in range(1, count + 1)])
                self.assertTrue(torch.equal(latent["samples"], refs[0]))
                self.assertEqual(len(vae.calls), count)
                self.assertEqual([event[0] for event in events], ["vae"] * count)
                self.assertEqual(pos[0][1]["keep"], "yes")
                self.assertEqual(neg[0][1]["keep"], "yes")
                self.assertEqual(positive[0][1], originals[0])
                self.assertEqual(negative[0][1], originals[1])

    def test_numeric_order_resize_grid_and_validation(self):
        # image_2 must precede image_10, independent of dictionary insertion order.
        images = {"image_10": make_image(10, h=40, w=40),
                  "image_2": make_image(2, h=20, w=80),
                  "image_0": make_image(1, h=32, w=48)}
        result, vae, _, _, _, _ = self._encode(3, images)
        refs = result.args[0][0][1]["reference_latents"]
        self.assertEqual([round(float(ref[0, 0, 0, 0, 0]), 1) for ref in refs], [0.1, 0.2, 1.0])
        self.assertEqual(len({(shape[1], shape[2]) for _, shape in vae.calls}), 1)
        target_h, target_w = vae.calls[0][1][1:3]
        self.assertEqual((target_h % 16, target_w % 16), (0, 0))

        self.upscale_calls.clear()
        self._encode(3, images)
        self.assertEqual([call[4] for call in self.upscale_calls], ["disabled", "center", "center"])
        self.assertTrue(all(call[3] == "lanczos" for call in self.upscale_calls))
        self.assertTrue(all((call[1] % 16, call[2] % 16) == (0, 0) for call in self.upscale_calls))

        same_size = make_image(1, h=160, w=160)
        self.upscale_calls.clear()
        self._encode(2, {"image_0": same_size, "image_1": same_size}, megapixels=160 * 160 / (1024 * 1024))
        self.assertEqual(self.upscale_calls, [])

        with self.assertRaisesRegex(ValueError, "at least one reference image"):
            mod.CosmosReferenceImagesEncode.execute(conditioning("p"), conditioning("n"), FakeVAE(), images={})
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            mod.CosmosReferenceImagesEncode.execute(conditioning("p"), conditioning("n"), FakeVAE(),
                                                     images={"image_0": make_image(1, batch=2)})
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            mod.CosmosReferenceImagesEncode.execute(conditioning("p"), conditioning("n"), FakeVAE(),
                                                     images={"image_0": make_image(1), "image_1": make_image(2, batch=2)})

    def test_legacy_chaining_keeps_exactly_n_refs_and_single_wrapper(self):
        patcher = FakePatcher()
        node = mod.ApplyCosmosReferenceLatent()
        first, = node.apply(patcher, make_latent(1))
        first_refs = first.model_options["cosmos_ref_latents_list"]
        self.assertEqual(len(first_refs), 1)

        latest = first
        for value in (2, 3, 4):
            latest, = node.apply(latest, make_latent(value))
            self.assertIs(latest.model_options["model_function_wrapper"], first.model_options["model_function_wrapper"])
            self.assertEqual(len(first.model_options["cosmos_ref_latents_list"]), 1)

        self.assertEqual(len(latest.model_options["cosmos_ref_latents_list"]), 4)
        self.assertEqual(len(latest.object_patches["cosmos_ref_latents"]), 4)
        self.assertEqual(len(latest.model.process_calls), 4)

        class Recorder:
            def __init__(self, refs):
                self.cosmos_ref_latents = refs
                self.calls = []

            def apply(self, x, timestep, **kwargs):
                self.calls.append((x.clone(), kwargs))
                return x

        recorder = Recorder(latest.object_patches["cosmos_ref_latents"])
        wrapper = latest.model_options["model_function_wrapper"]
        x = torch.zeros(2, 16, 2, 2, 2, dtype=torch.float16)
        args = {"input": x, "timestep": torch.ones(2), "c": {"tag": "cond"}, "runtime": "kept"}
        out = wrapper(recorder.apply, args)
        self.assertEqual(out.shape[2], 2)
        self.assertEqual(len(recorder.calls), 1)
        passed_x, passed_kwargs = recorder.calls[0]
        self.assertEqual(passed_x.shape[2], 6)
        self.assertEqual(passed_x.dtype, x.dtype)
        self.assertEqual([float(passed_x[0, 0, i, 0, 0]) for i in range(2, 6)], [2, 4, 6, 8])
        self.assertEqual(passed_kwargs, {"tag": "cond", "runtime": "kept"})
        self.assertIs(args["input"], x)

    def test_unrelated_wrapper_and_no_reference_fallback(self):
        calls = []

        def previous(model_apply, kwargs):
            calls.append(kwargs["input"].shape[2])
            return model_apply(kwargs["input"], kwargs["timestep"])

        patcher = FakePatcher(model_options={"model_function_wrapper": previous})
        (patched,) = mod.ApplyCosmosReferenceModelPatch().patch(patcher)
        wrapper = patched.model_options["model_function_wrapper"]

        class Recorder:
            def apply(self, x, timestep):
                return x

        x = torch.zeros(1, 16, 2, 1, 1)
        result = wrapper(Recorder().apply, {"input": x, "timestep": torch.ones(1), "c": {}})
        self.assertIs(result, x)
        self.assertEqual(calls, [2])

    def test_conditioning_refs_are_processed_once_and_cfg_expanded(self):
        events = []
        vae = FakeVAE(events)
        positive = conditioning("p")
        negative = conditioning("n")
        enc = mod.CosmosReferenceImagesEncode.execute(
            positive, negative, vae, megapixels=0.01,
            images={f"image_{i - 1}": make_image(i) for i in range(1, 4)},
        )
        self.assertEqual(len(vae.calls), 3)
        model = FakePatcher()
        (patched,) = mod.ApplyCosmosReferenceModelPatch().patch(model)
        extra_conds = patched.object_patches["extra_conds"]
        converted = extra_conds(reference_latents=enc.args[0][0][1]["reference_latents"])
        self.assertEqual(len(patched.model.process_calls), 3)
        refs = converted["ref_latents"].cond

        class Recorder:
            def __init__(self):
                self.calls = []

            def apply(self, x, timestep, **kwargs):
                self.calls.append((x.clone(), kwargs))
                return x

        recorder = Recorder()
        x = torch.zeros(2, 16, 2, *refs[0].shape[-2:], dtype=torch.float16)
        original_kwargs = {"input": x, "timestep": torch.ones(2), "c": {"ref_latents": refs, "tag": 1}}
        out = patched.model_options["model_function_wrapper"](recorder.apply, original_kwargs)
        patched.model_options["model_function_wrapper"](recorder.apply, original_kwargs)
        self.assertEqual(out.shape[2], 2)
        self.assertEqual(len(recorder.calls), 2)
        combined = recorder.calls[0][0]
        self.assertEqual(combined.shape[2], 5)
        self.assertEqual(combined.dtype, torch.float16)
        self.assertEqual(combined.device, x.device)
        self.assertIs(original_kwargs["input"], x)
        self.assertEqual(len(patched.model.process_calls), 3)

    def test_v3_dynamic_inputs_and_schema(self):
        schema = mod.CosmosReferenceImagesEncode.define_schema()
        schema.validate()
        self.assertEqual(schema.node_id, "CosmosReferenceImagesEncode")
        self.assertEqual(schema.category, "Cosmos/Reference")
        grow = schema.inputs[-1]
        self.assertEqual(grow.id, "images")
        self.assertEqual(grow.template.names[:4], [f"image_{i}" for i in range(4)])
        self.assertEqual(grow.template.min, 1)
        info = schema.get_v1_info(mod.CosmosReferenceImagesEncode)
        flat = {f"images.image_{i}": make_image(i + 1) for i in range(4)}
        _, _, v3_data = _io.get_finalized_class_inputs(
            info.input, flat
        )
        nested = _io.build_nested_inputs(flat, v3_data)
        self.assertEqual(sorted(nested["images"]), [f"image_{i}" for i in range(4)])
        result = mod.CosmosReferenceImagesEncode.execute(
            conditioning("p"), conditioning("n"), FakeVAE(), megapixels=0.01, **nested
        )
        self.assertEqual(len(result.args), 3)
        self.assertEqual(set(result.args[2]), {"samples"})
        self.assertEqual(mod.NODE_CLASS_MAPPINGS["CosmosReferenceImagesEncode"], mod.CosmosReferenceImagesEncode)


if __name__ == "__main__":
    unittest.main()
