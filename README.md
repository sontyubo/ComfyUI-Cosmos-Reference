# ComfyUI-Cosmos-Reference

Reference image support for Cosmos models and models based on Cosmos, such as Anima.

## `Cosmos Reference Images Encode`

Connect `image_0`, `image_1`, `image_2`, `image_3`, and so on in reference order
(API keys: `images.image_0`, etc.). The first image sets the output
pixel grid and is returned as the latent for `KSampler.latent_image`. Additional images
are resized and center-cropped to that grid. The same ordered VAE latents are attached to
both positive and negative conditioning, so each reference is encoded once and shared.

For this workflow, use this node in place of `VAEEncode` for the initial reference latent,
then connect its positive and negative outputs to the sampler. Put
`Apply Cosmos Reference Model Patch` once between the model and sampler. Three- and
four-reference generation quality has not been validated.

`Apply Cosmos Reference Latent` remains available for existing workflows that chain
pre-encoded latents. The model patch handles that legacy chain without installing the
reference wrapper more than once.

Inspired by https://github.com/levzzz5154/ComfyUI/tree/ref-latents-support, with assistance from Gemini.
