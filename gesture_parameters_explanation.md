# Gesture Parameters Explanation

This note summarizes what is stored in the PEAR minimal feature files, what the camera parameter means, how these features compare to 2D skeletons, and how they can be used for transformer-based sign language translation.

## What Is Saved In Minimal PEAR Features

In minimal mode, each clip stores structured SMPL-X / FLAME parameters rather than RGB video frames.

Typical saved arrays include:

- `body/global_pose`
- `body/body_pose`
- `body/left_hand_pose`
- `body/right_hand_pose`
- `body/exp`
- `body/shape`
- `body/hand_scale`
- `body/head_scale`
- `flame/expression_params`
- `flame/eye_pose_params`
- `flame/eyelid_params`
- `flame/jaw_params`
- `flame/pose_params`
- `flame/shape_params`
- `camera/pd_cam`

These are dense continuous parameters describing a 3D articulated body and face model.

## What These Parameters Mean

### Body Parameters

- `body/global_pose`
  - Global orientation of the body root.
- `body/body_pose`
  - Rotation of the main body joints such as torso, shoulders, elbows, and wrists.
- `body/left_hand_pose`, `body/right_hand_pose`
  - Fine-grained finger articulation for each hand.
- `body/exp`
  - Expression-related coefficients used in the body-side model output.
- `body/shape`
  - Body identity / shape coefficients.
- `body/hand_scale`
  - Learned scale terms for the hands.
- `body/head_scale`
  - Learned scale terms for the head.

### Face Parameters

- `flame/expression_params`
  - Facial expression coefficients.
- `flame/eye_pose_params`
  - Eye pose parameters.
- `flame/eyelid_params`
  - Eyelid motion coefficients.
- `flame/jaw_params`
  - Jaw motion parameters.
- `flame/pose_params`
  - Head / face pose parameters in the FLAME branch.
- `flame/shape_params`
  - Face identity / shape coefficients.

## Camera Parameter

`camera/pd_cam` is saved as a per-frame `4x4` transform matrix.

It is not the original camera calibration from the source video. It is the model's predicted render/extrinsic camera transform in PEAR's canonical camera convention.

In the model:

- the network predicts a 3D camera vector
- the z translation is derived from a scale-like term
- this is converted into a full `4x4` transform
- that transform is later used for projection and mesh rendering

So `camera/pd_cam` is useful for:

- reconstructing rendered SMPL-X videos
- preserving the model's view convention

It is not a physical camera measurement from the dataset.

## Are These Features Normalized

### Input Normalization

Before prediction, the input frames are normalized by:

- resizing/padding to a fixed spatial size
- image normalization with ImageNet mean/std
- a fixed crop convention used by the model

### Output Normalization

The saved outputs are not statistically normalized for downstream learning.

They are:

- structured in a canonical 3D parametric model space
- more geometry-aware than 2D keypoints
- still raw continuous parameters

So for transformer training, it is usually useful to apply your own feature normalization, for example:

- per-dimension mean/std normalization
- rotation conversion to a compact representation
- separate handling of static and dynamic features

## Comparison To 2D Skeleton Features

2D skeleton features usually contain:

- `(x, y, confidence)` for each joint

Examples:

- 17 body joints: about `51` numbers per frame
- 59 body+hands joints: about `177` numbers per frame
- 133 whole-body keypoints: about `399` numbers per frame

PEAR minimal features are much denser. For one sample, the per-frame dimensionality is about:

- `body/global_pose`: `1 x 3 x 3 = 9`
- `body/body_pose`: `21 x 3 x 3 = 189`
- `body/left_hand_pose`: `15 x 3 x 3 = 135`
- `body/right_hand_pose`: `15 x 3 x 3 = 135`
- `body/exp`: `50`
- `body/shape`: `200`
- `body/hand_scale`: `3`
- `body/head_scale`: `3`
- `camera/pd_cam`: `4 x 4 = 16`
- `flame/expression_params`: `50`
- `flame/eye_pose_params`: `6`
- `flame/eyelid_params`: `2`
- `flame/jaw_params`: `3`
- `flame/pose_params`: `3`
- `flame/shape_params`: `300`

Total: about `1104` numbers per frame.

### Practical Difference

2D skeletons:

- smaller
- easier to store
- viewpoint-dependent
- image-plane only
- easier to extract cheaply

PEAR parameters:

- larger
- 3D and parametric
- preserve hand articulation and face detail better
- renderable back into SMPL-X
- more structured for geometry-aware modeling

## Why PEAR Can Be Larger Than MP4

MP4 is aggressively compressed video. It exploits:

- repeated content across frames
- static background
- perceptual compression

PEAR stores dense numeric arrays per frame. Even with `fp16`, this can be larger than compressed MP4 because:

- pose is stored as dense rotations
- shape and face coefficients are stored directly
- many values are saved for every frame

So PEAR is paying storage for explicit motion structure, not appearance compression.

## Why `fp16` Helps

Using `fp16` reduces the size of floating-point tensors by about half.

From the lightweight comparison:

- fp32 outputs were about twice the size of fp16 outputs
- numerical differences were small
- rendered SMPL-X videos remained visually close

So `--minimal --fp16` is a reasonable setting when:

- the goal is sign language translation
- gesture parameters matter more than hidden features
- storage efficiency is important

## Transformer Encoding

### Simple Frame-As-Token Encoding

Flatten all dynamic per-frame parameters into one vector:

```python
x_t = concat([
    global_pose.flatten(),
    body_pose.flatten(),
    left_hand_pose.flatten(),
    right_hand_pose.flatten(),
    body_exp,
    body_shape,
    hand_scale,
    head_scale,
    pd_cam.flatten(),
    flame_expression,
    eye_pose,
    eyelid,
    jaw,
    flame_pose,
    flame_shape,
])  # [T, D]
```

Then project into transformer dimension:

```python
z = Linear(D, d_model)(x_t)
z = z + positional_encoding(T)
y = TransformerEncoder(z)
```

### Better Practical Encoding

A more compact approach is:

- convert rotation matrices to 6D or axis-angle
- treat static identity-like parameters separately
- optionally remove or separately encode camera

For example, dynamic per-frame features could include:

- global pose
- body pose
- left hand pose
- right hand pose
- face expression
- jaw / eye / eyelid
- optional camera

Static clip-level features could include:

- `body/shape`
- `flame/shape_params`
- average scales

These static features can be:

- added once as a clip token, or
- used as conditioning features

Example:

```python
clip_token = Linear(D_static, d_model)(clip_static)
frame_tokens = Linear(D_dyn, d_model)(dynamic_frames)
sequence = [clip_token, frame_1, frame_2, ..., frame_T]
memory = TransformerEncoder(sequence + positional_encoding)
```

### Part-Based Encoding

Another strong option for sign language:

- body token
- left hand token
- right hand token
- face token
- optional camera/global token

This lets the transformer attend across parts more explicitly and can be more effective for sign language than a single fully flattened frame vector.

## Summary

- Minimal PEAR features store explicit body, hand, face, and camera parameters.
- They are more structured and expressive than 2D skeletons.
- They are not automatically normalized for downstream modeling.
- `camera/pd_cam` is the model's predicted canonical render camera transform.
- These features can be encoded into a transformer as frame tokens, part tokens, and optionally a clip-level static token.
- For SLT, `--minimal --fp16` is a practical storage-efficient choice when hidden features are not needed.
