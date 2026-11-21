import numpy as np

from src.preprocessing.crop_black_borders import crop_black_borders, field_of_view_mask
from src.preprocessing.resize_normalize import looks_preprocessed, prepare_image


def test_crop_removes_black_border_and_keeps_last_row(raw_image):
    cropped = crop_black_borders(raw_image)
    assert cropped.shape[0] < raw_image.shape[0] or cropped.shape[1] < raw_image.shape[1]
    # the retina touches all four sides after cropping
    assert cropped[0].max() > 10 and cropped[-1].max() > 10
    assert cropped[:, 0].max() > 10 and cropped[:, -1].max() > 10


def test_crop_all_black_image_is_returned_unchanged():
    img = np.zeros((50, 50, 3), np.uint8)
    assert crop_black_borders(img).shape == img.shape


def test_fov_mask_covers_retina_not_corners(raw_image):
    mask = field_of_view_mask(crop_black_borders(raw_image))
    assert mask[mask.shape[0] // 2, mask.shape[1] // 2]
    assert not mask[0, 0] and not mask[-1, -1]
    assert 0.5 < mask.mean() < 0.85


def test_detects_raw_vs_preprocessed(raw_image, processed_image):
    assert not looks_preprocessed(raw_image)
    assert looks_preprocessed(processed_image)


def test_prepare_image_shapes(raw_image):
    prep = prepare_image(raw_image, 300, 512)
    assert prep.model_input.shape == (300, 300, 3)
    assert prep.display.shape == (512, 512, 3)
    assert prep.enhanced.shape == (512, 512, 3)
    assert prep.fov_mask.shape == (512, 512) and prep.fov_mask.dtype == bool
    assert prep.already_preprocessed is False


def test_prepare_image_respects_override(raw_image):
    assert prepare_image(raw_image, 300, 512, already_preprocessed=True).already_preprocessed is True
