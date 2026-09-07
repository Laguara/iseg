from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.data import (
    RAW_TO_INTERNAL_LABELS,
    load_analyze_volume,
    remap_labels,
    remove_trailing_singleton_dimension,
    validate_matching_shapes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_loads_synthetic_analyze_volume(tmp_path: Path) -> None:
    values = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    header_path = tmp_path / "sample.hdr"
    nib.AnalyzeImage(values, affine=np.eye(4)).to_filename(str(header_path))

    loaded = load_analyze_volume(header_path)

    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == values.shape
    np.testing.assert_array_equal(loaded, values)


def test_removes_only_trailing_singleton_dimension() -> None:
    values = np.arange(6, dtype=np.int16).reshape(1, 2, 3, 1)

    loaded = remove_trailing_singleton_dimension(values)

    assert loaded.shape == (1, 2, 3)
    np.testing.assert_array_equal(loaded, values[..., 0])


def test_leaves_volume_without_trailing_singleton_unchanged() -> None:
    values = np.arange(24, dtype=np.int16).reshape(2, 3, 4)

    loaded = remove_trailing_singleton_dimension(values)

    assert loaded is values
    assert loaded.shape == values.shape
    np.testing.assert_array_equal(loaded, values)


def test_accepts_matching_training_shapes() -> None:
    t1 = np.zeros((2, 3, 4))
    t2 = np.ones((2, 3, 4))
    labels = np.zeros((2, 3, 4), dtype=np.int16)

    validate_matching_shapes(t1, t2, labels)


def test_accepts_matching_test_modalities_without_labels() -> None:
    t1 = np.zeros((2, 3, 4))
    t2 = np.ones((2, 3, 4))

    validate_matching_shapes(t1, t2)


def test_rejects_mismatched_modalities() -> None:
    t1 = np.zeros((2, 3, 4))
    t2 = np.ones((2, 5, 4))

    with pytest.raises(ValueError, match=r"T1=\(2, 3, 4\).*T2=\(2, 5, 4\)"):
        validate_matching_shapes(t1, t2)


def test_rejects_mismatched_labels() -> None:
    t1 = np.zeros((2, 3, 4))
    t2 = np.ones((2, 3, 4))
    labels = np.zeros((2, 3, 5), dtype=np.int16)

    with pytest.raises(ValueError, match=r"labels=\(2, 3, 5\)"):
        validate_matching_shapes(t1, t2, labels)


def test_remaps_raw_labels_to_internal_class_indices() -> None:
    labels = np.array([[0, 10], [150, 250]], dtype=np.uint8)
    original = labels.copy()

    remapped = remap_labels(labels)

    np.testing.assert_array_equal(remapped, np.array([[0, 1], [2, 3]], dtype=np.uint8))
    assert remapped.shape == labels.shape
    assert remapped.dtype == np.uint8
    np.testing.assert_array_equal(labels, original)


def test_rejects_unknown_label_values() -> None:
    labels = np.array([[0, 42]], dtype=np.uint8)

    with pytest.raises(ValueError, match=r"Unknown label values: \[42\]"):
        remap_labels(labels)


def test_rejects_missing_header(tmp_path: Path) -> None:
    header_path = tmp_path / "missing.hdr"

    with pytest.raises(FileNotFoundError, match="header file not found"):
        load_analyze_volume(header_path)


def test_rejects_missing_image(tmp_path: Path) -> None:
    header_path = tmp_path / "sample.hdr"
    header_path.write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="image file not found"):
        load_analyze_volume(header_path)


def test_rejects_non_header_path(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.img"

    with pytest.raises(ValueError, match="Expected an Analyze .hdr path"):
        load_analyze_volume(image_path)


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("data/training/subject-1-T1.hdr"),
        Path("data/testing/subject-23-T1.hdr"),
        Path("data/testing/subject-23-T2.hdr"),
    ],
)
def test_loads_real_analyze_volumes(relative_path: Path) -> None:
    header_path = PROJECT_ROOT / relative_path
    if not header_path.is_file():
        pytest.skip(f"Local dataset not available: {header_path}")

    loaded = load_analyze_volume(header_path)

    assert isinstance(loaded, np.ndarray)
    assert loaded.ndim >= 2
    assert loaded.size > 0
    assert np.issubdtype(loaded.dtype, np.number)

    reference = np.asanyarray(nib.load(str(header_path)).dataobj)
    expected = remove_trailing_singleton_dimension(reference)
    assert loaded.shape == expected.shape
    np.testing.assert_array_equal(loaded, expected)


def test_validates_training_subject_1() -> None:
    relative_paths = (
        Path("data/training/subject-1-T1.hdr"),
        Path("data/training/subject-1-T2.hdr"),
        Path("data/training/subject-1-label.hdr"),
    )
    header_paths = [PROJECT_ROOT / relative_path for relative_path in relative_paths]
    if not all(path.is_file() for path in header_paths):
        pytest.skip("Local training dataset not available")

    t1, t2, labels = [load_analyze_volume(path) for path in header_paths]

    validate_matching_shapes(t1, t2, labels)


def test_validates_testing_subject_23_without_labels() -> None:
    relative_paths = (
        Path("data/testing/subject-23-T1.hdr"),
        Path("data/testing/subject-23-T2.hdr"),
    )
    header_paths = [PROJECT_ROOT / relative_path for relative_path in relative_paths]
    if not all(path.is_file() for path in header_paths):
        pytest.skip("Local testing dataset not available")

    t1, t2 = [load_analyze_volume(path) for path in header_paths]

    validate_matching_shapes(t1, t2)


def test_remaps_real_training_labels_subject_1() -> None:
    header_path = PROJECT_ROOT / "data/training/subject-1-label.hdr"
    if not header_path.is_file():
        pytest.skip("Local training dataset not available")

    labels = load_analyze_volume(header_path)
    remapped = remap_labels(labels)

    assert sorted(np.unique(labels).tolist()) == sorted(RAW_TO_INTERNAL_LABELS)
    assert sorted(np.unique(remapped).tolist()) == [0, 1, 2, 3]
    assert remapped.shape == labels.shape
