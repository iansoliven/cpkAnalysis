"""
Tests for outlier filtering functionality.

Tests cover:
- IQR and standard deviation methods
- NaN and Inf value preservation
- Edge cases (zero variance, single values)
- Multiple file groups
- Negative k values
"""
import numpy as np
import pandas as pd
import pytest

from cpkanalysis.outliers import apply_outlier_filter


def test_iqr_basic_outlier_removal():
    """Test that IQR method correctly removes outliers."""
    data = pd.DataFrame({
        'file': ['test.stdf'] * 100,
        'test_name': ['VDD'] * 100,
        'test_number': ['1'] * 100,
        'value': list(range(90)) + [1000] * 10  # 10 clear outliers
    })

    filtered, summary = apply_outlier_filter(data, 'iqr', 1.5)

    assert summary['method'] == 'iqr'
    assert summary['k'] == 1.5
    assert summary['removed'] == 10
    assert len(filtered) == 90


def test_stdev_basic_outlier_removal():
    """Test that standard deviation method correctly removes outliers."""
    # Create data with clear outliers
    data = pd.DataFrame({
        'file': ['test.stdf'] * 100,
        'test_name': ['VDD'] * 100,
        'test_number': ['1'] * 100,
        'value': [50.0] * 95 + [10.0, 90.0, 100.0, 5.0, 95.0]  # 5 outliers
    })

    filtered, summary = apply_outlier_filter(data, 'stdev', 2.0)

    assert summary['method'] == 'stdev'
    assert summary['k'] == 2.0
    assert summary['removed'] == 5
    assert len(filtered) == 95


def test_nan_values_preserved():
    """Test that NaN values are preserved during outlier filtering."""
    data = pd.DataFrame({
        'file': ['test.stdf'] * 12,
        'test_name': ['VDD'] * 12,
        'test_number': ['1'] * 12,
        'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, np.nan, np.nan, 100.0]
    })

    filtered, summary = apply_outlier_filter(data, 'iqr', 1.5)

    # 100.0 should be removed as outlier, NaN values should be preserved
    assert filtered['value'].isna().sum() == 2
    assert 100.0 not in filtered['value'].values
    assert summary['removed'] == 1  # Only 100.0 removed


def test_inf_values_preserved():
    """Test that Inf values are preserved during outlier filtering."""
    data = pd.DataFrame({
        'file': ['test.stdf'] * 12,
        'test_name': ['VDD'] * 12,
        'test_number': ['1'] * 12,
        'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, np.inf, -np.inf, 100.0]
    })

    filtered, summary = apply_outlier_filter(data, 'iqr', 1.5)

    # 100.0 should be removed as outlier, Inf values should be preserved
    assert np.isinf(filtered['value']).sum() == 2
    assert 100.0 not in filtered['value'].values
    assert summary['removed'] == 1  # Only 100.0 removed


def test_zero_variance_keeps_all_values():
    """Test that zero variance (all same values) keeps all data."""
    data = pd.DataFrame({
        'file': ['test.stdf'] * 50,
        'test_name': ['VDD'] * 50,
        'test_number': ['1'] * 50,
        'value': [5.0] * 50
    })

    filtered, summary = apply_outlier_filter(data, 'iqr', 1.5)

    assert len(filtered) == 50
    assert summary['removed'] == 0


def test_single_value_per_group():
    """Test that groups with single values are preserved."""
    data = pd.DataFrame({
        'file': ['test.stdf'],
        'test_name': ['VDD'],
        'test_number': ['1'],
        'value': [5.0]
    })

    filtered, summary = apply_outlier_filter(data, 'iqr', 1.5)

    assert len(filtered) == 1
    assert summary['removed'] == 0


def test_negative_k_returns_original():
    """Test that negative k value disables filtering."""
    data = pd.DataFrame({
        'file': ['test.stdf'] * 10,
        'test_name': ['VDD'] * 10,
        'test_number': ['1'] * 10,
        'value': [1, 2, 3, 4, 5, 100, 200, 300, 400, 500]
    })

    filtered, summary = apply_outlier_filter(data, 'iqr', -1.0)

    assert len(filtered) == 10
    assert summary['method'] == 'none'
    assert summary['removed'] == 0


def test_zero_k_returns_original():
    """Test that zero k value disables filtering."""
    data = pd.DataFrame({
        'file': ['test.stdf'] * 10,
        'test_name': ['VDD'] * 10,
        'test_number': ['1'] * 10,
        'value': [1, 2, 3, 4, 5, 100, 200, 300, 400, 500]
    })

    filtered, summary = apply_outlier_filter(data, 'iqr', 0.0)

    assert len(filtered) == 10
    assert summary['method'] == 'none'
    assert summary['removed'] == 0


def test_none_method_returns_original():
    """Test that 'none' method disables filtering."""
    data = pd.DataFrame({
        'file': ['test.stdf'] * 10,
        'test_name': ['VDD'] * 10,
        'test_number': ['1'] * 10,
        'value': [1, 2, 3, 4, 5, 100, 200, 300, 400, 500]
    })

    filtered, summary = apply_outlier_filter(data, 'none', 1.5)

    assert len(filtered) == 10
    assert summary['method'] == 'none'
    assert summary['removed'] == 0


def test_empty_dataframe():
    """Test that empty DataFrames are handled gracefully."""
    data = pd.DataFrame({
        'file': [],
        'test_name': [],
        'test_number': [],
        'value': []
    })

    filtered, summary = apply_outlier_filter(data, 'iqr', 1.5)

    assert len(filtered) == 0
    assert summary['method'] == 'none'
    assert summary['removed'] == 0


def test_multiple_file_groups():
    """Test that filtering is applied independently to each file group."""
    data = pd.DataFrame({
        'file': ['test1.stdf'] * 50 + ['test2.stdf'] * 50,
        'test_name': ['VDD'] * 100,
        'test_number': ['1'] * 100,
        'value': list(range(50)) + list(range(50))
    })

    filtered, summary = apply_outlier_filter(data, 'iqr', 1.5)

    # Each group should be filtered independently
    assert len(filtered) == 100  # No outliers in this clean data
    assert summary['removed'] == 0


def test_multiple_test_groups():
    """Test that filtering is applied independently to each test group."""
    data = pd.DataFrame({
        'file': ['test.stdf'] * 200,
        'test_name': ['VDD'] * 100 + ['VSS'] * 100,
        'test_number': ['1'] * 100 + ['2'] * 100,
        'value': list(range(100)) + list(range(100))
    })

    filtered, summary = apply_outlier_filter(data, 'iqr', 1.5)

    # Each test should be filtered independently
    assert len(filtered) == 200
    assert summary['removed'] == 0


def test_non_numeric_values_coerced():
    """Test that non-numeric values are handled gracefully with coercion."""
    data = pd.DataFrame({
        'file': ['test.stdf'] * 5,
        'test_name': ['VDD'] * 5,
        'test_number': ['1'] * 5,
        'value': ['bad', 'data', '3.0', '4.0', '5.0']
    })

    # Non-numeric strings will be coerced to NaN and preserved
    filtered, summary = apply_outlier_filter(data, 'iqr', 1.5)

    # 'bad' and 'data' become NaN and are preserved
    # '3.0', '4.0', '5.0' are converted to numbers
    assert len(filtered) >= 3  # At least the valid numbers should remain


def test_mixed_finite_and_nonfinite():
    """Test comprehensive case with NaN, Inf, and regular outliers."""
    data = pd.DataFrame({
        'file': ['test.stdf'] * 15,
        'test_name': ['VDD'] * 15,
        'test_number': ['1'] * 15,
        'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                  np.nan, np.inf, -np.inf, 1000.0, 2000.0]  # 2 outliers
    })

    filtered, summary = apply_outlier_filter(data, 'iqr', 1.5)

    # NaN and Inf should be preserved, 1000.0 and 2000.0 should be removed
    assert filtered['value'].isna().sum() >= 1  # At least the original NaN
    assert np.isinf(filtered['value']).sum() >= 2  # Both Inf values
    assert 1000.0 not in filtered['value'].values
    assert 2000.0 not in filtered['value'].values
    assert summary['removed'] == 2


def test_iqr_with_k_values():
    """Test IQR method with different k values."""
    data = pd.DataFrame({
        'file': ['test.stdf'] * 100,
        'test_name': ['VDD'] * 100,
        'test_number': ['1'] * 100,
        'value': list(range(90)) + [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
    })

    # k=1.5 (standard)
    filtered_15, summary_15 = apply_outlier_filter(data, 'iqr', 1.5)

    # k=3.0 (more permissive)
    filtered_30, summary_30 = apply_outlier_filter(data, 'iqr', 3.0)

    # More permissive k should remove fewer outliers
    assert summary_15['removed'] >= summary_30['removed']
    assert len(filtered_15) <= len(filtered_30)


def test_stdev_with_k_values():
    """Test standard deviation method with different k values."""
    data = pd.DataFrame({
        'file': ['test.stdf'] * 100,
        'test_name': ['VDD'] * 100,
        'test_number': ['1'] * 100,
        'value': [50.0] * 90 + [10.0, 20.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
    })

    # k=2.0 (standard)
    filtered_20, summary_20 = apply_outlier_filter(data, 'stdev', 2.0)

    # k=3.0 (more permissive)
    filtered_30, summary_30 = apply_outlier_filter(data, 'stdev', 3.0)

    # More permissive k should remove fewer outliers
    assert summary_20['removed'] >= summary_30['removed']
    assert len(filtered_20) <= len(filtered_30)
