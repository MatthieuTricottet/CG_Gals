from src.utils import graphics_utils as gu


def test_latex_number_uses_latex_scientific_notation():
    assert gu.latex_number(9.100024523062283e10, precision=3) == r"$9.1\times 10^{10}$"
    assert gu.latex_number(1.2809854309366843e11, precision=3) == r"$1.28\times 10^{11}$"
    assert gu.latex_number(7.800612032865688e-05, precision=2) == r"$7.8\times 10^{-5}$"


def test_latex_number_keeps_plain_decimal_for_moderate_values():
    assert gu.latex_number(123.456, precision=3) == r"$123$"
    assert gu.latex_number(12.3456, precision=3) == r"$12.3$"
    assert gu.latex_number(0.0123456, precision=3) == r"$0.0123$"


def test_latex_float_remains_unwrapped_for_axis_labels():
    assert gu.latex_float(9.100024523062283e10) == r"9.1\times 10^{10}"
