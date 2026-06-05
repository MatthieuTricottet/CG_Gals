from src.utils import labels_utils as lu


def test_formatted_text_label_omits_units_for_prose():
    assert lu.formatted_label("Vdisp") == r"$\sigma_v$ (km s$^{-1}$)"
    assert lu.formatted_text_label("Vdisp") == r"$\sigma_v$"

    assert lu.formatted_label("size_Group_Bary_kpc") == r"$\left\langle R_{ij}\right\rangle$ (kpc)"
    assert lu.formatted_text_label("size_Group_Bary_kpc") == r"$\left\langle R_{ij}\right\rangle$"


def test_formatted_unit_returns_only_value_units():
    assert lu.formatted_unit("Vdisp") == r"km s$^{-1}$"
    assert lu.formatted_unit("size_Group_Bary_kpc") == "kpc"
    assert lu.formatted_unit("Lum_group") == r"$L_\odot$"
    assert lu.formatted_unit("Offset_Bary") == ""
