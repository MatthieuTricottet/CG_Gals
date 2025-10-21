import numpy as np


#* safe_log_ratio() avoids invalid operations like log(0/0) by returning NaN
def safe_log_ratio(numerator, denominator):
    """Compute log10(numerator/denominator) safely, returning NaN when invalid."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(
            (denominator > 0) & (numerator > 0),
            np.log10(numerator / denominator),
            np.nan
        )