from __future__ import annotations
import builtins as __builtins__
__all__: list[str] = ['CuvsException', 'check_cuvs', 'get_last_error_text']
class CuvsException(Exception):
    pass
def check_cuvs(status: cuvsError_t):
    """
    check_cuvs(cuvsError_t status: cuvsError_t)
     Converts a status code into an exception
    """
def get_last_error_text():
    """
    get_last_error_text()
     returns the last error description from the cuvs c-api
    """
__test__: dict = {}
