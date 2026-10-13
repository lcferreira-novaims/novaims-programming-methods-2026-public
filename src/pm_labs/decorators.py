"""Function decorators from Lab 5 (Functional Python)."""

import functools
import time


def timer(func):
    """Print how long a call to `func` took."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.6f}s")
        return result
    return wrapper


def retry(func):
    """Retry `func` up to 3 times (0.5s apart) if it raises, then re-raise."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(1, 4):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                print(f"attempt {attempt} failed: {exc}")
                if attempt < 3:
                    time.sleep(0.5)
        raise last_exc
    return wrapper
