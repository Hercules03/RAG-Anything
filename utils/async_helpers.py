"""
Async Helpers - Streamlit-compatible async execution utilities

Provides safe async execution within Streamlit's event loop context.
Prevents "PriorityQueue is bound to a different event loop" errors.
"""

import asyncio
import sys
from typing import Any, Coroutine


def run_async(coro: Coroutine) -> Any:
    """
    Run async coroutine in Streamlit-compatible way

    This function safely runs async code within Streamlit by:
    1. Reusing existing event loop if available
    2. Creating new loop only when necessary
    3. Properly handling loop lifecycle

    Args:
        coro: Async coroutine to execute

    Returns:
        Result from the coroutine

    Raises:
        Any exception raised by the coroutine
    """
    # Try to get the current event loop
    try:
        loop = asyncio.get_event_loop()

        # Check if loop is running (e.g., in Jupyter/IPython)
        if loop.is_running():
            # If loop is already running, we need to use a different approach
            # Create a new event loop in a separate thread
            import threading
            import queue

            result_queue = queue.Queue()
            exception_queue = queue.Queue()

            def run_in_thread():
                try:
                    # Create new loop for this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(coro)
                        result_queue.put(result)
                    finally:
                        new_loop.close()
                except Exception as e:
                    exception_queue.put(e)

            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()

            # Check for exceptions
            if not exception_queue.empty():
                raise exception_queue.get()

            return result_queue.get()
        else:
            # Loop exists but not running, use it
            return loop.run_until_complete(coro)

    except RuntimeError:
        # No event loop exists, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            # Don't close the loop, might be needed again
            pass


def run_async_safe(coro: Coroutine, fallback_value: Any = None) -> Any:
    """
    Run async coroutine with error handling

    Same as run_async but catches exceptions and returns fallback value

    Args:
        coro: Async coroutine to execute
        fallback_value: Value to return if coroutine fails

    Returns:
        Result from coroutine or fallback_value on error
    """
    try:
        return run_async(coro)
    except Exception as e:
        print(f"Error in async execution: {e}")
        return fallback_value


# For backwards compatibility
async_run = run_async
