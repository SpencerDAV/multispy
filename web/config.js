/* Which front end this page talks to.
 *
 * The local Python server serves this file unchanged. The static build replaces
 * it with `pyodide`, which makes the page load CPython and run the same
 * `multispy` package in-process -- see tools/build_static.py and backend.js.
 *
 * Stated explicitly rather than sniffed by probing for /api/state, so the page
 * never has to interpret a 404 as an architecture.
 */
window.MULTISPY_MODE = 'server';
