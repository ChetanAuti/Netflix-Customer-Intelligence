"""Minimal Streamlit stub: lets us execute each page script to catch logic errors without streamlit installed."""
import sys, types
from contextlib import contextmanager

_ST = {}
class Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name):
        if name in _ST: return _ST[name]
        return lambda *a, **k: Ctx()
    def __iter__(self): return iter([Ctx(), Ctx()])
class Session(dict):
    __getattr__ = dict.get
calls = []
def mk():
    m = types.ModuleType("streamlit")
    def cache_data(*a, **k):
        if a and callable(a[0]): return a[0]
        return lambda f: f
    m.cache_data = cache_data
    m.session_state = Session()
    m.sidebar = Ctx()
    def columns(n, **k): return [Ctx() for _ in range(n if isinstance(n, int) else len(n))]
    def tabs(names): return [Ctx() for _ in names]
    m.columns, m.tabs = columns, tabs
    m.selectbox = lambda label, options=(), index=0, **k: list(options)[index] if len(list(options)) else None
    m.multiselect = lambda *a, **k: []
    m.radio = lambda label, options, index=0, **k: options[index]
    m.text_input = lambda label, value="", **k: value
    m.date_input = lambda label, value=None, **k: value
    m.button = lambda *a, **k: True
    m.expander = lambda *a, **k: Ctx()
    m.spinner = lambda *a, **k: Ctx()
    def default(name):
        def f(*a, **k): calls.append(name); return Ctx()
        return f
    for n in ["set_page_config", "markdown", "title", "subheader", "caption", "write", "metric", "dataframe", "bar_chart", "line_chart", "code",
              "info", "success", "warning", "error", "json", "image", "divider", "download_button", "table"]:
        setattr(m, n, default(n))
    _ST.update({k: v for k, v in vars(m).items() if callable(v)})
    return m
