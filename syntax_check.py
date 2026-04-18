import ast

files = [
    "src/trace_aml/store/vector_store.py",
    "src/trace_aml/service/app.py",
    "src/trace_aml/store/portrait_store.py",
    "src/trace_aml/pipeline/session.py",
]
for f in files:
    with open(f, encoding="utf-8") as fh:
        src = fh.read()
    try:
        ast.parse(src)
        print("OK", f)
    except SyntaxError as e:
        print("FAIL", f, e)
