from app.sandbox.run_pandas import run_pandas


def test_rejects_import_builtin():
    assert "error" in run_pandas("__import__('os')")


def test_rejects_import_statement():
    assert "error" in run_pandas("import os")


def test_rejects_file_access():
    assert "error" in run_pandas("open('/etc/passwd')")


def test_rejects_dunder_attribute_walk():
    assert "error" in run_pandas("df.__class__.__mro__")


def test_rejects_eval():
    assert "error" in run_pandas('eval("1+1")')


def test_rejects_exec():
    assert "error" in run_pandas('exec("import os")')


def test_allows_legitimate_groupby():
    result = run_pandas("df.groupby('Contract')['MonthlyCharges'].mean()")
    assert result["result_type"] == "Series"
    assert "Month-to-month" in result["result"]


def test_multi_statement_code_returns_last_expression():
    result = run_pandas("long_tenure = df[df['tenure'] > 60]\nlen(long_tenure)")
    assert result["result"] == "1407"


def test_runtime_error_is_returned_not_raised():
    result = run_pandas("df['does_not_exist'].mean()")
    assert "error" in result and "KeyError" in result["error"]


def test_sandbox_cannot_mutate_shared_frame():
    run_pandas("df['tenure'] = 0")
    assert "72" in run_pandas("df['tenure'].max()")["result"]
