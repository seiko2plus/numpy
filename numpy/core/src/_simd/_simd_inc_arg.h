#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

/************************************
 ** Protected Definitions
 ************************************/
static int
simd_arg_parse(PyObject *py_args, simd_arg *args, int args_n)
{
    Py_ssize_t cur_arg = 0;
    if (py_args == NULL) {
        if (args_n == 0) {
            return 1;
        }
        goto wrong_size;
    }
    int arg_pos = 0;
    if (!PyTuple_Check(py_args)) {
        if (args_n != 0) {
            goto wrong_size;
        }
        simd_arg *a = &args[0]; a->obj = py_args;
        a->data = simd_obj2data(py_args, a->type);
        if (PyErr_Occurred()) {
            goto wrong_data;
        }
        return 1;
    }
    cur_arg = PyTuple_GET_SIZE(py_args);
    if (cur_arg != args_n) {
        goto wrong_size;
    }
    for (; arg_pos < args_n; ++arg_pos) {
        simd_arg *a = &args[arg_pos];
        a->obj = PyTuple_GET_ITEM(py_args, arg_pos);
        if (a->obj == NULL) {
            if (!PyErr_Occurred()) {
                PyErr_Format(PyExc_RuntimeError,
                    "unable to get argument '%d'", arg_pos+1
                );
            }
            goto wrong_data;
        }
        a->data = simd_obj2data(a->obj, a->type);
        if (PyErr_Occurred()) {
            goto wrong_data;
        }
    }
    return 1;
wrong_size:
    PyErr_Format(PyExc_TypeError,
        "expected %d argument (%d given)", args_n, cur_arg
    );
    return 0;
wrong_data:
    // TODO: print
    return 0;
}

static void
simd_arg_clear(simd_arg *args, int args_n)
{
    for (int i = 0; i < args_n; ++i) {
        simd_arg *a = &args[i];
        simd_obj2data_clear(a->data, a->type);
    }
}
