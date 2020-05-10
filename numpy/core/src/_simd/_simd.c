#include "_simd.h"

void simd__sep_multitarget(char *target)
{
    char *ptr = target;
    int prev_score = 0;
    while(*(ptr++)) {
        if ((*ptr) != '_') {
            prev_score = 0;
            continue;
        }
        if ((*ptr) == '_' && prev_score++) {
            // multi target
            ptr[-1] = ' ';
            strcpy(ptr, ptr + 1);
        }
    }
}

static PyObject *
simd__targets(void)
{
    PyObject *dict = PyDict_New(), *mod;
    if (dict == NULL) {
        return NULL;
    }
    #define LOAD_TARGET(CHK, TNAME, DUMMY)       \
        if (CHK) {                               \
            mod = NPY_CAT(                       \
                simd_module_create_, TNAME       \
            )();                                 \
        } else {                                 \
            Py_INCREF(Py_None);                  \
            mod = Py_None;                       \
        }                                        \
        if (mod == NULL) goto err;               \
        else {                                   \
            char tname[] = NPY_TOSTRING(TNAME);  \
            simd__sep_multitarget(tname);        \
            if (PyDict_SetItemString(            \
                dict, tname, mod) < 0            \
            ) goto err;                          \
        }

    #define LOAD_BASELINE(DUMMY)                 \
        mod = simd_module_create();              \
        if (mod == NULL) goto err;               \
        if (PyDict_SetItemString(                \
            dict, "baseline", mod) < 0           \
        ) goto err;

    NPY__CPU_DISPATCH_CALL(NPY_CPU_HAVE, LOAD_TARGET, DUMMY)
    NPY__CPU_DISPATCH_BASELINE_CALL(LOAD_BASELINE, DUMMY)
    return dict;
err:
    Py_XDECREF(mod);
    Py_DECREF(dict);
    return NULL;
}

PyMODINIT_FUNC PyInit__simd(void)
{
    static struct PyModuleDef defs = {
        .m_base = PyModuleDef_HEAD_INIT,
        .m_name = "_simd",
        .m_size = -1
    };
    if (npy_cpu_init() < 0) {
        return NULL;
    }
    PyObject *m = PyModule_Create(&defs);
    if (m == NULL) {
        goto err;
    }
    PyObject *targets = simd__targets();
    if (targets == NULL) {
        goto err;
    }
    if (PyModule_AddObject(m, "targets", targets) < 0) {
        Py_DECREF(targets);
        goto err;
    }
    return m;
err:
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "cannot load _simd module.");
    }
    Py_XDECREF(m);
    return NULL;
}
