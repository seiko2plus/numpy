#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

/************************************
 ** Private Declarations
 ************************************/
// convert Python int and float to simd_data
static simd_data
simd__obj2data_scalar(PyObject *obj, simd_data_type dtype);
// convert Python (list, tuple, set, etc) to simd_data
static simd_data
simd__obj2data_sequence(PyObject *obj, simd_data_type dtype);
// convert c scalar to Python scalar object (int, float)
static PyObject *
simd__data2obj_scalar(simd_data data, simd_data_type dtype);
/**
 * allocating aligned memory depend on SIMD width
 * (NPY_SIMD_WIDTH).
 */
static void *
simd__alloc(size_t size);

static size_t
simd__alloc_size(void *ptr);

// free allocated aligned memory
static void
simd__free(void *ptr);

/************************************
 ** Protected Definitions
 ************************************/
static simd_data
simd_obj2data(PyObject *obj, simd_data_type dtype)
{
    if (simd_data_is_scalar(dtype)) {
        return simd__obj2data_scalar(obj, dtype);
    }
    if (simd_data_is_sequence(dtype)) {
        return simd__obj2data_sequence(obj, dtype);
    }
    if (simd_data_is_vector(dtype)) {
        return simd_vector2data((simd_vector*)obj, dtype);
    }
    simd_data data = {.u64 = 0};
    int is_x2 = simd_data_is_vectorx2(dtype);
    int is_x3 = simd_data_is_vectorx3(dtype);
    if (is_x2 || is_x3) {
        int tuple_len = is_x3 ? 3 : 2;
        simd_data_type vdtype = simd_data_cvt2vector(dtype);
        if (!PyTuple_Check(obj) || PyTuple_GET_SIZE(obj) != tuple_len) {
            PyErr_Format(PyExc_TypeError,
                "a tuple of %d vector type %s is required",
                tuple_len, simd_data_getinfo(vdtype)->pyname
            );
            return data;
        }
        for (int i = 0; i < tuple_len; ++i) {
            PyObject *item = PyTuple_GET_ITEM(obj, i);
            if (item == NULL) {
                PyErr_Format(PyExc_RuntimeError,
                    "unable to get item %d, expected to get a %s",
                    i, simd_data_getinfo(vdtype)->pyname
                );
                return data;
            }
            // let the compiler casting for us
            if (i == 2) {
                data.vu64x3.val[2] = simd_vector2data((simd_vector*)item, dtype).vu64;
            } else {
                data.vu64x2.val[i] = simd_vector2data((simd_vector*)item, dtype).vu64;
            }
            // TODO: improve log add item number
            if (PyErr_Occurred()) {
                return data;
            }
        }
        return data;
    }
    PyErr_Format(PyExc_RuntimeError,
        "unhandled obj2data type id:%d, name:%s",
        dtype, simd_data_getinfo(dtype)->pyname
    );
    return data;
}

static PyObject *
simd_data2obj(simd_data data, simd_data_type dtype)
{
    if (simd_data_is_scalar(dtype)) {
        return simd__data2obj_scalar(data, dtype);
    }
    if (simd_data_is_sequence(dtype)) {
        int lane_size = simd_data_getinfo(dtype)->lane_size;
        Py_ssize_t dseq_size = simd__alloc_size(data.qu8) / lane_size;
        PyObject *list = PyList_New(dseq_size);
        if (list == NULL) {
            return NULL;
        }
        if (!simd_data2seq(list, data, dtype)) {
            Py_DECREF(list);
            return NULL;
        }
        return list;
    }
    if (simd_data_is_vector(dtype)) {
        return (PyObject*)simd_data2vector(data, dtype);
    }
    int is_x2 = simd_data_is_vectorx2(dtype);
    int is_x3 = simd_data_is_vectorx3(dtype);
    if (is_x2 || is_x3) {
        int tuple_len = is_x3 ? 3 : 2;
        PyObject *tuple = PyTuple_New(tuple_len);
        if (tuple == NULL) {
            return NULL;
        }
        simd_data_type vdtype = simd_data_cvt2vector(dtype);
        for (int i = 0; i < tuple_len; ++i) {
            simd_data vdata;
            if (i == 2) {
                vdata.vu64 = data.vu64x3.val[2];
            } else {
                vdata.vu64 = data.vu64x2.val[i];
            }
            PyObject *item = (PyObject*)simd_data2vector(vdata, vdtype);
            if (item == NULL) {
                // TODO: improve log add item number
                Py_DECREF(tuple);
                return NULL;
            }
            PyTuple_SET_ITEM(tuple, i, item);
        }
        return tuple;
    }
    /**end repeat**/
    PyErr_Format(PyExc_RuntimeError,
        "unhandled data2obj type id:%d, name:%s", dtype,
        simd_data_getinfo(dtype)->pyname
    );
    return NULL;
}

static int
simd_data2seq(PyObject *seq, simd_data data, simd_data_type dtype)
{
    if (!simd_data_is_sequence(dtype) || data.qu8 == NULL) {
        PyErr_Format(PyExc_RuntimeError,
            "expected a sequence data type, given(%s:%d)",
            simd_data_getinfo(dtype)->pyname, dtype
        );
        return 0;
    }
    if (!PySequence_Check(seq)) {
        PyErr_Format(PyExc_TypeError,
            "a sequence %s is required",
            simd_data_getinfo(dtype)->pyname
        );
        return 0;
    }
    Py_ssize_t seq_size = PySequence_Size(seq);
    if (seq_size < 0) {
        PyErr_Format(PyExc_RuntimeError,
            "couldn't get the sequence size"
        );
        return 0;
    }
    int lane_size = simd_data_getinfo(dtype)->lane_size;
    Py_ssize_t dseq_size = simd__alloc_size(data.qu8) / lane_size;
    seq_size = seq_size > dseq_size ? dseq_size : seq_size;

    simd_data_type stype = simd_data_cvt2scalar(dtype);
    for (int i = 0; i < seq_size; ++i) {
        simd_data d = {.u64 = *((npy_uint64*)(data.qu8 + i * lane_size))};
        PyObject *item = simd__data2obj_scalar(d, stype);
        if (item == NULL) {
            return 0;
        }
        if (PySequence_SetItem(seq, i, item) < 0) {
            Py_DECREF(item);
            return 0;
        }
    }
    return 1;
}

static void
simd_obj2data_clear(simd_data data, simd_data_type dtype)
{
    if (simd_data_is_sequence(dtype) && data.qu8 != NULL) {
        simd__free(data.qu8);
    }
}

/************************************
 ** Private Definitions
 ************************************/
static simd_data
simd__obj2data_scalar(PyObject *obj, simd_data_type dtype)
{
    simd_data data;
    switch(dtype) {
    case simd_data_s8:
        data.s8 = (npyv_lanetype_s8)PyLong_AsUnsignedLongLongMask(obj);
        break;
    case simd_data_s16:
        data.s16 = (npyv_lanetype_s16)PyLong_AsUnsignedLongLongMask(obj);
        break;
    case simd_data_s32:
        data.s32 = (npyv_lanetype_s32)PyLong_AsUnsignedLongLongMask(obj);
        break;
    case simd_data_s64:
        data.s64 = (npyv_lanetype_s64)PyLong_AsUnsignedLongLongMask(obj);
        break;
    case simd_data_f32:
        data.f32 = (float)PyFloat_AsDouble(obj);
        break;
    case simd_data_f64:
        data.f64 = PyFloat_AsDouble(obj);
        break;
    default:
        data.u64 = PyLong_AsUnsignedLongLongMask(obj);
    }
    return data;
}

static simd_data
simd__obj2data_sequence(PyObject *obj, simd_data_type dtype)
{
    simd_data data = {.qu8 = NULL};
    PyObject *seq_obj = PySequence_Fast(obj, "expected a sequence");
    if (seq_obj == NULL) {
        return data;
    }
    Py_ssize_t seq_size = PySequence_Fast_GET_SIZE(seq_obj);
    const simd_data_info *dinfo = simd_data_getinfo(dtype);
    if (seq_size < dinfo->nlanes) {
        PyErr_Format(PyExc_ValueError,
            "minimum acceptable size of the sequence is %d, given(%d)",
            dinfo->nlanes, seq_size
        );
        Py_DECREF(seq_obj);
        return data;
    }
    data.qu8 = simd__alloc(
        seq_size * dinfo->lane_size
    );
    if (data.qu8 == NULL) {
        Py_DECREF(seq_obj);
        return data;
    }
    PyObject **seq_items = PySequence_Fast_ITEMS(seq_obj);
    for (Py_ssize_t i = 0; i < seq_size; ++i) {
        PyObject *item = seq_items[i];
        switch (dtype) {
        case simd_data_qs8:
            data.qs8[i] = (npyv_lanetype_s8)PyLong_AsUnsignedLongLongMask(item);
            break;
        case simd_data_qs16:
            data.qs16[i] = (npyv_lanetype_s16)PyLong_AsUnsignedLongLongMask(item);
            break;
        case simd_data_qs32:
            data.qs32[i] = (npyv_lanetype_s32)PyLong_AsUnsignedLongLongMask(item);
            break;
        case simd_data_qs64:
            data.qs64[i] = (npyv_lanetype_s64)PyLong_AsUnsignedLongLongMask(item);
            break;
        case simd_data_qu16:
            data.qu16[i] = (npyv_lanetype_u16)PyLong_AsUnsignedLongLongMask(item);
            break;
        case simd_data_qu32:
            data.qu32[i] = (npyv_lanetype_u32)PyLong_AsUnsignedLongLongMask(item);
            break;
        case simd_data_qu64:
            data.qu64[i] = PyLong_AsUnsignedLongLongMask(item);
            break;
        case simd_data_qf32:
            data.qf32[i] = (float)PyFloat_AsDouble(item);
            break;
        case simd_data_qf64:
            data.qf64[i] = PyFloat_AsDouble(item);
            break;
        default:
            data.qu8[i] = (npyv_lanetype_u8)PyLong_AsUnsignedLongLongMask(item);
        }
    }
    if (PyErr_Occurred()) {
        simd__free(data.qu8);
        data.qu8 = NULL;
    }
    Py_DECREF(seq_obj);
    return data;
}

static PyObject *
simd__data2obj_scalar(simd_data data, simd_data_type dtype)
{
    switch(dtype) {
    case simd_data_u8:
        return PyLong_FromUnsignedLong(data.u8);
    case simd_data_u16:
        return PyLong_FromUnsignedLong(data.u16);
    case simd_data_u32:
        return PyLong_FromUnsignedLong(data.u32);
    case simd_data_u64:
        return PyLong_FromUnsignedLongLong(data.u64);
    case simd_data_s8:
        return PyLong_FromLong(data.s8);
    case simd_data_s16:
        return PyLong_FromLong(data.s16);
    case simd_data_s32:
        return PyLong_FromLong(data.s16);
    case simd_data_s64:
        return PyLong_FromLongLong(data.s64);
    case simd_data_f32:
        return PyFloat_FromDouble(data.f32);
    case simd_data_f64:
        return PyFloat_FromDouble(data.f64);
    default:
        PyErr_Format(PyExc_RuntimeError,
            "unhandled scalar type id:%d, name:%s", dtype,
            simd_data_getinfo(dtype)->pyname
        );
    }
    return NULL;
}

static void *
simd__alloc(size_t size)
{
    size_t *ptr = malloc(size + NPY_SIMD_WIDTH + sizeof(size_t) + sizeof(size_t*));
    if (ptr == NULL) {
        return PyErr_NoMemory();
    }
    *(ptr++) = size;
    size_t **a_ptr = (size_t**)(
        ((size_t)ptr + NPY_SIMD_WIDTH) & ~(size_t)(NPY_SIMD_WIDTH-1)
    );
    a_ptr[-1] = ptr;
    return a_ptr;
}

static size_t
simd__alloc_size(void *ptr)
{
    size_t *ptrz = ((size_t**)ptr)[-1];
    return *(ptrz-1);
}

static void
simd__free(void *ptr)
{
    size_t *ptrz = ((size_t**)ptr)[-1];
    free(ptrz-1);
}
