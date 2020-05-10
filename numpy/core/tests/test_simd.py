import unittest

class _TestSIMD(object):
    """
    Please avoid the use of numpy.testing since NPYV intrinsics
    may be involved in their functionality.
    """
    npyv = None
    sfx  = None
    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName)
        unittest.TestCase.maxDiff = 1024

    def test_memory(self):
        data = self._data()
        simd_load = self.load(data)
        self.assertEqual(simd_load, data)
        simd_loada = self.loada(data)
        self.assertEqual(simd_loada, data)
        simd_loads = self.loads(data)
        self.assertEqual(simd_loads, data)
        # test load lower part
        simd_loadl = self.loadl(data)
        simd_loadl_half = list(simd_loadl)[:self.nlanes//2]
        data_half = data[:self.nlanes//2]
        self.assertEqual(simd_loadl_half, data_half)
        self.assertNotEqual(simd_loadl, data) # detect overflow

        vdata   = self.load(data)
        store   = [0] * self.nlanes
        self.store(store, vdata)
        self.assertEqual(store, data)
        store_a = [0] * self.nlanes
        self.storea(store_a, vdata)
        self.assertEqual(store_a, data)
        store_s = [0] * self.nlanes
        self.stores(store_s, vdata)
        self.assertEqual(store_s, data)
        # test store lower part
        store_l = [0] * self.nlanes
        self.storel(store_l, vdata)
        self.assertEqual(store_l[:self.nlanes//2], data[:self.nlanes//2])
        self.assertNotEqual(store_l, data) # detect overflow
        # test store higher part
        store_h = [0] * self.nlanes
        self.storeh(store_h, vdata)
        self.assertEqual(store_h[:self.nlanes//2], data[self.nlanes//2:])
        self.assertNotEqual(store_h, data) # detect overflow

    def test_misc(self):
        self.assertEqual(self.zero(),    [0] * self.nlanes)
        self.assertEqual(self.setall(1), [1] * self.nlanes)

        data_a  = self._data()
        data_b  = self._data(reverse=True)
        vdata_a = self.load(data_a)
        vdata_b = self.load(data_b)
        select_a = self.select(self.cmpeq(self.zero(), self.zero()), vdata_a, vdata_b)
        self.assertEqual(select_a, data_a)
        select_b = self.select(self.cmpneq(self.zero(), self.zero()), vdata_a, vdata_b)
        self.assertEqual(select_b, data_b)

        # We're testing the sainty of _simd's type-vector,
        # reinterpret* intrinsics itself are tested via compiler
        sfxes = ["u8", "s8", "u16", "s16", "u32", "s32", "u64", "s64", "f32"]
        if self.npyv.simd_f64:
            sfxes.append("f64")
        for sfx in sfxes:
            vec_name = getattr(self, "reinterpret_" + sfx)(vdata_a).__name__
            self.assertEqual(vec_name, "npyv_" + sfx)

        simd_set = self.set(*data_a)
        self.assertEqual(simd_set, data_a)
        simd_setf = self.setf(10, *data_a)
        self.assertEqual(simd_setf, data_a)

        # cleanup intrinsic is only used with AVX for
        # zeroing registers to avoid the AVX-SSE transition penalty,
        # so nothing to test here
        self.npyv.cleanup()

    def test_reorder(self):
        data_a  = self._data()
        data_al = data_a[:self.nlanes//2]
        data_ah = data_a[self.nlanes//2:]
        data_b  = self._data(reverse=True)
        data_bl = data_b[:self.nlanes//2]
        data_bh = data_b[self.nlanes//2:]
        vdata_a = self.load(data_a)
        vdata_b = self.load(data_b)

        data_combinel = data_al + data_bl
        simd_combinel = self.combinel(vdata_a, vdata_b)
        self.assertEqual(simd_combinel, data_combinel)
        data_combineh = data_ah + data_bh
        simd_combineh = self.combineh(vdata_a, vdata_b)
        self.assertEqual(simd_combineh, data_combineh)
        simd_combine  = self.combine(vdata_a, vdata_b)
        self.assertEqual(simd_combine, (data_combinel, data_combineh))

        data_zipl = [v for p in zip(data_al, data_bl) for v in p]
        data_ziph = [v for p in zip(data_ah, data_bh) for v in p]
        simd_zip  = self.zip(vdata_a, vdata_b)
        self.assertEqual(simd_zip, (data_zipl, data_ziph))

    def test_operators(self):
        if self._is_precision():
            data_a = self._data()
            data_b = self._data(reverse=True)
        else:
            data_a = self._data(self._int_max() - self.nlanes)
            data_b = self._data(self._int_min(), reverse=True)

        vdata_a = self.load(data_a)
        vdata_b = self.load(data_b)

        ## shifting
        if self.sfx not in ("u8", "s8", "f32", "f64"):
            for count in range(self._scalar_size()):
                data_shl_a = self.load([a << count for a in data_a]) # load to cast
                self.assertEqual(self.shl(vdata_a, count), data_shl_a)
                self.assertEqual(self.shli(vdata_a, count), data_shl_a)

                data_shr_a = self.load([a >> count for a in data_a])
                self.assertEqual(self.shr(vdata_a, count), data_shr_a)
                self.assertEqual(self.shri(vdata_a, count), data_shr_a)

        ## logical
        if self._is_precision():
            data_cast_a = self._to_unsigned(vdata_a)
            data_cast_b = self._to_unsigned(vdata_b)
            cast = self._to_unsigned
            cast_data = self._to_unsigned
        else:
            data_cast_a = data_a
            data_cast_b = data_b
            cast = lambda a: a
            cast_data = self.load

        data_xor = cast_data([a ^ b for a, b in zip(data_cast_a, data_cast_b)])
        simd_xor = cast(self.xor(vdata_a, vdata_b))
        self.assertEqual(simd_xor, data_xor)
        data_or  = cast_data([a | b for a, b in zip(data_cast_a, data_cast_b)])
        simd_or  = cast(getattr(self, "or")(vdata_a, vdata_b))
        self.assertEqual(simd_or, data_or)
        data_and = cast_data([a & b for a, b in zip(data_cast_a, data_cast_b)])
        simd_and = cast(getattr(self, "and")(vdata_a, vdata_b))
        self.assertEqual(simd_and, data_and)
        data_not = cast_data([~a for a in data_cast_a])
        simd_not = cast(getattr(self, "not")(vdata_a))
        self.assertEqual(simd_not, data_not)

        ## comparison
        def to_bool(vector):
            return [lane != 0 for lane in vector]

        data_eq  = [a == b for a, b in zip(data_a, data_b)]
        simd_eq  = to_bool(self.cmpeq(vdata_a, vdata_b))
        self.assertEqual(simd_eq, data_eq)
        data_neq = [a != b for a, b in zip(data_a, data_b)]
        simd_neq = to_bool(self.cmpneq(vdata_a, vdata_b))
        self.assertEqual(simd_neq, data_neq)
        data_gt  = [a > b for a, b in zip(data_a, data_b)]
        simd_gt  = to_bool(self.cmpgt(vdata_a, vdata_b))
        self.assertEqual(simd_gt, data_gt)
        data_ge  = [a >= b for a, b in zip(data_a, data_b)]
        simd_ge  = to_bool(self.cmpge(vdata_a, vdata_b))
        self.assertEqual(simd_ge, data_ge)
        data_lt  = [a < b for a, b in zip(data_a, data_b)]
        simd_lt  = to_bool(self.cmplt(vdata_a, vdata_b))
        self.assertEqual(simd_lt, data_lt)
        data_le  = [a <= b for a, b in zip(data_a, data_b)]
        simd_le  = to_bool(self.cmple(vdata_a, vdata_b))
        self.assertEqual(simd_le, data_le)

    def test_conversion(self):
        ## boolean vectors
        bsfx = "b" + self.sfx[1:]
        to_boolean = getattr(self.npyv, "cvt_%s_%s" % (bsfx, self.sfx))
        from_boolean = getattr(self.npyv, "cvt_%s_%s" % (self.sfx, bsfx))

        false_vb = to_boolean(self.setall(0))
        true_vb  = self.cmpeq(self.setall(0), self.setall(0))
        self.assertNotEqual(false_vb, true_vb)
        false_vsfx = from_boolean(false_vb)
        true_vsfx = from_boolean(true_vb)
        self.assertNotEqual(false_vsfx, true_vsfx)


    def test_arithmetic(self):
        if self._is_precision():
            data_a = self._data()
            data_b = self._data(reverse=True)
        else:
            data_a = self._data(self._int_max() - self.nlanes)
            data_b = self._data(self._int_min(), reverse=True)

        vdata_a = self.load(data_a)
        vdata_b = self.load(data_b)
        cast_data = self.load

        # non-saturated
        data_add  = cast_data([a + b for a, b in zip(data_a, data_b)])
        simd_add  = self.add(vdata_a, vdata_b)
        self.assertEqual(simd_add, data_add)
        data_sub  = cast_data([a - b for a, b in zip(data_a, data_b)])
        simd_sub  = self.sub(vdata_a, vdata_b)
        self.assertEqual(simd_sub, data_sub)

        if self.sfx not in ("u64", "s64"):
            data_mul = cast_data([a * b for a, b in zip(data_a, data_b)])
            simd_mul = self.mul(vdata_a, vdata_b)
            self.assertEqual(simd_mul, data_mul)

        if self._is_precision():
            data_div = [a / b for a, b in zip(data_a, data_b)]
            simd_div = self.div(vdata_a, vdata_b)
            self._assertAlmostEqual(simd_div, data_div, places=6)

        ## saturated
        if self.sfx not in ("u32", "s32", "u64", "s64", "f32", "f64"):
            data_adds = self._int_crop([a + b for a, b in zip(data_a, data_b)])
            simd_adds = self.adds(vdata_a, vdata_b)
            self.assertEqual(simd_adds, data_adds)
            data_subs = self._int_crop([a - b for a, b in zip(data_a, data_b)])
            simd_subs = self.subs(vdata_a, vdata_b)
            self.assertEqual(simd_subs, data_subs)

    def _data(self, n=1, reverse=False):
        rng = range(n, n + self.nlanes)
        if reverse:
            rng = reversed(rng)
        if self._is_precision():
            return [x / 1.0 for x in rng]
        return list(rng)

    def _vdata(self, n=1):
        return self.load(self._data(n))

    def _scalar_true(self):
        # module '_simd' avoid overflow checking for all integer types
        # so it's okay to pass negative integer
        return self._to_unsigned(self.npyv.setall_u64(-1))[0]

    def _is_unsigned(self):
        return self.sfx[0] == 'u'

    def _is_signed(self):
        return self.sfx[0] == 's'

    def _is_precision(self):
        return self.sfx[0] == 'f'

    def _scalar_size(self):
        return int(self.sfx[1:])

    def _int_crop(self, seq):
        if self._is_precision():
            return seq
        max_int = self._int_max()
        min_int = self._int_min()
        return [min(max(v, min_int), max_int) for v in seq]

    def _int_max(self):
        if self._is_precision():
            return 0
        max_u = self._to_unsigned(self.setall(-1))[0]
        if self._is_signed():
            return max_u // 2
        return max_u

    def _int_min(self):
        if self._is_precision():
            return 0
        if self._is_unsigned():
            return 0
        return -(self._int_max() + 1)

    def _to_unsigned(self, vector):
        if isinstance(vector, (list, tuple)):
            return getattr(self.npyv, "load_u" + self.sfx[1:])(vector)
        else:
            sfx = vector.__name__.replace("npyv_", "")
            if sfx[0] == "b":
                cvt_intrin = "cvt_u{0}_b{0}"
            else:
                cvt_intrin = "reinterpret_u{0}_{1}"
            return getattr(self.npyv, cvt_intrin.format(sfx[1:], sfx))(vector)

    def _assertAlmostEqual(self, seq_a, seq_b, **kwargs):
        for a, b in zip(seq_a, seq_b):
            self.assertAlmostEqual(a, b, **kwargs)

    def __getattr__(self, attr):
        nattr = getattr(self.npyv, attr + "_" + self.sfx)
        if callable(nattr):
            return lambda *args: nattr(*args)
        return nattr

from numpy.core._simd import targets
for name, npyv in targets.items():
    skip = ""
    skip_sfx = dict()
    if not npyv:
        skip = "target '%s' isn't supported by current machine" % name
    elif not npyv.simd:
        skip = "target '%s' isn't supported by NPYV" % name
    elif not npyv.simd_f64:
        skip_sfx["f64"] = "target '%s' doesn't support double-precision"  % name

    sfxes = ["u8", "s8", "u16", "s16", "u32", "s32", "u64", "s64", "f32", "f64"]
    for sfx in sfxes:
        skip_m = skip_sfx.get(sfx, skip)
        if skip_m:
            skip_m = '@unittest.skip("%s")' % skip_m

        the_class = (
            "{skip}\n"
            "class Test_SIMD{simd}_{name}_{sfx}(_TestSIMD, unittest.TestCase):\n"
            "   npyv = {npyv}\n"
            "   sfx  = '{sfx}'\n"
        ).format(
            skip=skip_m, simd=npyv.simd if npyv else '',
            name=name.replace(' ', '__'), # multi-target
            npyv="targets['%s']" % name, sfx=sfx
        )
        exec(the_class)
