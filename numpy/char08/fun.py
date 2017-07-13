#encoding=utf8
import numpy as np

'''
assert_almost_equal, 近似程度达不到指定精度
assert_approx_equal, 近似程序达不到指定有效数字
assert_array_almost_equal
assert_array_equal
assert_array_less, 两数组必须形状一致，且第一个数组的元素严格小于第二个数组的元素
assert_equal, 两个对象相同
assert_raises, 没抛出指定的异常，出错
assert_warns
assert_string_equal
assert_allclose
'''

#指定精度的近似相等
try:
    print "Decimal 6:", np.testing.assert_almost_equal(0.123456789, 0.123456780, decimal=7)
    print "Decimal 7:", np.testing.assert_almost_equal(0.123456789, 0.123456780, decimal=8)
except AssertionError, ae:
    print ae

#近似相等
# abs(actual - expected) >= 10**(significant - 1)
try:
    print "Significant 8:", np.testing.assert_approx_equal(0.123456789, 0.123456780, significant=8)
    print "Significant 10:", np.testing.assert_approx_equal(0.123456789, 0.123456780, significant=10)
except AssertionError, ae:
    print ae

#数组近似相等
#1. 两个数组形状相同
#2. 每个元素对： |expected - actual| < 0.5 * 10**(-decimal)
try:
    print "Decimal 8:", np.testing.assert_array_almost_equal([0, 0.123456789], [0, 0.123456780], decimal=8) 
    print "Decimal 9:", np.testing.assert_array_almost_equal([0, 0.123456789], [0, 0.123456780], decimal=9) 
except AssertionError, ae:
    print ae

#assert_allclose: atol:绝对容差限；rtol: 相对容差限
# |a-b| <= (atol + rtol * |b|)
try:
    print "Pass:", np.testing.assert_allclose([0, 0.123456789, np.nan], [0, 0.123456780, np.nan], rtol=1e-7, atol=0)
    print "Fail:", np.testing.assert_allclose([0, 0.123456789, np.nan], [0, 0.123456780, np.nan], rtol=1e-7, atol=0)
except AssertionError, ae:
    print ae

#数组排序
#两个一致，第2个严格小于第1个
try:
    print "Pass:", np.testing.assert_array_less([0, 0.123456789, np.nan], [1, 0.123456780, np.nan])
    print "Pass:", np.testing.assert_array_less([0, 0.123456789, np.nan], [0, 0.123456780, np.nan])
except AssertionError, ae:
    print ae

#对象比较
try:
    print "Equal?", np.testing.assert_equal((1,2), (1,2))
    print "Equal?", np.testing.assert_equal((1,2), (1,3))
except AssertionError, ae:
    print ae

#字符串比较
try:
    print "Pass:", np.testing.assert_string_equal("Numpy", "Numpy")
    print "Fail:", np.testing.assert_string_equal("Numpy", "NumPy")
except AssertionError, ae:
    print ae

#浮点数比较
#单eps, 浮点数精度（因为浮点数在机器里以不精确的方式表示
try:
    eps = np.finfo(float).eps
    print "EPS:", eps

    print "1", np.testing.assert_array_almost_equal_nulp(1.0, 1.0 + eps)
    print "2", np.testing.assert_array_almost_equal_nulp(1.0, 1.0 + 2*eps)
except AssertionError, ae:
    print ae

#多ULP的浮点数比较
try:
    print "1", np.testing.assert_array_max_ulp(1.0, 1.0 + eps)
    print "2", np.testing.assert_array_max_ulp(1.0, 1.0 + 2*eps, maxulp=2)
except AssertionError, ae:
    print ae

