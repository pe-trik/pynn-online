#%%
import numpy as np
import matplotlib.pyplot as plt# create mock data

attn=[[[2.2717e-01, 1.7285e-01, 1.4160e-01, 9.2896e-02, 6.1890e-02,
          3.9642e-02, 2.7344e-02, 2.4323e-02, 1.9669e-02, 1.2970e-02,
          7.7438e-03, 5.3558e-03, 4.6883e-03, 3.9597e-03, 4.1771e-03,
          5.5008e-03, 7.8659e-03, 1.0567e-02, 1.2619e-02, 1.1650e-02,
          7.0114e-03, 4.7722e-03, 5.6114e-03, 7.6675e-03, 8.7051e-03,
          7.7744e-03, 7.0114e-03, 9.2163e-03, 1.0834e-02, 1.0300e-02,
          8.7967e-03, 6.8626e-03, 3.2349e-03, 1.6747e-03, 1.1425e-03,
          8.4400e-04, 5.6028e-04, 3.3331e-04, 2.1744e-04, 1.6284e-04,
          1.3781e-04, 1.2314e-04, 1.1319e-04, 1.0544e-04, 1.0389e-04,
          9.9719e-05, 9.9301e-05, 9.9003e-05, 9.8944e-05, 9.5904e-05,
          9.5308e-05, 9.9123e-05, 1.0359e-04, 1.1683e-04, 1.2267e-04,
          1.1212e-04, 1.5748e-04, 1.8644e-04, 7.4804e-05, 4.2737e-05,
          3.0458e-05, 2.1636e-05, 1.6510e-05, 1.6868e-05, 1.8775e-05,
          2.7657e-05, 5.6863e-05, 1.4424e-04, 1.9753e-04],
         [1.3752e-03, 1.7519e-03, 6.0768e-03, 2.4612e-02, 4.8737e-02,
          6.8481e-02, 7.3730e-02, 7.0923e-02, 6.2622e-02, 5.7007e-02,
          5.2490e-02, 5.0903e-02, 5.2490e-02, 5.2917e-02, 6.0425e-02,
          7.4890e-02, 8.1665e-02, 7.4341e-02, 4.3701e-02, 1.5762e-02,
          4.2114e-03, 1.5221e-03, 1.1663e-03, 1.1530e-03, 1.2321e-03,
          1.4019e-03, 1.2760e-03, 1.7900e-03, 2.3212e-03, 2.3708e-03,
          1.9970e-03, 1.5068e-03, 7.9393e-04, 4.9686e-04, 3.8671e-04,
          3.0375e-04, 1.9944e-04, 1.0908e-04, 6.4731e-05, 4.5121e-05,
          3.6597e-05, 3.2246e-05, 2.9504e-05, 2.7239e-05, 2.6286e-05,
          2.5213e-05, 2.5213e-05, 2.5272e-05, 2.5511e-05, 2.5272e-05,
          2.5868e-05, 2.7001e-05, 2.8729e-05, 3.2067e-05, 3.5703e-05,
          3.9876e-05, 5.9009e-05, 5.2214e-05, 1.3888e-05, 9.9540e-06,
          1.1027e-05, 1.1623e-05, 9.9540e-06, 9.8348e-06, 1.0550e-05,
          1.2159e-05, 1.5557e-05, 2.5094e-05, 2.5630e-05],
         [3.4869e-05, 4.1842e-05, 9.5487e-05, 1.7965e-04, 1.9455e-04,
          1.6379e-04, 1.2636e-04, 9.7394e-05, 5.4955e-05, 2.4557e-05,
          2.0266e-05, 1.9073e-05, 2.1815e-05, 2.9922e-05, 6.1154e-05,
          1.9801e-04, 1.1873e-03, 4.9286e-03, 1.4267e-02, 2.8824e-02,
          5.8685e-02, 1.0583e-01, 1.4136e-01, 1.4355e-01, 1.3745e-01,
          1.2274e-01, 8.5388e-02, 5.7526e-02, 3.1158e-02, 1.5244e-02,
          1.1154e-02, 1.1070e-02, 9.2850e-03, 4.8904e-03, 2.9049e-03,
          1.8101e-03, 1.1663e-03, 7.6485e-04, 5.3930e-04, 4.1628e-04,
          3.5501e-04, 3.2449e-04, 3.0398e-04, 2.8133e-04, 2.7466e-04,
          2.6274e-04, 2.5916e-04, 2.5272e-04, 2.4235e-04, 2.3472e-04,
          2.4509e-04, 2.6488e-04, 3.0398e-04, 3.6931e-04, 4.3154e-04,
          4.9877e-04, 5.5838e-04, 3.5119e-04, 1.5104e-04, 7.3135e-05,
          3.1233e-05, 1.6153e-05, 1.5020e-05, 1.5676e-05, 2.3723e-05,
          4.9174e-05, 7.8619e-05, 1.1539e-04, 1.5581e-04],
         [1.0490e-05, 1.0729e-05, 2.7478e-05, 5.7280e-05, 7.0989e-05,
          6.9559e-05, 6.4194e-05, 5.5492e-05, 3.8624e-05, 2.6286e-05,
          2.3663e-05, 2.2233e-05, 2.5094e-05, 3.0518e-05, 4.1962e-05,
          7.0691e-05, 1.7703e-04, 4.2272e-04, 6.5851e-04, 6.3992e-04,
          4.7636e-04, 3.4952e-04, 3.1447e-04, 3.4809e-04, 1.3771e-03,
          1.1131e-02, 3.1952e-02, 5.7190e-02, 8.5510e-02, 1.1780e-01,
          1.4600e-01, 1.5613e-01, 1.2012e-01, 7.5500e-02, 4.9316e-02,
          3.1097e-02, 1.8280e-02, 1.0132e-02, 6.1455e-03, 4.4899e-03,
          3.8853e-03, 3.6411e-03, 3.4752e-03, 3.2902e-03, 3.2654e-03,
          3.1509e-03, 3.1509e-03, 3.1204e-03, 3.1338e-03, 3.2330e-03,
          3.4485e-03, 3.5782e-03, 3.8242e-03, 4.1847e-03, 4.5395e-03,
          5.7297e-03, 8.3008e-03, 4.1771e-03, 8.8549e-04, 5.3072e-04,
          3.3522e-04, 2.0695e-04, 1.3721e-04, 9.5427e-05, 8.9228e-05,
          9.0241e-05, 8.7440e-05, 1.0562e-04, 1.1045e-04],
         [4.0531e-06, 2.3246e-06, 4.8876e-06, 1.0133e-05, 1.2457e-05,
          1.2279e-05, 1.1861e-05, 1.0729e-05, 8.4043e-06, 4.5896e-06,
          3.3975e-06, 2.8610e-06, 3.0398e-06, 3.6955e-06, 5.1856e-06,
          8.8215e-06, 2.0385e-05, 4.6313e-05, 7.8440e-05, 6.6817e-05,
          3.5405e-05, 1.8895e-05, 1.2279e-05, 1.0073e-05, 2.8849e-05,
          1.8179e-04, 3.9101e-04, 4.7207e-04, 4.1127e-04, 4.6468e-04,
          1.0138e-03, 3.9864e-03, 1.2970e-02, 1.8219e-02, 2.0721e-02,
          2.3026e-02, 2.7145e-02, 3.2227e-02, 3.5950e-02, 3.8116e-02,
          3.9185e-02, 3.9642e-02, 3.9490e-02, 3.8879e-02, 3.9185e-02,
          3.8727e-02, 3.9337e-02, 3.9490e-02, 3.9795e-02, 4.0100e-02,
          4.1718e-02, 4.4220e-02, 4.7272e-02, 5.1910e-02, 5.6793e-02,
          5.5237e-02, 5.6122e-02, 2.7039e-02, 4.5547e-03, 2.1591e-03,
          1.2598e-03, 6.0892e-04, 3.1137e-04, 2.3162e-04, 2.3162e-04,
          2.3234e-04, 1.4257e-04, 1.7715e-04, 2.7418e-04],
         [8.1658e-06, 7.2122e-06, 1.1384e-05, 1.7822e-05, 2.1279e-05,
          2.2173e-05, 2.1517e-05, 1.5914e-05, 7.4506e-06, 2.3246e-06,
          1.2517e-06, 1.0133e-06, 1.0133e-06, 1.1921e-06, 1.4901e-06,
          2.5630e-06, 6.8545e-06, 1.5140e-05, 1.5438e-05, 7.6890e-06,
          3.0994e-06, 1.5497e-06, 1.0729e-06, 8.9407e-07, 1.3113e-06,
          3.6955e-06, 9.2387e-06, 1.6153e-05, 1.3590e-05, 1.3351e-05,
          1.5318e-05, 2.5570e-05, 4.6551e-05, 7.6830e-05, 1.2130e-04,
          1.5223e-04, 1.3113e-04, 8.1956e-05, 5.2631e-05, 4.0233e-05,
          3.6180e-05, 3.6240e-05, 3.7253e-05, 3.7789e-05, 3.8862e-05,
          3.9816e-05, 4.2140e-05, 4.4525e-05, 4.7624e-05, 5.4419e-05,
          6.7830e-05, 8.4817e-05, 1.1724e-04, 1.8084e-04, 3.3784e-04,
          1.6212e-03, 1.4648e-02, 6.7993e-02, 1.2122e-01, 1.6833e-01,
          2.1863e-01, 1.4001e-01, 9.5154e-02, 6.3354e-02, 4.7852e-02,
          3.1616e-02, 1.3184e-02, 7.9956e-03, 6.2637e-03],
         [1.2732e-04, 6.6757e-05, 6.8069e-05, 7.8797e-05, 1.0419e-04,
          9.8646e-05, 8.2254e-05, 6.6519e-05, 6.1035e-05, 5.6744e-05,
          4.7088e-05, 3.8207e-05, 3.9399e-05, 3.7789e-05, 3.6776e-05,
          3.6716e-05, 4.8101e-05, 7.2062e-05, 1.2159e-04, 1.0484e-04,
          5.2929e-05, 2.1696e-05, 9.8944e-06, 6.4969e-06, 1.1325e-05,
          3.8803e-05, 5.9128e-05, 7.2777e-05, 7.2300e-05, 6.7830e-05,
          7.1347e-05, 9.3043e-05, 1.0180e-04, 8.8811e-05, 1.0526e-04,
          1.1355e-04, 9.3222e-05, 6.1989e-05, 4.2081e-05, 3.2246e-05,
          2.9266e-05, 3.0518e-05, 3.2485e-05, 3.3677e-05, 3.6657e-05,
          3.9220e-05, 4.3511e-05, 4.7803e-05, 5.1796e-05, 5.6744e-05,
          7.1466e-05, 9.4354e-05, 1.2183e-04, 1.7238e-04, 2.6679e-04,
          5.5265e-04, 1.4868e-03, 3.1261e-03, 2.3365e-03, 5.3940e-03,
          1.3840e-02, 4.3457e-02, 5.3040e-02, 9.2346e-02, 1.5051e-01,
          2.3401e-01, 1.8738e-01, 1.5527e-01, 5.3467e-02],
         [1.4715e-03, 1.3952e-03, 9.8133e-04, 5.1308e-04, 3.7813e-04,
          2.5249e-04, 1.6296e-04, 9.9719e-05, 4.1425e-05, 1.7345e-05,
          1.0848e-05, 8.8811e-06, 9.0599e-06, 8.8811e-06, 1.0073e-05,
          1.3530e-05, 2.8670e-05, 5.0902e-05, 7.6175e-05, 7.5400e-05,
          5.9247e-05, 4.5419e-05, 3.8087e-05, 3.4153e-05, 3.3557e-05,
          4.3929e-05, 6.2466e-05, 7.2241e-05, 4.5538e-05, 2.7418e-05,
          2.1160e-05, 2.4676e-05, 2.5630e-05, 2.2531e-05, 2.3782e-05,
          2.0802e-05, 1.2398e-05, 6.1393e-06, 3.5763e-06, 2.5034e-06,
          2.0266e-06, 1.9073e-06, 1.9073e-06, 1.8477e-06, 1.9073e-06,
          1.9670e-06, 2.0862e-06, 2.2054e-06, 2.3842e-06, 2.6822e-06,
          3.2187e-06, 3.6955e-06, 4.3511e-06, 5.7817e-06, 7.9870e-06,
          1.5974e-05, 4.6134e-05, 1.0616e-04, 1.1456e-04, 1.3208e-04,
          1.2219e-04, 1.4341e-04, 1.8036e-04, 3.6263e-04, 8.3828e-04,
          3.6583e-03, 2.3300e-02, 3.3716e-01, 6.2744e-01],
         [7.3051e-04, 7.1430e-04, 4.3821e-04, 1.8811e-04, 9.0420e-05,
          4.8637e-05, 2.6345e-05, 1.3530e-05, 4.8876e-06, 1.5497e-06,
          8.3447e-07, 5.9605e-07, 4.1723e-07, 3.5763e-07, 4.1723e-07,
          7.1526e-07, 2.5034e-06, 9.7752e-06, 1.9014e-05, 1.3351e-05,
          5.5432e-06, 2.8610e-06, 2.5034e-06, 2.6822e-06, 2.8014e-06,
          3.6359e-06, 6.4373e-06, 8.2850e-06, 4.6492e-06, 2.8014e-06,
          2.3842e-06, 3.5763e-06, 4.8876e-06, 4.6492e-06, 4.7088e-06,
          4.1127e-06, 2.4438e-06, 1.0729e-06, 5.3644e-07, 3.5763e-07,
          2.9802e-07, 2.9802e-07, 2.9802e-07, 2.9802e-07, 2.9802e-07,
          2.9802e-07, 2.9802e-07, 2.9802e-07, 2.9802e-07, 3.5763e-07,
          4.7684e-07, 5.9605e-07, 7.1526e-07, 9.5367e-07, 1.4305e-06,
          4.4107e-06, 1.8477e-05, 2.8670e-05, 2.1815e-05, 2.4021e-05,
          3.2485e-05, 4.3154e-05, 5.0783e-05, 8.6665e-05, 2.3150e-04,
          1.3018e-03, 1.3962e-02, 2.1936e-01, 7.6270e-01],
         [9.8515e-04, 9.7656e-04, 6.8617e-04, 5.4073e-04, 5.5504e-04,
          5.1975e-04, 3.9124e-04, 2.7394e-04, 2.2447e-04, 2.3019e-04,
          2.0838e-04, 1.9956e-04, 2.1625e-04, 2.4056e-04, 2.8539e-04,
          3.5000e-04, 4.3845e-04, 5.8270e-04, 4.1771e-04, 1.4603e-04,
          4.5836e-05, 1.8477e-05, 1.2398e-05, 1.1206e-05, 1.5259e-05,
          3.7134e-05, 7.1466e-05, 9.8407e-05, 8.7738e-05, 7.5340e-05,
          7.0095e-05, 8.3387e-05, 8.7917e-05, 7.2181e-05, 6.7532e-05,
          5.5552e-05, 3.4988e-05, 1.8060e-05, 1.0550e-05, 7.5102e-06,
          6.2585e-06, 6.0201e-06, 6.0201e-06, 5.9009e-06, 6.0797e-06,
          6.2585e-06, 6.4969e-06, 6.6161e-06, 6.6161e-06, 7.0333e-06,
          8.4639e-06, 9.8944e-06, 1.2100e-05, 1.5557e-05, 2.2173e-05,
          4.8637e-05, 1.3220e-04, 1.8358e-04, 1.0419e-04, 1.3196e-04,
          1.8823e-04, 3.0255e-04, 4.4799e-04, 5.8365e-04, 1.0691e-03,
          3.4866e-03, 2.2324e-02, 2.2803e-01, 7.3340e-01]]]

#%%
plt.figure(figsize=(10, 10))
plt.pcolormesh(attn[0])
plt.colorbar()

#%%
padding=0.05
a = attn[0][1:]
cs = np.cumsum(a, axis=1)
ep = np.less_equal(cs, 1-padding)#.sum(axis=1)
sp = sp = np.less_equal(cs, padding)#.sum(axis=1)
plt.figure(figsize=(10, 10))
plt.pcolormesh(ep^sp)
plt.colorbar()