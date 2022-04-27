def histo_eq(image):
    img_hsv = cv.cvtColor(image,cv.COLOR_RGB2HSV)
    hist,bins = np.histogram(img_hsv.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img = cdf[image]
    return img
