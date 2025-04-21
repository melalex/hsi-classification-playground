from sklearn.discriminant_analysis import StandardScaler


def scale_image(image):
    h, w, c = image.shape
    img_reshaped = image.reshape(-1, c)
    scaler = StandardScaler()
    img_scaled = scaler.fit_transform(img_reshaped)

    return scaler, img_scaled.reshape(h, w, c)
