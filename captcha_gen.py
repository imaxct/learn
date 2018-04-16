from captcha.image import ImageCaptcha

img = ImageCaptcha(fonts=['./Times_New_Roman_Bold.ttf'], width=60, height=20, font_sizes=[14])
data = img.generate('1234')
img.write('1234', 'out.png')